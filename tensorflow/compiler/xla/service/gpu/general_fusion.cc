/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/general_fusion.h"

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

bool IsParameter(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kParameter;
}

bool IsFusion(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kFusion;
}

bool IsGTE(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kGetTupleElement;
}

bool IsTuple(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kTuple;
}

bool IsRoot(const HloInstruction* inst) {
  return inst->parent()->root_instruction() == inst;
}

void FuseTwoSiblings(HloInstruction* left_sibling, HloInstruction* right_sibling) {
  LOG(INFO) << "Fusing... " << left_sibling->name() << " and " << right_sibling->name();

  // Create the replacement map
  // replace parameter with the real external input
  std::unordered_map<HloInstruction*, HloInstruction*> real_external;
  for (int i = 0; i < left_sibling->operand_count(); ++i) {
    HloInstruction* operand = left_sibling->mutable_operand(i);
    if (IsFusion(left_sibling)) {
      HloComputation* left_comp = left_sibling->fused_instructions_computation();
      real_external[left_comp->parameter_instruction(i)] = operand;
    }
    else {
      real_external[operand] = operand;
    }
  }
  for (int i = 0; i < right_sibling->operand_count(); ++i) {
    HloInstruction* operand = right_sibling->mutable_operand(i);
    if (IsFusion(right_sibling)) {
      HloComputation* right_comp = right_sibling->fused_instructions_computation();
      real_external[right_comp->parameter_instruction(i)] = operand;
    }
    else {
      real_external[operand] = operand;
    }    
  }

  std::vector<HloInstruction*> post_order;
  // Get list of new tuple operand params
  std::vector<HloInstruction*> tuple_operands;
  // extern_rep represents the instruction that external users connect to. 
  // Usually it's just the instruction, but for a fusion, it refers to the 
  // gte for that tuple operand.  
  std::unordered_map<HloInstruction*, HloInstruction*> extern_rep; 

  auto process_sibling = [&](HloInstruction* sibling) {
    // Put everything in top order
    if (IsFusion(sibling)) {
      auto sibling_post_order =
          sibling->fused_instructions_computation()->MakeInstructionPostOrder();
      post_order.insert(post_order.end(),
                        std::make_move_iterator(sibling_post_order.begin()),
                        std::make_move_iterator(sibling_post_order.end()));
    } else {
      post_order.push_back(sibling);
    }

    // Assemble new tuple operands and get the external rep of the tuple operand
    // external rep = what users of that operand actually connect to, e.g. a 
    //                GTE that accesses that tuple operand if it's multi
    //                output, else just the instruction itself
    if (IsFusion(sibling)) {
      HloInstruction* sibling_root = sibling->fused_expression_root();

      if (IsTuple(sibling_root)) {
        // Get external users
        for (auto* user_inst : sibling->users()) {
          auto* gte_inst = Cast<HloGetTupleElementInstruction>(user_inst);
          auto* tuple_operand = sibling_root->mutable_operand(gte_inst->tuple_index());
          extern_rep[tuple_operand] = gte_inst;
        }

        for (int i = 0; i < sibling_root->operand_count(); ++i) {
          HloInstruction* tuple_operand = sibling_root->mutable_operand(i);
          tuple_operands.push_back(tuple_operand);
        }
      }
      else {
        tuple_operands.push_back(sibling_root);
        extern_rep[sibling_root] = sibling;
      }
    }
    else {
      tuple_operands.push_back(sibling);
      extern_rep[sibling] = sibling;
    }
  };

  process_sibling(left_sibling);
  process_sibling(right_sibling);

  // now make params and clone instrutions
  HloComputation::Builder builder("fuse_two_siblings");
  int param_index = 0;
  std::vector<HloInstruction*> uniq_extern_operands;
  std::unordered_map<HloInstruction*, HloInstruction*> extern_operand_param_map;
  std::unordered_map<HloInstruction*, HloInstruction*> old_to_new;

  for (auto* inst : post_order) {
    // don't clone parameter & tuple
    if (IsParameter(inst) || IsTuple(inst)) {
      continue;
    }

    const auto& operands = inst->operands();
    std::vector<HloInstruction*> new_operands;
    HloInstruction* new_operand;
    for (auto* operand : operands) {
      if (old_to_new.find(operand) != old_to_new.end()) {
        // Then replace with the newly cloned instructions
        new_operand = old_to_new[operand];
      } 
      // Because we are in post order, if we haven't seen it yet then that 
      // means that this is an external instruction
      else {
        HloInstruction* external_inst = real_external[operand];
        if (extern_operand_param_map.find(external_inst) != extern_operand_param_map.end()) {
          new_operand = extern_operand_param_map[external_inst];
        }
        else {
          // Otherwise create a new param for it
          auto* param = builder.AddInstruction(HloInstruction::CreateParameter(
              param_index, external_inst->shape(),
              "param" + std::to_string(param_index)));
          extern_operand_param_map[external_inst] = param;

          uniq_extern_operands.push_back(external_inst);
          
          new_operand = param;
          param_index++;
        }
      } 
      new_operands.push_back(new_operand);
    }
    old_to_new[inst] = builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), new_operands));
  }

  std::vector<HloInstruction*> cloned_tuple_operands;
  for (auto* inst : tuple_operands) {
    cloned_tuple_operands.push_back(old_to_new[inst]);
  }


  HloInstruction* root = nullptr;
  CHECK_GT(tuple_operands.size(), 1);
  root = builder.AddInstruction(HloInstruction::CreateTuple(cloned_tuple_operands));

  HloInstruction::FusionKind fusion_kind = (IsInputFusible(*left_sibling) || IsInputFusible(*right_sibling))
                                                ? HloInstruction::FusionKind::kInput
                                                : HloInstruction::FusionKind::kLoop;

  HloComputation* comp =
      left_sibling->parent()->parent()->AddEmbeddedComputation(builder.Build(root));

  HloInstruction* fusion =
      left_sibling->parent()->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), fusion_kind,
          uniq_extern_operands, comp));

  LOG(INFO) << "Sibling Fused into new fusion: " << fusion->name();

  for (int i = 0; i < tuple_operands.size(); ++i) {
    auto* tuple_operand = tuple_operands[i];

    if (extern_rep.find(tuple_operand) != extern_rep.end()) {
      HloInstruction* new_gte = left_sibling->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(fusion, i));
      extern_rep[tuple_operand]->ReplaceAllUsesWith(new_gte);
    }
  }
}

void FuseTwo(HloInstruction* consumer, HloInstruction* producer) {
  std::unordered_map<HloInstruction*, HloInstruction*> replacements;
  std::vector<HloInstruction*> extra;

  // Create the replacement map
  // replace parameter with the real external input
  for (int i = 0; i < producer->operand_count(); ++i) {
    HloInstruction* operand = producer->mutable_operand(i);
    if (IsFusion(producer)) {
      HloComputation* p_comp = producer->fused_instructions_computation();
      replacements[p_comp->parameter_instruction(i)] = operand;
    }
  }

  for (int i = 0; i < consumer->operand_count(); ++i) {
    HloInstruction* operand = consumer->mutable_operand(i);
    HloInstruction* key;
    HloInstruction* value;
    if (IsFusion(consumer)) {
      HloComputation* c_comp = consumer->fused_instructions_computation();
      key = c_comp->parameter_instruction(i);
    } else {
      key = operand;
    }
    if (!IsGTE(operand)) {  // not a GTE
      if (operand == producer) {
        if (IsFusion(producer)) {
          value = producer->fused_expression_root();
        } else {
          value = producer;
        }
      } else {
        value = operand;
      }
    } else {  // operand is a GTE
      if (operand->operand(0) == producer) {
        if (IsFusion(producer)) {
          // If producer is a fused computation, connect directly without GTE
          value = producer->fused_expression_root()->mutable_operand(
              operand->tuple_index());
        } else {
          // when this is an instruction that generates a tuple
          value = operand;
          // gte is useful only when the instruction provides a tuple output
          // but is not fusion
          extra.push_back(operand);
        }
      } else {
        value = operand;
      }
    }
    if (key != value) {
      replacements[key] = value;
    }
  }

  std::unordered_map<
      HloInstruction*,
      std::unordered_map<HloInstruction*, std::vector<HloInstruction*>>>
      replace_users;
  // new fusion root
  std::vector<HloInstruction*> new_fused_root;
  // indices of the instructions in the new_fusion_root that makes the root of
  // the computation
  std::vector<int> root_indices;

  auto track_external_users =
      [&](HloInstruction* inst, HloInstruction* to_replace, bool for_producer) {
        std::vector<HloInstruction*> users;
        users.reserve(to_replace->user_count());
        if (for_producer) {
          for (auto* u : to_replace->users()) {
            if (u != consumer && (u->dry() || u->rewrite())) {
              users.push_back(u);
              LOG(ERROR) << "inst: " << inst->name() << " user " << u->name();
            }
          }
        }
        // keep only external user
        if (for_producer && !users.empty()) {
          // Only add to the new root instruction tuple if the shape is 
          // compatible. Since we need to ensure that the output shapes
          // are consistent.
          if (ShapesCompatibleForMultiOutputFusion(*inst, *consumer)) {
            new_fused_root.push_back(inst);
            replace_users[inst][to_replace] = users;
          }
        } else if (!for_producer) {
          new_fused_root.push_back(inst);
          std::vector<HloInstruction*> users;
          std::copy_if(to_replace->users().begin(), to_replace->users().end(),
                       std::back_inserter(users),
                       [](HloInstruction* inst) { return inst->dry() || inst->rewrite(); });
          replace_users[inst][to_replace] = users;
        }
      };

  // replacement for producer
  auto track_uses = [&](HloInstruction* inst) {
    bool is_consumer = (inst == consumer);
    int num_root_inst = new_fused_root.size();
    if (IsFusion(inst)) {
      HloInstruction* root = inst->fused_expression_root();
      if (IsTuple(root)) {
        std::vector<bool> not_seen(root->mutable_operands().size(), true);
        for (auto* gte : inst->users()) {
          if (!(gte->dry() || gte->rewrite())) {
            continue;
          }
          auto* key = root->mutable_operand(gte->tuple_index());
          LOG(INFO) << key->name();
          track_external_users(key, gte, !is_consumer);
          not_seen.at(gte->tuple_index()) = false;
        }

        // Add back outputs that do not get used
        for (int i = 0; i < not_seen.size(); i++) {
          if (not_seen.at(i)) {
            new_fused_root.push_back(root->mutable_operand(i));
          }
        }

      } else {
        LOG(INFO) << root->name();
        track_external_users(root, inst, !is_consumer);
      }
    } else {
      LOG(INFO) << inst->name();
      track_external_users(inst, inst, !is_consumer);
    }
    if (is_consumer) {
      if (IsRoot(inst)) {
        root_indices.resize(new_fused_root.size() - num_root_inst);
        std::iota(root_indices.begin(), root_indices.end(), num_root_inst);
      }
    }
  };
  track_uses(producer);
  track_uses(consumer);
  for (auto* gte : extra) {
    track_external_users(gte, gte, true);
  }
  CHECK(!new_fused_root.empty());

  std::vector<HloInstruction*> post_order;
  std::unordered_map<HloInstruction*, HloInstruction*> old_to_new;
  // put everything in topo order
  if (IsFusion(producer)) {
    post_order =
        producer->fused_instructions_computation()->MakeInstructionPostOrder();
  } else {
    post_order.push_back(producer);
  }
  post_order.insert(post_order.end(), extra.begin(), extra.end());
  if (IsFusion(consumer)) {
    auto consumer_comp =
        consumer->fused_instructions_computation()->MakeInstructionPostOrder();
    post_order.insert(post_order.end(),
                      std::make_move_iterator(consumer_comp.begin()),
                      std::make_move_iterator(consumer_comp.end()));
  } else {
    post_order.push_back(consumer);
  }

  // now clone
  HloComputation::Builder builder("fuse_two");
  int param_index = 0;
  std::unordered_map<HloInstruction*, HloInstruction*> extern_to_param;
  std::vector<HloInstruction*> kfusion_operands;
  for (auto* inst : post_order) {
    // don't clone parameter & tuple
    if (IsParameter(inst) || IsTuple(inst)) {
      continue;
    }
    const auto& operands = inst->operands();
    std::vector<HloInstruction*> new_operands;
    for (auto* operand : operands) {
      // First find the replacement
      if (replacements.find(operand) != replacements.end()) {
        operand = replacements[operand];
      }

      if (old_to_new.find(operand) != old_to_new.end()) {
        // Then replace with the newly cloned instructions
        operand = old_to_new[operand];
      } else if (extern_to_param.find(operand) != extern_to_param.end()) {
        // If not found, means it is external instruction,
        // If a param is already associated with it, reuse the param
        operand = extern_to_param[operand];
      } else {
        // Otherwise create a new param for it
        auto* param = builder.AddInstruction(HloInstruction::CreateParameter(
            param_index, operand->shape(),
            "param" + std::to_string(param_index)));
        extern_to_param[operand] = param;
        kfusion_operands.push_back(operand);
        operand = param;
        param_index++;
      }
      new_operands.push_back(operand);
    }
    old_to_new[inst] = builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), new_operands));
    LOG(INFO) << "Cloning " << inst->name() << " to "
              << old_to_new[inst]->name();
  }

  HloInstruction* root = nullptr;
  std::vector<HloInstruction*> tuple_operands;
  for (auto* inst : new_fused_root) {
    tuple_operands.push_back(old_to_new[inst]);
  }
  if (tuple_operands.size() > 1) {
    root = builder.AddInstruction(HloInstruction::CreateTuple(tuple_operands));
  } else {
    root = tuple_operands[0];
  }
  HloComputation* comp =
      producer->parent()->parent()->AddEmbeddedComputation(builder.Build(root));
  HloInstruction* fusion =
      consumer->parent()->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), ChooseFusionKind(*producer, *consumer),
          kfusion_operands, comp));
  LOG(INFO) << "Fused into new fusion: " << fusion->name();

  for (int i = 0; i < new_fused_root.size(); ++i) {
    auto* inst = new_fused_root[i];
    for (auto& kv : replace_users[inst]) {
      auto* to_replace = kv.first;
      auto& users = kv.second;
      if (new_fused_root.size() > 1) {
        HloInstruction* gte = consumer->parent()->AddInstruction(
            HloInstruction::CreateGetTupleElement(fusion, i));
        to_replace->ReplaceUsesWith(users, gte);
      } else {
        to_replace->ReplaceUsesWith(users, fusion);
      }
    }
  }

  if (IsRoot(consumer)) {
    HloInstruction* new_root = nullptr;
    if (root_indices.size() == new_fused_root.size()) {
      new_root = fusion;
    } else {
      std::vector<HloInstruction*> operands;
      for (int i : root_indices) {
        HloInstruction* gte = consumer->parent()->AddInstruction(
            HloInstruction::CreateGetTupleElement(fusion, i));
        operands.push_back(gte);
      }
      if (operands.size() == 1) {
        new_root = operands[0];
      } else {
        new_root = consumer->parent()->AddInstruction(
            HloInstruction::CreateTuple(operands));
      }
    }
    consumer->parent()->set_root_instruction(new_root);
  }
}

bool GeneralFusion::HasCycle(HloInstruction* producer,
                             HloInstruction* consumer) {
  for (const HloInstruction* operand : consumer->operands()) {
    if (operand == producer) {
      continue;
    }
    else if (operand->opcode() == HloOpcode::kGetTupleElement) {
      if (operand->operand(0) == producer) {
        continue;
      }
    }
    // For consumer's every other operand, check if reachable by producer.
    // If so then creates a cycle.
    if (reachability_->IsPresent(producer) &&
        reachability_->IsPresent(operand) &&
        reachability_->IsReachable(producer, operand)) {
      return true;
    }
  }
  return false;
}

bool GeneralFusion::ShouldFuseProducerIntoConsumer(HloInstruction* producer,
                                                   HloInstruction* consumer) {
  if (!producer->IsFusible()) {
    return false;
  }
  if (HasCycle(producer, consumer)) {
    return false;
  }
  if (producer->opcode() == HloOpcode::kCustomCall ||
      consumer->opcode() == HloOpcode::kCustomCall) {
    return false;
  }
  bool fusible = true;

  int external_user_count = 0;
  // We need to check only for users that aren't parameters in the consumer.
  for (auto* user : producer->users()) {
    if (IsGTE(user)) {
      for (auto* gte_user : user->users()) {
        if (gte_user != consumer) {
          external_user_count += 1;
        }
      }
    }
    else {
      if (user != consumer) {
        external_user_count += 1;
      }
    }
  }

  bool shapes_compatible = ShapesCompatibleForMultiOutputFusion(*consumer, 
                                                                *producer);
  // If shapes are not compatible, we don't fuse in the other consumer->user
  // outputs as multiple outputs
  // This is to ensure multi-output fusion has identical output shapes
  if ((external_user_count > 0) && shapes_compatible) {
    // MultiOutputFusion
    fusible = IsProducerConsumerMultiOutputFusible(*producer, *consumer, 
                                                   /*general_fusion*/ true);
  } else {
    FusionDecision decision = IsProducerConsumerFusible(*producer, *consumer,
                                                   /*general_fusion*/ true);
    fusible = decision.CanFuse();
  }
  return fusible;
}

bool IsSiblingFusionValidCandidate(HloInstruction* candidate) {
  if (candidate->opcode() == HloOpcode::kFusion) {
    if (candidate->fused_expression_root()->opcode() ==
          HloOpcode::kDynamicUpdateSlice) {
      return false;
    }
  }

  if (!IsFusibleAsMultiOutputFusionRoot(*candidate)) {
    return false;
  }

  if (candidate->opcode() == HloOpcode::kCustomCall) {
    return false;
  }

  return true;
}

// Checks if its legal to fuse when left <-> right, specific to left/right order
// 
bool LegalToSiblingFuse(HloInstruction* left, HloInstruction* right) {
  // If we're fusing fusions only do it if the fusion kind matches. Loop fusions
  // merge into bigger loop fusions and input (reduce) fusions become fusions
  // with multiple reduce outputs. We could fuse reduce and loop fusions
  // together too (the result being an input fusion) if we find cases where this
  // improves things. Also disable fusing standalone input-fusible reduces into
  // loop fusions.
  if (!IsSiblingFusionValidCandidate(left) || 
      !IsSiblingFusionValidCandidate(right)) {
    return false;
  }

  if (left->opcode() == HloOpcode::kFusion && 
      right->opcode() == HloOpcode::kFusion &&
      left->fusion_kind() != right->fusion_kind()) {
    return false;
  }

  if (IsReductionFromOrToContiguousDimensions(*right) &&
       left->IsLoopFusion()) {
    return false;
  }
  if (IsReductionFromOrToContiguousDimensions(*left) &&
       right->IsLoopFusion()) {
    return false;
  }

  if (!ShapesCompatibleForMultiOutputFusion(*left, *right)) {
    return false;
  }

  if (left == right) {
    return false;
  }

  return true;
}

bool GeneralFusion::DoGeneralFusion(HloComputation* comp) {
  reachability_.reset();
  reachability_ = HloReachabilityMap::Build(comp);
  std::vector<HloInstruction*> post_order = comp->MakeInstructionPostOrder();
  int count = 0;
  LOG(INFO) << "Performing Vertical fusion for comp " << comp->name();
  while (!post_order.empty()) {
    HloInstruction* inst = post_order.back();
    post_order.pop_back();
    if (IsGTE(inst) || IsTuple(inst) || !inst->IsFusible()) {
      continue;
    }
    LOG(INFO) << "Considering consumer " << inst->name();
    HloInstructionSet uniq_operands;
    for (const auto* const_operand : inst->unique_operands()) {
      auto* operand = const_cast<HloInstruction*>(const_operand);
      if (IsGTE(operand)) {
        LOG(INFO) << "Considering GTE producer " << operand->name();
        // If GTE, trace back one more step
        uniq_operands.insert(operand->mutable_operand(0));
      } else {
        LOG(INFO) << "Considering normal producer " << operand->name();
        uniq_operands.insert(operand);
      }
    }
    for (auto* operand : uniq_operands) {
      LOG(INFO) << "Considering " << operand->name() << " and " << inst->name();
      if (!ShouldFuseProducerIntoConsumer(/*producer=*/operand,
                                          /*consumer=*/inst)) {
        continue;
      }
      LOG(INFO) << "Fusing " << operand->name() << " into " << inst->name();
      FuseTwo(/*consumer=*/inst, /*producer=*/operand);
      count++;
    }
  }

  LOG(INFO) << "Performing Sibling fusion for comp " << comp->name();
  post_order = comp->MakeInstructionPostOrder();
  while (!post_order.empty()) {
    HloInstruction* inst = post_order.back();
    post_order.pop_back();

    if (!(inst->dry() || inst->rewrite())) {
      continue;
    }

    if (inst->IsDead()) {
      continue;
    }
    std::vector<HloInstruction*> siblings;
    for (auto* user : inst->users()) {
      if (IsGTE(user)) {
        LOG(INFO) << "Collecting siblings: GTE user " << user->name();
        // If GTE, trace back one more step
        if (user->users().size() > 0) {
          HloInstruction* actual_sibling = user->users()[0];
          if (actual_sibling->dry() || actual_sibling->rewrite()) {
            siblings.push_back(actual_sibling);
          }
        }
      } else {
        LOG(INFO) << "Collecting siblings: normal user " << user->name();
        if (user->dry() || user->rewrite()) {
          siblings.push_back(user);
        }
      }
    }

    for (auto i = siblings.begin(); i != siblings.end(); ++i) {
      for (auto j = i + 1; j != siblings.end(); ++j) {
        HloInstruction* left = *i;
        HloInstruction* right = *j;

        CHECK(left->dry() || left->rewrite());
        CHECK(right->dry() || right->rewrite());

        LOG(INFO) << "Considering " << left->name() << " and " << right->name();
        if (reachability_->IsConnected(left, right) || 
            !LegalToSiblingFuse(left, right)) {
          continue;
        }
        FuseTwoSiblings(/*left_sibling=*/left, /*right_sibling=*/right);
        count++;
      }
    }    

  }
  LOG(INFO) << "Done with General Fusion for " << comp->name();

  return count > 0;
}

StatusOr<bool> GeneralFusion::Run(HloModule* module) {
  bool changed = false;
  for (auto* computation : module->MakeNonfusionComputations()) {
    changed |= DoGeneralFusion(computation);
  }
  VLOG(1) << "After running GeneralFusion for module: " << module->name();
  XLA_VLOG_LINES(3, module->ToString());
  LOG(ERROR) << "Run Finish " << changed;
  return changed;
}

}  // namespace gpu
}  // namespace xla

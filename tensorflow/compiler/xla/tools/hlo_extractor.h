/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_HLO_EXTRACTOR_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_HLO_EXTRACTOR_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

// Visitor that build a new HLO module with an entry computation and a root that
// is provided to the visit function. Only HLOs that are reachable from the new
// root instruction are included in the new module.
//
// The constructor allows specifying a set of boundary HLOs to prune the HLO
// graph. HLOs at the boundary are replaced with parameters. Can be nullptr
// which means no boundary, i.e. no HLOs are replaced with parameters.
class ExtractionVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  explicit ExtractionVisitor(
      const HloModule& old_module,
      absl::flat_hash_set<const HloInstruction*>* boundary)
      : old_module_(old_module),
        module_(absl::make_unique<HloModule>("extracted", config_)),
        clone_context_(module_.get()),
        builder_("entry_computation"),
        boundary_(boundary) {}

  Status HandleParameter(const HloInstruction* parameter) override {
    // Entry parameters need renumbering.
    auto new_parameter = HloInstruction::CreateParameter(
        parameter_number_++, parameter->shape(), parameter->name());
    clone_context_.MapInstruction(parameter, new_parameter.get());
    builder_.AddInstruction(std::move(new_parameter));
    return Status::OK();
  }

  Status DefaultAction(const HloInstruction* hlo) override {
    // Replace instructions at the boundary with parameters, but leave constants
    // untouched.
    if (boundary_ != nullptr && boundary_->count(hlo) > 0) {
      auto new_parameter = HloInstruction::CreateParameter(
          parameter_number_, hlo->shape(), hlo->name());
      parameter_number_++;
      clone_context_.MapInstruction(hlo, new_parameter.get());
      builder_.AddInstruction(std::move(new_parameter));
      return Status::OK();
    }
    std::vector<HloInstruction*> new_operands;
    for (auto operand : hlo->operands()) {
      new_operands.push_back(clone_context_.GetInstruction(operand));
    }
    auto instruction =
        hlo->CloneWithNewOperands(hlo->shape(), new_operands, &clone_context_);
    builder_.AddInstruction(std::move(instruction));
    return Status::OK();
  }

  Status FinishVisit(const HloInstruction* /*root*/) override {
    module_->AddEntryComputation(builder_.Build());
    // Rename HLOs so that their name matches the original. By default,
    // HLOs get new unique names when adding a new entry computation to
    // a module.
    for (auto computation : old_module_.MakeComputationPostOrder()) {
      for (auto old_instruction : computation->MakeInstructionPostOrder()) {
        if (auto new_instruction =
                clone_context_.FindInstruction(old_instruction)) {
          new_instruction->SetAndSanitizeName(old_instruction->name());
          // If this has already been cloned before by an extraction, then
          // utilize the cloned instructions orig_unique_id.
          // This way even if an instruction gets cloned multiple time it still
          // gets set to the original instructions id
          int orig_id = old_instruction->orig_unique_id() == -1 ?
                        old_instruction->unique_id() :
                        old_instruction->orig_unique_id();
          new_instruction->SetOriginalId(orig_id);
        }
      }
    }
    return Status::OK();
  }

  HloModule* module() { return module_.get(); }

  std::unique_ptr<HloModule> ConsumeModule() { return std::move(module_); }

 private:
  const HloModule& old_module_;
  HloModuleConfig config_;
  std::unique_ptr<HloModule> module_;
  HloCloneContext clone_context_;
  HloComputation::Builder builder_;
  absl::flat_hash_set<const HloInstruction*>* boundary_;
  int64_t parameter_number_ = 0;
};

// Creates a new HLO module rooted with an entry computation rooted at the given
// instruction.
//
//  By default (height == -1), the new computation includes all transitive
//  operands of `root`.  If you specify a different height, the new computation
//  will include all instructions <= `height` hops away from `root`.
//  Instructions at the boundary are replaced by parameters.
std::unique_ptr<HloModule> ExtractModule(HloInstruction* instruction,
                                         int64_t height = -1);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_HLO_EXTRACTOR_H_

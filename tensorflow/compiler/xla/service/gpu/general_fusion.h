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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GENERAL_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GENERAL_FUSION_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {
namespace gpu {

// An HLO pass that does general fusion.

class GeneralFusion : public HloModulePass {
 public:
  absl::string_view name() const override { return "general_fusion"; }

  StatusOr<bool> Run(HloModule* module) override;

  int fused_count() { return fused_count_; }

 private:
  bool FuseSiblings(std::vector<HloInstruction*>& siblings);
  bool HasCycle(HloInstruction* producer, HloInstruction* consumer);
  bool DoGeneralFusion(HloComputation* comp);
  bool ShouldFuseProducerIntoConsumer(HloInstruction* producer, HloInstruction* consumer);

  // Current computation the pass is dealing with.
  HloComputation* computation_;

  // Reachability map of current computation.
  std::unique_ptr<HloReachabilityMap> reachability_;

  int fused_count_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GENERAL_FUSION_H_

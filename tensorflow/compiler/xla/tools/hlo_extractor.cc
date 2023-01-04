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

#include "tensorflow/compiler/xla/tools/hlo_extractor.h"

#include <stdio.h>
#include <unistd.h>

namespace xla {
namespace {

void ComputeBoundary(const HloInstruction* root, int64_t limit,
                     absl::flat_hash_set<const HloInstruction*>* boundary) {
  std::deque<const HloInstruction*> worklist;
  absl::flat_hash_map<const HloInstruction*, int64_t> visited;
  worklist.push_back(root);
  visited.emplace(root, 0);
  while (!worklist.empty()) {
    auto hlo = worklist.front();
    worklist.pop_front();
    int64_t hops = visited[hlo];
    if (hops > limit) {
      boundary->insert(hlo);
      continue;
    }
    for (const HloInstruction* operand : hlo->operands()) {
      if (visited.count(operand)) {
        continue;
      }
      worklist.push_back(operand);
      visited.emplace(operand, hops + 1);
    }
  }
}

}  // namespace

std::unique_ptr<HloModule> ExtractModule(HloInstruction* instruction,
                                         int64_t height) {
  absl::flat_hash_set<const HloInstruction*> boundary;
  if (height != -1) {
    ComputeBoundary(instruction, height, &boundary);
  }
  ExtractionVisitor visitor(*instruction->GetModule(), &boundary);
  CHECK(instruction->Accept(&visitor).ok());

  // The first pass may leave unused parameter instructions. Do another
  // extraction pass to remove unused parameters. This is done because
  // HloComputation does not allow removing parameters after the computation has
  // been built.
  ExtractionVisitor cleanup_visitor(*visitor.module(), /*boundary=*/nullptr);
  TF_CHECK_OK(visitor.module()->entry_computation()->root_instruction()->Accept(
      &cleanup_visitor));

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision=*/true);
  TF_CHECK_OK(verifier.Run(cleanup_visitor.module()).status());
  return cleanup_visitor.ConsumeModule();
}

}  // namespace xla

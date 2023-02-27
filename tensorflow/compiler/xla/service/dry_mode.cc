#include "tensorflow/compiler/xla/service/dry_mode.h"

#include <fstream>
#include <random>
#include <sstream>

#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

StatusOr<bool> DryModeOn::Run(HloModule* module) {
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    computation->set_dry(true);
  }
  return false;
}

StatusOr<bool> DryModeOff::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    computation->set_dry(false);
  }
  // std::cout << module->ToString() << std::endl;
  for (auto* comp : module->MakeNonfusionComputations()) {
    std::cout << comp->ToString() << std::endl;
  }
  std::random_device device;
  std::mt19937 generator(device());
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    int num_replace = 0;
    for (HloInstruction* inst : computation->instructions()) {
      std::uniform_int_distribution<int> rand_selection(
          0, inst->operand_count() - 1);
      if (inst->opcode() == HloOpcode::kAlternatives) {
        // Select a random index for alternatives
        static_cast<HloAlternatives*>(inst)->Select(rand_selection(generator));
        num_replace++;
      }
    }
    if (num_replace > 0) {
      changed = true;
      LOG(ERROR) << num_replace << " alternatives replaced";
    }
    // Remove the residues
    computation->Prune();
  }
  return changed;
}

}  // namespace xla

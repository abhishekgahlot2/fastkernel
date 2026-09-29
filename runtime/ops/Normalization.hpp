// Modified by meowkernels.
#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>

namespace splash::ops {

class Normalization final {
public:
  static void addRms(metal::CommandGraph &graph, metal::MetalBuffer input,
                     metal::MetalBuffer weight, metal::MetalBuffer output,
                     uint32_t width, uint32_t rows,
                     LinearScratch scratch = {});

  // Staged RMS that also writes the next affine Q4 projection's input group
  // sums as [group * rows + row] (SPLASH_INPUT_FUSED_SUMS, decode M8).
  static void addRmsStagedSplitSums(metal::CommandGraph &graph, metal::MetalBuffer input,
                                    metal::MetalBuffer weight, metal::MetalBuffer output,
                                    metal::MetalBuffer sums, uint32_t width, uint32_t rows);

  // Fused RMS normalization plus Q4 input-group sums for packed prefill.
  static void addRmsWithQ4Sums(metal::CommandGraph &graph,
                               metal::MetalBuffer input,
                               metal::MetalBuffer weight,
                               metal::MetalBuffer output,
                               metal::MetalBuffer sums, uint32_t width,
                               uint32_t rows);
};

} // namespace splash::ops

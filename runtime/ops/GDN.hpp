// Modified by meowkernels.
#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// Tensor geometry mapped to a compiled Metal variant during graph construction.
struct GdnShape final {
  uint32_t keyHeads = 0;
  uint32_t valueHeads = 0;
  uint32_t headDimension = 0;
  uint32_t convolutionDimension = 0;
  uint32_t packedWidth = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return keyHeads && valueHeads && valueHeads % keyHeads == 0 &&
           headDimension && convolutionDimension && packedWidth &&
           convolutionDimension ==
               (uint64_t{2} * keyHeads + valueHeads) * headDimension &&
           packedWidth >= convolutionDimension +
                              uint64_t{valueHeads} * headDimension +
                              uint64_t{2} * valueHeads;
  }

  bool operator==(const GdnShape &) const = default;
};

struct GdnStateStrides final {
  uint64_t convolutionLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convolutionStateBytes = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return convolutionLayerBytes && recurrentLayerBytes &&
           convolutionStateBytes;
  }
};

struct GdnPrefillBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer convolutionIn;
  metal::MetalBuffer convolutionOut;
  metal::MetalBuffer queries;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  metal::MetalBuffer recurrentIn;
  metal::MetalBuffer recurrentOut;
  metal::MetalBuffer recurrentRows;
  metal::MetalBuffer mixerNorm;
  metal::MetalBuffer hidden;
};

struct GdnDecodeBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer mixerNorm;
  metal::MetalBuffer hidden;
  metal::MetalBuffer arrived;
  metal::MetalBuffer generation;
  LinearScratch linearScratch{};
  bool precomputeSplitSums = false;
};

struct GdnCommitBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer retainedCounts;
};

// Per-request scratch for the default-off wide lookup: one request's 16 (tiles
// = 2) or 32 (tiles = 4, SPLASH_WIDE_LOOKUP32) rows as M8 tiles. Forward needs
// one layer's convolution carry; commit needs every layer's; four tiles
// alternate two carry slots. Packed GDN tensors hold the tiles in physical
// lanes 0..tiles-1. Wide decode uses currentStates[0]/nextStates[0], and
// arrived/generation each hold one uint per tile.
[[nodiscard]] constexpr uint64_t
gdnDecode16ConvolutionScratchBytes(GdnStateStrides state,
                                   uint32_t tiles = 2) noexcept {
  return (tiles > 2 ? 2 : 1) * state.convolutionLayerBytes;
}

[[nodiscard]] constexpr uint64_t
gdnCommit16ConvolutionScratchBytes(GdnStateStrides state,
                                   uint32_t tiles = 2) noexcept {
  return (tiles > 2 ? 2 : 1) * state.convolutionStateBytes;
}

// A wide lookup's GDN route: the chained M8 tiles, one dispatch per layer
// (SPLASH_WIDE_GDN_SINGLE), or that single pass split into four value parts
// plus a finalize (VH48; other shapes fall back to Single).
enum class WideGdn : uint8_t { Chain, Single, SingleParts };

class GDN final {
public:
  static void addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                         GdnShape shape, uint32_t tokens);
  static void addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                        GdnShape shape, uint32_t lanes, uint32_t layer,
                        GdnStateStrides state);
  static void addDecode16(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                          metal::MetalBuffer convolutionScratch,
                          GdnShape shape, uint32_t layer,
                          GdnStateStrides state, uint32_t tiles = 2,
                          WideGdn route = WideGdn::Chain);
  static void addCommit(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                        GdnShape shape, uint32_t layers, uint32_t lanes,
                        GdnStateStrides state);
  static void addCommit16(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                          metal::MetalBuffer convolutionScratch,
                          GdnShape shape, uint32_t layers,
                          GdnStateStrides state, uint32_t tiles = 2);
};

} // namespace splash::ops

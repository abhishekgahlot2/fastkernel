// Modified by meowkernels.
#include "ops/GDN.hpp"

#include "metal/EnvSwitch.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/LaneBindings.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(GDNDecodeBatchParams, conv_layer_bytes) == 16);
static_assert(offsetof(GDNBatchCommitParams, conv_layer_bytes) == 32);

enum class KernelLayout : uint8_t { Value48, Value32 };

[[nodiscard]] KernelLayout kernelShape(const GdnShape &shape) {
  if (!shape.valid())
    throw std::invalid_argument("invalid GDN shape");
  if (shape == GdnShape{16, 48, 128, 10240, 16640})
    return KernelLayout::Value48;
  if (shape == GdnShape{16, 32, 128, 8192, 12544})
    return KernelLayout::Value32;
  throw std::invalid_argument("unsupported compiled GDN shape");
}

[[nodiscard]] const char *kernelName(KernelLayout shape,
                                     const char *value48,
                                     const char *value32) noexcept {
  return shape == KernelLayout::Value48 ? value48 : value32;
}

// SPLASH_GDN_VALUE_PARTS=4 (default; any other value turns it off): the GDN scan
// in four value partitions. Serving +1.07%, texts 18/18 identical.
// Needs the Split32 mixer-out tile (N/128 <= cores): a no-op below 40 cores for the 27B.
[[nodiscard]] bool valueParts4Enabled() noexcept {
  static const bool enabled = metal::envSwitch("SPLASH_GDN_VALUE_PARTS", "4");
  return enabled;
}

} // namespace

void GDN::addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                     GdnShape shape, uint32_t tokens) {
  if (!tokens)
    throw std::invalid_argument("invalid GDN prefill geometry");
  const KernelLayout kernel = kernelShape(shape);
  const GDNPreparePrefillParams prepare{tokens, shape.packedWidth};
  graph.add(kernelName(kernel, "prefill_gdn_prepare",
                       "prefill_gdn_prepare_vh32"),
            {buffers.packed, buffers.convolutionWeights,
             buffers.convolutionIn, buffers.convolutionOut, buffers.queries,
             buffers.keys, buffers.values, buffers.decayWeights,
             buffers.timeBias, buffers.decay, buffers.beta},
            prepare, {uint64_t{tokens} * shape.keyHeads, 1, 1},
            {shape.headDimension, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_scan",
                       "prefill_gdn_scan_vh32"),
            {buffers.queries, buffers.keys, buffers.values, buffers.decay,
             buffers.beta, buffers.recurrentIn, buffers.recurrentOut,
             buffers.recurrentRows},
            GDNPrefillParams{tokens},
            {uint64_t{shape.valueHeads} * shape.headDimension /
                 SPLASH_GDN_SCAN_STATE_ROWS,
             1, 1},
            {SPLASH_GDN_SCAN_THREADS, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_gate",
                       "prefill_gdn_gate_vh32"),
            {buffers.recurrentRows, buffers.packed, buffers.mixerNorm,
             buffers.hidden},
            prepare, {uint64_t{tokens} * shape.valueHeads, 1, 1},
            {128, 1, 1});
}

void GDN::addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                    GdnShape shape, uint32_t lanes, uint32_t layer,
                    GdnStateStrides state) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !state.valid())
    throw std::invalid_argument("invalid GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  const bool prepare = bool(buffers.linearScratch.input);
  const bool splitSums = buffers.precomputeSplitSums;
  if (splitSums && (prepare || lanes != 1 || shape.valueHeads != 48 ||
                    shape.headDimension != 128 ||
                    buffers.linearScratch.sums.sizeBytes() <
                        uint64_t{shape.valueHeads} * shape.headDimension / 2))
    throw std::invalid_argument("invalid GDN split-sum preparation");
  const uint64_t outputWidth = uint64_t{shape.valueHeads} * shape.headDimension;
  if (prepare && (buffers.linearScratch.input.sizeBytes() < outputWidth * 16 * lanes ||
                  buffers.linearScratch.sums.sizeBytes() < outputWidth / 2 * lanes))
    throw std::invalid_argument("Q4 GDN preparation scratch is below requirement");
  const GDNDecodeBatchParams params{0,
                                    shape.packedWidth,
                                    lanes,
                                    layer,
                                    state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  if (valueParts4Enabled() && lanes == 1 && kernel == KernelLayout::Value48 &&
      splitSums && !prepare) {
    if (buffers.currentStates.size() != SPLASH_MAXIMUM_BATCH_WIDTH ||
        buffers.nextStates.size() != SPLASH_MAXIMUM_BATCH_WIDTH)
      throw std::invalid_argument("lane bindings must cover every lane");
    graph.add("verify_gdn_value_parts4_scan",
              {buffers.packed, buffers.convolutionWeights,
               buffers.currentStates[0], buffers.nextStates[0], buffers.mixed,
               buffers.decayWeights, buffers.timeBias, buffers.decay,
               buffers.beta, buffers.recurrent},
              params, {uint64_t{shape.valueHeads} * 4, 1, 1});
    graph.add("verify_gdn_value_parts4_finalize",
              {buffers.packed, buffers.recurrent, buffers.mixerNorm,
               buffers.hidden, buffers.arrived, buffers.generation,
               buffers.linearScratch.sums},
              params, {shape.valueHeads, 1, 1});
    return;
  }
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  bindings.reserve(prepare ? 22 : splitSums ? 21 : 20);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.mixed, buffers.decayWeights, buffers.timeBias,
                   buffers.decay, buffers.beta, buffers.recurrent,
                   buffers.mixerNorm, buffers.hidden, buffers.arrived,
                   buffers.generation});
  if (prepare)
    bindings.insert(bindings.end(), {buffers.linearScratch.input, buffers.linearScratch.sums});
  if (splitSums) bindings.push_back(buffers.linearScratch.sums);
  graph.add(splitSums ? "verify_gdn_fused_split_sums"
                    : prepare ? kernelName(kernel, "verify_gdn_fused_q4", "verify_gdn_fused_q4_vh32")
                    : kernelName(kernel, "verify_gdn_fused", "verify_gdn_fused_vh32"),
            std::move(bindings), params, {shape.valueHeads, lanes, 1});
}

// The wide tiles' kernels: 16 rows keep their first/second names.
[[nodiscard]] static std::string wideKernelName(KernelLayout kernel, const char *stage,
                                                uint32_t tile, uint32_t tiles) {
  std::string name = tiles == 2
      ? std::string(stage) + "16_" + (tile ? "second" : "first")
      : std::string(stage) + "32_t" + std::to_string(tile);
  return kernel == KernelLayout::Value48 ? name : name + "_vh32";
}

void GDN::addDecode16(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                      metal::MetalBuffer convolutionScratch, GdnShape shape,
                      uint32_t layer, GdnStateStrides state, uint32_t tiles,
                      WideGdn route) {
  if ((tiles != 2 && tiles != 4) || !state.valid() || buffers.currentStates.empty() ||
      buffers.nextStates.empty() || buffers.arrived.sizeBytes() < tiles * sizeof(uint32_t) ||
      buffers.generation.sizeBytes() < tiles * sizeof(uint32_t) ||
      convolutionScratch.sizeBytes() <
          gdnDecode16ConvolutionScratchBytes(state, tiles) ||
      buffers.linearScratch.input || buffers.linearScratch.sums)
    throw std::invalid_argument("invalid wide GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed,          buffers.convolutionWeights,
      buffers.currentStates[0], buffers.nextStates[0],
      buffers.mixed,           buffers.decayWeights,
      buffers.timeBias,        buffers.decay,
      buffers.beta,            buffers.recurrent,
      buffers.mixerNorm,       buffers.hidden,
      buffers.arrived,         buffers.generation,
      convolutionScratch};
  const GDNDecodeBatchParams params{0,
                                    shape.packedWidth,
                                    1,
                                    layer,
                                    state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  // SPLASH_WIDE_GDN_SINGLE=parts: the single pass as 4 value parts per head,
  // then a finalize for the complete-head gated norm (VH48 only).
  if (route == WideGdn::SingleParts && kernel == KernelLayout::Value48) {
    const std::string name = tiles == 2 ? "verify_gdn_wide16_parts4" : "verify_gdn_wide32_parts4";
    graph.add(name + "_scan",
              {buffers.packed, buffers.convolutionWeights,
               buffers.currentStates[0], buffers.nextStates[0], buffers.mixed,
               buffers.decayWeights, buffers.timeBias, buffers.decay,
               buffers.beta, buffers.recurrent},
              params, {uint64_t{shape.valueHeads} * 4, 1, 1});
    graph.add(name + "_finalize",
              {buffers.packed, buffers.recurrent, buffers.mixerNorm,
               buffers.hidden, buffers.arrived, buffers.generation},
              params, {shape.valueHeads, 1, 1});
    return;
  }
  // SPLASH_WIDE_GDN_SINGLE: all tiles in one dispatch (current -> next).
  if (route != WideGdn::Chain) {
    const char *name = tiles == 2 ? "verify_gdn_wide16" : "verify_gdn_wide32";
    graph.add(kernel == KernelLayout::Value48 ? std::string(name) : std::string(name) + "_vh32",
              std::move(bindings), params, {shape.valueHeads, 1, 1});
    return;
  }
  for (uint32_t tile = 0; tile < tiles; ++tile)
    graph.add(wideKernelName(kernel, "verify_gdn_fused", tile, tiles),
              bindings, params, {shape.valueHeads, 1, 1});
}

void GDN::addCommit(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                    GdnShape shape, uint32_t layers, uint32_t lanes,
                    GdnStateStrides state) {
  if (!layers || !lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH ||
      !state.valid())
    throw std::invalid_argument("invalid GDN commit geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed, buffers.mixed, buffers.decay, buffers.beta};
  bindings.reserve(13);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.push_back(buffers.retainedCounts);
  constexpr uint32_t rows = SPLASH_TARGET_VERIFY_ROWS;
  const GDNBatchCommitParams params{shape.valueHeads,
                                    shape.packedWidth,
                                    lanes,
                                    rows * shape.packedWidth,
                                    rows * shape.convolutionDimension,
                                    rows * shape.valueHeads,
                                    rows * shape.valueHeads,
                                    0,
                                    state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  graph.add(kernelName(kernel, "verify_gdn_commit",
                       "verify_gdn_commit_vh32"),
            std::move(bindings), params,
            {shape.valueHeads, layers, lanes});
}

void GDN::addCommit16(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                      metal::MetalBuffer convolutionScratch, GdnShape shape,
                      uint32_t layers, GdnStateStrides state, uint32_t tiles) {
  if ((tiles != 2 && tiles != 4) || !layers || !state.valid() || buffers.currentStates.empty() ||
      buffers.nextStates.empty() || buffers.retainedCounts.sizeBytes() < sizeof(uint32_t) ||
      convolutionScratch.sizeBytes() <
          gdnCommit16ConvolutionScratchBytes(state, tiles))
    throw std::invalid_argument("invalid wide GDN commit geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed,          buffers.mixed,
      buffers.decay,           buffers.beta,
      buffers.currentStates[0], buffers.nextStates[0],
      convolutionScratch,      buffers.retainedCounts};
  constexpr uint32_t rows = SPLASH_TARGET_VERIFY_ROWS;
  const GDNBatchCommitParams params{shape.valueHeads,
                                    shape.packedWidth,
                                    1,
                                    rows * shape.packedWidth,
                                    rows * shape.convolutionDimension,
                                    rows * shape.valueHeads,
                                    rows * shape.valueHeads,
                                    0,
                                    state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  for (uint32_t tile = 0; tile < tiles; ++tile)
    graph.add(wideKernelName(kernel, "verify_gdn_commit", tile, tiles),
              bindings, params, {shape.valueHeads, layers, 1});
}

} // namespace splash::ops

// Modified by meowkernels.
#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Linear.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace splash::ops {

// Immutable views of one packed Q4 projection.  StorageN is part of the
// package ABI; the operator may choose a different compute tile at runtime.
struct Q4Projection final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

// Q8 affine projections use per-64-input quantization and StorageN=256 order.
// Used by the MoE router and shared-expert gate.
struct Q8Projection final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

// Expert-major Q4 slabs keep one complete StorageN-packed projection per
// expert. The operator selects expertStrideBytes directly; no per-expert
// MetalBuffer objects or weight copies are created at runtime.
struct ExpertQ4Projection final {
  metal::MetalBuffer packed;
  uint32_t experts = 0;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint64_t expertStrideBytes = 0;
};

struct LinearMatrix final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  auto operator<=>(const LinearMatrix &) const = default;
};

enum class LinearPhase : uint8_t { Prefill, Decode };
enum class LinearEpilogue : uint8_t { None, Residual, GateUp, UpWithGate };
// Compute tiles over the StorageN=256 packing. Paired tiles pipeline two
// quant groups of one lane. Split tiles keep one 8-row tile per threadgroup
// and split K into four partitions whose fp32 partial sums are reduced before
// the bf16 rounding; they take one lane, K % 1024 == 0 and one threadgroup
// per tile. Split32PrecomputedSums also supports the qualified 16-row residual
// shapes with eight simdgroups. Paired256 is the four-simdgroup N256 paired tile. Simdgroup
// uses bf16 8x8 matrix operations and an explicit activation/split workspace.
enum class LinearTile : uint8_t {
  N128, N256, Paired128, Split32, Split64, Paired256, Simdgroup,
  Split32LocalSync, Split32PrecomputedSums
};
enum class LinearSimdgroups : uint8_t { Four = 4, Eight = 8 };

struct LinearWorkload final {
  LinearMatrix matrix;
  uint32_t rows = 0;
  LinearPhase phase = LinearPhase::Decode;
  LinearEpilogue epilogue = LinearEpilogue::None;
  auto operator<=>(const LinearWorkload &) const = default;
};

struct LinearConfig final {
  LinearTile tile = LinearTile::N128;
  // Decode grid size. Prefill uses its matrix grid and requires zero here.
  uint32_t groups = 0;
  // Simdgroups per threadgroup, independent of the persistent grid size: the
  // cooperative scope of one tile, or for split tiles the four partitions
  // together (Split32 variants are 4 x 1, Split64 is 4 x 2; Paired256 runs
  // four). Split32PrecomputedSums also owns a device input-sums buffer.
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
  // Cross-threadgroup K partitions for Simdgroup; all other tiles use one.
  uint32_t splits = 1;
  bool operator==(const LinearConfig &) const = default;
};

struct LinearChoice final {
  LinearWorkload workload;
  LinearConfig configuration;
};

// Reused serially within one decode command stream. Counters are zeroed at
// allocation and restored by each completed split dispatch. Never share this
// workspace between concurrent command streams. Within a batched dispatch,
// each eight-row tile owns disjoint input, sums, partials and counters.
struct LinearScratch final {
  metal::MetalBuffer input;
  metal::MetalBuffer sums;
  metal::MetalBuffer partials;
  metal::MetalBuffer counters;
};
struct LinearScratchSize final {
  uint64_t input = 0, sums = 0, partials = 0, counters = 0;
  [[nodiscard]] uint64_t bytes() const noexcept { return input + sums + partials + counters; }
};

class LinearPlan final {
public:
  [[nodiscard]] LinearWorkload workload() const noexcept { return workload_; }
  [[nodiscard]] LinearConfig configuration() const noexcept { return config_; }
  [[nodiscard]] uint32_t storageRows() const noexcept;
  [[nodiscard]] uint32_t tileColumns() const noexcept;
  [[nodiscard]] uint32_t threadsPerThreadgroup() const noexcept;
  // fp32 partial sums the kernel reduces before the single bf16 rounding of
  // the projection: 1 for the sequential tiles, whose outputs are bitwise
  // identical for a workload; 4 for split tiles; 1-8 for Simdgroup. The latter
  // also reassociates within each quantization group, even with one split.
  [[nodiscard]] uint32_t partialSums() const noexcept;
  [[nodiscard]] bool usesSimdgroup() const noexcept;
  [[nodiscard]] LinearScratchSize scratchSize() const noexcept;
  [[nodiscard]] uint64_t sumsBytes() const noexcept;
  [[nodiscard]] uint64_t gateScratchBytes() const noexcept;
  [[nodiscard]] uint64_t downSumsBytes() const noexcept;
  [[nodiscard]] std::string_view pipeline() const noexcept { return pipeline_; }
  [[nodiscard]] std::string_view secondPipeline() const noexcept {
    return secondPipeline_;
  }

private:
  friend class Q4Linear;
  LinearPlan(LinearWorkload workload, LinearConfig config);
  LinearWorkload workload_;
  LinearConfig config_;
  std::string_view pipeline_;
  std::string_view secondPipeline_;
};

// The plan defines which fields are used and how much scratch they require.
struct LinearBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer output;
  metal::MetalBuffer sums;
  metal::MetalBuffer residual;
  metal::MetalBuffer gateScratch;
  metal::MetalBuffer downSums;
  LinearScratch scratch{};
  // The scratch table and sums already describe input (for example after
  // fused RMSNorm). They must survive unchanged until this dispatch.
  bool inputPrepared = false;
  // Exact group-major Split32 sums are available without a swizzled table.
  bool sumsPrepared = false;
  // SPLASH_SPLIT4_M16 (B2 only, set per encode by the caller): the residual M16
  // split4 body runs as its footer + metadata-hoist twin.
  bool m16HoistFooter = false;
};

struct Q4DispatchStats final {
  uint64_t fusedSourceOperations = 0;
  uint64_t m16Dispatches = 0;
  uint64_t m24Dispatches = 0;
  uint64_t m32Dispatches = 0;
};

// Owns Q4 pipeline selection and dispatch. Device policy uses GPU family,
// core count and workload tile counts.
class Q4Linear final {
public:
  explicit Q4Linear(const DeviceCapabilities &device) noexcept;

  // One lane: at most 3 tiles * 4 group counts, 4 split tiles, 2 paired
  // N256 grids, and 4 Apple9 simdgroup K splits (including its baseline):
  // 3 * 4 + 4 + 2 + 4 = 22. Other families have no simdgroup candidates
  // and at most one additional baseline (19). M24 replaces Paired128 with
  // N128/four-simdgroup candidates and adds up to four matrix K splits,
  // with no one-lane tiles (at most 17).
  static constexpr std::size_t kMaximumCandidates = 22;

  [[nodiscard]] LinearPlan plan(LinearWorkload workload) const;
  [[nodiscard]] LinearScratchSize decodeScratchSize(LinearWorkload workload) const;
  [[nodiscard]] static LinearPlan plan(LinearWorkload workload, LinearConfig config);
  [[nodiscard]] std::vector<LinearPlan> candidates(LinearWorkload workload) const;
  // Installed only at startup; encoding does a read-only lookup, never tuning.
  void setChoices(std::span<const LinearChoice> choices);
  void add(metal::CommandGraph &graph, LinearBuffers buffers,
           const Q4Projection &projection, const LinearPlan &plan,
           const Q4Projection *gate = nullptr,
           Q4DispatchStats *stats = nullptr) const;

  void addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input,
                      metal::MetalBuffer sums, LinearMatrix matrix,
                      uint32_t rows) const;
  void addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const Q4Projection &projection, metal::MetalBuffer output,
                  metal::MetalBuffer sums, LinearMatrix matrix,
                  uint32_t rows) const;
  void addPrefillUpWithGate(
      metal::CommandGraph &graph, metal::MetalBuffer input,
      const Q4Projection &up, metal::MetalBuffer gateScratch,
      metal::MetalBuffer output, metal::MetalBuffer sums,
      metal::MetalBuffer downSums, LinearMatrix matrix,
      uint32_t rows) const;
  void addPrefillResidual(metal::CommandGraph &graph,
                          metal::MetalBuffer input,
                          const Q4Projection &projection,
                          metal::MetalBuffer residual,
                          metal::MetalBuffer output, metal::MetalBuffer sums,
                          LinearMatrix matrix, uint32_t rows) const;

  void addDecode(metal::CommandGraph &graph,
                 metal::MetalBuffer input, const Q4Projection &projection,
                 metal::MetalBuffer output, LinearMatrix matrix,
                 LinearScratch scratch = {}) const;
  void addDecodeBatch(metal::CommandGraph &graph,
                      metal::MetalBuffer input,
                      const Q4Projection &projection,
                      metal::MetalBuffer output, LinearMatrix matrix,
                      uint32_t lanes, Q4DispatchStats &stats,
                      LinearScratch scratch = {}, bool inputPrepared = false,
                      bool sumsPrepared = false) const;
  void addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const Q4Projection &gate, const Q4Projection &up,
                      metal::MetalBuffer gateScratch,
                      metal::MetalBuffer output, LinearMatrix matrix,
                      uint32_t lanes, Q4DispatchStats &stats,
                      LinearScratch scratch = {}, bool inputPrepared = false,
                      bool prepareDownSums = false) const;
  void addResidualBatch(metal::CommandGraph &graph,
                        metal::MetalBuffer input,
                        const Q4Projection &projection,
                        metal::MetalBuffer residual,
                        metal::MetalBuffer output, LinearMatrix matrix,
                        uint32_t lanes, Q4DispatchStats &stats,
                        LinearScratch scratch = {}, bool inputPrepared = false,
                        bool sumsPrepared = false, bool m16HoistFooter = false) const;

  // SPLASH_GROUPED_CONTEXT_KV (the drafter context commit; exact): the commit reads only the K/V suffix of each
  // layer's qkv output (columns kvColumn..N-1; both drafters: 4096..6143 of 6144), so only those tiles are computed,
  // by the unchanged tile of the plan the full projection uses. Admitted for 1 to kMaxContextKvLayers layers, any
  // input width, a 128-aligned kvColumn, at one lane when that plan is M8 Paired128 at 256 threads (n128_paired) over
  // any persistent grid, and at 2-4 lanes when it runs one 128-column tile per group over all N/128 tiles: M16 N128
  // (n128_m16), M24 N128 four simdgroups (n128_m24_sg4), M32 N128 (n128_m32). addContextKv first checks every
  // projection and output exactly as add() would, plus outputs disjoint from each other and from the input, and throws
  // before appending anything; then it appends one layers x kv-tiles dispatch (no layer) or, at one lane only, that
  // layer's kv-tiles dispatch.
  static constexpr uint32_t kMaxContextKvLayers = SPLASH_CONTEXT_KV_MAX_LAYERS;  // the kernels' layer slots
  [[nodiscard]] bool admitsContextKv(LinearMatrix matrix, uint32_t lanes, uint32_t layers,
                                     uint32_t kvColumn) const;
  void addContextKv(metal::CommandGraph &graph, metal::MetalBuffer input,
                    std::span<const Q4Projection *const> projections,
                    std::span<const metal::MetalBuffer> outputs, LinearMatrix matrix, uint32_t kvColumn,
                    uint32_t lanes, LinearScratch scratch, std::optional<uint32_t> layer) const;

private:
  // add()'s buffer checks: the projection, input, output and scratch fit the plan.
  static void requireBuffers(const LinearBuffers &buffers, const Q4Projection &projection,
                             const LinearPlan &plan, const Q4Projection *gate);
  [[nodiscard]] LinearConfig baseline(LinearWorkload workload) const;
  [[nodiscard]] bool narrowSplitShape(LinearWorkload workload) const noexcept;
  uint32_t appleGpuFamily_ = 0;
  uint32_t gpuCores_ = 0;
  bool narrowSplit16_ = false;
  bool narrowSplit24_ = false;
  bool wideResidualSplit16_ = false;
  bool inputSg4_ = false;
  std::vector<LinearChoice> choices_;
};

} // namespace splash::ops

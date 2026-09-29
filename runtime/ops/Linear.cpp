// Modified by meowkernels.
#include "Linear.hpp"

#include "metal/EnvSwitch.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Linear.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kPrefillRows = 32;
constexpr uint32_t kQuantGroup = 64;
constexpr uint32_t kMaximumSimdgroupSplits = 8;
static_assert(SPLASH_TARGET_VERIFY_ROWS == 8,
              "simdgroup Q4 tiles require eight verify rows per lane");
// Split tiles hold four K partitions, each a whole number of the kernels'
// four-quant-group (256-input) input-sum blocks.
constexpr uint32_t kSplitPartitions = 4;
constexpr uint32_t kSplitInputBlock = kSplitPartitions * 4 * kQuantGroup;
static_assert(sizeof(LinearMatrix) == 8);

bool splitTile(LinearTile tile) noexcept {
  return tile == LinearTile::Split32 || tile == LinearTile::Split64 ||
      tile == LinearTile::Split32LocalSync ||
      tile == LinearTile::Split32PrecomputedSums;
}
bool oneLaneTile(LinearTile tile) noexcept {
  return tile == LinearTile::Paired128 || tile == LinearTile::Paired256 || splitTile(tile);
}
// Simdgroups fixed by the kernel instance: split tiles run four partitions of
// one (N32) or two (N64) simdgroups; the paired N256 tile runs four.
std::optional<LinearSimdgroups> fixedSimdgroups(LinearTile tile, uint32_t rows) noexcept {
  if (tile == LinearTile::Split32PrecomputedSums && rows >= 16)
    return LinearSimdgroups::Eight;
  switch (tile) {
  case LinearTile::Split32:
  case LinearTile::Split32LocalSync:
  case LinearTile::Split32PrecomputedSums:
  case LinearTile::Simdgroup:
  case LinearTile::Paired256: return LinearSimdgroups::Four;
  case LinearTile::Split64: return LinearSimdgroups::Eight;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128: return std::nullopt;
  }
  return std::nullopt;
}

void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid Q4 linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid Q4 prefill workload");
  } else if (w.phase == LinearPhase::Decode) {
    if (w.matrix.inputSize % 256 || !w.rows || w.rows % SPLASH_TARGET_VERIFY_ROWS ||
        w.rows > SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid Q4 decode workload");
  } else {
    throw std::invalid_argument("invalid Q4 linear phase");
  }
  if (w.epilogue != LinearEpilogue::None && w.epilogue != LinearEpilogue::Residual &&
      w.epilogue != LinearEpilogue::GateUp && w.epilogue != LinearEpilogue::UpWithGate)
    throw std::invalid_argument("invalid Q4 linear epilogue");
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes) {
  if (bytes && (!buffer || buffer.sizeBytes() < bytes))
    throw std::invalid_argument("Q4 buffer is below plan requirement");
}

void requireProjection(const Q4Projection &p, LinearMatrix matrix) {
  if (p.outputSize != matrix.outputSize || p.inputSize != matrix.inputSize)
    throw std::invalid_argument("Q4 projection does not match plan");
  requireBytes(p.weights, uint64_t{matrix.outputSize} * matrix.inputSize / 2);
  const uint64_t bytes = uint64_t{matrix.outputSize} * (matrix.inputSize / kQuantGroup) * 2;
  requireBytes(p.scales, bytes);
  requireBytes(p.biases, bytes);
}

void account(Q4DispatchStats &stats, uint32_t lanes, uint32_t count) noexcept {
  if (lanes == 1) return;
  stats.fusedSourceOperations += uint64_t{lanes} * count;
  if (lanes == 2) stats.m16Dispatches += count;
  else if (lanes == 3) stats.m24Dispatches += count;
  else stats.m32Dispatches += count;
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid Q4 decode batch width");
  return {matrix, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
}

// The four-simdgroup kernels: every prefill N128 tile, the decode M24 N128
// plain and residual projections, all matrix row tiles, and the one-lane
// Split32 variants (plain and residual; the original also gate/up) and
// Paired256 (plain) tiles.
bool supportsFourSimdgroups(LinearWorkload w, LinearTile tile) noexcept {
  if (tile == LinearTile::Simdgroup) return w.phase == LinearPhase::Decode;
  if (tile == LinearTile::Split32 || tile == LinearTile::Split32LocalSync ||
      tile == LinearTile::Split32PrecomputedSums)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS;
  // Only the affine paired N256 kernel is instantiated: this tile is used
  // for wide plain projections; residual and gate/up retain their own tiles.
  // Paired N128 has an affine four-SIMD-group kernel too (SPLASH_INPUT_SG4).
  if (tile == LinearTile::Paired256 || tile == LinearTile::Paired128)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS &&
        w.epilogue == LinearEpilogue::None;
  if (tile != LinearTile::N128) return false;
  return w.phase == LinearPhase::Prefill ||
      (w.rows == 24 && (w.epilogue == LinearEpilogue::None ||
                        w.epilogue == LinearEpilogue::Residual));
}

} // namespace

uint32_t LinearPlan::storageRows() const noexcept {
  return workload_.phase == LinearPhase::Prefill
      ? ((workload_.rows + kPrefillRows - 1) / kPrefillRows) * kPrefillRows : workload_.rows;
}
uint32_t LinearPlan::tileColumns() const noexcept {
  switch (config_.tile) {
  case LinearTile::Simdgroup: return workload_.epilogue == LinearEpilogue::GateUp ? 32 : 64;
  case LinearTile::Split32: return 32;
  case LinearTile::Split32LocalSync: return 32;
  case LinearTile::Split32PrecomputedSums: return 32;
  case LinearTile::Split64: return 64;
  case LinearTile::N256:
  case LinearTile::Paired256: return 256;
  case LinearTile::N128:
  case LinearTile::Paired128: return 128;
  }
  return 0;
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  return static_cast<uint32_t>(config_.simdgroups) * 32;
}
uint32_t LinearPlan::partialSums() const noexcept {
  return usesSimdgroup() ? config_.splits : splitTile(config_.tile) ? kSplitPartitions : 1;
}
bool LinearPlan::usesSimdgroup() const noexcept { return config_.tile == LinearTile::Simdgroup; }
LinearScratchSize LinearPlan::scratchSize() const noexcept {
  if (config_.tile == LinearTile::Split32PrecomputedSums)
    return {0, uint64_t{workload_.rows} * (workload_.matrix.inputSize / kQuantGroup) *
                   sizeof(float), 0, 0};
  if (!usesSimdgroup()) return {};
  const auto [n, k] = workload_.matrix;
  const uint64_t rows = workload_.rows;
  const uint64_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  // Each row tile owns two fp32 fragment streams per K partition and one
  // completion counter per column tile. Single-partition kernels use neither.
  return {rows * k * sizeof(uint16_t), rows * (k / kQuantGroup) * sizeof(float),
          config_.splits > 1 ? config_.splits * 2 * rows * n * sizeof(float) : sizeof(float),
          config_.splits > 1 ? lanes * (n / tileColumns()) * sizeof(uint32_t) : sizeof(uint32_t)};
}

uint64_t LinearPlan::sumsBytes() const noexcept {
  return workload_.phase == LinearPhase::Prefill
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp && !secondPipeline_.empty());
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config)
    : workload_(w), config_(config) {
  validate(w);
  if (config.tile != LinearTile::Simdgroup && config.splits != 1)
    throw std::invalid_argument("K splits require the simdgroup Q4 tile");
  if (config.tile != LinearTile::N128 && config.tile != LinearTile::N256 &&
      config.tile != LinearTile::Simdgroup && !oneLaneTile(config.tile))
    throw std::invalid_argument("invalid Q4 linear tile");
  if ((config.simdgroups != LinearSimdgroups::Four &&
       config.simdgroups != LinearSimdgroups::Eight) ||
      (config.simdgroups == LinearSimdgroups::Four &&
       !supportsFourSimdgroups(w, config.tile)))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (const auto fixed = fixedSimdgroups(config.tile, w.rows); fixed && config.simdgroups != *fixed)
    throw std::invalid_argument("Q4 tile requires its kernel's simdgroup count");
  if (w.matrix.outputSize % tileColumns())
    throw std::invalid_argument("Q4 matrix is not divisible by tile columns");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  if (w.phase == LinearPhase::Prefill) {
    if (config.groups || oneLaneTile(config.tile) || usesSimdgroup())
      throw std::invalid_argument("invalid Q4 prefill configuration");
    if (four) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? "prefill_linear_q4_n128_up_silu_sums_sg4"
          : residual ? "prefill_linear_q4_n128_residual_sg4" : "prefill_linear_q4_n128_sg4";
    } else if (w.epilogue == LinearEpilogue::UpWithGate) {
      if (config.tile != LinearTile::N256)
        throw std::invalid_argument(
            "Q4 fused prefill up requires N256 or four simdgroups");
      pipeline_ = "prefill_linear_q4_n256_up_silu_sums";
    } else if (residual) {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128_residual" : "prefill_linear_q4_n256_residual";
    } else {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128" : "prefill_linear_q4_n256";
    }
    return;
  }
  if (!config.groups || config.groups > w.matrix.outputSize / tileColumns())
    throw std::invalid_argument("invalid Q4 decode group count");
  const uint32_t lane = w.rows / SPLASH_TARGET_VERIFY_ROWS - 1;
  const bool m16Split = config.tile == LinearTile::Split32PrecomputedSums && w.rows >= 16;
  if (oneLaneTile(config.tile) && !m16Split && (lane != 0 || w.matrix.outputSize % 256))
    throw std::invalid_argument("paired or split Q4 tile requires one lane and paired columns");
  if (usesSimdgroup()) {
    const uint32_t groups = w.matrix.inputSize / kQuantGroup;
    if (config.groups != w.matrix.outputSize / tileColumns() ||
        !config.splits || config.splits > kMaximumSimdgroupSplits || (config.splits & (config.splits - 1)) ||
        groups % config.splits)
      throw std::invalid_argument("simdgroup Q4 requires full column grid and whole power-of-two K partitions");
    pipeline_ = w.epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
        residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
    return;
  }
  if (splitTile(config.tile)) {
    // Each partition takes a quarter of K in whole 256-input blocks, and the
    // split kernels are dispatched one threadgroup per tile.
    if (w.matrix.inputSize % kSplitInputBlock ||
        config.groups != w.matrix.outputSize / tileColumns())
      throw std::invalid_argument("split Q4 tile requires K % 1024 == 0 and the full grid");
    if (w.epilogue == LinearEpilogue::GateUp) {
      // Only the N32 two-stream split kernel is instantiated.
      if (config.tile != LinearTile::Split32)
        throw std::invalid_argument("Q4 split gate/up requires Split32");
      pipeline_ = "decode_linear_q4_n32_split4_gate_up";
    } else if (config.tile == LinearTile::Split32LocalSync) {
      pipeline_ = residual ? "decode_linear_q4_n32_split4_local_sync_residual"
                           : "decode_linear_q4_n32_split4_local_sync";
    } else if (config.tile == LinearTile::Split32PrecomputedSums) {
      // Multi-lane rows 16/24/32 (validate() admits only multiples of 8 up to 32).
      constexpr std::array multiResidual{"decode_linear_q4_n32_split4_precomputed_sums_residual_m16",
          "decode_linear_q4_n32_split4_precomputed_sums_residual_m24",
          "decode_linear_q4_n32_split4_precomputed_sums_residual_m32"};
      constexpr std::array multiPlain{"decode_linear_q4_n32_split4_precomputed_sums_m16",
          "decode_linear_q4_n32_split4_precomputed_sums_m24",
          "decode_linear_q4_n32_split4_precomputed_sums_m32"};
      pipeline_ = m16Split ? (residual ? multiResidual : multiPlain)[w.rows / 8 - 2]
                  : residual ? "decode_linear_q4_n32_split4_precomputed_sums_residual"
                           : "decode_linear_q4_n32_split4_precomputed_sums";
    } else if (residual) {
      pipeline_ = config.tile == LinearTile::Split32
          ? "decode_linear_q4_n32_split4_residual"
          : "decode_linear_q4_n64_split4_residual";
    } else {
      pipeline_ = config.tile == LinearTile::Split32
          ? "decode_linear_q4_n32_split4" : "decode_linear_q4_n64_split4";
    }
    return;
  }
  if (config.tile == LinearTile::Paired256) {
    pipeline_ = "decode_linear_q4_n256_paired_sg4";
    return;
  }
  if (config.tile == LinearTile::Paired128 && four) {
    // Only the affine form exists; supportsFourSimdgroups() rejects the rest.
    pipeline_ = "decode_linear_q4_n128_paired_sg4";
    return;
  }
  if (four) {
    pipeline_ = residual ? "decode_linear_q4_n128_residual_m24_sg4"
                         : "decode_linear_q4_n128_m24_sg4";
    return;
  }
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (config.tile != LinearTile::N256)
      throw std::invalid_argument("Q4 gate/up requires N256");
    constexpr std::array names{"decode_linear_q4_n256_gate_up", "decode_linear_q4_n256_gate_up_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
    if (lane >= 2)
      secondPipeline_ = lane == 2 ? "decode_linear_q4_n256_up_silu_m24"
                                  : "decode_linear_q4_n256_up_silu_m32";
  } else if (residual) {
    if (config.tile == LinearTile::N256)
      throw std::invalid_argument("Q4 decode residual requires N128");
    constexpr std::array names{"decode_linear_q4_n128_residual", "decode_linear_q4_n128_residual_m16",
        "decode_linear_q4_n128_residual_m24", "decode_linear_q4_n128_residual_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
        ? "decode_linear_q4_n128_residual_paired" : names[lane];
  } else if (config.tile == LinearTile::N256) {
    constexpr std::array names{"decode_linear_q4_n256", "decode_linear_q4_n256_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
  } else {
    constexpr std::array names{"decode_linear_q4_n128", "decode_linear_q4_n128_m16",
        "decode_linear_q4_n128_m24", "decode_linear_q4_n128_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
                    ? "decode_linear_q4_n128_paired" : names[lane];
  }
}

namespace {

// Decode groups stream output tiles. Under round-robin group placement, the
// most loaded core sets dispatch latency. Use the full grid for small workloads,
// balanced two-tile groups at intermediate sizes, and one resident wave for
// longer chains; sufficiently large grids balance themselves.
struct DecodeGroupPolicy final {
  // The one-tile grid wins up to this many groups per core.
  uint32_t fullGridGroupsPerCore;
  // Resident groups per core: one wave for this kernel's register footprint.
  uint32_t waveGroupsPerCore;
  // From this many tiles per core the many-wave grid wins again.
  uint32_t manyWaveTilesPerCore;
};
// Resident-wave and full-grid thresholds measured on 16/20-core Apple10 GPUs.
// Gate/up uses the conservative limit shared by both devices. Its many-wave
// threshold follows N256; the four-simdgroup threshold scales from N128. Those
// two extrapolations remain unmeasured.
// Resident groups per core of the 128-thread M8 prepared-sums split4 body, measured on
// M5 Max (40 cores: about 497 of 520 groups start in the first wave);
// other GPUs unmeasured.
constexpr uint32_t kSplit4InputWaveGroupsPerCore = 12;
constexpr DecodeGroupPolicy kN128Groups{4, 4, 12}, kN128M16Groups{5, 4, 12},
    kN256Groups{3, 3, 8}, kGateUpGroups{3, 3, 8},
    kFourSimdgroupGroups{8, 8, 24};
// Apple9 retains its measured gate/up clamp. The round-robin policy above was
// measured on Apple10; applying it to Apple9 requires separate calibration.
constexpr double kApple9GateUpGroupsPerCore = 2.25;

// Tiles on the most loaded core when `groups` threadgroups are placed
// round-robin on `cores` and group g streams tiles g, g + groups, ...
uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) noexcept {
  uint32_t worst = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint32_t load = 0;
    for (uint32_t group = core; group < groups; group += cores)
      load += (tiles - group + groups - 1) / groups;
    worst = std::max(worst, load);
  }
  return worst;
}

uint32_t decodeGroups(uint32_t tiles, uint32_t cores,
                      DecodeGroupPolicy policy) noexcept {
  const uint32_t wave = policy.waveGroupsPerCore * cores;
  if (tiles <= policy.fullGridGroupsPerCore * cores ||
      tiles >= policy.manyWaveTilesPerCore * cores)
    return tiles;
  const uint32_t twoTile = (tiles + 1) / 2;
  // Here wave < twoTile <= tiles, so the wave is a valid count (LinearPlan
  // rejects more groups than tiles) whatever the per-core constants are.
  if (twoTile > wave) return wave;
  // The smallest balanced two-tile count keeping three quarters of the
  // full-grid limit resident. A multiple of the core count is always
  // balanced, so the search ends within `cores` steps and below `tiles`.
  const uint32_t balanced = (tiles + cores - 1) / cores;
  uint32_t groups =
      std::max(twoTile, policy.fullGridGroupsPerCore * cores * 3 / 4);
  while (maxCoreTiles(tiles, groups, cores) != balanced) ++groups;
  return groups;
}
// A multi-row N256 decode tile halves the input re-reads of N128 but also
// halves the grid; it pays only while the N256 grid keeps two tiles per core.
constexpr uint32_t kWideDecodeTilesPerCore = 2;
// Apple9 N256 prefill needs eight threadgroups per core to amortize its larger
// tile. Paired-A/B tuning (tune-kernels) and the per-shape microprofile
// (benchmark-prefill) on a 32-core Apple9 GPU (M4 Max) measured the
// four-simdgroup N128 tile ahead of N256 on every prefill shape and probed
// row count: +6..10% GPU wherever the margin cleared the tuning threshold,
// never behind. Apple9 GPUs at or below that measured core count therefore
// share the Apple10 prefill rule. Larger Apple9 GPUs (40-core class) keep the
// wide-tile rule below; it was sized for them and remains unremeasured there.
constexpr uint32_t kApple9MeasuredPrefillCores = 32;
constexpr double kApple9WidePrefillGroupsPerCore = 8.0;
// Missing core metadata uses one intermediate estimate for all families.
// This is a fallback, not a calibrated optimum. Reported counts always win.
constexpr uint32_t kAssumedGpuCores = 32;

// Apple10 wide plain projections reduce input re-reads with paired N256
// tiles at one resident wave, measured on 16/20-core GPUs. Split-K remains
// an offline candidate: its reassociation reduced speculative acceptance
// on some measured prompts. Apple9's simdgroup policy is independent.
constexpr uint32_t kPaired256TilesPerCore = 8;
constexpr uint32_t kPaired256WaveGroupsPerCore = 4;

std::optional<LinearConfig> apple10OneLaneConfig(LinearWorkload w, uint32_t cores) {
  // validate() requires outputSize % 256 == 0, so every tile width divides it.
  const uint32_t n = w.matrix.outputSize;
  const uint32_t tiles256 = n / 256;
  // meowkernels fork: at most one N128 tile per core over a long K leaves the
  // GPU idle; tune-kernels on M5 Max 40 measured Split32 at 4 groups/core
  // +34% (K 17408/25600) and +38..42% (K 4096/6144) GPU per projection.
  // Narrow outputs (N < 4096) keep their own policy. Reassociates K sums, so
  // acceptance must be re-verified end to end. Don't add new reassociating shapes:
  // since 2026-09-28 matmul changes must be scheduling-only, byte-identical to the
  // path they replace (oracle). The split-K rules here predate that; quality-gated.
  // SPLASH_NARROW_SPLIT (default on) extends split-K to narrow outputs (N >= 256): one-lane
  // Paired128 runs them on N/128 groups that each stream all of K serially
  // (N=1280: 10 groups, ~62 us for 3.8 MB). Today only the drafter has them, so
  // the reassociation changes draft proposals, never verified output. Serving
  // ms/step -1.24%, greedy texts identical.
  static const bool narrowSplit = metal::envSwitch("SPLASH_NARROW_SPLIT");
  if ((w.epilogue == LinearEpilogue::None || w.epilogue == LinearEpilogue::Residual) &&
      n / 128 <= cores && n >= (narrowSplit ? 256u : 4096u) && w.matrix.inputSize >= 4096 &&
      w.matrix.inputSize % kSplitInputBlock == 0)
    return LinearConfig{LinearTile::Split32PrecomputedSums, n / 32, LinearSimdgroups::Four, 1};
  if (w.epilogue == LinearEpilogue::None && tiles256 >= kPaired256TilesPerCore * cores)
    return LinearConfig{LinearTile::Paired256,
                        std::min(tiles256, kPaired256WaveGroupsPerCore * cores),
                        LinearSimdgroups::Four};
  return std::nullopt;
}

} // namespace

Q4Linear::Q4Linear(const DeviceCapabilities &device) noexcept
    : appleGpuFamily_(device.appleGpuFamily),
      gpuCores_(device.gpuCoreCount ? device.gpuCoreCount : kAssumedGpuCores) {
  // Read per planner, not per process, so tests can build both policies.
  narrowSplit16_ = metal::envSwitch("SPLASH_M16_NARROW_SPLIT");
  narrowSplit24_ = metal::envSwitch("SPLASH_M24_NARROW_SPLIT");
  // Wide prompt lookup verifies 16 rows of one request; each must equal the
  // 8-row verify byte for byte, and the one-lane rule splits the same residual
  // shapes, so wide lookup keeps the 16-row residual split on.
  wideResidualSplit16_ = metal::envSwitch("SPLASH_WIDE_PROMPT_LOOKUP");
  const char *inputSg4 = std::getenv("SPLASH_INPUT_SG4");
  inputSg4_ = inputSg4 && std::string_view(inputSg4) == "1";
}

// SPLASH_M16_NARROW_SPLIT (16 rows) / SPLASH_M24_NARROW_SPLIT (24, 32 rows),
// both default on: multi-lane projections with at most one N128 tile per core
// over a long K take the one-lane split-K rule too (four K partitions of two
// simdgroups, precomputed sums, 256 threads). Two-request cycles -2.9%
//; 3-4 request cycles -20%, pooled +27%.
// Split-K rounding: texts can differ (quality-gated, not bitwise). The rule
// comes from the device (N/128 <= cores); it replaced a 40-core-only switch
// with fixed 27B shapes, which covered a subset of the same workloads.
bool Q4Linear::narrowSplitShape(LinearWorkload w) const noexcept {
  const uint32_t n = w.matrix.outputSize;
  const bool enabled = w.rows == 16
      ? narrowSplit16_ || (wideResidualSplit16_ && w.epilogue == LinearEpilogue::Residual)
      : (w.rows == 24 || w.rows == 32) && narrowSplit24_;
  return enabled && appleGpuFamily_ >= 10 && w.phase == LinearPhase::Decode &&
      (w.epilogue == LinearEpilogue::None || w.epilogue == LinearEpilogue::Residual) &&
      n / 128 <= gpuCores_ && n >= 256 && n % 256 == 0 && w.matrix.inputSize >= 4096 &&
      w.matrix.inputSize % kSplitInputBlock == 0;
}

// GPU family selects variants; core count and workload tile counts determine
// parallelism.
LinearConfig Q4Linear::baseline(LinearWorkload w) const {
  validate(w);
  if (narrowSplitShape(w))
    return {LinearTile::Split32PrecomputedSums, w.matrix.outputSize / 32,
            LinearSimdgroups::Eight};
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  if (w.phase == LinearPhase::Prefill) {
    if (appleGpuFamily_ >= 10 || gpuCores_ <= kApple9MeasuredPrefillCores)
      return {LinearTile::N128, 0, LinearSimdgroups::Four};
    const uint32_t rowTiles = (w.rows + kPrefillRows - 1) / kPrefillRows;
    const bool wide = double(rowTiles) * tiles256 >=
        kApple9WidePrefillGroupsPerCore * gpuCores_;
    return {w.epilogue == LinearEpilogue::UpWithGate || wide ? LinearTile::N256
                                                              : LinearTile::N128, 0};
  }
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  // Keep the existing broad-column plain projection path for wider batches:
  // independent row tiles repeat its weight stream. Reuse the existing
  // two-N256-tiles-per-core boundary rather than model-specific dimensions.
  const bool widePlain = lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * gpuCores_;
  if (appleGpuFamily_ == 9 && !widePlain) {
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    const uint32_t grid = w.matrix.outputSize / columns, groups = w.matrix.inputSize / 64;
    uint32_t splits = 1;
    // Aim for sixteen independent column/K groups per core, retaining at
    // least twelve quant groups per partition to amortize the reduction.
    while (splits < kMaximumSimdgroupSplits && uint64_t(grid) * splits < 16ULL * gpuCores_ &&
           groups % (2 * splits) == 0 && groups / (2 * splits) >= 12)
      splits *= 2;
    return {LinearTile::Simdgroup, grid, LinearSimdgroups::Four, splits};
  }
  if (appleGpuFamily_ >= 10 && lanes == 1)
    if (const auto config = apple10OneLaneConfig(w, gpuCores_)) return *config;
  // Apple9 keeps its one-tile grids (see kApple9GateUpGroupsPerCore).
  const auto groups = [&](uint32_t tiles, DecodeGroupPolicy policy) {
    return appleGpuFamily_ >= 10 ? decodeGroups(tiles, gpuCores_, policy)
                                 : tiles;
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (appleGpuFamily_ < 10) {
      const auto resident = static_cast<uint32_t>(
          std::max(1L, std::lround(kApple9GateUpGroupsPerCore * gpuCores_)));
      return {LinearTile::N256, std::min(tiles256, resident)};
    }
    return {LinearTile::N256, groups(tiles256, kGateUpGroups)};
  }
  // Pipelined N128 hides the latency of a single lane's weight stream.
  if (lanes == 1)
    return {LinearTile::Paired128, groups(tiles128, kN128Groups),
            // SG4 only for wide grids (>= 3 tiles per core): narrower grids lose
            // per-group throughput with half the SIMD groups (full-step A/B).
            inputSg4_ && w.epilogue == LinearEpilogue::None && tiles128 >= 3 * gpuCores_
                ? LinearSimdgroups::Four : LinearSimdgroups::Eight};
  // With at most one N128 tile per core, longer M24 dot products benefit
  // from eight groups. Short K and wider grids retain the four-group path.
  if (appleGpuFamily_ >= 10 && lanes == 3 && tiles128 <= gpuCores_ &&
      w.matrix.inputSize >= 4096)
    return {LinearTile::N128, tiles128, LinearSimdgroups::Eight};
  // M24 plain projections benefit from four SIMD groups on Apple9 too.
  // Apple9 residual projections retain eight groups with compact prefix
  // traversal; Apple10 uses four groups outside the narrow-grid case above.
  if (lanes == 3 && (appleGpuFamily_ >= 10 ||
      (appleGpuFamily_ == 9 && w.epilogue == LinearEpilogue::None)))
    return {LinearTile::N128, groups(tiles128, kFourSimdgroupGroups),
            LinearSimdgroups::Four};
  if (lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * gpuCores_)
    return {LinearTile::N256, groups(tiles256, kN256Groups)};
  return {LinearTile::N128,
          groups(tiles128, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

LinearPlan Q4Linear::plan(LinearWorkload workload) const {
  // The narrow split-K rule overrides tuned choices for its qualified workloads.
  if (narrowSplitShape(workload))
    return LinearPlan(workload, baseline(workload));
  const auto found = std::lower_bound(choices_.begin(), choices_.end(), workload,
      [](const LinearChoice &choice, LinearWorkload key) { return choice.workload < key; });
  return LinearPlan(workload, found != choices_.end() && found->workload == workload
      ? found->configuration : baseline(workload));
}
LinearPlan Q4Linear::plan(LinearWorkload workload, LinearConfig config) {
  return LinearPlan(workload, config);
}
void Q4Linear::setChoices(std::span<const LinearChoice> choices) {
  std::vector<LinearChoice> pending(choices.begin(), choices.end());
  for (const auto &choice : pending) {
    if (choice.workload.rows == 16 &&
        choice.configuration.tile == LinearTile::Split32PrecomputedSums &&
        !narrowSplitShape(choice.workload))
      throw std::invalid_argument("M16 split tuning choice requires the narrow split rule");
    (void)plan(choice.workload, choice.configuration);
  }
  std::sort(pending.begin(), pending.end(), [](const auto &a, const auto &b) {
    return a.workload < b.workload;
  });
  for (size_t i = 1; i < pending.size(); ++i)
    if (pending[i - 1].workload == pending[i].workload)
      throw std::invalid_argument("duplicate Q4 linear choice");
  choices_ = std::move(pending);
}

std::vector<LinearPlan> Q4Linear::candidates(LinearWorkload w) const {
  std::vector<LinearPlan> result;
  result.reserve(kMaximumCandidates);
  result.push_back(LinearPlan(w, baseline(w)));
  const auto append = [&](LinearConfig config) {
    for (const auto &existing : result)
      if (existing.configuration() == config) return;
    result.push_back(LinearPlan(w, config));
  };
  for (const auto tile : {LinearTile::N128, LinearTile::N256, LinearTile::Paired128}) {
    const uint32_t columns = tile == LinearTile::N256 ? 256 : 128;
    if (w.matrix.outputSize % columns ||
        (tile == LinearTile::Paired128 && (w.phase != LinearPhase::Decode ||
         w.rows != SPLASH_TARGET_VERIFY_ROWS || w.matrix.outputSize % 256)) ||
        (w.epilogue == LinearEpilogue::GateUp && tile != LinearTile::N256) ||
        (w.phase == LinearPhase::Decode && w.epilogue == LinearEpilogue::Residual && tile == LinearTile::N256))
      continue;
    if (w.phase == LinearPhase::Prefill) {
      // The fused up projection has no eight-simdgroup N128 kernel.
      if (tile == LinearTile::N256 || w.epilogue != LinearEpilogue::UpWithGate)
        append({tile, 0});
      if (supportsFourSimdgroups(w, tile))
        append({tile, 0, LinearSimdgroups::Four});
    } else {
      const uint32_t tiles = w.matrix.outputSize / columns;
      // Sample two, three and four groups per core plus the full grid.
      // Always retain the measured baseline above, including its balanced
      // group count. Fixed counts tied to one GPU miss these waves elsewhere.
      for (const uint32_t groups : {2 * gpuCores_, 3 * gpuCores_, 4 * gpuCores_, tiles}) {
        append({tile, std::min(groups, tiles)});
        // Paired N128 four-simdgroup tiles are the SPLASH_INPUT_SG4 experiment:
        // offered only when it is on, like baseline(), which keeps the list
        // within kMaximumCandidates on Apple9.
        if (supportsFourSimdgroups(w, tile) && (tile != LinearTile::Paired128 || inputSg4_))
          append({tile, std::min(groups, tiles), LinearSimdgroups::Four});
      }
    }
  }
  if (w.phase == LinearPhase::Decode && appleGpuFamily_ == 9) {
    const uint32_t n = w.matrix.outputSize;
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    for (uint32_t splits = 1; splits <= kMaximumSimdgroupSplits; splits *= 2)
      if ((w.matrix.inputSize / kQuantGroup) % splits == 0)
        append({LinearTile::Simdgroup, n / columns, LinearSimdgroups::Four, splits});
  }
  // One-lane tiles: the split forms at their full grid and the paired N256
  // tile at one resident wave and at its full grid.
  if (w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS) {
    const uint32_t n = w.matrix.outputSize;
    if (w.matrix.inputSize % kSplitInputBlock == 0) {
      append({LinearTile::Split32, n / 32, LinearSimdgroups::Four});
      if (w.epilogue != LinearEpilogue::GateUp) {
        append({LinearTile::Split32LocalSync, n / 32, LinearSimdgroups::Four});
        append({LinearTile::Split32PrecomputedSums, n / 32,
                LinearSimdgroups::Four});
        append({LinearTile::Split64, n / 64, LinearSimdgroups::Eight});
      }
    }
    if (w.epilogue == LinearEpilogue::None)
      for (const uint32_t groups : {kPaired256WaveGroupsPerCore * gpuCores_, n / 256})
        append({LinearTile::Paired256, std::min(groups, n / 256), LinearSimdgroups::Four});
  }
  return result;
}

LinearScratchSize Q4Linear::decodeScratchSize(LinearWorkload w) const {
  auto size = LinearPlan(w, baseline(w)).scratchSize();
  const auto selected = plan(w).scratchSize();
  size.input = std::max(size.input, selected.input);
  size.sums = std::max(size.sums, selected.sums);
  size.partials = std::max(size.partials, selected.partials);
  size.counters = std::max(size.counters, selected.counters);
  return size;
}

void Q4Linear::requireBuffers(const LinearBuffers &b, const Q4Projection &p,
    const LinearPlan &selected, const Q4Projection *gate) {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  requireProjection(p, w.matrix);
  requireBytes(b.input, uint64_t{selected.storageRows()} * k * 2);
  requireBytes(b.output, uint64_t{selected.storageRows()} * n * 2);
  requireBytes(b.sums, selected.sumsBytes());
  requireBytes(b.gateScratch, selected.gateScratchBytes());
  requireBytes(b.downSums, selected.downSumsBytes());
  if (w.epilogue == LinearEpilogue::Residual)
    requireBytes(b.residual, uint64_t{selected.storageRows()} * n * 2);
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!gate) throw std::invalid_argument("Q4 gate projection is missing");
    requireProjection(*gate, w.matrix);
  } else if (gate) throw std::invalid_argument("unexpected Q4 gate projection");
  const auto scratchSize = selected.scratchSize();
  requireBytes(b.scratch.input, scratchSize.input);
  requireBytes(b.scratch.sums, scratchSize.sums);
  requireBytes(b.scratch.partials, scratchSize.partials);
  requireBytes(b.scratch.counters, scratchSize.counters);
}

void Q4Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Q4Projection &p, const LinearPlan &selected, const Q4Projection *gate,
    Q4DispatchStats *stats) const {
  requireBuffers(b, p, selected, gate);
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  if (selected.usesSimdgroup()) {
    if (!b.inputPrepared)
      graph.add("decode_linear_q4_prepare", {b.input, b.scratch.input, b.scratch.sums},
                k, {k / 32, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1}, {128, 1, 1});
    const auto &first = gate ? *gate : p;
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, first.weights,
        first.scales, first.biases, b.output, b.scratch.sums,
        b.scratch.partials, b.scratch.counters};
    if (gate) bindings.insert(bindings.end(), {p.weights, p.scales, p.biases});
    else if (w.epilogue == LinearEpilogue::Residual) bindings.push_back(b.residual);
    graph.add(selected.pipeline(), std::move(bindings),
        Q4Params{n, k, selected.configuration().splits},
        {selected.configuration().groups, selected.configuration().splits,
         w.rows / SPLASH_TARGET_VERIFY_ROWS}, {128, 1, 1});
    if (stats) account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1);
    return;
  }
  const bool precomputedSums =
      selected.configuration().tile == LinearTile::Split32PrecomputedSums;
  if (precomputedSums && !b.sumsPrepared)
    graph.add(std::array{"decode_linear_q4_split_sums", "decode_linear_q4_split_sums_m16",
                         "decode_linear_q4_split_sums_m24", "decode_linear_q4_split_sums_m32"}[w.rows / 8 - 1],
              {b.input, b.scratch.sums}, k,
              {k / 256, 1, 1}, {selected.threadsPerThreadgroup(), 1, 1});
  const auto dispatch = [&](std::string_view name,
      std::initializer_list<metal::MetalBuffer> bindings) {
    if (w.phase == LinearPhase::Prefill)
      graph.add(name, bindings,
          Q4PrefillParams{w.matrix.outputSize, w.matrix.inputSize},
          {selected.storageRows() / kPrefillRows, n / selected.tileColumns(), 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    else {
      const uint32_t groups = selected.configuration().groups;
      graph.add(name, bindings, Q4Params{n, k, groups}, {groups, 1, 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    }
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (b.downSums) {
      // SPLASH_M16_FFN_SUMS: the 16-row gate/up publishes the down input sums too.
      const bool m16 = selected.pipeline() == "decode_linear_q4_n256_gate_up_m16";
      if (selected.pipeline() != "decode_linear_q4_n256_gate_up" && !m16)
        throw std::invalid_argument("output sums require the M8/M16 N256 gate/up kernel");
      requireBytes(b.downSums, uint64_t{n} / 64 * (m16 ? 16 : 8) * sizeof(float));
      dispatch(m16 ? "decode_linear_q4_n256_gate_up_m16_sums" : "decode_linear_q4_n256_gate_up_sums",
          {b.input, gate->weights, gate->scales, gate->biases,
           b.output, p.weights, p.scales, p.biases, b.downSums});
    } else if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales, gate->biases,
          b.output, p.weights, p.scales, p.biases});
    else {
      dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales, gate->biases, b.gateScratch});
      dispatch(selected.secondPipeline(), {b.input, p.weights, p.scales, p.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
        b.gateScratch, b.output, b.sums, b.downSums});
  else if (w.epilogue == LinearEpilogue::Residual) {
    if (w.phase == LinearPhase::Prefill)
      dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.residual, b.output, b.sums});
    else if (precomputedSums)
      dispatch(b.m16HoistFooter &&
                       selected.pipeline() == "decode_linear_q4_n32_split4_precomputed_sums_residual_m16"
                   ? "decode_linear_q4_n32_split4_precomputed_sums_residual_m16_hoist_ftr"
                   : selected.pipeline(),
          {b.input, p.weights, p.scales, p.biases, b.residual, b.output, b.scratch.sums});
    else dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
        b.residual, b.output});
  } else if (w.phase == LinearPhase::Prefill)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.output, b.sums});
  else if (precomputedSums)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
        b.output, b.scratch.sums});
  else if (b.sumsPrepared && (selected.configuration().tile == LinearTile::Paired128 ||
                              (w.rows == 16 && selected.configuration().tile == LinearTile::N128) ||
                              // SPLASH_M24_INPUT_SUMS: 24/32 rows, sequential N128/N256 plans.
                              (w.rows >= 24 && (selected.configuration().tile == LinearTile::N128 ||
                                                selected.configuration().tile == LinearTile::N256)))) {
    // SPLASH_INPUT_FUSED_SUMS (M8) / SPLASH_M16_INPUT_SUMS (M16): the preceding
    // RMS already wrote this input's sums ([group * rows + row]). The split-K N32
    // consumer reads them directly: a full-K paired consumer without its own sum
    // refills stalls on every core reading the same input lines in lockstep
    //. Reassociates K sums.
    const bool multi = w.rows > SPLASH_TARGET_VERIFY_ROWS;
    if ((w.rows != 8 && w.rows != 16 && w.rows != 24 && w.rows != 32) || k > 5120 ||
        k % kSplitInputBlock)
      throw std::invalid_argument("prepared input sums require 8-32 rows, staged width and K % 1024 == 0");
    requireBytes(b.scratch.sums, uint64_t{w.rows} * (k / kQuantGroup) * sizeof(float));
    // SPLASH_SPLIT4_INPUT_DIV (default 1 = divisor 4; =N uses divisor N; 0 = off; read at every
    // encode): an M8 input whose N/32 one-tile groups overflow a resident wave
    // (more than kSplit4InputWaveGroupsPerCore = 12 per core, measured on 40 cores: the GDN input's
    // 520 does, attention's 448 doesn't) runs as N/32/4 groups of 4 tiles each. Exact: host-only
    // grid change, the same tile body (it already loops tile += persistent_groups); oracle
    // identical. B1 lockstep -0.203 ms/step. Below one wave it changes nothing.
    const uint32_t tiles = n / 32;
    uint32_t groups = tiles;
    if (!multi) {
      const char *value = std::getenv("SPLASH_SPLIT4_INPUT_DIV");
      const uint32_t d = !value || std::string_view(value) == "1"
          ? 4u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
      if (d > 1 && tiles % d == 0 && tiles > kSplit4InputWaveGroupsPerCore * gpuCores_)
        groups = tiles / d;
    }
    graph.add(std::array{"decode_linear_q4_n32_split4_precomputed_sums",
                         "decode_linear_q4_n32_split4_precomputed_sums_m16",
                         "decode_linear_q4_n32_split4_precomputed_sums_m24",
                         "decode_linear_q4_n32_split4_precomputed_sums_m32"}[w.rows / 8 - 1],
              {b.input, p.weights, p.scales, p.biases, b.output, b.scratch.sums},
              Q4Params{n, k, groups}, {groups, 1, 1}, {multi ? 256u : 128u, 1, 1});
  } else dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.output});
  if (stats && w.phase == LinearPhase::Decode)
    account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, selected.secondPipeline().empty() ? 1 : 2);
}

namespace {
// The full projection each lane count's grouped K/V entry replaces (linear_q4_context_kv.metal): at 2-4 lanes one
// 128-column tile per group over all N/128 tiles, so both run the same tile call on the same tiles.
struct ContextKvRoute final {
  LinearTile tile;
  LinearSimdgroups simdgroups;
  uint32_t threads;
  std::string_view full, grouped;
};
constexpr std::array<ContextKvRoute, 4> kContextKvRoutes{{
    {LinearTile::Paired128, LinearSimdgroups::Eight, 256, "decode_linear_q4_n128_paired",
     "decode_linear_q4_n128_paired_context_grouped"},
    {LinearTile::N128, LinearSimdgroups::Eight, 256, "decode_linear_q4_n128_m16",
     "decode_linear_q4_n128_m16_context_grouped"},
    {LinearTile::N128, LinearSimdgroups::Four, 128, "decode_linear_q4_n128_m24_sg4",
     "decode_linear_q4_n128_m24_sg4_context_grouped"},
    {LinearTile::N128, LinearSimdgroups::Eight, 256, "decode_linear_q4_n128_m32",
     "decode_linear_q4_n128_m32_context_grouped"},
}};
} // namespace

bool Q4Linear::admitsContextKv(LinearMatrix matrix, uint32_t lanes, uint32_t layers, uint32_t kvColumn) const {
  const uint32_t n = matrix.outputSize, k = matrix.inputSize;
  // The decode shapes validate() accepts (plan() would throw on others): another drafter shape falls back to the full
  // projections instead of throwing.
  if (!lanes || lanes > kContextKvRoutes.size() || !layers || layers > kMaxContextKvLayers || kvColumn % 128 ||
      !n || n % 256 || kvColumn >= n || !k || k % 256)
    return false;
  const LinearPlan selected = plan(decode(matrix, lanes, LinearEpilogue::None));
  const ContextKvRoute &route = kContextKvRoutes[lanes - 1];
  const LinearConfig config = selected.configuration();
  // One lane: any persistent Paired128 grid, since a Paired128 tile's bytes don't depend on how many groups
  // stream the tiles, and the K/V kernels compute the K/V tiles one per group. Two to four lanes: only the
  // route's one-tile-per-group plan over all N/128 tiles, the one gated.
  const bool grid = lanes == 1 ? config.tile == route.tile
                               : config.tile == route.tile && config.simdgroups == route.simdgroups &&
                                     config.groups == n / 128 && config.splits == 1;
  return grid && selected.threadsPerThreadgroup() == route.threads && selected.pipeline() == route.full &&
         selected.secondPipeline().empty();
}

void Q4Linear::addContextKv(metal::CommandGraph &graph, metal::MetalBuffer input,
    std::span<const Q4Projection *const> projections, std::span<const metal::MetalBuffer> outputs,
    LinearMatrix matrix, uint32_t kvColumn, uint32_t lanes, LinearScratch scratch,
    std::optional<uint32_t> layer) const {
  const auto layers = static_cast<uint32_t>(projections.size());
  if (!admitsContextKv(matrix, lanes, layers, kvColumn) || outputs.size() != layers ||
      (layer && (*layer >= layers || lanes != 1)))
    throw std::invalid_argument("context K/V projection is not admitted");
  const LinearPlan selected = plan(decode(matrix, lanes, LinearEpilogue::None));
  for (uint32_t i = 0; i < layers; ++i) {
    if (!projections[i]) throw std::invalid_argument("context K/V projection is missing");
    // the buffers addDecodeBatch(..., inputPrepared = true) gives add() for the full projection
    requireBuffers({input, outputs[i], {}, {}, {}, {}, scratch, true}, *projections[i], selected, nullptr);
    if (outputs[i].overlaps(input)) throw std::invalid_argument("context K/V output overlaps its input");
    for (uint32_t j = 0; j < i; ++j)
      if (outputs[i].overlaps(outputs[j])) throw std::invalid_argument("context K/V outputs overlap");
  }
  // The K/V kernels guard on the full projection's one-tile-per-group grid (N/128), whatever persistent grid the
  // full projection itself runs, and compute tiles kvColumn/128 .. N/128-1.
  const uint32_t tiles = matrix.outputSize / 128, kvFirst = kvColumn / 128, kvTiles = tiles - kvFirst;
  const uint32_t threads = selected.threadsPerThreadgroup();
  if (layer) {
    const Q4Projection &p = *projections[*layer];
    graph.add("decode_linear_q4_n128_paired_context_tail", {input, p.weights, p.scales, p.biases, outputs[*layer]},
              Q4ContextKvParams{matrix.outputSize, matrix.inputSize, tiles, 1, kvFirst, kvTiles}, {kvTiles, 1, 1},
              {threads, 1, 1});
    return;
  }
  // Seven slots: unused ones repeat the last layer's buffers, which no group of the dispatch reads.
  std::vector<metal::MetalBuffer> bindings{input};
  for (uint32_t i = 0; i < kMaxContextKvLayers; ++i) {
    const Q4Projection &p = *projections[std::min(i, layers - 1)];
    bindings.insert(bindings.end(), {p.weights, p.scales, p.biases});
  }
  for (uint32_t i = 0; i < kMaxContextKvLayers; ++i) bindings.push_back(outputs[std::min(i, layers - 1)]);
  graph.add(kContextKvRoutes[lanes - 1].grouped, std::move(bindings),
            Q4ContextKvParams{matrix.outputSize, matrix.inputSize, tiles, layers, kvFirst, kvTiles},
            {layers * kvTiles, 1, 1}, {threads, 1, 1});
}

void Q4Linear::addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input,
    metal::MetalBuffer sums, LinearMatrix matrix, uint32_t rows) const {
  validate({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None});
  const uint32_t tiles = (rows + kPrefillRows - 1) / kPrefillRows;
  const uint64_t storageRows = uint64_t{tiles} * kPrefillRows;
  requireBytes(input, storageRows * matrix.inputSize * 2);
  requireBytes(sums, storageRows * (matrix.inputSize / kQuantGroup) * 4);
  graph.add("prefill_linear_q4_sums32", {input, sums},
      Q4PrefillParams{matrix.outputSize, matrix.inputSize}, {tiles, 1, 1});
}
void Q4Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, metal::MetalBuffer sums,
    LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, {}, {}, {}}, p,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None}));
}
void Q4Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer residual, metal::MetalBuffer output,
    metal::MetalBuffer sums, LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, residual, {}, {}}, p,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::Residual}));
}
void Q4Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &up, metal::MetalBuffer gateScratch, metal::MetalBuffer output,
    metal::MetalBuffer sums, metal::MetalBuffer downSums, LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, {}, gateScratch, downSums}, up,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::UpWithGate}));
}
void Q4Linear::addDecode(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, LinearMatrix matrix, LinearScratch scratch) const {
  add(graph, {input, output, {}, {}, {}, {}, scratch}, p, plan(decode(matrix, 1, LinearEpilogue::None)));
}
void Q4Linear::addDecodeBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, LinearMatrix matrix,
    uint32_t lanes, Q4DispatchStats &stats, LinearScratch scratch, bool inputPrepared,
    bool sumsPrepared) const {
  add(graph, {input, output, {}, {}, {}, {}, scratch, inputPrepared, sumsPrepared}, p,
      plan(decode(matrix, lanes, LinearEpilogue::None)), nullptr, &stats);
}
void Q4Linear::addResidualBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer residual, metal::MetalBuffer output,
    LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats, LinearScratch scratch,
    bool inputPrepared, bool sumsPrepared, bool m16HoistFooter) const {
  add(graph, {input, output, {}, residual, {}, {}, scratch, inputPrepared, sumsPrepared,
      m16HoistFooter}, p,
      plan(decode(matrix, lanes, LinearEpilogue::Residual)), nullptr, &stats);
}
void Q4Linear::addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &gate, const Q4Projection &up, metal::MetalBuffer gateScratch,
    metal::MetalBuffer output, LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats,
    LinearScratch scratch, bool inputPrepared, bool prepareDownSums) const {
  add(graph, {input, output, {}, {}, gateScratch,
      prepareDownSums ? scratch.sums : metal::MetalBuffer{}, scratch, inputPrepared},
      up, plan(decode(matrix, lanes, LinearEpilogue::GateUp)), &gate, &stats);
}

} // namespace splash::ops

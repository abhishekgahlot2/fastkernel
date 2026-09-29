#pragma once

#include "model/ModelFactory.hpp"
#include "tuning/LinearTuning.hpp"
#include "tuning/MoeTuning.hpp"

#include <array>

namespace splash::ops::tuning {

// Tune the full prefill budget plus the chunk sizes the scheduler actually
// emits: contended halves of the budget, and the small tails around state
// boundaries. Row counts outside this set keep the shipped operator defaults
// and still execute their actual row count.
inline constexpr std::array<uint32_t, 5> kPrefillProbeRows{
    64, 256, 512, 1024, SPLASH_PREFILL_TOKEN_BUDGET};
inline constexpr std::array<uint32_t, 4> kDecodeProbeWidths{1, 2, 3, 4};

} // namespace splash::ops::tuning

namespace splash::model {

struct TuningWorkloads final {
  std::vector<ops::tuning::LinearTuningInput> linear;
  std::vector<ops::tuning::MoeTuningInput> moe;
  ops::AttentionShape targetAttention;
  ops::DraftAttentionShape draftAttention;
};

// Describe the semantic operations used by the loaded target/draft pair.
// Operators supply the bounded probe sizes and own measurement/selection.
// Equal operation shapes retain evenly spaced distinct immutable weight
// bundles across layer order, capped at the operator's representative limit.
// Tied and re-created views deduplicate by allocation/offset/length,
// without reading weight data. No weights, dispatch code, device policy or
// model-name table is duplicated.
[[nodiscard]] TuningWorkloads collectTuningWorkloads(
    const ModelPackage &package, std::span<const uint32_t> prefillRows,
    std::span<const uint32_t> decodeWidths);

} // namespace splash::model

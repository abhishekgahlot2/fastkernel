#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace splash::ops::tuning {

// IDs refer to operator-owned typed configurations and measured workloads.
// This helper never interprets a kernel configuration or encodes GPU work.
struct CandidateId final {
  uint32_t value = 0;
  bool operator==(const CandidateId &) const = default;
};

struct WorkloadId final {
  uint32_t value = 0;
  bool operator==(const WorkloadId &) const = default;
};

inline constexpr CandidateId kBaseline{};
inline constexpr size_t kMinPairedSamples = 12;
inline constexpr size_t kMaxPairedSamples = 64;

enum class MeasurementOrder : uint8_t { BaselineFirst, CandidateFirst };

[[nodiscard]] constexpr MeasurementOrder measurementOrder(size_t pair) noexcept {
  return pair % 2 == 0 ? MeasurementOrder::BaselineFirst
                       : MeasurementOrder::CandidateFirst;
}

// Each pair must measure the same warmed workload through the production
// entry point. Record execution order rather than sorting samples by time.
// Either order may start a run, but consecutive pairs must reverse it.
struct PairedTiming final {
  double baselineSeconds = 0;
  double candidateSeconds = 0;
  MeasurementOrder first = MeasurementOrder::BaselineFirst;
  bool underPressure = false;
};

// These are conservative engineering thresholds, not a statistical confidence
// interval or a guarantee about unmeasured workloads. Timing spread is the
// central 80% range / median; paired gain spread is the central 80% gain range.
// The conservative gain subtracts that entire paired range from median gain.
struct Policy final {
  // A policy may require more samples, but cannot weaken the 12-pair floor.
  size_t minimumPairs = kMinPairedSamples;
  double maximumRelativeTimingSpread = 0.10;
  double maximumPairedGainSpread = 0.05;
  double minimumMeanImprovement = 0.03;
};

enum class TimingVerdict : uint8_t {
  Improved,
  Stable,
  InvalidPolicy,
  InsufficientSamples,
  TooManySamples,
  InvalidTiming,
  InvalidOrder,
  UnderPressure,
  Noisy,
  Regressed,
  Uncertain,
};

[[nodiscard]] constexpr std::string_view timingVerdictName(TimingVerdict verdict) noexcept {
  switch (verdict) {
  case TimingVerdict::Improved: return "improved";
  case TimingVerdict::Stable: return "stable";
  case TimingVerdict::InvalidPolicy: return "invalid_policy";
  case TimingVerdict::InsufficientSamples: return "insufficient_samples";
  case TimingVerdict::TooManySamples: return "too_many_samples";
  case TimingVerdict::InvalidTiming: return "invalid_timing";
  case TimingVerdict::InvalidOrder: return "invalid_order";
  case TimingVerdict::UnderPressure: return "under_pressure";
  case TimingVerdict::Noisy: return "noisy";
  case TimingVerdict::Regressed: return "regressed";
  case TimingVerdict::Uncertain: return "uncertain";
  }
  return "unknown";
}

struct TimingAssessment final {
  TimingVerdict verdict = TimingVerdict::InsufficientSamples;
  double baselineMedianSeconds = 0;
  double candidateMedianSeconds = 0;
  double medianPairedGain = 0;
  double pairedGainSpread = 0;
  double baselineRelativeSpread = 0;
  double candidateRelativeSpread = 0;
  double conservativeGain = 0;

  [[nodiscard]] bool qualified() const noexcept {
    return verdict == TimingVerdict::Improved || verdict == TimingVerdict::Stable;
  }
};

[[nodiscard]] TimingAssessment evaluate(std::span<const PairedTiming> samples,
                                        const Policy &policy = {}) noexcept;

struct WorkloadMeasurements final {
  WorkloadId id;
  std::span<const PairedTiming> samples;
  // Caller-proven structural identity, not an inference from neutral/noisy
  // timings. Identity requires no samples and contributes exactly zero gain.
  bool equivalentToBaseline = false;
};

struct CandidateMeasurements final {
  CandidateId id;
  std::span<const WorkloadMeasurements> workloads;
};

enum class SelectionVerdict : uint8_t { Baseline, Selected, InvalidInput };

struct Selection final {
  SelectionVerdict verdict = SelectionVerdict::Baseline;
  CandidateId candidate = kBaseline;
  double conservativeMeanGain = 0;
  double worstWorkloadGain = 0;
};

// Every required ID must appear exactly once in each candidate, with no extra
// IDs. Missing/duplicate workloads disqualify that candidate. Duplicate
// candidate IDs, baseline IDs, duplicate required IDs or invalid policy make
// the whole input invalid. All such outcomes retain the shipped baseline.
// Every workload must qualify or explicitly be structurally identical with
// empty samples; identity still counts in the equally weighted mean and worst
// gain. Their conservative mean must meet minimumMeanImprovement, so an
// all-identical candidate cannot win. Ties prefer the greater worst-workload
// gain, then the smaller candidate ID, independently of input order.
// The winner still requires caller-owned production graph confirmation.
[[nodiscard]] Selection
selectCandidate(std::span<const CandidateMeasurements> candidates,
                std::span<const WorkloadId> requiredWorkloads,
                const Policy &policy = {}) noexcept;

} // namespace splash::ops::tuning

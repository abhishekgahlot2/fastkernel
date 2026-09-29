#pragma once

#include "ops/ExecutionPlans.hpp"
#include "tuning/Measurement.hpp"

namespace splash::ops::tuning {

// Exact fixture inputs, deliberately distinct from installable policy keys.
struct PrefillAttentionWorkload final {
  AttentionShape shape;
  uint32_t rows = 0;
  uint32_t historyTokens = 0;
  auto operator<=>(const PrefillAttentionWorkload &) const = default;
};
struct VerifyAttentionWorkload final {
  AttentionShape shape;
  uint32_t lanes = 0;
  std::array<uint32_t, 4> historyTokens{};
  auto operator<=>(const VerifyAttentionWorkload &) const = default;
};

struct PrefillAttentionTuningResult final {
  struct ProbeChoice { PrefillAttentionWorkload workload; PrefillAttentionConfig configuration; } choice;
  std::vector<MeasurementResult> measurements;
  bool complete = false;
  std::exception_ptr failure;
  // Raw measurement pairs time this many complete production attention graphs
  // in ONE command. They are batch timings, not per-token latency.
  uint32_t repetitions = 1;
  // Explicit typed-plan equality to baseline; there are no timing samples for
  // these IDs. An empty measurement vector alone never means equivalence.
  std::vector<CandidateId> equivalentCandidates{};
};
struct VerifyAttentionTuningResult final {
  struct ProbeChoice { VerifyAttentionWorkload workload; VerifyAttentionConfig configuration; } choice;
  std::vector<MeasurementResult> measurements;
  bool complete = false;
  std::exception_ptr failure;
  uint32_t repetitions = 1;
  std::vector<CandidateId> equivalentCandidates{};
};

struct PrefillAttentionPolicyResult final {
  PrefillAttentionChoice choice;
  std::vector<PrefillAttentionTuningResult> probes;
  bool complete = false;
  std::exception_ptr failure;
};
struct VerifyAttentionPolicyResult final {
  VerifyAttentionChoice choice;
  std::vector<VerifyAttentionTuningResult> probes;
  bool complete = false;
  std::exception_ptr failure;
};

// Prefill calibrates only the fixed 2048-row chunk over four histories.
// Candidates change the balanced split count and its explicit scratch bound.
// Verify includes mixed lane orders. Histories are never serving keys.
[[nodiscard]] std::array<PrefillAttentionWorkload, 4>
prefillAttentionPolicyWorkloads(AttentionShape shape);
[[nodiscard]] std::vector<VerifyAttentionWorkload>
verifyAttentionPolicyWorkloads(VerifyAttentionPolicy policy);

// Candidate IDs are baseline-first indices. Preserve operator order after the
// selected baseline; a baseline outside the precompiled candidates is rejected.
[[nodiscard]] std::vector<VerifyAttentionConfig>
verifyAttentionTuningCandidates(VerifyAttentionConfig baseline);

// CPU-only aggregation. Every required probe must be complete; every candidate
// must qualify on every measured case, with explicit graph-equivalent cases
// contributing unchanged evidence rather than fabricated timing pairs. Verify
// evidence is read in the tuning order of the given baseline, which every
// probe must have been measured against; insufficient evidence keeps it.
[[nodiscard]] PrefillAttentionConfig selectPrefillAttentionPolicy(
    std::span<const PrefillAttentionTuningResult> probes, const Policy &policy = {});
[[nodiscard]] VerifyAttentionConfig selectVerifyAttentionPolicy(
    std::span<const VerifyAttentionTuningResult> probes, VerifyAttentionConfig baseline,
    const Policy &policy = {});

// Uses the same exact-workload samplers below, one admitted fixture at a time.
// options.maximumWallSeconds bounds the entire policy suite, not each probe.
[[nodiscard]] PrefillAttentionPolicyResult tunePrefillAttentionPolicy(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    PrefillAttentionPolicy policy, const MeasurementOptions &options = {},
    const MeasurementStop &underPressure = {}, const MeasurementStop &shouldStop = {});
[[nodiscard]] VerifyAttentionPolicyResult tuneVerifyAttentionPolicy(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    VerifyAttentionPolicy policy, const MeasurementOptions &options = {},
    const MeasurementStop &underPressure = {}, const MeasurementStop &shouldStop = {});

// Exact, aligned single-allocation requirements, including the reference
// outputs for baseline equivalence and repeat-idempotence qualification.
// These CPU-only helpers reject unsupported shapes/rows/histories.
[[nodiscard]] uint64_t prefillAttentionTuningFixtureBytes(
    PrefillAttentionWorkload workload);
[[nodiscard]] uint64_t verifyAttentionTuningFixtureBytes(
    VerifyAttentionWorkload workload);

// Offline only. Each trial uses deterministic Page32 Q8 history and the same
// production store/split/reduce graph as serving. Fixture initialization and
// restoration and output qualification are outside the recorded GPU/wall
// timings; wall time includes production graph encoding and submission.
// Single/batched baseline and candidate outputs are qualified BEFORE paired
// sampling; no full CPU output scan runs between its timed commands. Each
// candidate and its repeated graph must satisfy the same established
// numerical-equivalence tolerance; bitwise output identity is not required.
// One baseline qualification pilot selects a fixed 1..16 repetition count for
// both IDs, targeting approximately 5ms of GPU work per measured command.
// Admission denial, cancellation, pressure or run failure
// leaves the baseline choice and complete=false; no backend failure is retried.
// Independent GPU/wall selections must agree on the same qualified winner.
[[nodiscard]] PrefillAttentionTuningResult tunePrefillAttention(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    PrefillAttentionWorkload workload,
    const MeasurementOptions &options = {},
    const MeasurementStop &underPressure = {},
    const MeasurementStop &shouldStop = {});
[[nodiscard]] VerifyAttentionTuningResult tuneVerifyAttention(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    VerifyAttentionWorkload workload,
    const MeasurementOptions &options = {},
    const MeasurementStop &underPressure = {},
    const MeasurementStop &shouldStop = {});

} // namespace splash::ops::tuning

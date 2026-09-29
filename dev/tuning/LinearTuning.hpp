#pragma once

#include "ops/Linear.hpp"
#include "tuning/Measurement.hpp"

#include <optional>

namespace splash::ops::tuning {

inline constexpr size_t kMaximumLinearTuningRepresentatives = 8;

struct LinearTuningWeights final {
  Q4Projection projection;
  // Required only for GateUp. UpWithGate consumes deterministic gate scratch
  // as its semantic input; it does not include an external gate projection.
  std::optional<Q4Projection> gate;
};

struct LinearTuningInput final {
  LinearWorkload workload;
  // 1..8 immutable representative views supplied by the model. Each timed
  // batch visits them in this fixed order before repeating the ring. The
  // operator never copies weights or invents a cache-flushing workload.
  std::vector<LinearTuningWeights> weights;
};

struct LinearTuningResult final {
  LinearChoice choice;
  std::vector<MeasurementResult> measurements;
  bool complete = false;
  std::exception_ptr failure;
  // Actual complete operators per timed command and supplied weight views.
  // Paired seconds cover the whole batch, not a divided per-operator estimate.
  // repetitions remains zero if qualification/pilot did not finish.
  uint32_t repetitions = 0;
  uint32_t representativeCount = 0;
};

// Exact admitted shared-fixture bytes, including the reference outputs and
// maximum workspace of the bounded candidate set. CPU-only; invalid shapes
// and a fixture exceeding a supplied nonzero device buffer limit throw.
[[nodiscard]] uint64_t linearTuningFixtureBytes(
    const DeviceCapabilities &device, LinearWorkload workload);

// Offline only: real supplied weights, deterministic BF16 inputs and the
// production Q4Linear graph. Candidate IDs are their baseline-first plan index.
// Every representative is qualified against its own baseline before timing.
// Existing baseline qualification timings select a fixed batch of 1..16 whole
// operators, rounded to complete representative rings, targeting about 5 ms.
// A final batched qualification verifies that scratch reuse is equivalent.
// Restoring scratch/output and exact finite output checks are outside timing;
// wall time covers graph encoding and synchronous submission. The wall budget
// covers this entire sweep, including warmup and output qualification.
// Both timing metrics must independently select the same non-baseline winner.
// complete includes timing rejections, but excludes interrupted/failed sweeps.
// Metal failures are preserved and never retried. No result is persisted here;
// any winner still requires caller-owned production graph confirmation.
[[nodiscard]] LinearTuningResult tuneLinear(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    const LinearTuningInput &input,
    const MeasurementOptions &options = {},
    const MeasurementStop &underPressure = {},
    const MeasurementStop &shouldStop = {});

} // namespace splash::ops::tuning

#pragma once

#include "ops/ExecutionPlans.hpp"
#include "tuning/Measurement.hpp"

namespace splash::ops::tuning {

struct DraftAttentionTuningResult final {
  DraftAttentionChoice choice;
  std::vector<MeasurementResult> measurements;
  bool complete = false;
  std::exception_ptr failure;
};

[[nodiscard]] uint64_t draftAttentionTuningFixtureBytes(
    DraftAttentionWorkload workload);

// Measures the operator-owned stages of one draft layer: two prepare and two
// residual convolutions, QKV preparation, the unchanged attention core, and
// reorder. Deterministic semantic inputs stand in for the intervening Linear
// outputs; no Linear dispatch is copied or measured here. Short and wrapped
// full-window histories must both qualify because this key has no history.
// Only the surrounding group count changes; the core tile/window never does.
// One exact admitted shared fixture includes all references/restoration data.
// Reset and finite, bit-exact qualification are outside timing; wall time
// includes production encoding and submission. Both metrics independently
// must select the same winner. Denial/control/failure retains baseline and
// complete=false; Metal failures are never retried. Nothing is persisted here.
[[nodiscard]] DraftAttentionTuningResult tuneDraftAttention(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    DraftAttentionWorkload workload,
    const MeasurementOptions &options = {},
    const MeasurementStop &underPressure = {},
    const MeasurementStop &shouldStop = {});

} // namespace splash::ops::tuning

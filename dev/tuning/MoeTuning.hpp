#pragma once

#include "ops/ExecutionPlans.hpp"
#include "tuning/Measurement.hpp"

#include <cstddef>
#include <vector>

namespace splash::ops::tuning {

inline constexpr size_t kMaximumMoeTuningRepresentatives = 8;

struct MoeTuningInput final {
  MoeWorkload workload;
  // Borrow the package's actual router and expert views. Tuning never copies
  // weight slabs or substitutes a routing distribution by changing weights.
  std::vector<MoeWeights> weights;
};

struct MoeTuningResult final {
  MoeChoice choice;
  // Candidate 1 is the alternate precompiled tile. Workload 0 uses distinct
  // deterministic rows; workload 1 repeats the same row. GPU and wall samples
  // remain separate, with their original paired execution order.
  std::vector<MeasurementResult> measurements;
  // All requested pairs finished; a rejected timing verdict can still mean
  // a complete sweep. This does not certify an improvement or authorize a
  // profile write without the caller's production-graph confirmation.
  bool complete = false;
  std::exception_ptr failure;
  // Raw pairs are whole-command timings across this many complete MoE graphs,
  // rotating borrowed representatives identically within each A/B pair.
  uint32_t repetitions = 1;
  uint32_t representativeCount = 0;
};

// Includes all candidate scratch maxima and two distribution baseline mirrors.
// Representatives share the same mutable scratch; fixture size does not grow
// with the number of borrowed weight views. The total is rounded to the
// physical shared-buffer allocation granule.
[[nodiscard]] uint64_t moeTuningFixtureBytes(const MoeWorkload &workload);

// Measures the complete production MoE operator, including GPU routing,
// grouping, gathering, projections and combine. Both input distributions
// must independently qualify on GPU and wall time before a choice changes.
// Learned routers determine the routes; no route values return to the CPU.
// All representatives qualify on both distributions before sampling. Batched
// outputs must exactly equal the last representative's single-run output.
// A baseline qualification mean chooses one fixed count of complete rings,
// totaling 1..16 repetitions and at least one visit per representative. Both
// A/B commands visit the same order. No CPU output scans occur between pairs.
// Denied fixture admission or interrupted measurement retains the baseline.
// Execution failures are retained in measurements; the caller decides whether
// its backend can continue. No retry, policy installation or persistence here.
// maximumWallSeconds bounds the whole call, including fixture setup and probes.
[[nodiscard]] MoeTuningResult tuneMoe(
    metal::MetalBackend &backend,
    const metal::AllocationAdmission &admitAllocation,
    const MoeTuningInput &input, const MeasurementOptions &options = {},
    const MeasurementStop &underPressure = {},
    const MeasurementStop &shouldStop = {});

} // namespace splash::ops::tuning

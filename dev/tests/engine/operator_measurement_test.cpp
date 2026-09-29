#include "tuning/Measurement.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace splash::ops::tuning;

constexpr CandidateId kCandidate{7};
constexpr WorkloadId kWorkload{19};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

RunTiming stableTiming(CandidateId candidate) {
  return candidate == kBaseline ? RunTiming{1, 2, false}
                                : RunTiming{0.875, 1.75, false};
}

void diagnosticNames() {
  constexpr std::array statuses{"completed", "invalid_input", "cancelled",
      "budget_exceeded", "under_pressure", "invalid_timing", "rejected", "run_failed"};
  for (size_t index = 0; index < statuses.size(); ++index)
    require(measurementStatusName(static_cast<MeasurementStatus>(index)) == statuses[index],
            "measurement diagnostic status name changed");
  constexpr std::array verdicts{"improved", "stable", "invalid_policy",
      "insufficient_samples", "too_many_samples", "invalid_timing", "invalid_order",
      "under_pressure", "noisy", "regressed", "uncertain"};
  for (size_t index = 0; index < verdicts.size(); ++index)
    require(timingVerdictName(static_cast<TimingVerdict>(index)) == verdicts[index],
            "timing diagnostic verdict name changed");
  require(measurementStatusName(static_cast<MeasurementStatus>(255)) == "unknown" &&
              timingVerdictName(static_cast<TimingVerdict>(255)) == "unknown",
          "invalid diagnostic enum did not have a bounded fallback");
}

void comparableWarmupPairs() {
  splash::model::WarmupStepResult baseline;
  baseline.completed = true;
  baseline.wallSeconds = 2;
  baseline.lanes.resize(4);
  for (size_t lane = 0; lane < baseline.lanes.size(); ++lane) {
    auto &result = baseline.lanes[lane];
    result.step.requestId = lane + 1;
    result.step.outputTokens = {10, 11, 12, 13};
    result.step.draftedTokens = 7;
    result.step.acceptedDraftTokens = 3;
    result.pendingToken = 14;
    result.committedTokens = 103;
  }
  auto candidate = baseline;
  candidate.wallSeconds = 0.25;
  for (auto &lane : candidate.lanes) {
    lane.step.outputTokens = {20, 21, 22, 23};
    lane.pendingToken = 24;
  }
  std::vector<PairedTiming> gpu, wall;
  for (size_t pair = 0; pair < kMinPairedSamples; ++pair) {
    const auto order = measurementOrder(pair);
    const auto [gpuPair, wallPair] =
        pairWarmupMeasurements(baseline, 1, candidate, 0.125, order);
    require(gpuPair.first == order && wallPair.first == order &&
                gpuPair.baselineSeconds == 1 && gpuPair.candidateSeconds == 0.125 &&
                wallPair.baselineSeconds == 2 && wallPair.candidateSeconds == 0.25,
            "comparable warmup pair lost its metric or execution order");
    gpu.push_back(gpuPair);
    wall.push_back(wallPair);
  }
  require(evaluate(gpu).verdict == TimingVerdict::Improved &&
              evaluate(wall).verdict == TimingVerdict::Improved,
          "token identity alone prevented a comparable-work confirmation");

  const auto reject = [&](const auto &before, const auto &after,
                           std::string_view diagnostic) {
    bool rejected = false;
    try {
      (void)pairWarmupMeasurements(before, 1, after, 0.125,
                                   MeasurementOrder::CandidateFirst);
    } catch (const std::runtime_error &error) {
      rejected = std::string_view(error.what()).find(diagnostic) != std::string_view::npos;
    }
    require(rejected, "different warmup work was admitted as a faster timing pair");
  };
  // Only the final lane differs: a first-lane-only check would accept every
  // apparently faster candidate below, despite its changed decode work.
  for (unsigned change = 0; change < 4; ++change) {
    auto different = candidate;
    auto &last = different.lanes.back();
    if (change == 0) --last.step.acceptedDraftTokens;
    if (change == 1) --last.committedTokens;
    if (change == 2) last.step.outputTokens.pop_back();
    if (change == 3) last.pendingToken.reset();
    reject(baseline, different, "different warmup work at lane 4");
  }
  auto different = candidate;
  different.lanes.pop_back();
  reject(baseline, different, "lane counts");
  different = candidate;
  different.completed = false;
  reject(baseline, different, "incomplete warmups");
  reject(different, candidate, "incomplete warmups");
  different = candidate;
  different.lanes.clear();
  reject(different, different, "lane counts");
}

void completeRun() {
  std::vector<CandidateId> invocations;
  const auto result = measureWorkload(kCandidate, kWorkload,
      [&](CandidateId candidate) {
        invocations.push_back(candidate);
        return stableTiming(candidate);
      });
  require(result.status == MeasurementStatus::Completed &&
              result.candidate == kCandidate && result.workload == kWorkload &&
              result.pairCount == 12 && result.warmup.attemptedCalls == 4 &&
              result.warmup.returnedCalls == 4 &&
              result.measurement.attemptedCalls == 24 &&
              result.measurement.returnedCalls == 24 && !result.failure,
          "default measurement did not run two warmup and twelve sample pairs");
  require(result.warmup.gpuSeconds == 3.75 && result.warmup.wallSeconds == 7.5 &&
              result.measurement.gpuSeconds == 22.5 &&
              result.measurement.wallSeconds == 45 &&
              std::isfinite(result.elapsedWallSeconds) &&
              result.elapsedWallSeconds >= 0,
          "whole-workload invocation accounting was incorrect");
  for (size_t call = 0; call < invocations.size(); ++call) {
    const size_t phaseCall = call < 4 ? call : call - 4;
    const bool baseline = ((phaseCall / 2) % 2 == 0) == (phaseCall % 2 == 0);
    require(invocations[call] == (baseline ? kBaseline : kCandidate),
            "baseline/candidate pair order did not alternate");
  }
  for (size_t pair = 0; pair < result.pairCount; ++pair) {
    require(result.gpuPairs[pair].baselineSeconds == 1 &&
                result.gpuPairs[pair].candidateSeconds == 0.875 &&
                result.wallPairs[pair].baselineSeconds == 2 &&
                result.wallPairs[pair].candidateSeconds == 1.75 &&
                result.gpuPairs[pair].first == measurementOrder(pair) &&
                result.wallPairs[pair].first == measurementOrder(pair) &&
                !result.gpuPairs[pair].underPressure,
            "raw samples lost matched timing or execution order");
  }
  require(result.gpuAssessment.verdict == TimingVerdict::Improved &&
              result.wallAssessment.verdict == TimingVerdict::Improved &&
              result.rawGpuSamples().size() == 12 &&
              result.rawWallSamples().size() == 12,
          "valid measurement did not expose its complete raw samples");
  const std::array workloads{WorkloadMeasurements{result.workload, result.rawGpuSamples()}};
  const std::array candidates{CandidateMeasurements{kCandidate, workloads}};
  const std::array required{kWorkload};
  require(selectCandidate(candidates, required).candidate == kCandidate,
          "measurement result did not integrate with candidate selection");

  MeasurementOptions maximum;
  maximum.warmupPairs = kMaximumWarmupPairs;
  maximum.samplePairs = kMaxPairedSamples;
  maximum.policy.minimumPairs = 64;
  const auto bounded = measureWorkload(kCandidate, kWorkload, stableTiming, maximum);
  require(bounded.status == MeasurementStatus::Completed &&
              bounded.warmup.returnedCalls == 8 &&
              bounded.measurement.returnedCalls == 128 && bounded.pairCount == 64,
          "maximum bounded measurement changed call counts");
}

void warmupsAndQualification() {
  size_t calls = 0;
  auto result = measureWorkload(kCandidate, kWorkload,
      [&](CandidateId candidate) {
        ++calls;
        return calls <= 4 ? RunTiming{double(calls * 20), double(calls * 30), false}
                          : stableTiming(candidate);
      });
  require(result.status == MeasurementStatus::Completed &&
              result.gpuAssessment.conservativeGain == 0.125,
          "warmup variation contaminated measured pairs");

  result = measureWorkload(kCandidate, kWorkload,
                           [](CandidateId) { return RunTiming{1, 2, false}; });
  require(result.status == MeasurementStatus::Completed &&
              result.gpuAssessment.verdict == TimingVerdict::Stable &&
              result.wallAssessment.verdict == TimingVerdict::Stable,
          "stable neutral workload could not participate in aggregate selection");
  const std::array workloads{WorkloadMeasurements{result.workload, result.rawWallSamples()}};
  const std::array candidates{CandidateMeasurements{kCandidate, workloads}};
  const std::array required{kWorkload};
  require(selectCandidate(candidates, required).candidate == kBaseline,
          "neutral measurement alone displaced baseline");

  for (const bool noisyGpu : {false, true}) {
    calls = 0;
    result = measureWorkload(kCandidate, kWorkload,
        [&](CandidateId candidate) {
          const size_t call = calls++;
          auto timing = stableTiming(candidate);
          if (call >= 4 && candidate == kCandidate && ((call - 4) / 2) % 2 == 0) {
            if (noisyGpu)
              timing.gpuSeconds = 1.125;
            else
              timing.wallSeconds = 2.25;
          }
          return timing;
        });
    require(result.status == MeasurementStatus::Rejected && result.pairCount == 12 &&
                result.rawGpuSamples().size() == 12 &&
                result.rawWallSamples().size() == 12 &&
                (noisyGpu ? result.gpuAssessment : result.wallAssessment).verdict ==
                    TimingVerdict::Noisy,
            "one noisy metric did not invalidate the entire workload");
  }
  result = measureWorkload(kCandidate, kWorkload,
      [](CandidateId candidate) {
        auto timing = stableTiming(candidate);
        if (candidate == kCandidate)
          timing.wallSeconds = 2.125;
        return timing;
      });
  require(result.status == MeasurementStatus::Rejected &&
              result.gpuAssessment.verdict == TimingVerdict::Improved &&
              result.wallAssessment.verdict == TimingVerdict::Regressed,
          "GPU improvement concealed a whole-workload wall regression");
}

void pressureAndInvalidTiming() {
  for (const size_t pressuredCall : {1U, 2U, 5U, 6U}) {
    size_t calls = 0;
    const auto result = measureWorkload(kCandidate, kWorkload,
        [&](CandidateId candidate) {
          auto timing = stableTiming(candidate);
          timing.underPressure = ++calls == pressuredCall;
          return timing;
        });
    require(result.status == MeasurementStatus::UnderPressure &&
                calls == pressuredCall && result.pairCount == (calls == 6 ? 1U : 0U),
            "pressure did not stop sampling immediately");
    if (result.pairCount)
      require(result.gpuPairs[0].underPressure && result.wallPairs[0].underPressure,
              "completed pressured pair was not retained as raw evidence");
  }
  for (const bool invalidGpu : {false, true}) {
    for (const double value : {0.0, -1.0,
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
      size_t calls = 0;
      const auto result = measureWorkload(kCandidate, kWorkload,
          [&](CandidateId candidate) {
            auto timing = stableTiming(candidate);
            if (++calls == 6) {
              if (invalidGpu)
                timing.gpuSeconds = value;
              else
                timing.wallSeconds = value;
            }
            return timing;
          });
      require(result.status == MeasurementStatus::InvalidTiming && calls == 6 &&
                  result.pairCount == 1 &&
                  result.measurement.returnedCalls == 2 &&
                  std::isfinite(result.measurement.gpuSeconds) &&
                  std::isfinite(result.measurement.wallSeconds),
              "invalid returned timing was accepted or destroyed accounting");
    }
  }
}

void cancellationAndFailures() {
  size_t calls = 0;
  auto result = measureWorkload(kCandidate, kWorkload,
      [&](CandidateId candidate) { ++calls; return stableTiming(candidate); }, {},
      [] { return true; });
  require(result.status == MeasurementStatus::Cancelled && calls == 0 &&
              result.warmup.attemptedCalls == 0 && result.pairCount == 0,
          "already cancelled workload invoked the production callback");

  MeasurementOptions options;
  options.samplePairs = 13;
  result = measureWorkload(kCandidate, kWorkload,
      [&](CandidateId candidate) { ++calls; return stableTiming(candidate); }, options,
      [&] { return calls == 29; });
  require(result.status == MeasurementStatus::Cancelled && calls == 29 &&
              result.pairCount == 12 && result.measurement.returnedCalls == 25 &&
              result.measurement.gpuSeconds == 23.5 &&
              result.gpuAssessment.qualified(),
          "interrupted unmatched run was fabricated into an eligible pair");

  calls = 0;
  result = measureWorkload(kCandidate, kWorkload,
      [&](CandidateId candidate) { ++calls; return stableTiming(candidate); }, {},
      [&] { return calls == 28; });
  require(result.status == MeasurementStatus::Cancelled && result.pairCount == 12,
          "cancellation after the final return lost raw pairs or stayed eligible");

  calls = 0;
  result = measureWorkload(kCandidate, kWorkload,
      [&](CandidateId candidate) {
        if (++calls == 29)
          throw std::runtime_error("production execution failed");
        return stableTiming(candidate);
      }, options);
  require(result.status == MeasurementStatus::RunFailed && result.failure &&
              result.pairCount == 12 && result.measurement.attemptedCalls == 25 &&
              result.measurement.returnedCalls == 24,
          "execution failure was retried or its unmatched half was retained");
  try {
    std::rethrow_exception(result.failure);
  } catch (const std::runtime_error &error) {
    require(std::string_view(error.what()) == "production execution failed",
            "original production failure detail was lost");
  }
  calls = 0;
  result = measureWorkload(kCandidate, kWorkload,
      [&](CandidateId candidate) { ++calls; return stableTiming(candidate); }, {},
      [&] {
        if (calls == 6)
          throw std::runtime_error("control failed after a returned command");
        return false;
      });
  require(result.status == MeasurementStatus::RunFailed && result.failure &&
              result.pairCount == 1 && result.measurement.returnedCalls == 2,
          "control failure discarded an already completed pair");
}

void boundsAndDeadline() {
  size_t calls = 0;
  const MeasurementRun run = [&](CandidateId candidate) {
    ++calls;
    return stableTiming(candidate);
  };
  std::array<MeasurementOptions, 8> invalid{};
  invalid[0].warmupPairs = 0;
  invalid[1].warmupPairs = 5;
  invalid[2].samplePairs = 11;
  invalid[3].samplePairs = 65;
  invalid[4].policy.minimumPairs = 13;
  invalid[5].policy.minimumMeanImprovement = 0;
  invalid[6].maximumWallSeconds = -1;
  invalid[7].maximumWallSeconds = std::numeric_limits<double>::infinity();
  for (const auto &options : invalid)
    require(measureWorkload(kCandidate, kWorkload, run, options).status ==
                MeasurementStatus::InvalidInput && calls == 0,
            "invalid measurement bounds invoked the production callback");
  require(measureWorkload(kBaseline, kWorkload, run).status ==
              MeasurementStatus::InvalidInput &&
              measureWorkload(kCandidate, kWorkload, {}).status ==
                  MeasurementStatus::InvalidInput && calls == 0,
          "missing callback or baseline-as-candidate was accepted");

  MeasurementOptions deadline;
  deadline.maximumWallSeconds = 0;
  auto result = measureWorkload(kCandidate, kWorkload, run, deadline);
  require(result.status == MeasurementStatus::BudgetExceeded && calls == 0,
          "expired deadline started a workload");
  deadline.maximumWallSeconds = 0.01;
  bool returned = false;
  result = measureWorkload(kCandidate, kWorkload,
      [&](CandidateId candidate) {
        ++calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        returned = true;
        return stableTiming(candidate);
      }, deadline);
  require(result.status == MeasurementStatus::BudgetExceeded && calls == 1 &&
              returned && result.warmup.returnedCalls == 1 &&
              result.warmup.gpuSeconds == 1 && result.pairCount == 0 &&
              result.elapsedWallSeconds >= deadline.maximumWallSeconds,
          "deadline interrupted an active workload or lost its accounting");
}

} // namespace

// A non-regression gate may accept Uncertain: median paired gain nonnegative
// and spreads within policy, but the conservative gain straddles zero.
// Selection keeps the default and still rejects it.
void uncertainAcceptance() {
  size_t candidateCalls = 0;
  auto uncertainTiming = [&](CandidateId candidate) {
    if (candidate == kBaseline)
      return RunTiming{1, 2, false};
    // Two warmup pairs precede twelve measured pairs: eight small wins
    // and four small losses leave a positive median inside the noise.
    const size_t index = candidateCalls++;
    const double scale = index < 2 || index - 2 < 8 ? 0.99 : 1.01;
    return RunTiming{scale, 2 * scale, false};
  };
  auto rejected = measureWorkload(kCandidate, kWorkload, uncertainTiming);
  require(rejected.status == MeasurementStatus::Rejected &&
              rejected.gpuAssessment.verdict == TimingVerdict::Uncertain &&
              rejected.wallAssessment.verdict == TimingVerdict::Uncertain,
          "selection accepted an uncertain candidate");
  candidateCalls = 0;
  MeasurementOptions options;
  options.acceptUncertain = true;
  auto accepted = measureWorkload(kCandidate, kWorkload, uncertainTiming, options);
  require(accepted.status == MeasurementStatus::Completed &&
              accepted.gpuAssessment.verdict == TimingVerdict::Uncertain &&
              accepted.wallAssessment.verdict == TimingVerdict::Uncertain &&
              accepted.pairCount == 12 &&
              accepted.gpuAssessment.medianPairedGain > 0 &&
              accepted.gpuAssessment.conservativeGain < 0,
          "non-regression gate rejected an uncertain candidate");
  // Regressed and Noisy outcomes are still rejected by the gate.
  auto regressedTiming = [](CandidateId candidate) {
    return candidate == kBaseline ? RunTiming{1, 2, false} : RunTiming{1.02, 2.04, false};
  };
  require(measureWorkload(kCandidate, kWorkload, regressedTiming, options).status ==
              MeasurementStatus::Rejected,
          "non-regression gate accepted a regression");
}

int main() {
  try {
    diagnosticNames();
    comparableWarmupPairs();
    uncertainAcceptance();
    require(measurementBatchRepetitions(0.010) == 1 &&
                measurementBatchRepetitions(0.005) == 1 &&
                measurementBatchRepetitions(0.001) == 5 &&
                measurementBatchRepetitions(0.00001) == 16 &&
                measurementBatchRepetitions(0) == 1 &&
                measurementBatchRepetitions(-1) == 1 &&
                measurementBatchRepetitions(std::numeric_limits<double>::infinity()) == 1 &&
                measurementBatchRepetitions(std::numeric_limits<double>::quiet_NaN()) == 1,
            "measurement batch size is unbounded or pilot-dependent incorrectly");
    completeRun();
    warmupsAndQualification();
    pressureAndInvalidTiming();
    cancellationAndFailures();
    boundsAndDeadline();
    std::cout << "operator measurement: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "operator measurement: FAIL: " << error.what() << '\n';
    return 1;
  }
}

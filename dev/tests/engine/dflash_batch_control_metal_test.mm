// Modified by meowkernels.
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "model/Model.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::Lookup16AcceptanceBuffers;
using splash::ops::Sampling;
using splash::ops::SamplingPolicy;

constexpr uint32_t kRows = splash::model::ExecutionLimits::targetVerifyRows;
constexpr uint32_t kProposals =
    splash::model::ExecutionLimits::draftProposalTokens;
constexpr uint32_t kLanes =
    splash::model::ExecutionLimits::maximumBatchWidth;

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

template <class T> T *contents(const MetalBuffer &buffer) {
  return static_cast<T *>(buffer.contents());
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void runWidth(MetalBackend &backend, uint32_t width,
              const std::array<uint32_t, kLanes> &acceptedReference,
              const std::array<uint32_t, kLanes> &remainingReference,
              uint32_t stopLane = kLanes, uint32_t stopRow = 0,
              bool deltaProposals = false) {
  require(width >= 1 && width <= kLanes, "invalid test width");
  MetalBuffer draft = shared(backend, kLanes * kProposals * sizeof(uint32_t),
                             "accept-draft");
  MetalBuffer draftIds =
      shared(backend, kLanes * kProposals * 16 * sizeof(uint32_t),
             "accept-draft-ids");
  MetalBuffer draftProbabilities =
      shared(backend, kLanes * kProposals * 16 * sizeof(float),
             "accept-draft-probabilities");
  MetalBuffer targetIds =
      shared(backend, kLanes * kRows * 32 * sizeof(uint32_t),
             "accept-target-ids");
  MetalBuffer targetProbabilities =
      shared(backend, kLanes * kRows * 32 * sizeof(float),
             "accept-target-probabilities");
  MetalBuffer uniforms =
      shared(backend, kLanes * 2 * kRows * sizeof(float), "accept-uniforms");
  MetalBuffer output = shared(backend, kLanes * kRows * sizeof(uint32_t),
                              "accept-output");
  MetalBuffer retained =
      shared(backend, kLanes * sizeof(uint32_t), "accept-retained");
  MetalBuffer next =
      shared(backend, kLanes * sizeof(uint32_t), "accept-next");
  MetalBuffer accepted =
      shared(backend, kLanes * sizeof(uint32_t), "accept-count");

  auto *draftTokens = contents<uint32_t>(draft);
  auto *targetTokens = contents<uint32_t>(output);
  std::memset(draftIds.contents(), 0, draftIds.sizeBytes());
  std::memset(draftProbabilities.contents(), 0,
              draftProbabilities.sizeBytes());
  std::memset(targetIds.contents(), 0, targetIds.sizeBytes());
  std::memset(targetProbabilities.contents(), 0,
              targetProbabilities.sizeBytes());
  std::memset(uniforms.contents(), 0, uniforms.sizeBytes());
  std::memset(retained.contents(), 0, retained.sizeBytes());
  std::memset(next.contents(), 0, next.sizeBytes());
  std::memset(accepted.contents(), 0, accepted.sizeBytes());
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    for (uint32_t token = 0; token < kProposals; ++token) {
      const uint32_t proposal = 1000 + lane * 100 + token;
      draftTokens[lane * kProposals + token] = proposal;
      targetTokens[lane * kRows + token] =
          token < acceptedReference[lane] ? proposal : proposal + 50;
    }
    targetTokens[lane * kRows + kRows - 1] = 9000 + lane;
  }

  constexpr uint32_t kStopToken = 248044;
  if (stopLane < width) {
    require(stopRow < kRows, "invalid stop row");
    targetTokens[stopLane * kRows + stopRow] = kStopToken;
    if (stopRow < acceptedReference[stopLane])
      draftTokens[stopLane * kProposals + stopRow] = kStopToken;
  }

  // Prompt lookup is deterministic q. Check the unchanged sampled verifier
  // with both an out-of-support proposal and rejection inside target support.
  const std::vector<uint32_t> expectedOutput(targetTokens,
                                            targetTokens + kLanes * kRows);
  if (deltaProposals) {
    std::memset(draftIds.contents(), 0xff, draftIds.sizeBytes());
    std::memset(targetIds.contents(), 0xff, targetIds.sizeBytes());
    std::fill_n(contents<float>(uniforms), kLanes * 2 * kRows, 0.5F);
    for (uint32_t lane = 0; lane < width; ++lane) {
      for (uint32_t row = 0; row < kRows; ++row) {
        const size_t p = (lane * kRows + row) * 32;
        contents<uint32_t>(targetIds)[p] = targetTokens[lane * kRows + row];
        contents<float>(targetProbabilities)[p] = 1.0F;
        if (row == kProposals)
          continue;
        const size_t q = (lane * kProposals + row) * 16;
        const uint32_t token = draftTokens[lane * kProposals + row];
        contents<uint32_t>(draftIds)[q] = token;
        contents<float>(draftProbabilities)[q] = 1.0F;
        if (row >= acceptedReference[lane] && row % 2) {
          contents<uint32_t>(targetIds)[p + 1] = token;
          contents<float>(targetProbabilities)[p + 1] = 0.25F;
          contents<float>(targetProbabilities)[p] = 0.75F;
        }
      }
    }
  }

  AcceptBatchParams params{};
  std::copy(remainingReference.begin(), remainingReference.end(),
            std::begin(params.remaining));
  params.stop_token_0 = kStopToken;
  params.stop_token_1 = 248046;
  params.lanes = width;
  params.sampling_mask = deltaProposals ? (1U << width) - 1 : 0;
  ComputeDispatch dispatch;
  dispatch.pipelineName = "decode_accept_dflash";
  dispatch.buffers = {{0, draft},
                      {1, draftIds},
                      {2, draftProbabilities},
                      {3, targetIds},
                      {4, targetProbabilities},
                      {5, uniforms},
                      {6, output},
                      {7, retained},
                      {8, next},
                      {9, accepted}};
  dispatch.bytes = {{10, &params, sizeof(params)}};
  dispatch.threadgroups = {width, 1, 1};
  dispatch.threadsPerThreadgroup = {1, 1, 1};
  static_cast<void>(backend.submit(dispatch));

  const auto *retainedCounts = contents<uint32_t>(retained);
  const auto *nextTokens = contents<uint32_t>(next);
  const auto *acceptedCounts = contents<uint32_t>(accepted);
  for (uint32_t lane = 0; lane < width; ++lane) {
    uint32_t expectedRetained =
        std::min(acceptedReference[lane] + 1, remainingReference[lane]);
    if (lane == stopLane && stopRow < expectedRetained)
      expectedRetained = stopRow + 1;
    require(acceptedCounts[lane] == acceptedReference[lane],
            "accepted proposal count mismatch");
    require(retainedCounts[lane] == expectedRetained,
            "retained token count mismatch");
    require(nextTokens[lane] ==
                expectedOutput[lane * kRows + expectedRetained - 1],
            "next anchor mismatch");
    for (uint32_t row = 0; row <= acceptedReference[lane]; ++row)
      require(targetTokens[lane * kRows + row] ==
                  expectedOutput[lane * kRows + row],
              "accepted prefix or residual differs from reference");
  }
}

// tiles = 2: the 16-row lookup; 4: SPLASH_WIDE_LOOKUP32's 32 rows. rejectRow
// (sampled, all accepted otherwise): that row's proposal has probability 0.5
// and its uniform is 0.75, so acceptance stops there with correction 4000; at
// row >= 16 the uniform sits in the second lane's half of the uniforms.
void runLookupWide(MetalBackend &backend, uint32_t tiles, uint32_t acceptedReference,
                   uint32_t remaining, bool sampled = false,
                   uint32_t stopRow = UINT32_MAX, float residualDraw = -1.0F,
                   uint32_t residualExpected = 0, uint32_t rejectRow = UINT32_MAX) {
  const uint32_t kLookupRows = tiles * kRows;
  const uint32_t kLookupProposals = kLookupRows - 1;
  stopRow = std::min(stopRow, kLookupRows);
  require(acceptedReference <= kLookupProposals, "invalid accepted count");
  require(remaining >= 1 && remaining <= kLookupRows,
          "invalid lookup retention limit");
  MetalBuffer input =
      shared(backend, kLookupRows * sizeof(uint32_t), "lookup16-input");
  MetalBuffer targetIds =
      shared(backend, kLookupRows * 32 * sizeof(uint32_t),
             "lookup16-target-ids");
  MetalBuffer targetProbabilities =
      shared(backend, kLookupRows * 32 * sizeof(float),
             "lookup16-target-probabilities");
  MetalBuffer uniforms =
      shared(backend, kLookupRows * sizeof(float), "lookup16-uniforms");
  MetalBuffer output =
      shared(backend, kLookupRows * sizeof(uint32_t), "lookup16-output");
  MetalBuffer retained =
      shared(backend, sizeof(uint32_t), "lookup16-retained");
  MetalBuffer next = shared(backend, sizeof(uint32_t), "lookup16-next");
  MetalBuffer accepted =
      shared(backend, sizeof(uint32_t), "lookup16-accepted");
  MetalBuffer halves =
      shared(backend, tiles * sizeof(uint32_t), "lookup16-halves");

  auto *inputTokens = contents<uint32_t>(input);
  auto *targetTokens = contents<uint32_t>(output);
  auto *ids = contents<uint32_t>(targetIds);
  auto *probabilities = contents<float>(targetProbabilities);
  inputTokens[0] = 77;
  std::memset(targetIds.contents(), 0xff, targetIds.sizeBytes());
  std::memset(targetProbabilities.contents(), 0,
              targetProbabilities.sizeBytes());
  std::fill_n(contents<float>(uniforms), kLookupRows, 0.5F);
  for (uint32_t row = 0; row < kLookupProposals; ++row) {
    const uint32_t proposal = 1000 + row;
    const uint32_t correction = 2000 + row;
    inputTokens[row + 1] = proposal;
    targetTokens[row] = row < acceptedReference ? proposal : correction;
    ids[row * 32] = targetTokens[row];
    probabilities[row * 32] = 1.0F;
  }
  targetTokens[kLookupProposals] = 3000;
  ids[kLookupProposals * 32] = targetTokens[kLookupProposals];
  probabilities[kLookupProposals * 32] = 1.0F;

  constexpr uint32_t kStopToken = 248044;
  if (stopRow < kLookupRows) {
    targetTokens[stopRow] = kStopToken;
    ids[stopRow * 32] = kStopToken;
    if (stopRow < acceptedReference)
      inputTokens[stopRow + 1] = kStopToken;
  }
  if (residualDraw >= 0.0F) {
    require(sampled && acceptedReference == 0 && stopRow == kLookupRows,
            "invalid lookup residual fixture");
    ids[0] = inputTokens[1];
    ids[1] = 20;
    ids[2] = 30;
    probabilities[0] = 0.25F;
    probabilities[1] = 0.25F;
    probabilities[2] = 0.5F;
    contents<float>(uniforms)[0] = 0.25F;
    contents<float>(uniforms)[kLookupProposals] = residualDraw;
    targetTokens[0] = residualExpected;
  }
  if (rejectRow < kLookupProposals) {
    require(sampled && acceptedReference == rejectRow && stopRow == kLookupRows,
            "invalid lookup reject fixture");
    for (uint32_t row = 0; row < kLookupProposals; ++row) {
      targetTokens[row] = inputTokens[row + 1];
      ids[row * 32] = targetTokens[row];
    }
    probabilities[rejectRow * 32] = 0.5F;
    ids[rejectRow * 32 + 1] = 4000;
    probabilities[rejectRow * 32 + 1] = 0.5F;
    contents<float>(uniforms)[rejectRow] = 0.75F;
    targetTokens[rejectRow] = 4000;
  }
  const std::vector<uint32_t> expected(targetTokens, targetTokens + kLookupRows);

  Lookup16AcceptanceBuffers buffers{input,    targetIds, targetProbabilities,
                                    uniforms, output,    retained,
                                    next,     accepted,  halves};
  Sampling sampling(backend, 4096, kRows);
  const SamplingPolicy policy{1, sampled ? 1.0F : 0.0F, 1.0F, false};
  if (acceptedReference == 0 && remaining == kLookupRows && !sampled &&
      stopRow == kLookupRows && residualDraw < 0.0F) {
    const auto expectInvalid = [](auto operation, const char *message) {
      try {
        operation();
      } catch (const std::invalid_argument &) {
        return;
      }
      throw std::runtime_error(message);
    };
    expectInvalid(
        [&] {
          CommandGraph graph;
          sampling.addLookup16Acceptance(graph, buffers, 0, policy, kStopToken,
                                         248046, tiles);
        },
        "lookup16 accepted zero remaining");
    expectInvalid(
        [&] {
          CommandGraph graph;
          Lookup16AcceptanceBuffers undersized = buffers;
          undersized.retainedHalves =
              shared(backend, sizeof(uint32_t), "lookup16-short-halves");
          sampling.addLookup16Acceptance(graph, undersized, kLookupRows,
                                         policy, kStopToken, 248046, tiles);
        },
        "lookup16 accepted undersized buffer");
  }
  CommandGraph graph;
  sampling.addLookup16Acceptance(graph, buffers, remaining, policy, kStopToken,
                                 248046, tiles);
  require(graph.dispatches().size() == 1,
          "lookup16 acceptance dispatch count mismatch");
  static_cast<void>(backend.submitCommand(graph.dispatches()));

  uint32_t expectedRetained = std::min(acceptedReference + 1, remaining);
  if (stopRow < expectedRetained)
    expectedRetained = stopRow + 1;
  require(contents<uint32_t>(accepted)[0] == acceptedReference,
          "lookup16 accepted count mismatch");
  require(contents<uint32_t>(retained)[0] == expectedRetained,
          "lookup16 retained count mismatch");
  require(contents<uint32_t>(next)[0] == expected[expectedRetained - 1],
          "lookup16 next anchor mismatch");
  for (uint32_t tile = 0; tile < tiles; ++tile)
    require(contents<uint32_t>(halves)[tile] ==
                std::min(expectedRetained > tile * kRows ? expectedRetained - tile * kRows : 0, kRows),
            "lookup retained tile counts mismatch");
  for (uint32_t row = 0; row <= acceptedReference; ++row)
    require(targetTokens[row] == expected[row],
            "lookup16 accepted prefix or correction mismatch");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: dflash-batch-control METALLIB");
    MetalBackend backend(argv[1]);
    constexpr std::array<uint32_t, kLanes> lowerAccepted{0, 1, 2, 3};
    constexpr std::array<uint32_t, kLanes> upperAccepted{4, 5, 6, 7};
    constexpr std::array<uint32_t, kLanes> fullRemaining{8, 8, 8, 8};
    for (uint32_t width = 1; width <= kLanes; ++width) {
      runWidth(backend, width, lowerAccepted, fullRemaining);
      runWidth(backend, width, upperAccepted, fullRemaining);
      runWidth(backend, width, lowerAccepted, fullRemaining, kLanes, 0, true);
      runWidth(backend, width, upperAccepted, fullRemaining, kLanes, 0, true);
    }

    // Output limits and stop tokens shorten the committed prefix without
    // changing the physical eight-row graph.  The accepted proposal count is
    // still seven in every lane; only retained rows and the next anchor move.
    constexpr std::array<uint32_t, kLanes> allAccepted{7, 7, 7, 7};
    constexpr std::array<uint32_t, kLanes> shortRemaining{1, 2, 3, 8};
    runWidth(backend, kLanes, allAccepted, shortRemaining, 3, 3);
    runWidth(backend, kLanes, allAccepted, shortRemaining, 3, 3, true);

    for (uint32_t tiles : {2u, 4u}) {
      const uint32_t kLookupRows = tiles * kRows;
      const uint32_t kLookupProposals = kLookupRows - 1;
      for (uint32_t accepted = 0; accepted <= kLookupProposals; ++accepted)
        runLookupWide(backend, tiles, accepted, kLookupRows);
      for (uint32_t remaining = 1; remaining <= kLookupRows; ++remaining)
        runLookupWide(backend, tiles, kLookupProposals, remaining);
      runLookupWide(backend, tiles, 0, kLookupRows, true);
      runLookupWide(backend, tiles, kLookupProposals, kLookupRows, true);
      for (uint32_t stop : {7u, 8u, 15u, kLookupRows - 1})
        runLookupWide(backend, tiles, kLookupProposals, kLookupRows, false, stop);
      runLookupWide(backend, tiles, 0, kLookupRows, true, kLookupRows, 0.0F, 20);
      runLookupWide(backend, tiles, 0, kLookupRows, true, kLookupRows, 0.32F, 20);
      runLookupWide(backend, tiles, 0, kLookupRows, true, kLookupRows, 0.34F, 30);
      runLookupWide(backend, tiles, 0, kLookupRows, true, kLookupRows, 0.99F, 30);
      for (uint32_t reject : {3u, 12u, 20u, 27u})
        if (reject < kLookupProposals)
          runLookupWide(backend, tiles, reject, kLookupRows, true, kLookupRows,
                        -1.0F, 0, reject);
    }
    std::cout << "dflash_batch_control_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "dflash_batch_control_metal_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}

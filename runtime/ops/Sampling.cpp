// Modified by meowkernels.
#include "ops/Sampling.hpp"

#include "metal/EnvSwitch.hpp"
#include "metal/abi/Sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kTargetShards = SPLASH_TARGET_SAMPLING_SHARDS;

// A lane that does not sample takes the argmax path: one candidate, unit
// temperature and top-p, whatever the request carried.
struct EffectivePolicy final {
  uint32_t topK;
  float temperature;
  float topP;
};
EffectivePolicy effectivePolicy(const SamplingPolicy &policy) noexcept {
  if (policy.samples()) return {policy.topK, policy.temperature, policy.topP};
  return {1, 1.0F, 1.0F};
}
constexpr uint32_t kTargetCandidates = kTargetSamplingCandidates;
constexpr uint32_t kDraftShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kDraftCandidates = 16;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

// SPLASH_DRAFT_TAU=t (default 0.85): the drafter proposes from
// q' = softmax(scores / (temperature * t)) over its 16 candidates. The same q'
// is written to the proposal probabilities that every acceptance rule reads,
// so sampled outputs keep the target distribution; greedy lanes (argmax) are
// unaffected. t = 1 multiplies by exactly 1.0f: the drafter's own q. 0.85 won
// the exact block-rule replay: +0.80% acceptance, every category >= 0; sampling
// checked by chi^2 over 800 seeds at tau 0.76 (tau).
float envFloat(const char *name, float fallback) {
  const char *value = std::getenv(name);
  if (!value)
    return fallback;
  char *end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (end == value || *end != '\0')
    throw std::invalid_argument(std::string(name) + " must be a number");
  return parsed;
}

float draftTau() {
  static const float tau = [] {
    const float parsed = envFloat("SPLASH_DRAFT_TAU", 0.85F);
    if (!std::isfinite(parsed) || parsed <= 0.0F)
      throw std::invalid_argument("SPLASH_DRAFT_TAU must be a positive number");
    return parsed;
  }();
  return tau;
}

// SPLASH_DRAFT_TOP_P=p (default 0.99; 1 = off): after tau, the drafter keeps its
// highest-probability candidates until their mass reaches p (always the
// argmax), renormalizes over them, samples from that q'' and writes it to the
// proposal probabilities every acceptance rule reads, so sampled outputs keep
// the target distribution. Greedy lanes are unaffected; p = 1 takes the
// unchanged kernel path. 0.99: block-rule replay +0.26% (27B) / +0.29% (35B),
// every category >= 0; selector bit-identical to its reference.
float draftTopP() {
  static const float topP = [] {
    const float parsed = envFloat("SPLASH_DRAFT_TOP_P", 0.99F);
    if (!(parsed > 0.0F && parsed <= 1.0F))
      throw std::invalid_argument("SPLASH_DRAFT_TOP_P must be in (0, 1]");
    return parsed;
  }();
  return topP;
}

// SPLASH_BLOCK_VERIFY (default on): block verification for all-sampled
// acceptance batches (Sun et al.). +1.29% tokens per cycle on the bench, exact vs a
// serial reference on 22,848 real draws (stack5).
bool blockVerificationEnabled() {
  static const bool enabled = metal::envSwitch("SPLASH_BLOCK_VERIFY");
  return enabled;
}

} // namespace

SamplingWorkspace Sampling::workspace(uint32_t rows) {
  if (!rows)
    throw std::invalid_argument("invalid sampling workspace row count");
  const uint64_t shards = uint64_t{rows} * kTargetShards;
  const uint64_t candidates = uint64_t{rows} * kTargetCandidates;
  return {shards * sizeof(float), shards * sizeof(uint32_t),
          shards * kTargetCandidates * sizeof(uint32_t),
          shards * kTargetCandidates * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float)};
}

DraftSelectorWorkspace Sampling::draftWorkspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * kDraftCandidates;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kDraftShards * sizeof(uint32_t),
          candidates * (kDraftShards + kDraftCandidates) * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(uint16_t),
          candidates * sizeof(float)};
}

Sampling::Sampling(metal::MetalBackend &backend, uint32_t vocabulary,
                   uint32_t rowsPerLane)
    : backend_(backend), vocabulary_(vocabulary), rowsPerLane_(rowsPerLane),
      maskWords_((vocabulary + 31) / 32),
      blockVerify_(blockVerificationEnabled()) {
  if (!vocabulary || !rowsPerLane)
    throw std::invalid_argument("invalid sampling geometry");
  static_cast<void>(draftTau());  // reject a bad SPLASH_DRAFT_TAU at startup
  static_cast<void>(draftTopP());  // and a bad SPLASH_DRAFT_TOP_P
}

void Sampling::addInitial(metal::CommandGraph &graph,
                          const SamplingPolicy &policy,
                          SamplingBuffers buffers,
                          uint32_t rowOffset) const {
  if (rowOffset >= rowsPerLane_)
    throw std::invalid_argument("invalid initial sampling row");
  if (policy.samples() || policy.constrained) {
    const EffectivePolicy effective = effectivePolicy(policy);
    const TargetSamplingParams params{
        vocabulary_, rowOffset, effective.topK, effective.temperature,
        effective.topP, maskWords_, 0, policy.constrained ? 1U : 0U};
    graph.add("decode_sample_top32_sharded",
              {buffers.logits, buffers.partialIds, buffers.partialValues,
               buffers.constraintMasks},
              params, {kTargetShards, 1, 1});
    graph.add("decode_sample_top32_probs",
              {buffers.partialIds, buffers.partialValues, buffers.topIds,
               buffers.topProbabilities},
              params, {1, 1, 1}, {1, 1, 1});
    if (policy.samples()) {
      graph.add("decode_sample_sparse_draw",
                {buffers.topIds, buffers.topProbabilities, buffers.uniforms,
                 buffers.outputTokens},
                {1, 1, 1}, {1, 1, 1});
    } else {
      graph.add("decode_sample_sparse_top1",
                {buffers.topIds, buffers.topProbabilities,
                 buffers.outputTokens},
                {1, 1, 1}, {1, 1, 1});
    }
    return;
  }

  metal::MetalBuffer logits = buffers.logits;
  if (rowOffset) {
    const uint64_t rowBytes = uint64_t{vocabulary_} * sizeof(uint16_t);
    logits = backend_.view(logits, uint64_t{rowOffset} * rowBytes, rowBytes);
  }
  graph.add("decode_sample_argmax_sharded",
            {std::move(logits), buffers.argmaxValues, buffers.argmaxIndices},
            vocabulary_, {kTargetShards, 1, 1});
  graph.add("decode_sample_argmax_reduce",
            {buffers.argmaxValues, buffers.argmaxIndices,
             buffers.outputTokens},
            {1, 1, 1}, {32, 1, 1});
}

void Sampling::addVerify(metal::CommandGraph &graph,
                         std::span<const SamplingPolicy> policies,
                         SamplingBuffers buffers) const {
  if (policies.empty() || policies.size() > kMaximumLanes)
    throw std::invalid_argument("invalid sampling batch width");
  const uint32_t lanes = static_cast<uint32_t>(policies.size());
  const bool constrained = std::any_of(
      policies.begin(), policies.end(),
      [](const SamplingPolicy &policy) { return policy.constrained; });
  const bool sampling = std::any_of(
      policies.begin(), policies.end(),
      [](const SamplingPolicy &policy) { return policy.samples(); });
  const bool greedy = std::any_of(
      policies.begin(), policies.end(),
      [](const SamplingPolicy &policy) { return !policy.samples(); });
  const bool distributed = constrained || sampling;
  const uint32_t rows = lanes * rowsPerLane_;

  if (!distributed) {
    graph.add("decode_sample_argmax_sharded",
              {buffers.logits, buffers.argmaxValues, buffers.argmaxIndices},
              vocabulary_, {uint64_t{rows} * kTargetShards, 1, 1});
    graph.add("decode_sample_argmax_reduce",
              {buffers.argmaxValues, buffers.argmaxIndices,
               buffers.outputTokens},
              {rows, 1, 1}, {32, 1, 1});
    return;
  }

  TargetSamplingBatchParams params{};
  params.vocabulary = vocabulary_;
  params.rows_per_lane = rowsPerLane_;
  params.lanes = lanes;
  params.mask_words = maskWords_;
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
    const SamplingPolicy &policy = policies[std::min(lane, lanes - 1)];
    const EffectivePolicy effective = effectivePolicy(policy);
    params.top_k[lane] = effective.topK;
    params.temperature[lane] = effective.temperature;
    params.top_p[lane] = effective.topP;
    if (lane < lanes && policy.constrained)
      params.constrained_mask |= uint32_t{1} << lane;
  }
  graph.add("decode_sample_top32_sharded_batch",
            {buffers.logits, buffers.partialIds, buffers.partialValues,
             buffers.constraintMasks},
            params, {uint64_t{rows} * kTargetShards, 1, 1});
  graph.add("decode_sample_top32_probs_batch",
            {buffers.partialIds, buffers.partialValues, buffers.topIds,
             buffers.topProbabilities},
            params, {rows, 1, 1}, {32, 1, 1});
  // Acceptance consumes argmax tokens for greedy lanes and distributions
  // for sampling lanes, including when both share the same target forward.
  if (constrained || greedy) {
    graph.add("decode_sample_sparse_top1",
              {buffers.topIds, buffers.topProbabilities,
               buffers.outputTokens},
              {rows, 1, 1}, {1, 1, 1});
  }
}

SelectorBatchParams
Sampling::draftSelectorParams(std::span<const uint32_t> anchors,
                              std::span<const SamplingPolicy> policies) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > kMaximumLanes)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.vocabulary = vocabulary_;
  params.top_p = draftTopP();
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
    const uint32_t source = std::min(lane, lanes - 1);
    params.anchor[lane] = anchors[source];
    params.temperature[lane] = policies[source].temperature * draftTau();
    if (lane < lanes && policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  return params;
}

void Sampling::addDraftSelector(
    metal::CommandGraph &graph, DraftSelectorBuffers buffers,
    std::span<const uint32_t> anchors,
    std::span<const SamplingPolicy> policies, uint32_t proposalTokens,
    metal::MetalBuffer deviceParams) const {
  if (!proposalTokens)
    throw std::invalid_argument("invalid draft selector batch");
  const SelectorBatchParams params = draftSelectorParams(anchors, policies);
  const uint32_t lanes = params.lanes;
  const uint32_t headRows = buffers.headRows ? buffers.headRows : vocabulary_;
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            headRows,
            {uint64_t{lanes} * proposalTokens * kDraftShards, 1, 1});
  if (buffers.headRows) {
    const uint32_t count = lanes * proposalTokens * kDraftShards * kDraftCandidates;
    graph.add("draft_map_head_ids", {buffers.partialIds, buffers.headIdMap},
              count, {(count + 255) / 256, 1, 1}, {256, 1, 1});
  }
  if (deviceParams) {
    graph.add("draft_select_edges",
              {buffers.partialIds, buffers.partialValues, buffers.candidates,
               buffers.unary, buffers.selectorHidden,
               buffers.predecessorCodebook, buffers.successorCodebook,
               deviceParams},
              {uint64_t{lanes} * proposalTokens, 1, 1}, {kEdgeThreads, 1, 1});
    graph.add("draft_select_dflash",
              {buffers.candidates, buffers.unary, buffers.partialValues,
               buffers.uniforms, buffers.proposedTokens,
               buffers.proposalProbabilities, std::move(deviceParams)},
              {lanes, 1, 1}, {1, 1, 1});
    return;
  }
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden,
             buffers.predecessorCodebook, buffers.successorCodebook},
            params, {uint64_t{lanes} * proposalTokens, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add("draft_select_dflash",
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities},
            params, {lanes, 1, 1}, {1, 1, 1});
}

void Sampling::addAcceptance(
    metal::CommandGraph &graph, AcceptanceBuffers buffers,
    std::span<const uint32_t> maximumRetained,
    std::span<const SamplingPolicy> policies, uint32_t stopToken0,
    uint32_t stopToken1) const {
  if (maximumRetained.empty() || maximumRetained.size() != policies.size() ||
      maximumRetained.size() > kMaximumLanes)
    throw std::invalid_argument("invalid DFlash acceptance batch");
  AcceptBatchParams params{};
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  params.lanes = static_cast<uint32_t>(maximumRetained.size());
  for (uint32_t lane = 0; lane < params.lanes; ++lane) {
    if (!maximumRetained[lane] || maximumRetained[lane] > rowsPerLane_)
      throw std::invalid_argument("invalid DFlash retention limit");
    params.remaining[lane] = maximumRetained[lane];
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  const uint32_t allLanesMask = (uint32_t{1} << params.lanes) - 1;
  const bool block = blockVerify_ && params.sampling_mask == allLanesMask;
  graph.add(block ? "decode_accept_dflash_block" : "decode_accept_dflash",
            {buffers.proposedTokens, buffers.candidates,
             buffers.proposalProbabilities, buffers.targetTopIds,
             buffers.targetTopProbabilities, buffers.uniforms,
             buffers.outputTokens, buffers.retainedCounts, buffers.nextAnchors,
             buffers.acceptedCounts},
            params, {params.lanes, 1, 1}, {block ? 32u : 1u, 1, 1});
}

void Sampling::addLookup16Acceptance(
    metal::CommandGraph &graph, Lookup16AcceptanceBuffers buffers,
    uint32_t maximumRetained, const SamplingPolicy &policy,
    uint32_t stopToken0, uint32_t stopToken1, uint32_t tiles) const {
  const uint64_t rows = uint64_t{tiles} * 8;
  if ((tiles != 2 && tiles != 4) || !maximumRetained || maximumRetained > rows ||
      buffers.inputTokens.sizeBytes() < rows * sizeof(uint32_t) ||
      buffers.targetTopIds.sizeBytes() <
          rows * kTargetCandidates * sizeof(uint32_t) ||
      buffers.targetTopProbabilities.sizeBytes() <
          rows * kTargetCandidates * sizeof(float) ||
      buffers.uniforms.sizeBytes() < rows * sizeof(float) ||
      buffers.outputTokens.sizeBytes() < rows * sizeof(uint32_t) ||
      buffers.retainedCount.sizeBytes() < sizeof(uint32_t) ||
      buffers.nextAnchor.sizeBytes() < sizeof(uint32_t) ||
      buffers.acceptedCount.sizeBytes() < sizeof(uint32_t) ||
      buffers.retainedHalves.sizeBytes() < tiles * sizeof(uint32_t)) {
    throw std::invalid_argument("invalid lookup16 acceptance buffers");
  }
  AcceptBatchParams params{};
  params.remaining[0] = maximumRetained;
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  params.lanes = 1;
  params.sampling_mask = policy.samples() ? 1U : 0U;
  graph.add(tiles == 2 ? "decode_accept_lookup16" : "decode_accept_lookup32",
            {buffers.inputTokens, buffers.targetTopIds,
             buffers.targetTopProbabilities, buffers.uniforms,
             buffers.outputTokens, buffers.retainedCount, buffers.nextAnchor,
             buffers.acceptedCount, buffers.retainedHalves},
            params, {1, 1, 1}, {1, 1, 1});
}

void Sampling::addVerifyInput(metal::CommandGraph &graph,
                              metal::MetalBuffer draftInputTokens,
                              metal::MetalBuffer proposedTokens,
                              metal::MetalBuffer verifyInputTokens,
                              uint32_t lanes) const {
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid verify input batch");
  const VerifyInputBatchParams params{lanes, vocabulary_};
  graph.add("verify_input_tokens",
            {std::move(draftInputTokens), std::move(proposedTokens),
             std::move(verifyInputTokens)},
            params, {uint64_t{lanes} * rowsPerLane_, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops

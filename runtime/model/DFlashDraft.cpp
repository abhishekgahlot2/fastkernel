// Modified by meowkernels.
#include "DFlashDraft.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

void validateLayout(const DFlashDraftLayout &layout) {
  if (!layout.layers || !layout.hiddenSize || !layout.vocabularySize ||
      !layout.dynamicSize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) ||
      !layout.targetHiddenSize || !layout.selectorRank ||
      !layout.kvHeads) {
    throw WeightStoreError("DFlash draft layout contains a zero dimension");
  }
  if (layout.selectorRank != 256) {
    throw WeightStoreError(
        "draft selector kernels are compiled for rank 256");
  }
  validateQ4Layout(layout.dynamicSize, layout.hiddenSize);
  validateQ4Layout(layout.qkvSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionSize);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
  validateQ4Layout(layout.selectorRank, layout.hiddenSize);
}

// SPLASH_DRAFT_HEAD_IDS / SPLASH_DRAFT_HEAD_ROWS: restricted head rows, 0 when unset.
uint32_t configuredHeadRows() {
  const char *path = std::getenv("SPLASH_DRAFT_HEAD_IDS");
  if (!path || !*path) return 0;
  const char *rowsValue = std::getenv("SPLASH_DRAFT_HEAD_ROWS");
  return rowsValue ? static_cast<uint32_t>(std::strtoul(rowsValue, nullptr, 10)) : 98304u;
}

} // namespace

uint64_t DFlashDraft::restrictedHeadPlannedBytes(const DFlashDraftLayout &layout) {
  // draftHead(): rows x (Q4 weights, bf16 scales, bf16 biases, u32 ids) over the hidden width.
  const uint64_t rows = configuredHeadRows();
  const auto pages = [](uint64_t bytes) { return (bytes + 16383) & ~uint64_t{16383}; };
  const uint64_t metadata = pages(rows * (layout.hiddenSize / 64) * 2);
  return rows ? pages(rows * layout.hiddenSize / 2) + 2 * metadata + pages(rows * 4) : 0;
}

void DFlashDraft::loadRestrictedHead(const ops::Q4Projection &target) const {
  if (!headIds_.empty())
    static_cast<void>(draftHead(target));
}

void DFlashDraft::disableRestrictedHead() noexcept {
  headIds_.clear();
  headKeep_.clear();
  headFiller_.clear();
}

uint64_t DFlashDraft::restrictedHeadAllocatedBytes() const noexcept {
  return restrictedHead_ ? restrictedHead_->weights.sizeBytes() + restrictedHead_->scales.sizeBytes() +
                               restrictedHead_->biases.sizeBytes() + headIdMap_.sizeBytes()
                         : 0;
}

DFlashDraftRing::DFlashDraftRing(
    metal::MetalBackend &backend, std::shared_ptr<StateAllocationTracker> tracker,
    DraftStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)), layers_(layout.layers) {
  if (!tracker_)
    throw std::invalid_argument("draft state allocation tracker is empty");
  if (!layout.valid() ||
      layout.tokens != ExecutionLimits::draftContextTokens) {
    throw std::invalid_argument("draft state layout is invalid");
  }
  const uint64_t before = backend.memoryStats().allocatedBytes;
  const metal::MetalBuffer base = backend.allocateBuffer(
      layout.ringBytes(), metal::BufferStorage::Shared, label);
  uint64_t cursor = 0;
  for (DFlashDraftRingLayer &layer : layers_) {
    layer.keys = backend.view(base, cursor, layout.tensorBytes());
    cursor += layout.tensorBytes();
    layer.values = backend.view(base, cursor, layout.tensorBytes());
    cursor += layout.tensorBytes();
  }
  if (cursor != layout.ringBytes())
    throw std::logic_error("draft ring accounting mismatch");
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.ringBytes())
    throw std::logic_error("draft ring allocation is below declared bytes");
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashDraftRing::~DFlashDraftRing() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashDraft::DFlashDraft(const DFlashDraftWeights &weights,
                         metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      selector_(backend, weights.layout.vocabularySize,
                ExecutionLimits::draftQueryRows) {
  validateLayout(weights_.layout);
  if (weights_.layers.size() != weights_.layout.layers ||
      !weights_.layout.stateLayout().valid()) {
    throw std::invalid_argument("draft weights do not match state geometry");
  }
  // SPLASH_DRAFT_HEAD_IDS stays an explicit path (off when unset): the ranked ids
  // are data for one tokenizer, and turning it on allocates ~270 MiB outside the
  // memory plan.
  if (const char *path = std::getenv("SPLASH_DRAFT_HEAD_IDS"); path && *path) {
    // Ranked ids (most frequent first, unique). The first rows - 256 form the
    // static head (sorted ascending); the rest fill each request's segment.
    std::ifstream file(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)), {});
    const uint32_t vocabulary = weights_.layout.vocabularySize;
    const uint32_t rows = configuredHeadRows();
    const size_t count = bytes.size() / 4;
    if (bytes.size() % 4 || rows % 256 || rows <= kHeadSegmentRows || rows >= vocabulary ||
        count < rows + kHeadSegmentRows)
      throw std::invalid_argument("draft head ids must rank at least SPLASH_DRAFT_HEAD_ROWS + 256 ids; rows % 256 == 0");
    std::vector<uint32_t> ranked(count);
    std::memcpy(ranked.data(), bytes.data(), count * 4);
    headKeep_.assign(vocabulary, 0);
    std::vector<uint8_t> seen(vocabulary, 0);
    for (const uint32_t id : ranked) {
      if (id >= vocabulary || seen[id])
        throw std::invalid_argument("draft head ids must be unique vocabulary ids");
      seen[id] = 1;
    }
    const size_t staticRows = rows - kHeadSegmentRows;
    headIds_.assign(ranked.begin(), ranked.begin() + staticRows);
    std::sort(headIds_.begin(), headIds_.end());
    for (const uint32_t id : headIds_) headKeep_[id] = 1;
    headFiller_.assign(ranked.begin() + staticRows, ranked.end());
  }
}

std::vector<uint32_t>
DFlashDraft::promptSegment(std::span<const uint32_t> prompt, uint32_t *promptIds) const {
  if (headIds_.empty() || prompt.empty())
    return {};
  std::vector<uint32_t> segment;
  std::vector<uint8_t> taken(headKeep_.size(), 0);
  for (const uint32_t token : prompt) {
    if (token >= headKeep_.size() || headKeep_[token] || taken[token])
      continue;
    if (segment.size() == kHeadSegmentRows)
      return {};  // too many rare prompt tokens: keep the full head
    taken[token] = 1;
    segment.push_back(token);
  }
  // Prompts where over 1% of tokens are rare (other languages) keep the full head:
  // their answers use rare tokens the prompt does not show yet.
  size_t outside = 0;
  for (const uint32_t token : prompt)
    outside += token < headKeep_.size() && !headKeep_[token];
  if (outside * 100 > prompt.size())
    return {};
  if (promptIds) *promptIds = static_cast<uint32_t>(segment.size());
  for (size_t i = 0; segment.size() < kHeadSegmentRows && i < headFiller_.size(); ++i)
    if (!taken[headFiller_[i]]) segment.push_back(headFiller_[i]);
  return segment;
}

void DFlashDraft::copyHeadRows(const ops::Q4Projection &target,
                               std::span<const uint32_t> ids,
                               uint32_t firstRow) const {
  // StorageN=256 packing: weights [N/256][K/64][256][32 B]; scales and biases
  // [N/256][K/64][256] BF16. Rows move independently, bytes unchanged.
  const auto *srcWeights = static_cast<const uint8_t *>(target.weights.contents());
  const auto *srcScales = static_cast<const uint16_t *>(target.scales.contents());
  const auto *srcBiases = static_cast<const uint16_t *>(target.biases.contents());
  auto *weights = static_cast<uint8_t *>(restrictedHead_->weights.contents());
  auto *scales = static_cast<uint16_t *>(restrictedHead_->scales.contents());
  auto *biases = static_cast<uint16_t *>(restrictedHead_->biases.contents());
  const uint64_t groups = target.inputSize / 64;
  for (uint32_t i = 0; i < ids.size(); ++i) {
    const uint64_t source = ids[i], row = firstRow + i;
    for (uint64_t group = 0; group < groups; ++group) {
      const uint64_t from = ((source / 256) * groups + group) * 256 + source % 256;
      const uint64_t to = ((row / 256) * groups + group) * 256 + row % 256;
      std::memcpy(weights + to * 32, srcWeights + from * 32, 32);
      scales[to] = srcScales[from];
      biases[to] = srcBiases[from];
    }
  }
  std::memcpy(static_cast<uint32_t *>(headIdMap_.contents()) + firstRow, ids.data(), ids.size() * 4);
}

void DFlashDraft::useHeadSegment(uint64_t owner, uint64_t version,
                                 std::span<const uint32_t> segment,
                                 const ops::Q4Projection &target) const {
  if (segment.size() != kHeadSegmentRows)
    throw std::invalid_argument("draft head segment must have 256 ids");
  static_cast<void>(draftHead(target));
  if (owner == segmentOwner_ && version == segmentVersion_)
    return;
  // Exact even if a command still reads the old rows: drafts are sampled from
  // the q computed in that command, and acceptance uses that same q.
  // Only rows whose id changed are copied (one 2.9 KB row per new rare token).
  const uint32_t first = static_cast<uint32_t>(headIds_.size());
  for (uint32_t i = 0; i < kHeadSegmentRows; ++i)
    if (loadedSegment_[i] != segment[i]) {
      copyHeadRows(target, segment.subspan(i, 1), first + i);
      loadedSegment_[i] = segment[i];
    }
  segmentOwner_ = owner;
  segmentVersion_ = version;
}

void DFlashDraft::addSelection(
    metal::CommandGraph &graph, DFlashSelectionBuffers buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies, uint32_t proposalTokens,
    bool restrictedHead, metal::MetalBuffer deviceSelectorParams) const {
  if (restrictedHead && headIds_.empty())
    throw std::logic_error("restricted draft head requested without SPLASH_DRAFT_HEAD_IDS");
  selector_.addDraftSelector(
      graph,
      {std::move(buffers.logits), std::move(buffers.partialIds),
       std::move(buffers.partialValues), std::move(buffers.candidates),
       std::move(buffers.unary), std::move(buffers.selectorHidden),
       weights_.predecessorCodebook, weights_.successorCodebook,
       std::move(buffers.uniforms), std::move(buffers.proposedTokens),
       std::move(buffers.proposalProbabilities),
       restrictedHead ? headIdMap_ : metal::MetalBuffer{},
       restrictedHead ? headRows_ : 0u},
      anchors, policies, proposalTokens, std::move(deviceSelectorParams));
}

const ops::Q4Projection &
DFlashDraft::draftHead(const ops::Q4Projection &target) const {
  if (restrictedHead_)
    return *restrictedHead_;
  if (headIds_.empty() || target.outputSize != weights_.layout.vocabularySize)
    throw std::logic_error("restricted draft head needs SPLASH_DRAFT_HEAD_IDS and the full target head");
  if (!target.weights.contents() || !target.scales.contents() || !target.biases.contents() ||
      target.inputSize % 64)
    throw std::runtime_error("target head is not CPU-visible Q4 for draft head gathering");
  const uint32_t count = static_cast<uint32_t>(headIds_.size()) + kHeadSegmentRows;
  const uint64_t groups = target.inputSize / 64;
  restrictedHead_ = ops::Q4Projection{
      backend_.allocateBuffer(uint64_t{count} * target.inputSize / 2, metal::BufferStorage::Shared, "draft-head-weights"),
      backend_.allocateBuffer(uint64_t{count} * groups * 2, metal::BufferStorage::Shared, "draft-head-scales"),
      backend_.allocateBuffer(uint64_t{count} * groups * 2, metal::BufferStorage::Shared, "draft-head-biases"),
      count, target.inputSize};
  headIdMap_ = backend_.allocateBuffer(uint64_t{count} * 4, metal::BufferStorage::Shared, "draft-head-ids");
  copyHeadRows(target, headIds_, 0);
  copyHeadRows(target, std::span(headFiller_).first(kHeadSegmentRows), static_cast<uint32_t>(headIds_.size()));
  loadedSegment_.assign(headFiller_.begin(), headFiller_.begin() + kHeadSegmentRows);
  segmentOwner_ = 0;
  headRows_ = count;
  return *restrictedHead_;
}

void DFlashDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  for (const DFlashPrefillSpan &span : spans) {
    if (span.ring.size() != layout.layers)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  }
  const ops::LinearMatrix context{layout.hiddenSize,
                                    layout.targetHiddenSize};
  operators_.linear().addPrefillSums(graph, buffers.capturedTargetHidden,
                     buffers.projectionSums, context, rows);
  operators_.linear().addPrefill(graph, buffers.capturedTargetHidden,
                 weights_.contextProjection, buffers.projected,
                 buffers.projectionSums, context, rows);
  ops::Normalization::addRmsWithQ4Sums(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden,
      buffers.projectionSums, layout.hiddenSize, rows);

  const ops::LinearMatrix qkv{layout.qkvSize, layout.hiddenSize};
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    operators_.linear().addPrefill(graph, buffers.hidden,
                      weights_.layers[layer].qkvProjection, buffers.qkv,
                      buffers.projectionSums, qkv, rows);
    for (const DFlashPrefillSpan &span : spans) {
      const uint64_t qkvOffset =
          uint64_t{span.compactRow} * layout.qkvSize * sizeof(uint16_t);
      const uint64_t ropeOffset =
          uint64_t{span.compactRow} * (layout.attentionHeadDimension / 2) * sizeof(float);
      ops::DraftAttention::addContextPrefill(
          graph,
          backend_.view(buffers.qkv, qkvOffset,
                        uint64_t{span.rows} * layout.qkvSize *
                            sizeof(uint16_t)),
          weights_.layers[layer].keyNorm,
          backend_.view(buffers.ropeCos, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          backend_.view(buffers.ropeSin, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          span.ring[layer].keys, span.ring[layer].values, span.rows,
          layout.stateLayout().tokens, span.startPosition,
          layout.attentionShape());
    }
  }
}

void DFlashDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Q4Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths, uint32_t lanes,
    ops::Q4DispatchStats &stats, bool restrictedHead,
    metal::MetalBuffer deviceAttentionParams) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      cacheLengths.size() != ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes);
  const ops::LinearMatrix qkv{layout.qkvSize, layout.hiddenSize};
  const ops::LinearMatrix dynamic{layout.dynamicSize, layout.hiddenSize};
  const ops::LinearMatrix output{layout.hiddenSize, layout.attentionSize};
  const ops::LinearMatrix gateUp{layout.intermediateSize, layout.hiddenSize};
  const ops::LinearMatrix down{layout.hiddenSize, layout.intermediateSize};

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const DFlashDraftLayerWeights &weights = weights_.layers[layer];
    ops::Normalization::addRms(graph, buffers.hidden[current],
                               weights.inputNorm,
                               buffers.normalized, layout.hiddenSize, rows, buffers.linearScratch);
    operators_.linear().addDecodeBatch(graph,
                       buffers.normalized, weights.attentionDynamic,
                       buffers.dynamic, dynamic, lanes, stats, buffers.linearScratch, true);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    operators_.linear().addDecodeBatch(graph, buffers.convolved,
                       weights.qkvProjection, buffers.proposalQkv, qkv, lanes,
                       stats, buffers.linearScratch);
    ops::DraftAttention::addPrepare(
        graph,
        {buffers.proposalQkv, buffers.attention, weights.queryNorm,
         weights.keyNorm, buffers.ropeCos, buffers.ropeSin, buffers.queryKeys,
         buffers.queryValues},
        attentionPlan);
    ops::DraftAttention::addDecode(
        graph,
        {buffers.attention, buffers.persistentKeys[layer],
         buffers.persistentValues[layer], buffers.queryKeys,
         buffers.queryValues},
        cacheLengths, layout.stateLayout().tokens, attentionPlan,
        deviceAttentionParams);
    ops::DraftAttention::addReorder(graph, buffers.attention,
                                    buffers.proposalQkv, attentionPlan);
    operators_.linear().addDecodeBatch(graph,
                       buffers.proposalQkv, weights.outputProjection,
                       buffers.projected, output, lanes, stats, buffers.linearScratch);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.residual},
        attentionPlan, ops::DraftConvolutionStage::Residual);
    ops::Normalization::addRms(graph, buffers.residual,
                               weights.postAttentionNorm, buffers.normalized,
                               layout.hiddenSize, rows, buffers.linearScratch);
    operators_.linear().addDecodeBatch(graph,
                       buffers.normalized, weights.mlpDynamic, buffers.dynamic,
                       dynamic, lanes, stats, buffers.linearScratch, true);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    operators_.linear().addGateUpBatch(graph, buffers.convolved, weights.gateProjection,
                       weights.upProjection, buffers.gateScratch,
                       buffers.intermediate, gateUp, lanes, stats, buffers.linearScratch);
    operators_.linear().addDecodeBatch(graph,
                       buffers.intermediate, weights.downProjection,
                       buffers.projected, down, lanes, stats, buffers.linearScratch);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.hidden[next]},
        attentionPlan, ops::DraftConvolutionStage::Residual);
  }

  ops::Normalization::addRms(graph,
                             buffers.hidden[weights_.layout.layers & 1],
                             weights_.finalNorm,
                             buffers.finalHidden, layout.hiddenSize, rows, buffers.linearScratch);
  const ops::Q4Projection &headWeights =
      restrictedHead ? draftHead(vocabularyProjection) : vocabularyProjection;
  const ops::LinearMatrix head{restrictedHead ? headRows_ : layout.vocabularySize, layout.hiddenSize};
  operators_.linear().addDecodeBatch(graph, buffers.finalHidden,
                     headWeights, buffers.logits, head, lanes, stats, buffers.linearScratch, true);
  const ops::LinearMatrix selector{layout.selectorRank, layout.hiddenSize};
  operators_.linear().addDecodeBatch(graph,
                     buffers.finalHidden, weights_.selectorProjection,
                     buffers.selectorHidden, selector, lanes, stats, buffers.linearScratch, true);
}

void DFlashDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      startPositions.size() != ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft context batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  const ops::LinearMatrix context{layout.hiddenSize, layout.targetHiddenSize};
  operators_.linear().addDecodeBatch(graph,
                     buffers.capturedTargetHidden, weights_.contextProjection,
                     buffers.projected, context, lanes, stats, buffers.linearScratch);
  ops::Normalization::addRms(graph, buffers.projected, weights_.hiddenNorm,
                             buffers.hidden, layout.hiddenSize, rows, buffers.linearScratch);

  const ops::LinearMatrix qkv{layout.qkvSize, layout.hiddenSize};
  if (!buffers.layerQkv.empty() && buffers.layerQkv.size() != layout.layers)
    throw std::invalid_argument("invalid per-layer draft context qkv");
  // SPLASH_GROUPED_CONTEXT_KV (exact; default "1", "0" = off; read per graph build so lockstep can flip it): the
  // commits read only the K/V columns of each layer's qkv output, so only those are
  // projected (Q4Linear::addContextKv, the full projection's own tile). Exactly "1": one layers x 16-group dispatch
  // for all layers (27B drafter: 5 layers, 80 groups; 35B: 6, 96), then the commits; "2", or "1" with
  // SPLASH_GROUPED_CONTEXT_KV_TAIL=1 (so a two-arm lockstep can flip it, one lane only): one 16-group dispatch per
  // layer, each followed by its commit (the tail control). Needs per-layer qkv views (E3's at one lane; Runtime's own at 2
  // lanes when the switch is "1") and a plan Q4Linear admits; otherwise the full projections run and the reason prints
  // once. B1 lockstep -0.219 ms/step, B2 -0.368 ms per B2 cycle; rings byte-identical, oracle identical
  // (linear-plan --grouped-context-kv).
  const auto env = [](const char *name, const char *fallback = "") {
    const char *value = std::getenv(name);
    return std::string_view(value ? value : fallback);
  };
  const std::string_view groupedSwitch = env("SPLASH_GROUPED_CONTEXT_KV", "1");
  const int groupedMode = groupedSwitch == "2" ? 2
      : groupedSwitch == "1" ? (env("SPLASH_GROUPED_CONTEXT_KV_TAIL") == "1" ? 2 : 1) : 0;
  contextKvWitness.commits.fetch_add(1, std::memory_order_relaxed);
  const bool seamViews = !buffers.layerQkv.empty();  // per-layer views: the admission scope
  // The commit reads qkv columns attentionSize.. (draft_context_kv_commit), so that is where the K/V tiles start.
  const bool kvOnly = groupedMode && seamViews && (groupedMode == 1 || lanes == 1) &&
                      operators_.linear().admitsContextKv(qkv, lanes, layout.layers, layout.attentionSize);
  // An explicit class and reason, never a silent full-projection fallback. Gates grep these texts; keep them stable
  // ("REFUSED inside E3 scope" also covers Runtime's 2-4 lane views).
  if (groupedMode && !kvOnly) {
    (seamViews ? contextKvWitness.refused : contextKvWitness.outside).fetch_add(1, std::memory_order_relaxed);
    static std::atomic<bool> outsideWitnessed{false}, refusedWitnessed{false};
    if (!(seamViews ? refusedWitnessed : outsideWitnessed).exchange(true)) {
      const auto full = operators_.linear().plan({qkv, rows, ops::LinearPhase::Decode, ops::LinearEpilogue::None});
      std::fprintf(stderr, "grouped context K/V: %s (mode %d, lanes %u, layers %u, per-layer qkv %zu, "
                   "plan %s x %u groups x %u threads); full projections kept\n",
                   seamViews ? "REFUSED inside E3 scope" : "outside admission", groupedMode, lanes, layout.layers,
                   buffers.layerQkv.size(), std::string(full.pipeline()).c_str(), full.configuration().groups,
                   full.threadsPerThreadgroup());
    }
  }
  std::array<const ops::Q4Projection *, ops::Q4Linear::kMaxContextKvLayers> slots{};
  const auto projections = std::span(slots).first(kvOnly ? layout.layers : 0);
  if (kvOnly) {
    for (uint32_t layer = 0; layer < layout.layers; ++layer)
      projections[layer] = &weights_.layers[layer].qkvProjection;
    if (groupedMode == 1)
      operators_.linear().addContextKv(graph, buffers.hidden, projections, buffers.layerQkv, qkv,
                                       layout.attentionSize, lanes, buffers.linearScratch, std::nullopt);
    contextKvWitness.built.fetch_add(1, std::memory_order_relaxed);
    if (lanes > 1) contextKvWitness.multi.fetch_add(1, std::memory_order_relaxed);
  }
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const metal::MetalBuffer &layerQkv =
        buffers.layerQkv.empty() ? buffers.qkv : buffers.layerQkv[layer];
    if (!kvOnly)
      operators_.linear().addDecodeBatch(graph, buffers.hidden,
                         weights_.layers[layer].qkvProjection, layerQkv, qkv,
                         lanes, stats, buffers.linearScratch, true);
    else if (groupedMode == 2)
      operators_.linear().addContextKv(graph, buffers.hidden, projections, buffers.layerQkv, qkv,
                                       layout.attentionSize, lanes, buffers.linearScratch, layer);
    ops::DraftAttention::addContextCommit(
        graph, layerQkv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedCounts,
        startPositions, layout.stateLayout().tokens, layout.attentionShape(),
        lanes);
  }
}

DFlashDraftWeights
loadDFlashDraftWeights(metal::MetalBackend &backend,
                       const std::filesystem::path &directory,
                       DFlashDraftLayout layout) {
  validateLayout(layout);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DFlashDraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t hiddenBytes = checkedWeightMultiply(
      layout.hiddenSize, kBFloat16Bytes, "draft norm bytes");
  const uint64_t convolutionBytes = checkedWeightMultiply(
      checkedWeightMultiply(4, layout.hiddenSize,
                            "draft convolution elements"),
      kBFloat16Bytes, "draft convolution bytes");
  const uint64_t headNormBytes = checkedWeightMultiply(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const std::string filename =
        "layer-" + std::to_string(layerIndex) + ".bin";
    WeightFile file(backend, directory / filename, "draft/" + filename,
                    kDFlashLayerMagic, layerIndex, 0);
    DFlashDraftLayerWeights layer;
    layer.inputNorm = file.section(hiddenBytes, "input-norm");
    layer.attentionConvolution =
        file.section(convolutionBytes, "attention-convolution");
    layer.attentionDynamic = readQ4Projection(
        file, backend, layout.dynamicSize, layout.hiddenSize,
        "attention-dynamic");
    layer.qkvProjection = readQ4Projection(
        file, backend, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readQ4Projection(
        file, backend, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        file.section(hiddenBytes, "post-attention-norm");
    layer.mlpConvolution = file.section(convolutionBytes, "mlp-convolution");
    layer.mlpDynamic = readQ4Projection(
        file, backend, layout.dynamicSize, layout.hiddenSize, "mlp-dynamic");
    layer.gateProjection = readQ4Projection(
        file, backend, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readQ4Projection(
        file, backend, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readQ4Projection(file, backend, layout.hiddenSize,
                                            layout.intermediateSize,
                                            "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.layers.push_back(std::move(layer));
  }

  {
    WeightFile file(backend, directory / "model.bin", "draft/model.bin",
                    kDFlashLayerMagic, layout.layers, 1);
    result.contextProjection = readQ4Projection(
        file, backend, layout.hiddenSize, layout.targetHiddenSize,
        "context-projection");
    result.hiddenNorm = file.section(hiddenBytes, "hidden-norm");
    result.finalNorm = file.section(hiddenBytes, "final-norm");
    result.selectorProjection = readQ4Projection(
        file, backend, layout.selectorRank, layout.hiddenSize, "selector");
    const uint64_t codebookBytes = checkedWeightMultiply(
        checkedWeightMultiply(layout.vocabularySize, layout.selectorRank,
                              "draft codebook elements"),
        kBFloat16Bytes, "draft codebook bytes");
    result.predecessorCodebook =
        file.section(codebookBytes, "predecessor-codebook");
    result.successorCodebook =
        file.section(codebookBytes, "successor-codebook");
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model

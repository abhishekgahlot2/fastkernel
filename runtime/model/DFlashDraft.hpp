// Modified by meowkernels.
#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Sampling.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

// SPLASH_GROUPED_CONTEXT_KV route witness. decode-profile's lockstep and block modes print each cycle's deltas as
// gkv=commits/built/grouped/tail/seams/outside/refused/multi:
//   commits  context commits encoded (any route)             built    of them with K/V-only projections
//   grouped, tail, seams  E3 seams placed with the grouped layout, the five-tail layout, and in total (Runtime
//            placeSeamSiblings, counted once placed)
//   outside  the switch was set but the commit has no E3 per-layer views (wide lookup, multi-lane, constrained, E3
//            off): stock full projections, by design     refused  the switch was set inside E3's scope but not admitted
//   multi    grouped K/V built at 2-4 lanes (per-layer views from Runtime, no seam: the commit stays in order)
struct ContextKvWitness final {
  std::atomic<uint64_t> commits{0}, built{0}, grouped{0}, tail{0}, seams{0}, outside{0}, refused{0}, multi{0};
};
inline ContextKvWitness contextKvWitness;

struct DFlashDraftRingLayer final {
  // K is [head][ring_position][dimension].
  metal::MetalBuffer keys;
  // V is [head][dimension][ring_position].
  metal::MetalBuffer values;
};

// Draft-owned persistent context, paired with target state in composite caches.
class DFlashDraftRing final {
public:
  DFlashDraftRing(metal::MetalBackend &backend,
                  std::shared_ptr<StateAllocationTracker> tracker,
                  DraftStateLayout layout,
                  std::string_view label);
  ~DFlashDraftRing();
  DFlashDraftRing(const DFlashDraftRing &) = delete;
  DFlashDraftRing &operator=(const DFlashDraftRing &) = delete;

  [[nodiscard]] const std::vector<DFlashDraftRingLayer> &layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept {
    return actualAllocatedBytes_;
  }

private:
  std::shared_ptr<StateAllocationTracker> tracker_;
  std::vector<DFlashDraftRingLayer> layers_;
  uint64_t actualAllocatedBytes_ = 0;
};

struct DFlashDraftLayout final {
  uint32_t layers = 5;
  uint32_t hiddenSize = 5120;
  uint32_t vocabularySize = 248320;
  uint32_t dynamicSize = 1280;
  uint32_t qkvSize = 6144;
  uint32_t attentionSize = 4096;
  uint32_t intermediateSize = 17408;
  uint32_t attentionHeadDimension = 128;
  float rotaryTheta = 10'000'000.0F;
  uint32_t targetHiddenSize = 25600;
  uint32_t selectorRank = 256;
  uint32_t kvHeads = 8;

  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    return {layers, kvHeads, ExecutionLimits::draftContextTokens,
            attentionHeadDimension};
  }
  [[nodiscard]] constexpr ops::DraftAttentionShape attentionShape() const noexcept {
    return {hiddenSize, dynamicSize, qkvSize, attentionSize,
            attentionSize / attentionHeadDimension, kvHeads,
            attentionHeadDimension};
  }

  bool operator==(const DFlashDraftLayout &) const = default;
};

struct DFlashDecodeBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer dynamic;
  metal::MetalBuffer convolved;
  metal::MetalBuffer proposalQkv;
  metal::MetalBuffer attention;
  metal::MetalBuffer projected;
  metal::MetalBuffer residual;
  metal::MetalBuffer intermediate;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer gateScratch;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashContextBuffers final {
  ops::LinearScratch linearScratch{};
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer qkv;
  // SPLASH_SEAM_SIBLING: one qkv buffer per drafter layer (empty: all layers
  // share `qkv`, projected and committed one layer at a time).
  std::vector<metal::MetalBuffer> layerQkv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer retainedCounts;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashPrefillSpan final {
  uint32_t compactRow = 0;
  uint32_t rows = 0;
  uint32_t startPosition = 0;
  std::span<const DFlashDraftRingLayer> ring;
};

struct DFlashPrefillBuffers final {
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer qkv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
};

struct DFlashSelectionBuffers final {
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
};

struct DFlashDraftLayerWeights final {
  metal::MetalBuffer inputNorm;
  metal::MetalBuffer attentionConvolution;
  ops::Q4Projection attentionDynamic;
  ops::Q4Projection qkvProjection;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Q4Projection outputProjection;
  metal::MetalBuffer postAttentionNorm;
  metal::MetalBuffer mlpConvolution;
  ops::Q4Projection mlpDynamic;
  ops::Q4Projection gateProjection;
  ops::Q4Projection upProjection;
  ops::Q4Projection downProjection;
};

struct DFlashDraftWeights final {
  DFlashDraftLayout layout;
  std::vector<DFlashDraftLayerWeights> layers;
  ops::Q4Projection contextProjection;
  metal::MetalBuffer hiddenNorm;
  metal::MetalBuffer finalNorm;
  ops::Q4Projection selectorProjection;
  metal::MetalBuffer predecessorCodebook;
  metal::MetalBuffer successorCodebook;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

inline constexpr std::string_view kDFlashLayerMagic = "MDFD0004";

[[nodiscard]] DFlashDraftWeights
loadDFlashDraftWeights(metal::MetalBackend &backend,
                       const std::filesystem::path &directory,
                       DFlashDraftLayout layout = {});

// Builds the draft layer graph from packed buffers and persistent context.
// Sampling and acceptance policy remain outside the model.
class DFlashDraft final {
public:
  DFlashDraft(const DFlashDraftWeights &weights, metal::MetalBackend &backend,
               const ops::ExecutionPlans &operators);

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;

  // deviceAttentionParams / deviceSelectorParams (SPLASH_DRAFT_AHEAD): GPU copies
  // of attentionParams() / selectionParams(), bound instead of host bytes.
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Q4Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths, uint32_t lanes,
                 ops::Q4DispatchStats &stats, bool restrictedHead = false,
                 metal::MetalBuffer deviceAttentionParams = {}) const;
  void addSelection(metal::CommandGraph &graph,
                    DFlashSelectionBuffers buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies,
                    uint32_t proposalTokens, bool restrictedHead = false,
                    metal::MetalBuffer deviceSelectorParams = {}) const;
  [[nodiscard]] DraftAttentionBatchParams
  attentionParams(std::span<const uint32_t> cacheLengths, uint32_t lanes) const {
    return ops::DraftAttention::decodeParams(
        cacheLengths, weights_.layout.stateLayout().tokens, lanes);
  }
  [[nodiscard]] SelectorBatchParams
  selectionParams(std::span<const uint32_t> anchors,
                  std::span<const ops::SamplingPolicy> policies) const {
    return selector_.draftSelectorParams(anchors, policies);
  }
  static constexpr uint32_t kHeadSegmentRows = 256;
  // SPLASH_DRAFT_HEAD_IDS: bytes the restricted head allocates (0 when unset), each
  // buffer rounded to 16 KiB pages. The memory plan counts them with the draft weights.
  [[nodiscard]] static uint64_t restrictedHeadPlannedBytes(const DFlashDraftLayout &layout);
  // Gathers the restricted head at startup (it is planned) instead of on the first draft.
  void loadRestrictedHead(const ops::Q4Projection &target) const;
  // The memory plan could not hold the restricted head: every request drafts with the full head.
  void disableRestrictedHead() noexcept;
  [[nodiscard]] uint64_t restrictedHeadAllocatedBytes() const noexcept;
  // Frequency-ranked head (SPLASH_DRAFT_HEAD_IDS): the 256 ids a request adds
  // to the static rows (its rare prompt tokens, then filler), or empty when
  // unconfigured or the prompt has more than 256 rare tokens (full head).
  // *promptIds receives how many leading ids came from the prompt (the rest is
  // filler that decode-time rare output tokens may replace).
  [[nodiscard]] std::vector<uint32_t> promptSegment(std::span<const uint32_t> prompt,
                                                    uint32_t *promptIds = nullptr) const;
  [[nodiscard]] bool staticHeadHas(uint32_t token) const noexcept {
    return token < headKeep_.size() && headKeep_[token];
  }
  // Loads a request's segment (owner, version) into the head's last 256 rows (B1 only).
  void useHeadSegment(uint64_t owner, uint64_t version, std::span<const uint32_t> segment,
                      const ops::Q4Projection &target) const;
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions,
                        uint32_t lanes,
                        ops::Q4DispatchStats &stats) const;

private:
  // SPLASH_DRAFT_HEAD_IDS=path (ascending u32 ids, count % 256 == 0): gather
  // those rows of the target head once; the draft head then reads only them.
  const ops::Q4Projection &draftHead(const ops::Q4Projection &target) const;
  void copyHeadRows(const ops::Q4Projection &target, std::span<const uint32_t> ids,
                    uint32_t firstRow) const;

  const DFlashDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::Sampling selector_;
  std::vector<uint32_t> headIds_;
  std::vector<uint8_t> headKeep_;
  std::vector<uint32_t> headFiller_;
  mutable uint64_t segmentOwner_ = 0;
  mutable uint64_t segmentVersion_ = 0;
  mutable std::vector<uint32_t> loadedSegment_;
  mutable std::optional<ops::Q4Projection> restrictedHead_;
  mutable metal::MetalBuffer headIdMap_;
  mutable uint32_t headRows_ = 0;
};

} // namespace splash::model

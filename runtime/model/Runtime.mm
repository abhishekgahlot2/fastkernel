// Modified by meowkernels.
#include "model/Runtime.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTarget.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/PromptLookup.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/EnvSwitch.hpp"
#include "metal/HostPhase.hpp"
#include "metal/abi/DraftAttention.h"
#include "ops/DraftAttention.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"
#include "ops/RoPE.hpp"
#include "ops/Sampling.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

// Fixed reserves the memory plan carries beside the planned arenas: Metal
// pipeline objects and encoder scratch, and the process's own runtime overhead.
// Capped at 1% and 2% of the device's recommended working set so small devices
// are not charged large-device constants; working sets of 25 GiB or more keep
// the full amounts.
constexpr uint64_t kPipelineReserveBytes = 256ULL << 20;
constexpr uint64_t kRuntimeOverheadReserveBytes = 512ULL << 20;

using metal::BufferStorage;
using metal::CommandGraph;
using metal::CommandTicket;
using metal::CommandTiming;
using metal::MetalBackend;
using metal::MetalBuffer;

// All-Macs G0: wide lookups only up to the rows that keep every projection's
// 8-row K reduction (Apple9 plans its 32-row input projections and head as
// plain tiles); the result is logged once per process.
uint32_t loggedStableVerifyRows(const RuntimeGeometry &geometry,
                                const ops::ExecutionPlans &operators) {
  const auto scratch = DecodeArena::linearScratchSize(geometry, operators);
  const uint32_t rows = rowStableVerifyRows(operators.linear(), geometry.target,
                                            scratch.sums && !scratch.input);
  [[maybe_unused]] static const bool logged = [rows] {
    std::fprintf(stderr, "row-stable verify rows: %u\n", rows);
    return true;
  }();
  return rows;
}

class DeferredMetalTicket final : public ModelBatchTicket {
public:
  using Completion = std::function<std::vector<ModelStepResult>(CommandTiming)>;

  DeferredMetalTicket(CommandTicket ticket, Completion completion,
                      double priorWallMilliseconds = 0.0,
                      bool representativePrefillTiming = true)
      : ticket_(std::move(ticket)), completion_(std::move(completion)),
        wallMilliseconds_(priorWallMilliseconds),
        representativePrefillTiming_(representativePrefillTiming) {}

  bool ready() const noexcept override { return ticket_.ready(); }

  std::vector<ModelStepResult> wait() override {
    if (!completion_) {
      throw std::logic_error("Metal ticket was already consumed");
    }
    CommandTiming timing = ticket_.wait();
    wallMilliseconds_ += timing.wallSeconds * 1000.0;
    Completion completion = std::move(completion_);
    return completion(timing);
  }

  double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }
  bool prefillTimingIsRepresentative() const noexcept override {
    return representativePrefillTiming_;
  }

private:
  CommandTicket ticket_;
  Completion completion_;
  double wallMilliseconds_ = 0.0;
  bool representativePrefillTiming_;
};

class ReadyModelTicket final : public ModelBatchTicket {
public:
  ReadyModelTicket(std::vector<ModelStepResult> results,
                   double wallMilliseconds)
      : results_(std::move(results)), wallMilliseconds_(wallMilliseconds) {}

  bool ready() const noexcept override { return true; }

  std::vector<ModelStepResult> wait() override {
    if (!results_) {
      throw std::logic_error("ready model ticket was already consumed");
    }
    std::vector<ModelStepResult> results = std::move(*results_);
    results_.reset();
    return results;
  }

  double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }

private:
  std::optional<std::vector<ModelStepResult>> results_;
  double wallMilliseconds_ = 0.0;
};

using kv::Q8ChunkedPrefillParams;

bool isStopToken(const RuntimeGeometry &geometry, uint32_t token) noexcept {
  return token == geometry.target.stopTokens[0] ||
         token == geometry.target.stopTokens[1];
}

void requireShared(const MetalBuffer &buffer, std::string_view label) {
  if (!buffer || buffer.storage() != BufferStorage::Shared ||
      !buffer.contents()) {
    throw std::logic_error(std::string(label) + " is not CPU-visible");
  }
}

template <class T>
T *contents(const MetalBuffer &buffer, std::string_view label) {
  requireShared(buffer, label);
  return static_cast<T *>(buffer.contents());
}

void validatePlan(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                  WorkKind expected) {
  if (plan.kind != expected || plan.empty() || plan.width() > kLaneCount ||
      items.size() != plan.items.size()) {
    throw std::invalid_argument("model runtime received an invalid batch plan");
  }
  for (size_t index = 0; index < items.size(); ++index) {
    if (items[index].requestId != plan.items[index].requestId ||
        items[index].stateSlot >= kLaneCount ||
        (expected == WorkKind::Prefill &&
         (!plan.items[index].tokenCount ||
          plan.items[index].tokenCount != items[index].tokenCount ||
          items[index].inputTokens.size() != items[index].tokenCount)) ||
        (expected == WorkKind::Decode &&
         (plan.items[index].tokenCount || items[index].tokenCount ||
          !items[index].inputTokens.empty()))) {
      throw std::invalid_argument("batch items do not match explicit plan");
    }
  }
}

QwenStateStorage &requireQwenStateStorage(StateStorage &storage) {
  auto *qwen = dynamic_cast<QwenStateStorage *>(&storage);
  if (!qwen) {
    throw std::invalid_argument(
        "Qwen runtime requires Qwen composite state storage");
  }
  return *qwen;
}

// Any unassigned slot works: its buffers come from the storage's pool, and
// the governor is asked only for what the pool lacks.
template <class Activate>
StateAdmission admitIdleSlot(const QwenStateStorage &states,
                             Activate activate) {
  for (uint32_t slot = 0; slot < kLaneCount; ++slot) {
    if (states.metadata(slot).assigned)
      continue;
    const metal::AllocationResult admission = activate(slot);
    if (admission)
      return {slot, StateFailure::None};
    return {{}, StateFailure::MemoryPressure, admission.failure};
  }
  return {{}, StateFailure::ConcurrencyLimit};
}

} // namespace

struct Runtime::Impl {
  // Repeated placements share one encode and its buffers. Pixels are released
  // on completion; embeddings remain until every placement has finished.
  struct ImageData final {
    MetalBuffer pixels;
    MetalBuffer embeddings;
    bool encoding = false;
    bool encoded = false;
  };

  struct ImageState final {
    ImageSpan span;
    std::shared_ptr<ImageData> data;
  };

  struct Request final {
    uint64_t id = 0;
    uint32_t slot = 0;
    bool resident = false;
    bool promptComplete = false;
    // Rebuild state from already-emitted tokens without sampling an initial
    // anchor, consuming RNG, or replaying output to the caller.
    bool replayingGeneration = false;
    uint32_t promptTokens = 0;
    uint32_t maxNewTokens = 0;
    uint32_t generatedTokens = 0;
    BatchCohort cohort = BatchCohort::Greedy;
    SamplingParameters sampling;
    ConstraintMode constraint = ConstraintMode::None;
    std::optional<uint32_t> pendingToken;
    // Transient active-request hidden used only while a constrained request
    // waits for its first token mask. Composite cache state never stores it;
    // every cache hit replays one input token and regenerates this value.
    std::vector<uint16_t> finalTargetHidden;
    std::array<float, kSamplingUniformCount> cycleUniforms{};
    // Nonempty selects score-only mode: the final prefill chunk computes raw
    // logits at these token ids instead of selecting an anchor.
    std::vector<uint32_t> scoreTokens;
    std::vector<uint32_t> maskWords;
    // Set only while the current scheduler-owned ticket overlaps grammar-mask
    // computation with target verification. This is model runtime state, not a
    // scheduler decode stage.
    bool verifyMaskInFlight = false;
    // Frequency-ranked draft head segment for this prompt (empty: full head).
    std::vector<uint32_t> draftHeadSegment;
    uint32_t draftHeadSegmentUsed = 0;  // leading prompt/output ids; the rest is filler
    uint64_t draftHeadSegmentVersion = 0;
    uint32_t verifyRows = kDecodeRows;
    uint64_t rngCounter = 0;
    DecodeStage decodeStage = DecodeStage::Regular;
    bool draftContextValid = false;
    uint64_t draftContextThrough = 0;
    std::optional<DraftContextPlan> draftContextPlan;
    std::vector<ImageState> images;
    std::vector<uint32_t> lookupHistory;
    uint64_t lookupCycles = 0;
    uint64_t lookupHits = 0;
    uint64_t lookupRetained = 0;
    uint64_t wideLookupHits = 0;
    uint64_t wideLookupRetained = 0;
    // SPLASH_WIDE_LOOKUP32: this request may verify 32 rows, its last cycle
    // was a fully accepted wide lookup, its 32-row cycles, and a 32-row
    // cycle's second 16 uniforms.
    bool wide32 = false;
    bool wideFull = false;
    ops::WideGdn wideGdn = ops::WideGdn::Chain;  // SPLASH_WIDE_GDN_SINGLE
    uint64_t wideGdnCycles = 0;  // wide cycles encoded with that route (not Chain)
    uint64_t wide32Hits = 0;
    std::array<float, kSamplingUniformCount> wideUniforms{};
    size_t lookupMatch = 16;  // SPLASH_LOOKUP_ADAPTIVE per-request threshold
    // SPLASH_DRAFT_AHEAD blocks launched for / adopted by this request, and
    // those dropped because the next cycle chose a prompt lookup.
    uint64_t aheadLaunched = 0;
    uint64_t aheadAdopted = 0;
    uint64_t aheadLookupDrops = 0;
    // Cycles since this request's last prompt-lookup cycle (none yet: max).
    uint32_t cyclesSinceLookup = std::numeric_limits<uint32_t>::max();
  };

  struct DecodeLaneResult final {
    Request *request = nullptr;
    uint32_t retained = 0;
    uint32_t accepted = 0;
    uint32_t nextAnchor = 0;
    uint32_t currentAnchor = 0;
    uint32_t maximumRetained = 0;
    bool verify = false;
    bool draftForMask = false;
    bool draftComputed = false;
    bool promptLookup = false;
    bool wideLookup = false;
    uint32_t wideTiles = 2;  // M8 tiles of a wide lookup: 2 (16 rows) or 4 (32)
  };

  struct PageTableBinding final {
    uint64_t requestId = 0;
    uint64_t revision = 0;
    uint32_t entries = 0;
  };

  MetalBackend &backend;
  metal::AllocationAdmission admitAllocation;
  const ModelPackage &package;
  const RuntimeGeometry geometry;
  const ops::ExecutionPlans &operators;
  kv::PageStorage &kvPages;
  QwenStateStorage &states;
  std::unique_ptr<PrefillArena> prefillArena;
  std::unique_ptr<DecodeArena> decodeArena;
  // SPLASH_REUSE_DECODE_GRAPH (default on): keep the graph's host storage between
  // steps. Host only; texts 18/18 identical, tok/s +0.35%.
  const bool reuseDecodeGraph = metal::envSwitch("SPLASH_REUSE_DECODE_GRAPH");
  CommandGraph reusableDecodeGraph;
  bool decodeGraphInUse = false;

  class DecodeGraphLease final {
    std::optional<CommandGraph> local_;
    Impl &owner_;
  public:
    CommandGraph &graph;
    explicit DecodeGraphLease(Impl &owner)
        : owner_(owner), graph(owner.reuseDecodeGraph
              ? owner.reusableDecodeGraph : local_.emplace()) {
      if (owner_.reuseDecodeGraph && std::exchange(owner_.decodeGraphInUse, true))
        throw std::logic_error("decode graph is already being encoded");
    }
    DecodeGraphLease(const DecodeGraphLease &) = delete;
    DecodeGraphLease &operator=(const DecodeGraphLease &) = delete;
    ~DecodeGraphLease() {
      // SPLASH_STREAMED_SUBMIT: a head streamed from this graph whose
      // submission never came (the build threw).
      if (owner_.streamChunk && owner_.backend.streamedHeadPending())
        owner_.backend.abandonStreamedHead();
      if (owner_.reuseDecodeGraph) {
        graph.clear();
        owner_.decodeGraphInUse = false;
      }
    }
  };
  std::unordered_map<uint64_t, Request> requests;
  const bool wideLookupEnabled = widePromptLookupEnabled();
  const uint32_t stableVerifyRows = loggedStableVerifyRows(geometry, operators);
  const char *const wide32Mode = stableVerifyRows >= 4 * kDecodeRows ? wideLookup32Mode() : nullptr;
  uint64_t wide32Admitted = 0;  // =alt: requests admitted so far
  // SPLASH_WIDE_GDN_SINGLE (default parts): a wide lookup's GDN as one dispatch
  // per layer (=1) or as that pass in four value parts plus a finalize
  // (=parts, VH48 layers only); =alt / =alt-parts give it to every other request
  // (in-run A/B); any other value (e.g. 0) keeps the per-tile chain. parts vs the
  // chain: code-edit +2.39%, tool-copy +2.74%, exact.
  const std::string_view wideGdnMode = [] {
    const char *value = std::getenv("SPLASH_WIDE_GDN_SINGLE");
    return value ? std::string_view(value) : std::string_view("parts");
  }();
  // Exact values only (Codex 06:30): anything else stays Chain.
  const ops::WideGdn wideGdnRoute =
      wideGdnMode == "parts" || wideGdnMode == "alt-parts" ? ops::WideGdn::SingleParts
      : wideGdnMode == "1" || wideGdnMode == "alt"         ? ops::WideGdn::Single
                                                            : ops::WideGdn::Chain;
  const bool wideGdnAlternate = wideGdnMode == "alt" || wideGdnMode == "alt-parts";
  uint64_t wideGdnAdmitted = 0;
  // SPLASH_LOOKUP_MIN_MATCH (default 8) / SPLASH_WIDE_LOOKUP_MIN_MATCH
  // (default 16): shortest repeated suffix that may replace the drafter with a
  // lookup proposal. With SPLASH_LOOKUP_ADAPTIVE (default on) 8 is the floor of
  // the per-request cutoff (starts at 16); texts 18/18 identical.
  static size_t lookupMinMatchFromEnv(const char *name, size_t fallback) {
    const char *value = std::getenv(name);
    return value ? std::max<size_t>(2, std::strtoul(value, nullptr, 10)) : fallback;
  }
  const size_t lookupMinMatch = lookupMinMatchFromEnv("SPLASH_LOOKUP_MIN_MATCH", 8);
  const size_t wideLookupMinMatch = lookupMinMatchFromEnv("SPLASH_WIDE_LOOKUP_MIN_MATCH", 16);
  const bool adaptiveLookup = metal::envSwitch("SPLASH_LOOKUP_ADAPTIVE");
  // SPLASH_PROMPT_LOOKUP (default on): propose tokens copied from the prompt. A
  // proposal with q(x) = 1 under the target's acceptance rule keeps p exactly
  //; with 16 rows: rewrite +52%, edit +24% (stack12).
  const bool promptLookupEnabled =
      wideLookupEnabled || metal::envSwitch("SPLASH_PROMPT_LOOKUP");
  // SPLASH_DRAFT_AHEAD (default on): after a drafter cycle, the next cycle's draft block
  // is committed behind this cycle's command and runs while the host reads
  // the result. draft_ahead_prepare derives its anchor, positions and params
  // from the device acceptance. The next decodeAsync adopts it only when
  // today's path would run the same block; otherwise the host waits for it
  // before touching any lane buffer and runs today's path. Serving -0.257 ms/cycle
  // (-0.67%), oracle and 78 served outputs byte-identical.
  const bool draftAhead = draftAheadEnabled();
  // SPLASH_DRAFT_AHEAD_GRAMMAR (with SPLASH_DRAFT_AHEAD; default on):
  // grammar-constrained cycles launch and adopt blocks too. Their proposals
  // are already unconstrained (the host simulates the grammar over the drafted
  // tokens and masks only the target), so an adopted block feeds the mask
  // simulation exactly the proposals today's draft would. -0.34 ms per drafter
  // cycle on JSON, outputs identical (stack39).
  const bool grammarAhead = draftAhead && metal::envSwitch("SPLASH_DRAFT_AHEAD_GRAMMAR");
  // SPLASH_GRAMMAR_CHAIN (default on): a B1 grammar-constrained cycle
  // runs as one command; the host reads the proposals and writes the masks at
  // chain-event gates instead of between three commands. Tool calls +1.36% tok/s
  // [+0.17, +2.57], 12/12 identical (stack41).
  const bool grammarChain = metal::envSwitch("SPLASH_GRAMMAR_CHAIN");
  uint64_t chainValues = 0;  // chain event values taken so far
  // SPLASH_STREAMED_SUBMIT=N (default 48; 0 = off): a decode command's first
  // N dispatches commit as soon as they are built, while the host builds and
  // encodes the rest. A grammar-chain command streams its gated head instead:
  // a drafted cycle's draft at once, a lookup cycle's rope plus the forward's
  // first N dispatches. Exact: a decode graph build writes no buffer (page
  // tables sync in prepareDecodeLane, lane and policy buffers before the
  // build, masks behind the chain event). Oracle identical; -0.069 ms/cycle on
  // grammar+chat, -0.205 on grammar lookup cycles.
  const uint32_t streamChunk = [] {
    const char *value = std::getenv("SPLASH_STREAMED_SUBMIT");
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 48u;
  }();
  // SPLASH_ROW_HASH=1 (diagnostic, default off): finalizeDecode prints a hash of
  // the final hidden and logits of every accepted-path verify row (rows
  // 0..accepted, whose inputs are committed tokens), keyed by position, so runs
  // with different cycle schedules can be compared row by row.
  const bool rowHash = [] {
    const char *value = std::getenv("SPLASH_ROW_HASH");
    return value && std::string_view(value) == "1";
  }();
  struct DraftAhead final {
    uint32_t lanes = 0;
    std::array<uint64_t, kLaneCount> requestIds{};
    // Logical positions of the launching cycle; + retained once finalized.
    std::array<uint64_t, kLaneCount> positions{};
    std::array<uint64_t, kLaneCount> headVersions{};
    // NextAnchor of the launching cycle, as finalizeDecode reads it back.
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<std::array<float, kSamplingUniformCount>, kLaneCount> uniforms{};
    bool restrictedHead = false;
    bool finalized = false;
    ops::Q4DispatchStats stats;
  };
  std::optional<DraftAhead> ahead;
  CommandGraph aheadGraph;
  // SPLASH_DRAFT_AHEAD_LOOKUP_QUIET=K (default 8): launch a block only after K
  // cycles without a lookup hit. Lookup hits cluster (copying runs broken by
  // short drafter stretches), and a block dropped for a lookup costs ~3 ms
  // against ~0.3 ms saved by an adopted one, so blocks near lookups lose.
  const uint32_t aheadLookupQuiet = [] {
    const char *value = std::getenv("SPLASH_DRAFT_AHEAD_LOOKUP_QUIET");
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 8u;
  }();
  // Allocated for image cache misses and reclaimable once pending encodes
  // finish. Injecting already encoded rows needs no vision arena.
  std::unique_ptr<ops::Vision> vision;
  // Image buffers owned by the current admission attempt until a state cell
  // is activated. Failed attempts leave no image allocations behind.
  std::unordered_map<uint64_t, std::vector<ImageState>> stagedImages;
  // Encoded rows retained for reuse, including prefix hits that land inside
  // an image and still need its remaining rows. Byte-bounded LRU; the memory
  // reclaimer drops it entirely.
  struct CachedEmbeddings final {
    ImageSpan key;
    MetalBuffer embeddings;
  };
  static constexpr uint64_t kEmbeddingCacheBytes = 512ULL * 1024 * 1024;
  std::list<CachedEmbeddings> embeddingCache;
  uint64_t embeddingCacheBytes = 0;
  uint32_t maximumImagePatches = 0;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;
  std::array<PageTableBinding, kLaneCount> pageTableBindings{};
  ModelTelemetry counters;
  ops::Sampling sampling;
  QwenTarget targetModel;
  DFlashDraft draftModel;
  explicit Impl(RuntimeContext value)
      : backend(value.backend),
        admitAllocation(std::move(value.admitAllocation)),
        package(value.package),
        geometry(RuntimeGeometry::from(value.package, value.kvPages.layout().format)),
        operators(value.operators),
        kvPages(value.kvPages),
        states(requireQwenStateStorage(value.stateStorage)),
        maximumImagePatches(value.maximumImagePatches),
        pipelineReserveBytes(value.pipelineReserveBytes),
        runtimeOverheadReserveBytes(value.runtimeOverheadReserveBytes),
        sampling(value.backend, geometry.target.vocabularySize, kDecodeRows),
        targetModel(std::visit(
                        [&](const auto &weights) {
                          return QwenTarget(weights, value.backend, operators,
                                            value.kvPages.layout().format);
                        },
                        value.package.target)),
        draftModel(value.package.draft, value.backend, operators) {
    if (!admitAllocation)
      throw std::invalid_argument(
          "model runtime requires allocation admission");
    if (states.layout() != package.stateLayout() ||
        kvPages.layout() != package.targetKvLayout(kvPages.layout().format)) {
      throw std::invalid_argument(
          "model runtime resources do not match the loaded package");
    }
    prefillArena = std::make_unique<PrefillArena>(backend, geometry, operators);
    decodeArena = std::make_unique<DecodeArena>(backend, geometry, operators);
    // SPLASH_DRAFT_HEAD_IDS: the memory plan carries the restricted draft head with
    // the draft weights, so it is gathered now, before warmup and the memory audit.
    if (value.restrictedDraftHead)
      draftModel.loadRestrictedHead(targetModel.vocabularyProjection());
    else
      draftModel.disableRestrictedHead();
  }

  Request &request(uint64_t id) {
    auto found = requests.find(id);
    if (found == requests.end())
      throw std::out_of_range("unknown request");
    return found->second;
  }

  static bool samplingEnabled(const Request &entry) noexcept {
    return entry.sampling.temperature > 0.0F;
  }

  // Qwen3.5 M-RoPE: text rows advance one counter shared by all three axes;
  // an image's rows spread over (t, h, w) from the counter at the image start
  // and the counter then advances by max(merged height, merged width).
  static std::array<uint32_t, 3> ropePosition(const Request &entry,
                                              uint64_t logical) {
    int64_t delta = 0;
    for (const ImageState &image : entry.images) {
      const ImageSpan &span = image.span;
      if (logical < span.offset)
        break;
      const uint32_t mergedHeight = span.gridHeight / 2;
      const uint32_t mergedWidth = span.gridWidth / 2;
      const uint32_t start =
          static_cast<uint32_t>(static_cast<int64_t>(span.offset) + delta);
      if (logical < span.end()) {
        const uint32_t local = static_cast<uint32_t>(logical - span.offset);
        return {start, start + local / mergedWidth,
                start + local % mergedWidth};
      }
      delta += static_cast<int64_t>(std::max(mergedHeight, mergedWidth)) -
               static_cast<int64_t>(span.tokens);
    }
    const uint32_t position =
        static_cast<uint32_t>(static_cast<int64_t>(logical) + delta);
    return {position, position, position};
  }

  uint64_t embeddingBytes(const ImageSpan &span) const {
    return uint64_t{
               ops::Vision::embeddingRows({span.gridHeight, span.gridWidth})} *
           geometry.target.hiddenSize * sizeof(uint16_t);
  }

  static bool sameImage(const ImageSpan &left,
                        const ImageSpan &right) noexcept {
    return left.digestLo == right.digestLo && left.digestHi == right.digestHi &&
           left.gridHeight == right.gridHeight &&
           left.gridWidth == right.gridWidth;
  }

  // Encoded rows for an identical image, moved to the front of the LRU.
  MetalBuffer cachedEmbeddings(const ImageSpan &span) {
    for (auto entry = embeddingCache.begin(); entry != embeddingCache.end();
         ++entry) {
      if (!sameImage(entry->key, span))
        continue;
      embeddingCache.splice(embeddingCache.begin(), embeddingCache, entry);
      return entry->embeddings;
    }
    return {};
  }

  void retainEmbeddings(const ImageState &image) {
    if (!image.data || !image.data->encoded || !image.data->embeddings ||
        embeddingBytes(image.span) > kEmbeddingCacheBytes ||
        cachedEmbeddings(image.span)) {
      return;
    }
    embeddingCache.push_front({image.span, image.data->embeddings});
    embeddingCacheBytes += embeddingBytes(image.span);
    while (embeddingCacheBytes > kEmbeddingCacheBytes) {
      embeddingCacheBytes -= embeddingBytes(embeddingCache.back().key);
      embeddingCache.pop_back();
    }
  }

  // Rows served from the cache stay held by the request using them, so
  // dropping their entry frees nothing until that request ends.
  [[nodiscard]] bool
  embeddingsHeld(const MetalBuffer &embeddings) const noexcept {
    auto holds = [&](const std::vector<ImageState> &images) {
      for (const ImageState &image : images) {
        if (image.data && image.data->embeddings.sameView(embeddings))
          return true;
      }
      return false;
    };
    for (const auto &[_, images] : stagedImages) {
      if (holds(images))
        return true;
    }
    for (const auto &[_, entry] : requests) {
      if (holds(entry.images))
        return true;
    }
    return false;
  }

  uint64_t dropEmbeddingCache() noexcept {
    uint64_t released = 0;
    for (const CachedEmbeddings &entry : embeddingCache) {
      if (!embeddingsHeld(entry.embeddings))
        released += embeddingBytes(entry.key);
    }
    embeddingCache.clear();
    embeddingCacheBytes = 0;
    return released;
  }

  struct ImageAdmission final {
    Impl &runtime;
    uint64_t requestId;
    bool hadVision;
    bool committed = false;

    ImageAdmission(Impl &owner, uint64_t id)
        : runtime(owner), requestId(id), hadVision(bool(owner.vision)) {}
    ~ImageAdmission() {
      if (!committed) {
        runtime.stagedImages.erase(requestId);
        if (!hadVision)
          runtime.vision.reset();
      }
    }
  };

  // Admits the memory an image request needs before its state cell: the
  // shared vision scratch and per-image pixel and embedding buffers, all
  // through the governor, preserving the allocation refusal reason.
  metal::AllocationResult stageImages(const ModelRequest &request) {
    if (request.images.empty() || stagedImages.contains(request.id))
      return true;
    std::vector<ImageState> staged;
    staged.reserve(request.images.size());
    uint64_t bytes = 0;
    for (const ImageSpan &span : request.images) {
      ImageState image{span, {}};
      const auto duplicate = std::find_if(
          staged.begin(), staged.end(), [&](const ImageState &previous) {
            return sameImage(previous.span, span);
          });
      if (duplicate != staged.end()) {
        image.data = duplicate->data;
      } else {
        image.data = std::make_shared<ImageData>();
        image.data->embeddings = cachedEmbeddings(span);
        image.data->encoded = static_cast<bool>(image.data->embeddings);
        if (image.data->encoded)
          ++counters.imageEmbeddingReuses;
        else
          bytes += span.pixelBytes() + embeddingBytes(span);
      }
      staged.push_back(std::move(image));
    }
    // Keep cache references alive during admission. Only misses need the
    // encoder; cached rows can be injected after its arena has been reclaimed.
    if (bytes && !vision) {
      std::unique_ptr<ops::Vision> candidate;
      const auto admission = admitAllocation(
              ops::Vision::scratchBytes(package.vision.tensors.layout,
                                        maximumImagePatches),
              [&] {
                candidate = std::make_unique<ops::Vision>(
                    backend, package.vision.tensors, maximumImagePatches);
              });
      if (!admission)
        return admission;
      vision = std::move(candidate);
    }
    const uint8_t *pixels = request.imagePixels.data();
    const auto allocateImages = [&] {
      for (ImageState &image : staged) {
        const ImageSpan &span = image.span;
        if (!image.data->embeddings) {
          image.data->pixels = backend.allocateBuffer(
              span.pixelBytes(), BufferStorage::Shared, "image pixels");
          std::memcpy(contents<uint8_t>(image.data->pixels, "image pixels"), pixels,
                      static_cast<size_t>(span.pixelBytes()));
          image.data->embeddings = backend.allocateBuffer(
              embeddingBytes(span), BufferStorage::Private, "image embeddings");
        }
        pixels += span.pixelBytes();
      }
    };
    if (bytes) {
      if (auto admission = admitAllocation(bytes, allocateImages); !admission)
        return admission;
    }
    stagedImages.emplace(request.id, std::move(staged));
    return true;
  }

  [[nodiscard]] bool visionIdle() const noexcept {
    if (!stagedImages.empty())
      return false;
    for (const auto &[_, entry] : requests) {
      for (const ImageState &image : entry.images) {
        if (image.data && !image.data->encoded && image.data->embeddings)
          return false;
      }
    }
    return true;
  }

  // Encodes every image whose rows first appear in this chunk and overwrites
  // the chunk's placeholder embedding rows with the image rows. Text-only
  // requests add no dispatches.
  void addImageRows(CommandGraph &graph, Request &entry,
                    const ModelBatchItem &item, uint32_t rowBegin) {
    const uint64_t chunkBegin = item.promptOffset;
    const uint64_t chunkEnd = chunkBegin + item.tokenCount;
    for (ImageState &image : entry.images) {
      const uint64_t begin = std::max<uint64_t>(chunkBegin, image.span.offset);
      const uint64_t end = std::min<uint64_t>(chunkEnd, image.span.end());
      if (begin >= end || !image.data || !image.data->embeddings)
        continue;
      ImageData &data = *image.data;
      if (!data.encoded && !data.encoding) {
        if (!vision)
          throw std::logic_error("image request has no vision encoder");
        vision->encode(graph, {image.span.gridHeight, image.span.gridWidth},
                       data.pixels, data.embeddings);
        data.encoding = true;
        ++counters.imageEncodes;
      }
      const uint32_t rows = static_cast<uint32_t>(end - begin);
      ops::Vision::inject(
          graph, data.embeddings, prefillArena->get(PrefillTensor::Hidden0),
          package.vision.tensors.layout.outputHiddenSize,
          static_cast<uint32_t>(begin - image.span.offset),
          rowBegin + static_cast<uint32_t>(begin - chunkBegin), rows);
    }
  }

  static float uniformAt(uint64_t seed, uint64_t counter) noexcept {
    uint64_t value = seed + counter * 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return float(value >> 40) * 0x1p-24F;
  }

  static float nextUniform(Request &entry) noexcept {
    return uniformAt(entry.sampling.seed, ++entry.rngCounter);
  }

  static void stageSamplingCycle(Request &entry) noexcept {
    entry.cycleUniforms.fill(0.0F);
    for (uint32_t index = 1; index < entry.cycleUniforms.size(); ++index) {
      entry.cycleUniforms[index] = nextUniform(entry);
    }
  }

  // The uniforms the next regular cycle's loadPolicyBuffers will write,
  // computed without consuming them.
  static std::array<float, kSamplingUniformCount>
  nextCycleUniforms(const Request &entry) noexcept {
    if (!samplingEnabled(entry))
      return entry.cycleUniforms;
    std::array<float, kSamplingUniformCount> result{};
    for (uint32_t index = 1; index < result.size(); ++index)
      result[index] = uniformAt(entry.sampling.seed, entry.rngCounter + index);
    return result;
  }

  [[nodiscard]] uint64_t estimatedWarmupPeak() const {
    uint64_t result = 0;
    auto add = [&](uint64_t bytes, std::string_view label) {
      result = checkedAdd(result, bytes, label);
    };
    add(package.targetActualAllocatedBytes(), "warmup target weights");
    add(package.draft.actualAllocatedBytes, "warmup draft weights");
    add(draftModel.restrictedHeadAllocatedBytes(), "warmup draft head");
    add(package.vision.actualAllocatedBytes, "warmup vision weights");
    add(states.actualAllocatedBytes(), "warmup state slots");
    add(prefillArena->bytes(), "warmup prefill arena");
    add(decodeArena->bytes(), "warmup decode arena");
    add(kvPages.actualAllocatedBytes(), "warmup KV pool");
    add(pipelineReserveBytes, "warmup pipeline reserve");
    add(runtimeOverheadReserveBytes, "warmup runtime reserve");
    return result;
  }

  void copyPageTable(const MetalBuffer &destination,
                     std::span<const uint32_t> pages) const {
    if (pages.empty() || pages.size() > kMaximumPageTableEntries) {
      throw std::invalid_argument("request page table has invalid length");
    }
    auto *target = contents<uint32_t>(destination, "request page table");
    std::copy(pages.begin(), pages.end(), target);
  }

  [[nodiscard]] MetalBuffer synchronizedPageTable(Request &entry,
                                                  const ModelBatchItem &item) {
    if (entry.slot >= pageTableBindings.size())
      throw std::out_of_range("request state slot is outside page tables");
    PageTableBinding &binding = pageTableBindings[entry.slot];
    MetalBuffer destination =
        decodeArena->get(entry.slot, DecodeTensor::PageTable);
    const bool unversioned = item.pageTableRevision == 0;
    if (unversioned || binding.requestId != entry.id ||
        binding.revision != item.pageTableRevision ||
        binding.entries != item.pageTable.size()) {
      copyPageTable(destination, item.pageTable);
      binding = {entry.id, item.pageTableRevision,
                 static_cast<uint32_t>(item.pageTable.size())};
    }
    return destination;
  }

  void addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin) const {
    ops::RoPE::addTables(
        graph, std::move(targetPositions), std::move(draftPositions),
        prefillArena->get(PrefillTensor::TargetInverseFrequencies),
        prefillArena->get(PrefillTensor::DraftInverseFrequencies),
        std::move(targetCos), std::move(targetSin), std::move(draftCos),
        std::move(draftSin), {targetRows, draftRows}, kPrefillRows);
  }

  void captureFinalHidden(Request &entry, const MetalBuffer &rows,
                          uint32_t row) const {
    if (row >= kDecodeRows) {
      throw std::out_of_range("final hidden row is out of range");
    }
    const uint16_t *source =
        contents<uint16_t>(rows, "target final hidden source");
    entry.finalTargetHidden.assign(
        source + uint64_t{row} * geometry.target.hiddenSize,
        source + uint64_t{row + 1} * geometry.target.hiddenSize);
  }

  static DispatchDraftCapturePlan
  activeDraftCaptures(const Request &entry, const ModelBatchItem &item) {
    if (!entry.draftContextPlan) {
      throw std::logic_error("prefill request has no draft context plan");
    }
    const uint64_t next = item.logicalPosition + item.tokenCount;
    return draftCaptureSpansForDispatch(
        *entry.draftContextPlan, static_cast<uint32_t>(item.logicalPosition),
        static_cast<uint32_t>(next));
  }

  static uint32_t captureRows(const DispatchDraftCapturePlan &captures) {
    uint64_t rows = 0;
    for (const auto &capture : captures) {
      rows += capture.absoluteEnd - capture.absoluteBegin;
    }
    if (rows > kPrefillRows) {
      throw std::logic_error("draft capture exceeds packed prefill capacity");
    }
    return static_cast<uint32_t>(rows);
  }

  static QwenLogicalLengths
  advanceDraftContext(const QwenLogicalLengths &previous, uint64_t targetTokens,
                      const DispatchDraftCaptureSpan &capture) {
    QwenLogicalLengths next = previous;
    next.targetTokens = targetTokens;
    const uint32_t rows = capture.absoluteEnd - capture.absoluteBegin;
    if (!rows)
      return next;
    const bool continues =
        !capture.resetDraftState &&
        previous.draftEnd() == capture.absoluteBegin &&
        previous.draftCommitCursor == capture.absoluteBegin % kDraftCacheStride;
    const uint64_t combined =
        continues ? uint64_t{previous.draftLength} + rows : rows;
    next.draftLength =
        static_cast<uint32_t>(std::min<uint64_t>(combined, kDraftCacheStride));
    next.draftBase = capture.absoluteEnd - next.draftLength;
    next.draftCommitCursor =
        static_cast<uint32_t>(capture.absoluteEnd % kDraftCacheStride);
    return next;
  }

  void loadPolicyBuffers(Request &entry, uint32_t lane,
                         std::span<const uint32_t> masks) const {
    auto uniforms = decodeArena->get(lane, DecodeTensor::SamplingUniforms);
    auto *uniformData = contents<float>(uniforms, "sampling uniforms");
    std::copy(entry.cycleUniforms.begin(), entry.cycleUniforms.end(),
              uniformData);

    // Unconstrained sampling and argmax never consume the mask buffer.
    if (entry.constraint != ConstraintMode::TokenMask && masks.empty())
      return;

    auto constraint = decodeArena->get(lane, DecodeTensor::ConstraintMasks);
    auto *maskData = contents<uint32_t>(constraint, "constraint masks");
    const uint64_t capacity =
        uint64_t{ExecutionLimits::maximumStepTokens} * geometry.maskWords();
    std::fill(maskData, maskData + capacity,
              std::numeric_limits<uint32_t>::max());
    if (!masks.empty()) {
      if (masks.size() > capacity) {
        throw std::invalid_argument("constraint mask exceeds decode arena");
      }
      std::copy(masks.begin(), masks.end(), maskData);
    }
  }

  void loadLookup16PolicyBuffers(Request &entry,
                                  std::span<const uint32_t> masks) const {
    const size_t words = geometry.maskWords();
    const uint32_t tiles = entry.verifyRows / kDecodeRows;
    if (!masks.empty() && masks.size() != (entry.verifyRows + 1) * words)
      throw std::invalid_argument("wide lookup needs one mask row per row and one more");
    // Each tile's initial mask is the previous tile's final mask.
    for (uint32_t tile = 0; tile < tiles; ++tile)
      loadPolicyBuffers(entry, tile,
                        masks.empty() ? masks
                                      : masks.subspan(tile * kDecodeRows * words,
                                                      (kDecodeRows + 1) * words));
    // A 32-row acceptance reads 32 uniforms: lane 1 holds the second 16.
    if (tiles == 4)
      std::copy(entry.wideUniforms.begin(), entry.wideUniforms.end(),
                contents<float>(decodeArena->get(1, DecodeTensor::SamplingUniforms),
                                "wide uniforms"));
  }

  static ops::SamplingPolicy samplingPolicy(const Request &entry) noexcept {
    const bool enabled = samplingEnabled(entry);
    return {enabled ? entry.sampling.topK : 1,
            enabled ? entry.sampling.temperature : 0.0F,
            enabled ? entry.sampling.topP : 1.0F,
            entry.constraint == ConstraintMode::TokenMask};
  }

  template <class Get>
  static ops::SamplingBuffers samplingBuffersWith(Get d) {
    return {d(DecodeTensor::Logits),
            d(DecodeTensor::TargetTopPartialIds),
            d(DecodeTensor::TargetTopPartialValues),
            d(DecodeTensor::TargetTopIds),
            d(DecodeTensor::TargetTopProbs),
            d(DecodeTensor::SamplingUniforms),
            d(DecodeTensor::ConstraintMasks),
            d(DecodeTensor::OutputTokens),
            d(DecodeTensor::ArgmaxValues),
            d(DecodeTensor::ArgmaxIndices)};
  }

  ops::SamplingBuffers samplingBuffers(uint32_t lanes) const {
    return samplingBuffersWith(
        [&](DecodeTensor t) { return decodeArena->packed(t, lanes); });
  }

  ops::SamplingBuffers samplingBuffersForLane(uint32_t lane) const {
    return samplingBuffersWith(
        [&](DecodeTensor t) { return decodeArena->get(lane, t); });
  }

  void addInitialPolicySelection(CommandGraph &graph, Request &entry,
                                 uint32_t lane, uint32_t rowOffset) const {
    sampling.addInitial(graph, samplingPolicy(entry),
                        samplingBuffersForLane(lane), rowOffset);
  }

  CommandTiming selectPendingFromFinalHidden(Request &entry, uint32_t lane,
                                             std::span<const uint32_t> masks) {
    if (entry.finalTargetHidden.size() != geometry.target.hiddenSize) {
      throw std::logic_error("request has no policy-neutral final hidden");
    }
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->get(lane, tensor);
    };
    auto *hidden =
        contents<uint16_t>(d(DecodeTensor::Hidden0), "pending final hidden");
    for (uint32_t row = 0; row < kDecodeRows; ++row) {
      std::copy(entry.finalTargetHidden.begin(), entry.finalTargetHidden.end(),
                hidden + uint64_t{row} * geometry.target.hiddenSize);
    }
    entry.cycleUniforms.fill(0.0F);
    if (samplingEnabled(entry)) {
      entry.cycleUniforms[0] = nextUniform(entry);
    }
    loadPolicyBuffers(entry, lane, masks);

    CommandGraph graph;
    targetModel.addHead(graph, d(DecodeTensor::Hidden0),
                        d(DecodeTensor::FinalHidden), d(DecodeTensor::Logits),
                        kDecodeRows, decodeArena->linearScratch());
    addInitialPolicySelection(graph, entry, lane, 0);
    CommandTiming timing = backend.submitCommand(graph.dispatches());
    entry.pendingToken = *contents<uint32_t>(d(DecodeTensor::OutputTokens),
                                             "restored prefix next token");
    if (*entry.pendingToken >= geometry.target.vocabularySize) {
      throw std::runtime_error("target policy selected an invalid token");
    }
    return timing;
  }

  Q8ChunkedPrefillParams q8Params(uint64_t logicalPosition,
                                  uint32_t chunkTokens, uint32_t chunkStride,
                                  std::span<const uint32_t> pages) const {
    return ops::PagedAttention::prefillParams(
        logicalPosition, chunkTokens, chunkStride, pages, kvPages.pageCount());
  }

  struct PackedPrefillSequence final {
    Request *entry = nullptr;
    const ModelBatchItem *item = nullptr;
    uint32_t lane = 0;
    uint32_t rowBegin = 0;
    uint32_t attentionStride = 0;
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    uint32_t captureBegin = 0;
    Q8ChunkedPrefillParams q8;
    MetalBuffer pageTable;
    DispatchDraftCapturePlan captures;
  };

  struct PackedPrefillBatch final {
    std::vector<PackedPrefillSequence> sequences;
    uint32_t rows = 0;
    uint32_t capturedRows = 0;
  };

  MetalBuffer prefillU16(PrefillTensor tensor, uint32_t begin, uint32_t rows,
                         uint32_t width) const {
    return backend.view(prefillArena->get(tensor),
                        bytesFor<uint16_t>(uint64_t{begin} * width),
                        bytesFor<uint16_t>(uint64_t{rows} * width));
  }

  PackedPrefillBatch
  preparePackedPrefill(std::span<const ModelBatchItem> items,
                       std::array<Request *, kLaneCount> &entries) {
    PackedPrefillBatch batch;
    batch.sequences.reserve(items.size());
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      const ModelBatchItem &item = items[lane];
      Request &entry = request(item.requestId);
      if (item.tokenCount > kPrefillRows ||
          item.promptOffset > entry.promptTokens ||
          item.tokenCount > entry.promptTokens - item.promptOffset ||
          item.logicalPosition != item.promptOffset || !entry.resident ||
          entry.slot != item.stateSlot) {
        throw std::invalid_argument("invalid packed Qwen prefill item");
      }
      const QwenSlotMetadata &metadata = states.metadata(entry.slot);
      if (!metadata.assigned || metadata.requestId != entry.id ||
          metadata.lengths.targetTokens != item.logicalPosition) {
        throw std::logic_error("packed prefill state length is not exact");
      }
      if (item.tokenCount > kPrefillRows - batch.rows) {
        throw std::invalid_argument("packed prefill exceeds actual-row budget");
      }
      auto captures = activeDraftCaptures(entry, item);
      const uint32_t capturedRows = captureRows(captures);
      if (capturedRows > kPrefillRows - batch.capturedRows) {
        throw std::invalid_argument("packed draft capture exceeds row budget");
      }
      const uint32_t attentionStride =
          ((item.tokenCount + kTileRows - 1) / kTileRows) * kTileRows;
      const Q8ChunkedPrefillParams q8 =
          q8Params(item.logicalPosition, item.tokenCount, attentionStride,
                   item.pageTable);
      MetalBuffer pageTable = synchronizedPageTable(entry, item);
      batch.sequences.push_back({&entry, &item, lane, batch.rows,
                                 attentionStride, queryOffset, kvOffset,
                                 batch.capturedRows, q8, std::move(pageTable),
                                 std::move(captures)});
      entries[lane] = &entry;
      batch.rows += item.tokenCount;
      batch.capturedRows += capturedRows;
      queryOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionQueryHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
      kvOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionKvHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
    }
    if (!batch.rows ||
        queryOffset >
            prefillArena->get(PrefillTensor::FullQueries).sizeBytes() ||
        kvOffset > prefillArena->get(PrefillTensor::ChunkKeys).sizeBytes()) {
      throw std::logic_error("packed prefill scratch geometry overflowed");
    }

    auto *input =
        contents<uint32_t>(prefillArena->get(PrefillTensor::InputTokens),
                           "packed prefill input tokens");
    auto *targetPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::TargetPositions),
                           "target RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::DraftPositions),
                           "draft RoPE positions");
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      const ModelBatchItem &item = *sequence.item;
      std::copy(item.inputTokens.begin(), item.inputTokens.end(),
                input + sequence.rowBegin);
      for (uint32_t localRow = 0; localRow < item.tokenCount; ++localRow) {
        const uint32_t row = sequence.rowBegin + localRow;
        if (input[row] >= geometry.target.vocabularySize) {
          throw std::invalid_argument("prompt token is out of vocabulary");
        }
        const std::array<uint32_t, 3> rotary =
            ropePosition(*sequence.entry, item.logicalPosition + localRow);
        std::copy(rotary.begin(), rotary.end(), targetPositions + row * 3);
      }
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        for (uint32_t row = capture.absoluteBegin; row < capture.absoluteEnd;
             ++row) {
          const uint32_t compactRow = sequence.captureBegin +
                                      capture.compactDestinationRow + row -
                                      capture.absoluteBegin;
          draftPositions[compactRow] = row;
        }
      }
    }
    return batch;
  }

  void addPackedDraftContext(CommandGraph &graph,
                             const PackedPrefillBatch &batch) {
    if (!batch.capturedRows)
      return;
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };
    std::array<DFlashPrefillSpan, kLaneCount * 2> spans{};
    uint32_t spanCount = 0;
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      const QwenSlotBuffers &slot = states.buffers(sequence.entry->slot);
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        DFlashPrefillSpan &span = spans.at(spanCount++);
        span.compactRow = sequence.captureBegin + capture.compactDestinationRow;
        span.rows = capture.absoluteEnd - capture.absoluteBegin;
        span.startPosition = capture.absoluteBegin;
        span.ring = slot.draft;
      }
    }
    draftModel.addContextPrefill(
        graph,
        {p(PrefillTensor::Captured), p(PrefillTensor::ProjectionSums),
         p(PrefillTensor::ContextProjected), p(PrefillTensor::ContextHidden),
         p(PrefillTensor::ContextQkv), p(PrefillTensor::DraftRopeCos),
         p(PrefillTensor::DraftRopeSin)},
        batch.capturedRows, std::span(spans).first(spanCount));
  }

  void encodePackedPrefillGraph(CommandGraph &graph,
                                std::span<const ModelBatchItem> items,
                                std::array<Request *, kLaneCount> &entries) {
    PackedPrefillBatch batch = preparePackedPrefill(items, entries);
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };

    addRopeTables(graph, p(PrefillTensor::TargetPositions), batch.rows,
                  p(PrefillTensor::DraftPositions), batch.capturedRows,
                  p(PrefillTensor::RopeCos), p(PrefillTensor::RopeSin),
                  p(PrefillTensor::DraftRopeCos),
                  p(PrefillTensor::DraftRopeSin));

    targetModel.addEmbedding(graph, p(PrefillTensor::InputTokens),
                             p(PrefillTensor::Hidden0), batch.rows);
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      addImageRows(graph, *sequence.entry, *sequence.item, sequence.rowBegin);
    }

    std::array<QwenTargetPrefillSequence, kLaneCount> modelSequences{};
    const uint32_t modelSequenceCount =
        static_cast<uint32_t>(batch.sequences.size());
    const uint64_t stateBindingCount = uint64_t{modelSequenceCount} *
                                       geometry.target.stateLayout.layers;
    std::vector<MetalBuffer> convolutionIn(stateBindingCount);
    std::vector<MetalBuffer> convolutionOut(stateBindingCount);
    std::vector<MetalBuffer> recurrentIn(stateBindingCount);
    std::vector<MetalBuffer> recurrentOut(stateBindingCount);
    for (uint32_t lane = 0; lane < batch.sequences.size(); ++lane) {
      const PackedPrefillSequence &sequence = batch.sequences[lane];
      QwenTargetPrefillSequence &destination = modelSequences[lane];
      destination.rowBegin = sequence.rowBegin;
      destination.rows = sequence.item->tokenCount;
      destination.attentionStride = sequence.attentionStride;
      destination.queryOffset = sequence.queryOffset;
      destination.kvOffset = sequence.kvOffset;
      destination.q8 = sequence.q8;
      destination.pageTable = sequence.pageTable;
      const uint32_t gdnLayers = geometry.target.stateLayout.layers;
      const uint64_t stateBegin = uint64_t{lane} * gdnLayers;
      destination.convolutionIn =
          std::span(convolutionIn).subspan(stateBegin, gdnLayers);
      destination.convolutionOut =
          std::span(convolutionOut).subspan(stateBegin, gdnLayers);
      destination.recurrentIn =
          std::span(recurrentIn).subspan(stateBegin, gdnLayers);
      destination.recurrentOut =
          std::span(recurrentOut).subspan(stateBegin, gdnLayers);
      const QwenSlotMetadata &metadata = states.metadata(sequence.entry->slot);
      const QwenSlotBuffers &slot = states.buffers(sequence.entry->slot);
      for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
        convolutionIn[stateBegin + layer] =
            slot.gdn[metadata.activeParity].convolutionLayers[layer];
        convolutionOut[stateBegin + layer] =
            slot.gdn[metadata.activeParity ^ 1].convolutionLayers[layer];
        recurrentIn[stateBegin + layer] =
            slot.gdn[metadata.activeParity].recurrentLayers[layer];
        recurrentOut[stateBegin + layer] =
            slot.gdn[metadata.activeParity ^ 1].recurrentLayers[layer];
      }
      destination.captureCount = sequence.captures.size();
      for (uint32_t index = 0; index < sequence.captures.size(); ++index) {
        const DispatchDraftCaptureSpan &capture = sequence.captures[index];
        destination.captures[index] = {
            sequence.rowBegin +
                static_cast<uint32_t>(capture.absoluteBegin -
                                      sequence.item->logicalPosition),
            sequence.captureBegin + capture.compactDestinationRow,
            capture.absoluteEnd - capture.absoluteBegin};
      }
    }
    QwenTargetPrefillBuffers buffers;
    buffers.hidden = {p(PrefillTensor::Hidden0), p(PrefillTensor::Hidden1)};
    buffers.normalized = p(PrefillTensor::Normalized);
    buffers.captured = p(PrefillTensor::Captured);
    buffers.gdnPacked = p(PrefillTensor::GdnPacked);
    buffers.gdnQueries = p(PrefillTensor::GdnQueries);
    buffers.gdnKeys = p(PrefillTensor::GdnKeys);
    buffers.gdnValues = p(PrefillTensor::GdnValues);
    buffers.gdnDecay = p(PrefillTensor::GdnDecay);
    buffers.gdnBeta = p(PrefillTensor::GdnBeta);
    buffers.recurrent = p(PrefillTensor::Recurrent);
    buffers.gdnHidden = p(PrefillTensor::GdnHidden);
    buffers.gdnOutput = p(PrefillTensor::GdnOutput);
    buffers.denseGateScratch = p(PrefillTensor::GateIntermediate);
    buffers.denseIntermediate = p(PrefillTensor::Intermediate);
    buffers.fullPacked = p(PrefillTensor::FullPacked);
    buffers.fullQueries = p(PrefillTensor::FullQueries);
    buffers.fullAttention = p(PrefillTensor::FullAttention);
    buffers.attentionPartials = p(PrefillTensor::AttentionPartials);
    buffers.attentionStatistics = p(PrefillTensor::AttentionStatistics);
    buffers.attentionHidden = p(PrefillTensor::AttentionHidden);
    buffers.attentionOutput = p(PrefillTensor::AttentionOutput);
    buffers.projectionSums = p(PrefillTensor::ProjectionSums);
    buffers.downProjectionSums = p(PrefillTensor::DownProjectionSums);
    buffers.ropeCos = p(PrefillTensor::RopeCos);
    buffers.ropeSin = p(PrefillTensor::RopeSin);
    buffers.chunkKeys = p(PrefillTensor::ChunkKeys);
    buffers.chunkValues = p(PrefillTensor::ChunkValues);
    buffers.selectedExperts = p(PrefillTensor::MoeSelectedExperts);
    buffers.routingWeights = p(PrefillTensor::MoeRoutingWeights);
    buffers.tileDescriptors = p(PrefillTensor::MoeTileDescriptors);
    buffers.tileCount = p(PrefillTensor::MoeTileCount);
    buffers.groupedRoutes = p(PrefillTensor::MoeGroupedRoutes);
    buffers.routeRows = p(PrefillTensor::MoeRouteRows);
    buffers.groupedInput = p(PrefillTensor::MoeGroupedInput);
    buffers.expertIntermediate = p(PrefillTensor::MoeExpertIntermediate);
    buffers.expertOutput = p(PrefillTensor::MoeExpertOutput);
    std::vector<kv::LayerStorage> kvLayers(
        geometry.target.kvLayout.attentionLayers);
    for (uint32_t layer = 0; layer < kvLayers.size(); ++layer)
      kvLayers[layer] = kvPages.layer(layer);
    targetModel.addPrefill(
        graph, std::move(buffers),
        std::span(modelSequences).first(batch.sequences.size()), batch.rows,
        kvLayers);
    addPackedDraftContext(graph, batch);

    for (const PackedPrefillSequence &sequence : batch.sequences) {
      Request &entry = *sequence.entry;
      const ModelBatchItem &item = *sequence.item;
      if (entry.replayingGeneration ||
          item.logicalPosition + item.tokenCount != entry.promptTokens)
        continue;
      const bool scoring = !entry.scoreTokens.empty();
      auto d = [&](DecodeTensor tensor) {
        return decodeArena->get(sequence.lane, tensor);
      };
      const uint32_t lastRows = std::min(item.tokenCount, kDecodeRows);
      ops::DraftAttention::gatherLastRows(
          graph,
          prefillU16(PrefillTensor::Hidden0, sequence.rowBegin,
                     item.tokenCount, geometry.target.hiddenSize),
          d(DecodeTensor::Hidden0), item.tokenCount,
          geometry.target.hiddenSize);
      if (scoring) {
        // Score-only: compute raw logits at the final prompt position; no
        // policy selection, sampling, or anchor is produced.
        targetModel.addHead(graph, d(DecodeTensor::Hidden0),
                            d(DecodeTensor::FinalHidden),
                            d(DecodeTensor::Logits), lastRows, decodeArena->linearScratch());
      } else if (entry.constraint == ConstraintMode::None) {
        if (samplingEnabled(entry)) {
          entry.cycleUniforms.fill(0.0F);
          entry.cycleUniforms[0] = nextUniform(entry);
          loadPolicyBuffers(entry, sequence.lane, {});
        }
        addPrefillPolicy(graph, entry, sequence.lane, lastRows - 1);
      }
    }
  }

  // draftInputs false: an adopted SPLASH_DRAFT_AHEAD block already wrote the
  // draft-side inputs on the GPU (the same values).
  void prepareDecodeLane(Request &entry, const ModelBatchItem &item,
                         uint32_t lane, bool draftInputs = true) {
    if (!entry.resident || entry.slot != item.stateSlot ||
        !entry.promptComplete || !entry.pendingToken) {
      throw std::logic_error("decode request is not ready");
    }
    entry.verifyRows = kDecodeRows;
    const QwenSlotMetadata &metadata = states.metadata(entry.slot);
    if (metadata.lengths.targetTokens != item.logicalPosition ||
        !metadata.lengths.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::logic_error("decode state length is not exact");
    }
    static_cast<void>(synchronizedPageTable(entry, item));
    if (draftInputs) {
      auto *draftInput = contents<uint32_t>(
          decodeArena->get(lane, DecodeTensor::DraftInputTokens),
          "draft input tokens");
      draftInput[0] = *entry.pendingToken;
      std::fill(draftInput + 1, draftInput + kDecodeRows,
                geometry.target.maskToken);
    }

    auto *positions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Positions),
                           "decode RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::DraftPositions),
                           "decode draft RoPE positions");
    for (uint32_t row = 0; row < kDecodeRows; ++row) {
      const std::array<uint32_t, 3> rotary =
          ropePosition(entry, item.logicalPosition + row);
      std::copy(rotary.begin(), rotary.end(), positions + row * 3);
      // The draft is a text model over logical positions.
      if (draftInputs)
        draftPositions[row] = static_cast<uint32_t>(item.logicalPosition + row);
    }
    *contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Arrived),
                        "decode arrived") = 0;
    *contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Generation),
                        "decode generation") = 0;
  }

  struct WideProposal final {
    std::array<uint32_t, 31> tokens{};  // the first tiles * 8 - 1 are used
    uint32_t tiles = 2;
  };

  std::optional<WideProposal>
  lookup16Proposal(Request &entry, const ModelBatchItem &item) {
    // ponytail: use only already-admitted pages; fall back to M8 (or 16
    // rows) when an extra page would be needed instead of changing engine
    // admission.
    const auto fits = [&](uint32_t rows) {
      return entry.maxNewTokens - entry.generatedTokens >= rows &&
             item.logicalPosition + rows <= kv::kMaximumPhysicalTokens &&
             (item.logicalPosition + rows + kTileRows - 1) / kTileRows <=
                 item.pageTable.size();
    };
    if (!wideLookupEnabled || stableVerifyRows < 2 * kDecodeRows || entry.lookupHistory.empty() ||
        geometry.target.ffnKind != QwenFfnKind::Dense || !fits(2 * kDecodeRows))
      return std::nullopt;
    const size_t minMatch = adaptiveLookup ? entry.lookupMatch : wideLookupMinMatch;
    entry.lookupHistory.push_back(*entry.pendingToken);
    std::optional<WideProposal> result;
    // SPLASH_WIDE_LOOKUP32: 32 rows right after a fully accepted wide lookup, and
    // only then: a 32-row verify costs ~16.5 ms more than 16 rows, so it pays only
    // inside long copy runs. code-edit +12.7%, tool-copy +27.4%, chat -0.7% (n.s.)
    // ('Exact 32-row single-request lookup').
    if (entry.wide32 && entry.wideFull && fits(4 * kDecodeRows)) {
      if (const auto proposal = model::promptLookup<31>(entry.lookupHistory, minMatch))
        result = WideProposal{*proposal, 4};
    }
    if (!result) {
      if (const auto proposal = model::promptLookup<15>(entry.lookupHistory, minMatch)) {
        result.emplace();
        std::copy(proposal->begin(), proposal->end(), result->tokens.begin());
      }
    }
    entry.lookupHistory.pop_back();
    return result;
  }

  void prepareLookup16(Request &entry, const ModelBatchItem &item,
                       const WideProposal &proposal) {
    const uint32_t tiles = proposal.tiles;
    const uint32_t rows = tiles * kDecodeRows;
    auto *tokens = contents<uint32_t>(
        decodeArena->packed(DecodeTensor::InputTokens, tiles), "wide lookup input");
    tokens[0] = *entry.pendingToken;
    std::copy_n(proposal.tokens.begin(), rows - 1, tokens + 1);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
      auto *positions = contents<uint32_t>(
          decodeArena->get(tile, DecodeTensor::Positions), "wide positions");
      auto *draftPositions = contents<uint32_t>(
          decodeArena->get(tile, DecodeTensor::DraftPositions), "wide draft positions");
      for (uint32_t row = 0; row < kDecodeRows; ++row) {
        const uint64_t position = item.logicalPosition + tile * kDecodeRows + row;
        const auto rotary = ropePosition(entry, position);
        std::copy(rotary.begin(), rotary.end(), positions + row * 3);
        draftPositions[row] = static_cast<uint32_t>(position);
      }
      *contents<uint32_t>(decodeArena->get(tile, DecodeTensor::Arrived),
                          "wide arrived") = 0;
      *contents<uint32_t>(decodeArena->get(tile, DecodeTensor::Generation),
                          "wide generation") = 0;
    }
    entry.verifyRows = rows;
    // Legacy DFlash leaves slot zero unused; wide delta-q needs all 16 (32) draws.
    if (samplingEnabled(entry)) {
      entry.cycleUniforms[0] = nextUniform(entry);
      if (tiles == 4)
        for (float &uniform : entry.wideUniforms)
          uniform = nextUniform(entry);
    }
    loadLookup16PolicyBuffers(entry, {});
  }

  std::optional<std::array<uint32_t, 7>> promptLookupProposal(Request &entry) {
    if (entry.lookupHistory.empty())
      return std::nullopt;
    entry.lookupHistory.push_back(*entry.pendingToken);
    const auto proposal = model::promptLookup(
        entry.lookupHistory, adaptiveLookup ? entry.lookupMatch : lookupMinMatch);
    entry.lookupHistory.pop_back();
    return proposal;
  }

  void preparePromptLookup(const std::array<uint32_t, 7> &proposal) {
    static_assert(kDraftProposalTokens == 7);
    auto *tokens = contents<uint32_t>(
        decodeArena->get(0, DecodeTensor::ProposedTokens), "lookup proposals");
    auto *ids = contents<uint32_t>(
        decodeArena->get(0, DecodeTensor::Candidates), "lookup candidates");
    auto *probabilities = contents<float>(
        decodeArena->get(0, DecodeTensor::ProposalProbs), "lookup probabilities");
    std::copy(proposal.begin(), proposal.end(), tokens);
    std::fill_n(ids, 7 * 16, std::numeric_limits<uint32_t>::max());
    std::fill_n(probabilities, 7 * 16, 0.0F);
    for (size_t row = 0; row < proposal.size(); ++row) {
      ids[row * 16] = proposal[row];
      probabilities[row * 16] = 1.0F;
    }
  }

  // Batch lanes beyond the active width replay the last active request so
  // every padded M32 lane binds valid state.
  static Request &laneEntry(std::span<Request *const> entries, uint32_t lane) {
    Request *entry = entries[std::min<size_t>(lane, entries.size() - 1)];
    if (!entry)
      throw std::invalid_argument("empty decode batch lane");
    return *entry;
  }

  void bindDraftRings(
      std::span<Request *const> entries,
      std::vector<std::array<MetalBuffer, kLaneCount>> &keys,
      std::vector<std::array<MetalBuffer, kLaneCount>> &values) const {
    keys.resize(geometry.draft.layers);
    values.resize(geometry.draft.layers);
    for (uint32_t layer = 0; layer < geometry.draft.layers; ++layer) {
      for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
        const auto &ring =
            states.buffers(laneEntry(entries, lane).slot).draft[layer];
        keys[layer][lane] = ring.keys;
        values[layer][lane] = ring.values;
      }
    }
  }

  // The device params (SPLASH_DRAFT_AHEAD) replace the host attention and
  // selector params, whose cache lengths and anchors are then only templates.
  void encodeDraftBatchGraph(CommandGraph &graph,
                             std::span<Request *const> entries,
                             std::span<const uint64_t> logicalPositions,
                             ops::Q4DispatchStats &stats,
                             MetalBuffer deviceAttentionParams = {},
                             MetalBuffer deviceSelectorParams = {}) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != logicalPositions.size()) {
      throw std::invalid_argument("invalid draft decode batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };
    std::array<uint32_t, kLaneCount> cacheLengths{};
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      cacheLengths[lane] =
          static_cast<uint32_t>(logicalPositions[std::min(lane, lanes - 1)]);
    }

    DFlashDecodeBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    for (uint32_t hidden = 0; hidden < buffers.hidden.size(); ++hidden) {
      buffers.hidden[hidden] = d(static_cast<DecodeTensor>(
          static_cast<uint32_t>(DecodeTensor::DraftHidden0) + hidden));
    }
    buffers.normalized = d(DecodeTensor::DraftNormalized);
    buffers.dynamic = d(DecodeTensor::DraftDynamic);
    buffers.convolved = d(DecodeTensor::DraftConvolved);
    buffers.proposalQkv = d(DecodeTensor::DraftProposalQkv);
    buffers.attention = d(DecodeTensor::DraftAttention);
    buffers.projected = d(DecodeTensor::DraftProjected);
    buffers.residual = d(DecodeTensor::DraftResidual);
    buffers.intermediate = d(DecodeTensor::DraftIntermediate);
    buffers.finalHidden = d(DecodeTensor::DraftFinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.selectorHidden = d(DecodeTensor::SelectorHidden);
    buffers.queryKeys = d(DecodeTensor::DraftQueryKeys);
    buffers.queryValues = d(DecodeTensor::DraftQueryValues);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.gateScratch = decodeArena->gateScratch();
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    // B1 only: the head's segment rows belong to one request at a time.
    const bool restrictedHead =
        lanes == 1 && !laneEntry(entries, 0).draftHeadSegment.empty();
    if (restrictedHead)
      draftModel.useHeadSegment(laneEntry(entries, 0).id,
                                laneEntry(entries, 0).draftHeadSegmentVersion,
                                laneEntry(entries, 0).draftHeadSegment,
                                targetModel.vocabularyProjection());
    draftModel.addDecode(graph, std::move(buffers),
                         targetModel.vocabularyProjection(), cacheLengths,
                         lanes, stats, restrictedHead,
                         std::move(deviceAttentionParams));
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      Request &entry = laneEntry(entries, lane);
      if (!entry.pendingToken)
        throw std::invalid_argument("draft batch lane has no anchor");
      anchors[lane] = *entry.pendingToken;
      policies[lane] = samplingPolicy(entry);
    }
    draftModel.addSelection(
        graph,
        {d(DecodeTensor::Logits), d(DecodeTensor::TopPartialIds),
         d(DecodeTensor::TopPartialValues), d(DecodeTensor::Candidates),
         d(DecodeTensor::Unary), d(DecodeTensor::SelectorHidden),
         d(DecodeTensor::SamplingUniforms), d(DecodeTensor::ProposedTokens),
         d(DecodeTensor::ProposalProbs)},
        std::span(anchors).first(lanes), std::span(policies).first(lanes),
        kDraftProposalTokens, restrictedHead, std::move(deviceSelectorParams));
  }

  // SPLASH_DRAFT_AHEAD helpers.
  void retireAhead() {
    if (!ahead)
      return;
    ahead.reset();
    backend.awaitTrailing();
  }

  // A released request's ring must not stay pinned by a running block: its
  // bytes would still count as allocated while reclaim expects them freed.
  void retireAheadFor(uint64_t requestId) {
    if (ahead && std::find(ahead->requestIds.begin(),
                           ahead->requestIds.begin() + ahead->lanes,
                           requestId) != ahead->requestIds.begin() + ahead->lanes)
      retireAhead();
  }

  // Cheap checks before any lane buffer is written: same requests in the same
  // lanes at the positions the block was built for, and an unchanged head.
  bool aheadMayMatch(const BatchPlan &plan,
                     std::span<const ModelBatchItem> items) {
    if (!ahead->finalized ||
        (plan.cohort == BatchCohort::Constrained && !grammarAhead) ||
        plan.decodeStage != DecodeStage::Regular || items.size() != ahead->lanes)
      return false;
    for (uint32_t lane = 0; lane < ahead->lanes; ++lane) {
      if (items[lane].requestId != ahead->requestIds[lane] ||
          items[lane].logicalPosition != ahead->positions[lane])
        return false;
      const Request &entry = request(items[lane].requestId);
      if (entry.draftHeadSegmentVersion != ahead->headVersions[lane] ||
          !entry.pendingToken || *entry.pendingToken != ahead->anchors[lane])
        return false;
    }
    return ahead->restrictedHead ==
           (ahead->lanes == 1 &&
            !request(items[0].requestId).draftHeadSegment.empty());
  }

  // Launch rule for a block behind a cycle's final command: a verified
  // drafter cycle, no lookup hit for aheadLookupQuiet cycles, and every
  // resident request in this batch (a prefilling, other-cohort or mask-waiting
  // peer makes the next plan differ).
  bool aheadLaunchAllowed(std::span<Request *const> entries,
                          std::span<const ModelBatchItem> items, bool verified,
                          bool lookup) const {
    // SPLASH_ROW_HASH reads the verify's logits after the command completes;
    // an ahead block's drafter head writes the same Logits tensor behind it.
    return draftAhead && !rowHash && verified && !lookup &&
           std::all_of(entries.begin(), entries.end(),
                       [&](const Request *entry) {
                         return entry->cyclesSinceLookup >= aheadLookupQuiet;
                       }) &&
           std::all_of(requests.begin(), requests.end(),
                       [&](const auto &resident) {
                         return !resident.second.resident ||
                                std::any_of(items.begin(), items.end(),
                                            [&](const ModelBatchItem &item) {
                                              return item.requestId == resident.first;
                                            });
                       });
  }

  // Submits a cycle's final command; with `launch`, the next cycle's draft
  // block is built once it is committed and runs right behind it.
  CommandTicket submitWithAhead(std::span<const metal::ComputeDispatch> dispatches,
                                metal::CommandCompletion completion,
                                std::span<Request *const> entries,
                                std::span<const ModelBatchItem> items,
                                bool launch,
                                const metal::MetalBackend::ChainGates *gates = nullptr) {
    metal::hostPhase("graph", launch ? 1 : 0, gates ? 1 : 0);
    std::optional<DraftAhead> launched;
    const auto build = [&]() -> std::span<const metal::ComputeDispatch> {
      aheadGraph.clear();
      launched = encodeDraftAhead(aheadGraph, entries, items);
      return aheadGraph.dispatches();
    };
    bool committed = false;
    CommandTicket command = backend.submitCommandAsync(
        dispatches, std::move(completion),
        launch ? metal::MetalBackend::TrailingBuilder(build)
               : metal::MetalBackend::TrailingBuilder{},
        committed, gates);
    aheadGraph.clear();
    if (committed) {
      ahead = std::move(launched);
      for (Request *entry : entries)
        ++entry->aheadLaunched;
    }
    return command;
  }

  // The next cycle's draft block for these lanes, fed on the GPU by this
  // cycle's acceptance. Host-side cache lengths and anchors are templates.
  DraftAhead encodeDraftAhead(CommandGraph &graph,
                              std::span<Request *const> entries,
                              std::span<const ModelBatchItem> items) {
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    DraftAhead next;
    next.lanes = lanes;
    DraftAheadParams params{};
    std::array<uint32_t, kLaneCount> cacheLengths{};
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint64_t, kLaneCount> logicalPositions{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const Request &entry = *entries[lane];
      next.requestIds[lane] = entry.id;
      next.positions[lane] = items[lane].logicalPosition;
      next.headVersions[lane] = entry.draftHeadSegmentVersion;
      next.uniforms[lane] = nextCycleUniforms(entry);
      params.start_position[lane] =
          static_cast<uint32_t>(items[lane].logicalPosition);
      std::copy(next.uniforms[lane].begin(), next.uniforms[lane].end(),
                params.uniforms + lane * kSamplingUniformCount);
      anchors[lane] = *entry.pendingToken;
      policies[lane] = samplingPolicy(entry);
      logicalPositions[lane] = items[lane].logicalPosition;
    }
    for (uint32_t lane = 0; lane < kLaneCount; ++lane)
      cacheLengths[lane] = static_cast<uint32_t>(
          items[std::min(lane, lanes - 1)].logicalPosition);
    params.attention = draftModel.attentionParams(cacheLengths, lanes);
    params.selector = draftModel.selectionParams(
        std::span(anchors).first(lanes), std::span(policies).first(lanes));
    params.mask_token = geometry.target.maskToken;
    next.restrictedHead = lanes == 1 && !entries[0]->draftHeadSegment.empty();

    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };
    const MetalBuffer deviceParams =
        decodeArena->get(0, DecodeTensor::DraftAheadParams);
    MetalBuffer attention =
        backend.view(deviceParams, 0, sizeof(DraftAttentionBatchParams));
    MetalBuffer selector =
        backend.view(deviceParams, 64, sizeof(SelectorBatchParams));
    graph.add("draft_ahead_prepare",
              {d(DecodeTensor::RetainedCount), d(DecodeTensor::NextAnchor),
               d(DecodeTensor::DraftInputTokens),
               d(DecodeTensor::DraftPositions),
               d(DecodeTensor::SamplingUniforms), attention, selector},
              params, {1, 1, 1}, {64, 1, 1});
    const uint32_t rows = lanes * kDecodeRows;
    addRopeTables(graph, d(DecodeTensor::Positions), 0,
                  d(DecodeTensor::DraftPositions), rows, d(DecodeTensor::RopeCos),
                  d(DecodeTensor::RopeSin), d(DecodeTensor::DraftRopeCos),
                  d(DecodeTensor::DraftRopeSin));
    encodeBatchEmbedding(graph, DecodeTensor::DraftInputTokens,
                         DecodeTensor::DraftHidden0, lanes);
    encodeDraftBatchGraph(graph, entries, std::span(logicalPositions).first(lanes),
                          next.stats, std::move(attention), std::move(selector));
    return next;
  }

  void encodeTargetVerifyBatchForward(CommandGraph &graph,
                                      std::span<Request *const> entries,
                                      std::span<const ModelBatchItem> items,
                                      ops::Q4DispatchStats &stats,
                                      bool lookup16 = false) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid target verify batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };
    auto paddedItem = [&](uint32_t lane) -> const ModelBatchItem & {
      return items[std::min(lane, lanes - 1)];
    };

    std::array<Q8ChunkedPrefillParams, kLaneCount> q8{};
    std::array<kv::Q8VerifyAttentionParams, kLaneCount> verify{};
    const uint32_t gdnLayers = geometry.target.stateLayout.layers;
    const uint32_t attentionLayers =
        geometry.target.kvLayout.attentionLayers;
    std::vector<MetalBuffer> gdnPacked(gdnLayers);
    std::vector<MetalBuffer> gdnMixed(gdnLayers);
    std::vector<MetalBuffer> gdnDecay(gdnLayers);
    std::vector<MetalBuffer> gdnBeta(gdnLayers);
    std::vector<MetalBuffer> chunkKeys(attentionLayers);
    std::vector<MetalBuffer> chunkValues(attentionLayers);
    QwenTargetVerifyBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    buffers.hidden = {d(DecodeTensor::Hidden0), d(DecodeTensor::Hidden1)};
    buffers.normalized = d(DecodeTensor::Normalized);
    buffers.recurrent = d(DecodeTensor::Recurrent);
    buffers.gdnHidden = d(DecodeTensor::GdnHidden);
    buffers.gdnOutput = d(DecodeTensor::GdnOutput);
    buffers.denseIntermediate = d(DecodeTensor::Intermediate);
    buffers.fullPacked = d(DecodeTensor::FullPacked);
    buffers.fullQueries = d(DecodeTensor::FullQueries);
    buffers.attentionPartials = d(DecodeTensor::AttentionPartials);
    buffers.attentionStatistics = d(DecodeTensor::AttentionStatistics);
    buffers.fullAttention = d(DecodeTensor::FullAttention);
    buffers.attentionHidden = d(DecodeTensor::AttentionHidden);
    buffers.attentionOutput = d(DecodeTensor::AttentionOutput);
    buffers.ropeCos = d(DecodeTensor::RopeCos);
    buffers.ropeSin = d(DecodeTensor::RopeSin);
    buffers.arrived = d(DecodeTensor::Arrived);
    buffers.generation = d(DecodeTensor::Generation);
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.finalHidden = d(DecodeTensor::FinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.denseGateScratch = decodeArena->gateScratch();
    buffers.gdnPacked = gdnPacked;
    buffers.gdnMixed = gdnMixed;
    buffers.gdnDecay = gdnDecay;
    buffers.gdnBeta = gdnBeta;
    buffers.chunkKeys = chunkKeys;
    buffers.chunkValues = chunkValues;
    buffers.selectedExperts = d(DecodeTensor::MoeSelectedExperts);
    buffers.routingWeights = d(DecodeTensor::MoeRoutingWeights);
    buffers.tileDescriptors = d(DecodeTensor::MoeTileDescriptors);
    buffers.tileCount = d(DecodeTensor::MoeTileCount);
    buffers.groupedRoutes = d(DecodeTensor::MoeGroupedRoutes);
    buffers.routeRows = d(DecodeTensor::MoeRouteRows);
    buffers.groupedInput = d(DecodeTensor::MoeGroupedInput);
    buffers.expertIntermediate = d(DecodeTensor::MoeExpertIntermediate);
    buffers.expertOutput = d(DecodeTensor::MoeExpertOutput);
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      const ModelBatchItem &item = paddedItem(lane);
      q8[lane] = q8Params(item.logicalPosition, kDecodeRows, kTileRows,
                          item.pageTable);
      verify[lane] = kv::q8VerifyAttentionParams(
          q8[lane].committed_tokens, q8[lane].chunk_tokens,
          q8[lane].chunk_stride, q8[lane].page_table_entries,
          q8[lane].physical_page_count);
      if (!kv::q8VerifyAttentionValidationError(verify[lane]).empty())
        throw std::invalid_argument("invalid batched KV verify geometry");
      Request &entry = laneEntry(entries, lane);
      buffers.pageTables[lane] =
          decodeArena->get(entry.slot, DecodeTensor::PageTable);
      const uint32_t active = states.metadata(entry.slot).activeParity;
      buffers.currentGdnStates[lane] =
          states.buffers(entry.slot).gdn[active].stateBase;
      buffers.nextGdnStates[lane] =
          states.buffers(entry.slot).gdn[active ^ 1].stateBase;
    }
    // SPLASH_M24_PAD3: B3's input RMS and projections also cover the idle
    // fourth lane (QwenTarget decides per encode).
    std::vector<MetalBuffer> gdnPackedPadded;
    // Pad only into a fourth lane the arena has and nothing in flight uses
    // (an ahead block over four lanes is retired before a B3 step is encoded;
    // this is the safety net). Otherwise B3 keeps the M24 consumer.
    const char *padRefusal = !(lanes == 3 && !lookup16) ? nullptr
        : kLaneCount < 4 ? "the arena has no fourth lane"
        : ahead && !ahead->finalized && ahead->lanes > 3 ? "an ahead block holds the fourth lane"
        : nullptr;
    if (padRefusal) {
      static std::atomic<bool> logged{false};
      if (!logged.exchange(true))
        std::fprintf(stderr, "m24 pad3: B3 kept the M24 consumer (%s)\n", padRefusal);
    }
    if (lanes == 3 && !lookup16 && !padRefusal) {
      buffers.hiddenPadded = {decodeArena->packed(DecodeTensor::Hidden0, 4),
                              decodeArena->packed(DecodeTensor::Hidden1, 4)};
      buffers.normalizedPadded = decodeArena->packed(DecodeTensor::Normalized, 4);
      buffers.fullPackedPadded = decodeArena->packed(DecodeTensor::FullPacked, 4);
      gdnPackedPadded.resize(gdnLayers);
      for (uint32_t layer = 0; layer < gdnLayers; ++layer)
        gdnPackedPadded[layer] = decodeArena->gdnBatchSlice(
            DecodeTensor::VerifyPackedBase, layer, 4);
      buffers.gdnPackedPadded = gdnPackedPadded;
    }
    for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
      gdnPacked[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyPackedBase, layer, lanes);
      gdnMixed[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyMixedBase, layer, lanes);
      gdnDecay[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyDecayBase, layer, lanes);
      gdnBeta[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyBetaBase, layer, lanes);
    }
    std::vector<kv::LayerStorage> kvLayers(attentionLayers);
    for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
      chunkKeys[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkKeysBase, layer, lanes);
      chunkValues[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkValuesBase, layer, lanes);
      kvLayers[layer] = kvPages.layer(layer);
    }
    if (lookup16) {
      if ((lanes != 2 && lanes != 4) ||
          std::any_of(entries.begin(), entries.end(),
                      [&](Request *entry) { return entry != entries[0]; }))
        throw std::logic_error("wide lookup must be one aliased request");
      targetModel.addVerify16(
          graph, std::move(buffers), kvLayers, q8, verify, stats,
          decodeArena->wideConvolutionScratch(), lanes, entries[0]->wideGdn);
      if (entries[0]->wideGdn != ops::WideGdn::Chain)
        ++entries[0]->wideGdnCycles;
    } else {
      targetModel.addVerify(graph, std::move(buffers), kvLayers, q8, verify,
                            lanes, stats);
    }
  }

  void encodeTargetVerifyBatchPolicy(CommandGraph &graph,
                                     std::span<Request *const> entries) {
    if (entries.empty() || entries.size() > kLaneCount)
      throw std::invalid_argument("invalid target policy batch");
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!entries[lane])
        throw std::invalid_argument("empty target policy lane");
      policies[lane] = samplingPolicy(*entries[lane]);
    }
    sampling.addVerify(graph, std::span(policies).first(lanes),
                       samplingBuffers(lanes));
  }

  void addPrefillPolicy(CommandGraph &graph, Request &entry, uint32_t lane,
                        uint32_t finalRow) const {
    if (finalRow >= kDecodeRows || entry.constraint != ConstraintMode::None) {
      throw std::invalid_argument("invalid prefill policy boundary");
    }
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->get(lane, tensor);
    };
    targetModel.addHead(graph, d(DecodeTensor::Hidden0),
                        d(DecodeTensor::FinalHidden), d(DecodeTensor::Logits),
                        finalRow + 1, decodeArena->linearScratch());
    addInitialPolicySelection(graph, entry, lane, finalRow);
  }

  void encodeDraftStateCommitBatch(CommandGraph &graph,
                                   std::span<Request *const> entries,
                                   std::span<const ModelBatchItem> items,
                                   ops::Q4DispatchStats &stats,
                                   MetalBuffer retainedOverride = {},
                                   bool seamSibling = false,
                                   bool groupedLanes = false) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid draft state commit batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };

    std::array<uint32_t, kLaneCount> startPositions{};
    for (uint32_t lane = 0; lane < kLaneCount; ++lane)
      startPositions[lane] = static_cast<uint32_t>(
          items[std::min(lane, lanes - 1)].logicalPosition);
    DFlashContextBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.projected = d(DecodeTensor::ContextProjected);
    buffers.hidden = d(DecodeTensor::ContextHidden);
    buffers.qkv = d(DecodeTensor::ContextQkv);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.retainedCounts = retainedOverride ? retainedOverride
                                               : d(DecodeTensor::RetainedCount);
    if (seamSibling) {
      // Siblings of the target tail must not share its linear scratch, and
      // the drafter's qkv projections run long before their commits.
      if (lanes != 1) throw std::invalid_argument("seam siblings are one-lane only");
      buffers.linearScratch = {decodeArena->get(0, DecodeTensor::SeamScratchInput),
                               decodeArena->get(0, DecodeTensor::SeamScratchSums), {}, {}};
      const MetalBuffer all = decodeArena->get(0, DecodeTensor::SeamContextQkv);
      const uint64_t layerBytes =
          uint64_t{kDecodeRows} * geometry.draft.qkvSize * sizeof(uint16_t);
      for (uint32_t layer = 0; layer < geometry.draft.layers; ++layer)
        buffers.layerQkv.push_back(backend.view(all, layer * layerBytes, layerBytes));
    } else if (groupedLanes) {
      // SPLASH_GROUPED_CONTEXT_KV=1 at 2 separate requests (the ordinary unconstrained verify only; the caller decides,
      // B3/B4 wait for in-engine timing): one output of lanes x 8 rows per drafter layer for the grouped K/V projection, carved from the lanes'
      // packed SeamContextQkv (the seam only uses it in one-lane cycles). No siblings: the commit keeps its order after
      // the GDN commit. Wide lookup (aliased lanes of one request) and constrained commits get no views: outside
      // admission, stock projections.
      if (lanes < 2) throw std::invalid_argument("grouped context K/V lanes need 2-4 requests");
      const MetalBuffer all = d(DecodeTensor::SeamContextQkv);
      const uint64_t layerBytes =
          uint64_t{lanes} * kDecodeRows * geometry.draft.qkvSize * sizeof(uint16_t);
      for (uint32_t layer = 0; layer < geometry.draft.layers; ++layer)
        buffers.layerQkv.push_back(backend.view(all, layer * layerBytes, layerBytes));
    }
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    draftModel.addContextCommit(graph, std::move(buffers), startPositions,
                                lanes, stats);
  }

  // The grouped mode of SPLASH_GROUPED_CONTEXT_KV (unset or exactly "1", without SPLASH_GROUPED_CONTEXT_KV_TAIL=1,
  // as DFlashDraft reads it; per graph build): multi-lane context commits get per-layer qkv views so DFlashDraft can
  // admit the grouped K/V projection. The per-layer tail control stays one-lane and gets none.
  static bool groupedContextKvMultiLane() {
    const char *tail = std::getenv("SPLASH_GROUPED_CONTEXT_KV_TAIL");
    return metal::envSwitch("SPLASH_GROUPED_CONTEXT_KV") && !(tail && std::string_view(tail) == "1");
  }

  // SPLASH_SEAM_SIBLING (E3; default on, read per
  // submission): the drafter context commit reads only the last tapped
  // layer's hidden (capture_target_hidden) and, in its commit kernels, the
  // acceptance's RetainedCount. So it rides encode-after behind the verify
  // tail's own ops: each dependent seam stage (split sums, context projection,
  // norm) behind the next tail matmul, the context projection behind a gate/up,
  // one qkv projection behind each of the next matmuls (one per drafter layer), and
  // the commits behind the GDN commit (after acceptance). Anything unexpected keeps
  // today's serial order. Lockstep -0.096 ms/step, live route -0.283 (-0.77%),
  // oracle identical. Overlap goes after the op it doesn't depend on, never ahead.
  static bool seamSiblingEnabled() { return metal::envSwitch("SPLASH_SEAM_SIBLING"); }

  void placeSeamSiblings(CommandGraph &graph, size_t seamBegin) {
    static std::atomic<bool> skipWitnessed{false}, placeWitnessed{false}, groupedWitnessed{false}, tailWitnessed{false};
    const auto skip = [&](const char *why) {
      if (!skipWitnessed.exchange(true))
        std::fprintf(stderr, "seam sibling: kept serial (%s)\n", why);
    };
    const auto list = graph.dispatches();
    const auto starts = [](const std::string &name, std::string_view prefix) {
      return name.compare(0, prefix.size(), prefix) == 0;
    };
    size_t lastTap = SIZE_MAX, accept = SIZE_MAX, gdnCommit = SIZE_MAX;
    for (size_t index = 0; index < seamBegin; ++index) {
      const std::string &name = list[index].pipelineName;
      if (name == "capture_target_hidden") lastTap = index;
      else if (starts(name, "decode_accept_dflash")) accept = index;
      else if (name == "verify_gdn_commit") gdnCommit = index;
    }
    if (lastTap == SIZE_MAX || accept == SIZE_MAX || gdnCommit == SIZE_MAX ||
        !(lastTap < accept && accept < gdnCommit))
      return skip("tail ops not found");
    std::vector<size_t> tail;  // the verify tail's matmuls, tap to acceptance
    for (size_t index = lastTap + 1; index < accept; ++index) {
      const std::string &name = list[index].pipelineName;
      if (starts(name, "decode_linear_q4") && name.find("split_sums") == std::string::npos)
        tail.push_back(index);
    }
    const size_t layers = geometry.draft.layers, count = list.size() - seamBegin;
    // SPLASH_GROUPED_CONTEXT_KV grouped mode: one grouped K/V projection, then one commit per layer (prefix + 1 + layers
    // dispatches); it rides behind one tail matmul and the commits behind the GDN commit, as before.
    const bool grouped = count >= layers + 3 &&
        list[seamBegin + count - layers - 1].pipelineName == "decode_linear_q4_n128_paired_context_grouped";
    const size_t prefix = grouped ? count - layers - 1 : count - 2 * layers;  // sums?, context projection, norm
    size_t tailStages = 0;  // SPLASH_GROUPED_CONTEXT_KV per-layer tail layout: every projection K/V-only, or none
    if (grouped) {
      for (size_t layer = 0; layer < layers; ++layer)
        if (list[seamBegin + prefix + 1 + layer].pipelineName != "draft_context_kv_commit")
          return skip("unexpected grouped seam order");
    } else {
      if (count < 2 * layers + 2) return skip("unexpected seam size");
      for (size_t layer = 0; layer < layers; ++layer) {
        if (!starts(list[seamBegin + prefix + 2 * layer].pipelineName, "decode_linear_q4") ||
            list[seamBegin + prefix + 2 * layer + 1].pipelineName != "draft_context_kv_commit")
          return skip("unexpected seam order");
        tailStages += list[seamBegin + prefix + 2 * layer].pipelineName == "decode_linear_q4_n128_paired_context_tail";
      }
      if (tailStages && tailStages != layers) return skip("mixed K/V-only seam");
    }
    std::vector<size_t> partners(count);
    size_t next = 0;
    for (size_t stage = 0; stage < count; ++stage) {
      const std::string &name = list[seamBegin + stage].pipelineName;
      if (name == "draft_context_kv_commit") {
        partners[stage] = gdnCommit;
        continue;
      }
      // The context projection (the seam's largest dispatch) waits for a gate/up.
      if (stage < prefix && starts(name, "decode_linear_q4") && name != "decode_linear_q4_split_sums")
        while (next < tail.size() &&
               list[tail[next]].pipelineName.find("gate_up") == std::string::npos)
          ++next;
      if (next >= tail.size()) return skip("not enough tail matmuls");
      partners[stage] = tail[next++];
    }
    graph.placeSiblings(seamBegin, partners);
    contextKvWitness.seams.fetch_add(1, std::memory_order_relaxed);  // counted once placed (route witness)
    if (grouped) contextKvWitness.grouped.fetch_add(1, std::memory_order_relaxed);
    else if (tailStages) contextKvWitness.tail.fetch_add(1, std::memory_order_relaxed);
    if (!(grouped ? groupedWitnessed : tailStages ? tailWitnessed : placeWitnessed).exchange(true)) {  // once per layout
      std::fprintf(stderr, "seam sibling%s: %zu dispatches placed behind tail ops",
                   grouped ? " (grouped K/V)" : tailStages ? " (five-tail K/V)" : "", count);  // texts gates grep
      for (size_t stage = 0; stage < count; ++stage)
        std::fprintf(stderr, " %zu", partners[stage]);
      std::fprintf(stderr, " (tap %zu, acceptance %zu, GDN commit %zu)\n", lastTap, accept, gdnCommit);
    }
  }

  void encodeBatchAcceptance(CommandGraph &graph,
                             std::span<Request *const> lanes,
                             std::span<const uint32_t> maximumRetained) {
    if (lanes.empty() || lanes.size() > kLaneCount ||
        lanes.size() != maximumRetained.size()) {
      throw std::invalid_argument("invalid DFlash acceptance batch");
    }
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      if (!lanes[lane] || !maximumRetained[lane] ||
          maximumRetained[lane] > kDecodeRows) {
        throw std::invalid_argument("invalid DFlash acceptance lane");
      }
      policies[lane] = samplingPolicy(*lanes[lane]);
    }
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    sampling.addAcceptance(
        graph,
        {decodeArena->packed(DecodeTensor::ProposedTokens, width),
         decodeArena->packed(DecodeTensor::Candidates, width),
         decodeArena->packed(DecodeTensor::ProposalProbs, width),
         decodeArena->packed(DecodeTensor::TargetTopIds, width),
         decodeArena->packed(DecodeTensor::TargetTopProbs, width),
         decodeArena->packed(DecodeTensor::SamplingUniforms, width),
         decodeArena->packed(DecodeTensor::OutputTokens, width),
         decodeArena->packed(DecodeTensor::RetainedCount, width),
         decodeArena->packed(DecodeTensor::NextAnchor, width),
         decodeArena->packed(DecodeTensor::AcceptedCount, width)},
        maximumRetained, std::span(policies).first(width),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1]);
  }

  void encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid embedding batch width");
    const uint32_t rows = lanes * kDecodeRows;
    targetModel.addEmbedding(graph, decodeArena->packed(tokens, lanes),
                             decodeArena->packed(output, lanes), rows);
  }

  void encodeBatchVerifyInput(CommandGraph &graph, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid verify-input batch width");
    sampling.addVerifyInput(
        graph, decodeArena->packed(DecodeTensor::DraftInputTokens, lanes),
        decodeArena->packed(DecodeTensor::ProposedTokens, lanes),
        decodeArena->packed(DecodeTensor::InputTokens, lanes), lanes);
  }

  void encodeBatchGdnCommit(CommandGraph &graph,
                            std::span<Request *const> lanes,
                            bool lookup16 = false) {
    if (lanes.empty() || lanes.size() > kLaneCount)
      throw std::invalid_argument("invalid GDN commit batch");
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    std::array<MetalBuffer, kLaneCount> currentStates;
    std::array<MetalBuffer, kLaneCount> nextStates;
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      Request *entry = lanes[std::min(lane, width - 1)];
      if (!entry)
        throw std::invalid_argument("empty GDN commit lane");
      const uint32_t active = states.metadata(entry->slot).activeParity;
      const auto &gdn = states.buffers(entry->slot).gdn;
      currentStates[lane] = gdn[active].stateBase;
      nextStates[lane] = gdn[active ^ 1].stateBase;
    }
    QwenTargetCommitBuffers buffers{
         decodeArena->gdnStorage(DecodeTensor::VerifyPackedBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyMixedBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyDecayBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyBetaBase), currentStates,
         nextStates, decodeArena->packed(DecodeTensor::RetainedCount, width)};
    if (lookup16) {
      targetModel.addStateCommit16(graph, std::move(buffers),
                                   decodeArena->wideConvolutionScratch(), width);
    } else {
      targetModel.addStateCommit(graph, std::move(buffers), width);
    }
  }

  // A wide lookup's physical lanes: one request, tile t at position + 8t.
  struct WideLanes final {
    std::array<Request *, kLaneCount> entries{};
    std::array<ModelBatchItem, kLaneCount> items{};
    uint32_t tiles = 0;
    std::span<Request *const> entrySpan() const { return {entries.data(), tiles}; }
    std::span<const ModelBatchItem> itemSpan() const { return {items.data(), tiles}; }
  };

  static WideLanes wideLanes(Request &entry, const ModelBatchItem &item) {
    WideLanes lanes;
    lanes.tiles = entry.verifyRows / kDecodeRows;
    for (uint32_t tile = 0; tile < lanes.tiles; ++tile) {
      lanes.entries[tile] = &entry;
      lanes.items[tile] = item;
      lanes.items[tile].logicalPosition += tile * kDecodeRows;
    }
    return lanes;
  }

  void encodeLookup16Forward(CommandGraph &graph, Request &entry,
                              const ModelBatchItem &item,
                              ops::Q4DispatchStats &stats) {
    const WideLanes lanes = wideLanes(entry, item);
    encodeBatchEmbedding(graph, DecodeTensor::InputTokens,
                          DecodeTensor::Hidden0, lanes.tiles);
    encodeTargetVerifyBatchForward(graph, lanes.entrySpan(), lanes.itemSpan(), stats, true);
  }

  void encodeLookup16Commit(CommandGraph &graph, Request &entry,
                             const ModelBatchItem &item,
                             uint32_t maximumRetained,
                             ops::Q4DispatchStats &stats) {
    const WideLanes lanes = wideLanes(entry, item);
    const uint32_t tiles = lanes.tiles;
    encodeTargetVerifyBatchPolicy(graph, lanes.entrySpan());
    const auto halves = decodeArena->get(0, DecodeTensor::LookupRetainedHalves);
    sampling.addLookup16Acceptance(
        graph,
        {decodeArena->packed(DecodeTensor::InputTokens, tiles),
         decodeArena->packed(DecodeTensor::TargetTopIds, tiles),
         decodeArena->packed(DecodeTensor::TargetTopProbs, tiles),
         decodeArena->packed(DecodeTensor::SamplingUniforms, tiles / 2),
         decodeArena->packed(DecodeTensor::OutputTokens, tiles),
         decodeArena->get(0, DecodeTensor::RetainedCount),
         decodeArena->get(0, DecodeTensor::NextAnchor),
         decodeArena->get(0, DecodeTensor::AcceptedCount), halves},
        maximumRetained, samplingPolicy(entry),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1], tiles);
    encodeBatchGdnCommit(graph, lanes.entrySpan(), true);
    // SPLASH_CONTEXT_KV_WITNESS=1 (diagnostic, default off; the grouped K/V wide-path gate, lead 11:2x): one line per
    // wide commit with its context K/V witness deltas. Grouped K/V must never take this path (outside admission).
    const char *witnessValue = std::getenv("SPLASH_CONTEXT_KV_WITNESS");
    const bool witness = witnessValue && std::string_view(witnessValue) == "1";
    const auto &w = contextKvWitness;
    std::array<uint64_t, 6> before{};  // loaded only when the diagnostic is on
    if (witness)
      before = {w.commits.load(), w.built.load(), w.multi.load(), w.outside.load(), w.refused.load(),
                w.seams.load()};
    encodeDraftStateCommitBatch(graph, lanes.entrySpan(), lanes.itemSpan(), stats, halves);
    if (witness)
      std::fprintf(stderr, "context_kv_witness wide request=%llu tiles=%u commits=%llu built=%llu multi=%llu "
                   "outside=%llu refused=%llu seams=%llu\n", static_cast<unsigned long long>(entry.id), tiles,
                   static_cast<unsigned long long>(w.commits.load() - before[0]),
                   static_cast<unsigned long long>(w.built.load() - before[1]),
                   static_cast<unsigned long long>(w.multi.load() - before[2]),
                   static_cast<unsigned long long>(w.outside.load() - before[3]),
                   static_cast<unsigned long long>(w.refused.load() - before[4]),
                   static_cast<unsigned long long>(w.seams.load() - before[5]));
  }

  // A stop token or the last budgeted token needs no target work of its own:
  // the next cycle would only echo it as output. Emitting it as soon as it is
  // selected saves that cycle; the engine is told it has no KV row.
  bool emitTerminalAnchor(Request &entry, ModelStepResult &result) const {
    const bool stop = isStopToken(geometry, *entry.pendingToken);
    if (!stop && entry.maxNewTokens - entry.generatedTokens != 1)
      return false;
    result.outputTokens.push_back(*entry.pendingToken);
    result.outputTokensWithoutKv = 1;
    result.finished = stop;
    ++entry.generatedTokens;
    return true;
  }

  std::vector<ModelStepResult> finalizeDecode(
      std::span<DecodeLaneResult> lanes, std::vector<ModelStepResult> results,
      std::span<const ModelBatchItem> items, const ops::Q4DispatchStats &stats,
      uint32_t planWidth, CommandTiming timing) {
    if (lanes.size() != items.size() || results.size() != items.size())
      throw std::logic_error("decode completion shape changed");
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      if (!laneResult.verify)
        continue;
      auto d = [&](DecodeTensor tensor) {
        return decodeArena->get(lane, tensor);
      };
      const uint32_t generation =
          *contents<uint32_t>(d(DecodeTensor::Generation), "target generation");
      if (generation != geometry.target.stateLayout.layers)
        throw std::runtime_error("target verify resident grids did not finish");
      if (laneResult.wideLookup) {
        const uint32_t tiles = laneResult.wideTiles;
        const auto *generations = contents<uint32_t>(
            decodeArena->packed(DecodeTensor::Generation, tiles), "wide generations");
        const auto *arrivals = contents<uint32_t>(
            decodeArena->packed(DecodeTensor::Arrived, tiles), "wide arrivals");
        for (uint32_t tile = 0; tile < tiles; ++tile)
          if (generations[tile] != geometry.target.stateLayout.layers || arrivals[tile])
            throw std::runtime_error("wide target tiles did not all finish");
      }

      laneResult.retained = *contents<uint32_t>(d(DecodeTensor::RetainedCount),
                                                "GPU retained token count");
      laneResult.accepted = *contents<uint32_t>(d(DecodeTensor::AcceptedCount),
                                                "GPU accepted draft count");
      laneResult.nextAnchor =
          *contents<uint32_t>(d(DecodeTensor::NextAnchor), "GPU next anchor");
      const uint32_t verifyRows = laneResult.wideLookup
                                      ? laneResult.wideTiles * kDecodeRows
                                      : kDecodeRows;
      if (!laneResult.retained || laneResult.retained > verifyRows)
        throw std::runtime_error("target policy produced invalid retention");
      if (laneResult.accepted > verifyRows - 1 ||
          laneResult.nextAnchor >= geometry.target.vocabularySize) {
        throw std::runtime_error(
            "target policy selected an invalid next anchor");
      }
      if (rowHash) {
        // A wide lookup's rows span its aliased lanes; a multi-request batch
        // hashes each lane's own eight rows.
        const uint32_t tiles = laneResult.wideLookup ? laneResult.wideTiles : 1;
        const auto rowsOf = [&](DecodeTensor tensor) {
          return laneResult.wideLookup ? decodeArena->packed(tensor, tiles)
                                       : decodeArena->get(lane, tensor);
        };
        const MetalBuffer hidden = rowsOf(DecodeTensor::FinalHidden);
        const MetalBuffer logits = rowsOf(DecodeTensor::Logits);
        const auto *hiddenBytes = contents<char>(hidden, "row hash");
        const auto *logitBytes = contents<char>(logits, "row hash");
        const auto *tokens = contents<uint32_t>(rowsOf(DecodeTensor::InputTokens), "row hash tokens");
        const size_t hiddenRow = hidden.sizeBytes() / verifyRows;
        const size_t logitRow = logits.sizeBytes() / verifyRows;
        const std::hash<std::string_view> hash;
        for (uint32_t row = 0; row <= laneResult.accepted; ++row)
          std::fprintf(stderr, "row_hash request=%llu position=%llu token=%u wide=%u row=%u hidden=%zx logits=%zx lanes=%zu\n",
                       static_cast<unsigned long long>(items[lane].requestId),
                       static_cast<unsigned long long>(items[lane].logicalPosition + row), tokens[row],
                       tiles - 1, row, hash(std::string_view(hiddenBytes + row * hiddenRow, hiddenRow)),
                       hash(std::string_view(logitBytes + row * logitRow, logitRow)), items.size());
      }
    }

    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      if (!laneResult.verify)
        continue;
      Request &entry = *laneResult.request;
      const uint32_t *targetTokens =
          contents<uint32_t>(laneResult.wideLookup
                                 ? decodeArena->packed(DecodeTensor::OutputTokens,
                                                       laneResult.wideTiles)
                                 : decodeArena->get(lane, DecodeTensor::OutputTokens),
                             "target output tokens");
      std::vector<uint32_t> output;
      output.reserve(laneResult.retained);
      output.push_back(laneResult.currentAnchor);
      output.insert(output.end(), targetTokens,
                    targetTokens + (laneResult.retained - 1));
      // Rare output tokens (e.g. another language) join the draft head's
      // segment; once it is full the request falls back to the full head.
      if (!entry.draftHeadSegment.empty()) {
        auto &segment = entry.draftHeadSegment;
        for (const uint32_t token : output) {
          if (token >= geometry.target.vocabularySize || draftModel.staticHeadHas(token))
            continue;
          const size_t used = entry.draftHeadSegmentUsed;
          const size_t at = std::find(segment.begin(), segment.end(), token) - segment.begin();
          if (at < used)
            continue;  // already protected
          if (at < segment.size()) {
            std::swap(segment[at], segment[used]);  // filler id seen in output: protect it
          } else if (used == segment.size()) {
            segment.clear();  // more rare ids than rows: full head from now on
            break;
          } else {
            segment[used] = token;  // evicts a filler id
          }
          ++entry.draftHeadSegmentUsed;
          ++entry.draftHeadSegmentVersion;
        }
      }

      if (ahead && !ahead->finalized && lane < ahead->lanes &&
          ahead->requestIds[lane] == entry.id) {
        ahead->positions[lane] += laneResult.retained;
        ahead->anchors[lane] = laneResult.nextAnchor;
      }
      states.swapParity(entry.slot);
      const uint64_t nextLength =
          items[lane].logicalPosition + laneResult.retained;
      const QwenLogicalLengths previous = states.metadata(entry.slot).lengths;
      states.updateLengths(
          entry.slot, advanceDraftContext(
                          previous, nextLength,
                          {static_cast<uint32_t>(items[lane].logicalPosition),
                           static_cast<uint32_t>(nextLength), 0, false}));
      entry.generatedTokens += laneResult.retained;
      entry.pendingToken = laneResult.nextAnchor;
      entry.maskWords.clear();
      entry.verifyMaskInFlight = false;
      entry.verifyRows = kDecodeRows;
      entry.wideFull = laneResult.wideLookup &&
                       laneResult.accepted == laneResult.wideTiles * kDecodeRows - 1;
      entry.decodeStage = DecodeStage::Regular;
      ModelStepResult &result = results[lane];
      result = {entry.id,
                0,
                std::move(output),
                false,
                DecodeStage::Regular,
                laneResult.wideLookup ? laneResult.wideTiles * kDecodeRows - 1
                                      : kDraftProposalTokens,
                std::min(laneResult.accepted, laneResult.retained - 1)};
      if (entry.generatedTokens < entry.maxNewTokens)
        emitTerminalAnchor(entry, result);
      if (!entry.lookupHistory.empty()) {
        entry.cyclesSinceLookup =
            laneResult.promptLookup ? 0 : entry.cyclesSinceLookup + (entry.cyclesSinceLookup < UINT32_MAX);
        ++entry.lookupCycles;
        if (laneResult.promptLookup) {
          ++entry.lookupHits;
          entry.lookupRetained += laneResult.retained;
        }
        if (laneResult.wideLookup) {
          ++entry.wideLookupHits;
          entry.wideLookupRetained += laneResult.retained;
          entry.wide32Hits += laneResult.wideTiles == 4 ? 1 : 0;
        }
        if (adaptiveLookup && (laneResult.promptLookup || laneResult.wideLookup))
          entry.lookupMatch = model::adaptLookupMatch(entry.lookupMatch, laneResult.retained,
                                                      lookupMinMatch);
        entry.lookupHistory.insert(entry.lookupHistory.end(),
                                   result.outputTokens.begin(),
                                   result.outputTokens.end());
        if (entry.lookupHistory.size() > kPromptLookupHistory) {
          entry.lookupHistory.erase(entry.lookupHistory.begin(),
                                    entry.lookupHistory.end() - kPromptLookupHistory);
        }
      }
    }

    if (ahead)
      ahead->finalized = true;
    counters.lastDecodeWidth = planWidth;
    counters.lastDecodeFusedOperations = stats.fusedSourceOperations;
    counters.lastDecodeM16Dispatches = stats.m16Dispatches;
    counters.lastDecodeM24Dispatches = stats.m24Dispatches;
    counters.lastDecodeM32Dispatches = stats.m32Dispatches;
    counters.lastDecodeGpuSeconds = timing.gpuSeconds;
    counters.totalDecodeGpuSeconds += timing.gpuSeconds;
    counters.lastDecodeWallSeconds = timing.wallSeconds;
    counters.totalDecodeWallSeconds += timing.wallSeconds;
    return results;
  }

  // A constrained DFlash cycle has one host dependency between three Metal
  // commands: draft proposals define the grammar simulation, while the target
  // forward is independent of the resulting mask.  This ticket keeps the
  // scheduler batch (and therefore its DecodeArena lanes) owned across that
  // dependency.  All state transitions run on the engine thread; completion
  // handlers only wake it, so they capture the wake hook and never the ticket.
  class ConstrainedDecodeTicket final : public ModelBatchTicket {
  public:
    ConstrainedDecodeTicket(Impl &impl, std::vector<DecodeLaneResult> lanes,
                            std::vector<ModelStepResult> results,
                            std::span<const ModelBatchItem> items,
                            const ops::Q4DispatchStats &stats,
                            uint32_t planWidth, CommandTiming priorTiming,
                            CommandGraph &draft, bool adopted, bool chain,
                            std::function<void()> completion)
        : impl_(impl), lanes_(std::move(lanes)), results_(std::move(results)),
          items_(items.begin(), items.end()), stats_(stats),
          planWidth_(planWidth), adopted_(adopted), timing_(priorTiming),
          wake_(std::make_shared<std::function<void()>>(
              std::move(completion))) {
      if (!chain) {
        submit(draft);
        return;
      }
      // SPLASH_GRAMMAR_CHAIN (B1): the target forward and the commit join this
      // command. Its head raises the chain event once the proposals exist; the
      // commit part waits until the host has written the masks.
      beginVerify();
      metal::MetalBackend::ChainGates gates;
      gates.signalBefore = draft.dispatches().size();
      gates.signalValue = ++impl_.chainValues;
      gates.waitValue = ++impl_.chainValues;
      // SPLASH_STREAMED_SUBMIT: the gated head commits before the rest is
      // built. A drafted cycle's draft goes now; a lookup cycle's proposals are
      // host-written, so its head adds the forward's first chunk.
      if (impl_.streamChunk) {
        auto sink = [&impl = impl_, value = gates.signalValue](
                        std::span<const metal::ComputeDispatch> head) {
          impl.backend.streamHead(head, value);
        };
        if (lanes_[0].promptLookup)
          draft.streamAt(gates.signalBefore + impl_.streamChunk, std::move(sink));
        else
          sink(draft.dispatches());
      }
      encodeForward(draft);
      draft.streamAt(0, {});  // the head must end before the mask gate
      gates.waitBefore = draft.dispatches().size();
      encodeCommit(draft);
      chain_ = gates;
      submit(draft, entries(), launchAhead(), &gates);
      stage_ = Stage::ChainDraft;
      // Lookup proposals were written by the host; drafted ones exist once the
      // head has signalled.
      proposalsKnown_ = lanes_[0].promptLookup;
      if (!proposalsKnown_)
        impl_.backend.notifyChain(gates.signalValue, [wake = wake_] {
          if (*wake)
            (*wake)();
        });
    }

    // A gated command still waiting for masks must not wait forever: a
    // cancelled, failed or shut-down request releases it with this ticket.
    ~ConstrainedDecodeTicket() override {
      if (chain_ && stage_ != Stage::Commit && stage_ != Stage::Done)
        impl_.backend.signalChain(chain_->waitValue);
    }

    std::vector<ModelMaskRequest> takeMaskRequests() override {
      std::vector<ModelMaskRequest> requests;
      if (chain_) {
        if (stage_ == Stage::ChainDraft &&
            (proposalsKnown_ || impl_.backend.chainValue() >= chain_->signalValue)) {
          requests = maskRequests();
          maskWaitStarted_ = std::chrono::steady_clock::now();
          stage_ = Stage::WaitingMask;
        }
        if (stage_ == Stage::WaitingMask && masksReady()) {
          finishMaskWait();
          loadMasks();
          impl_.backend.signalChain(chain_->waitValue);
          stage_ = Stage::Commit;
        }
        return requests;
      }
      if (stage_ == Stage::Draft && command_.ready()) {
        addTiming(command_.wait());
        // An adopted draft-ahead block wrote this cycle's proposals.
        if (adopted_)
          impl_.backend.awaitTrailing();
        beginVerify();
        requests = maskRequests();
        DecodeGraphLease lease(impl_);
        encodeForward(lease.graph);
        submit(lease.graph);
        stage_ = Stage::TargetForward;
      }

      if (stage_ == Stage::TargetForward && command_.ready()) {
        const CommandTiming forward = command_.wait();
        addTiming(forward);
        targetForwardGpuSeconds_ += forward.gpuSeconds;
        maskWaitStarted_ = std::chrono::steady_clock::now();
        stage_ = Stage::WaitingMask;
      }

      if (stage_ == Stage::WaitingMask && masksReady()) {
        finishMaskWait();
        loadMasks();
        DecodeGraphLease lease(impl_);
        encodeCommit(lease.graph);
        // SPLASH_DRAFT_AHEAD_GRAMMAR: the next cycle's draft block runs
        // right behind the commit, as after a regular cycle.
        submit(lease.graph, entries(), launchAhead());
        stage_ = Stage::Commit;
      }
      return requests;
    }

    bool ownsMaskWait(uint64_t requestId) const noexcept override {
      if (stage_ == Stage::Draft || stage_ == Stage::ChainDraft ||
          stage_ == Stage::Done)
        return false;
      return std::any_of(lanes_.begin(), lanes_.end(),
                         [requestId](const DecodeLaneResult &lane) {
                           return lane.request->id == requestId;
                         });
    }

    void abandonMask(uint64_t requestId) noexcept override {
      if (stage_ == Stage::Done)
        return;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (lanes_[lane].request->id == requestId) {
          abandoned_[lane] = true;
          lanes_[lane].request->maskWords.clear();
          return;
        }
      }
    }

    bool ready() const noexcept override {
      return stage_ == Stage::Commit && command_.ready();
    }

    std::vector<ModelStepResult> wait() override {
      if (!ready())
        throw std::logic_error("constrained decode ticket is not complete");
      addTiming(command_.wait());
      stage_ = Stage::Done;
      ModelTelemetry &counters = impl_.counters;
      ++counters.constrainedMaskOverlapBatches;
      counters.constrainedMaskOverlapRequests += lanes_.size();
      counters.lastConstrainedTargetForwardGpuSeconds =
          targetForwardGpuSeconds_;
      counters.totalConstrainedTargetForwardGpuSeconds +=
          targetForwardGpuSeconds_;
      counters.lastConstrainedMaskWaitSeconds = maskWaitSeconds_;
      counters.totalConstrainedMaskWaitSeconds += maskWaitSeconds_;
      return impl_.finalizeDecode(lanes_, std::move(results_), items_, stats_,
                                  planWidth_, timing_);
    }

    double wallMilliseconds() const noexcept override {
      return timing_.wallSeconds * 1000.0;
    }

  private:
    enum class Stage : uint8_t {
      Draft,
      TargetForward,
      ChainDraft,  // SPLASH_GRAMMAR_CHAIN: one gated command, proposals pending
      WaitingMask,
      Commit,
      Done
    };

    // The cycle's anchor, commit bound and mask state, once its proposals are
    // fixed (before the target forward is encoded).
    void beginVerify() {
      for (DecodeLaneResult &laneResult : lanes_) {
        Request &entry = *laneResult.request;
        entry.maskWords.clear();
        entry.verifyMaskInFlight = true;
        const uint32_t remaining = entry.maxNewTokens - entry.generatedTokens;
        laneResult.currentAnchor = *entry.pendingToken;
        laneResult.maximumRetained = std::min(remaining, entry.verifyRows);
        laneResult.verify = true;
      }
    }

    // One grammar simulation request per live lane: the anchor and the
    // proposals (both tiles' inputs for a wide lookup).
    std::vector<ModelMaskRequest> maskRequests() const {
      std::vector<ModelMaskRequest> requests;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (abandoned_[lane])
          continue;
        const DecodeLaneResult &laneResult = lanes_[lane];
        const Request &entry = *laneResult.request;
        ModelMaskRequest request;
        request.requestId = entry.id;
        if (laneResult.wideLookup) {
          const auto *input = contents<uint32_t>(
              impl_.decodeArena->packed(DecodeTensor::InputTokens, laneResult.wideTiles),
              "wide mask simulation");
          request.simulationTokens.assign(input,
                                          input + laneResult.wideTiles * kDecodeRows);
        } else {
          const uint32_t *proposed = contents<uint32_t>(
              impl_.decodeArena->get(lane, DecodeTensor::ProposedTokens),
              "constrained draft proposals");
          request.simulationTokens.reserve(kDecodeRows);
          request.simulationTokens.push_back(laneResult.currentAnchor);
          request.simulationTokens.insert(request.simulationTokens.end(),
                                          proposed, proposed + kDraftProposalTokens);
        }
        requests.push_back(std::move(request));
      }
      return requests;
    }

    std::array<Request *, kLaneCount> entryArray() const {
      std::array<Request *, kLaneCount> result{};
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane)
        result[lane] = lanes_[lane].request;
      return result;
    }

    std::span<Request *const> entries() {
      entries_ = entryArray();
      return {entries_.data(), lanes_.size()};
    }

    // The target forward is independent of the masks.
    void encodeForward(CommandGraph &graph) {
      if (lanes_[0].wideLookup) {
        impl_.encodeLookup16Forward(graph, *lanes_[0].request, items_[0], stats_);
        return;
      }
      const uint32_t width = static_cast<uint32_t>(lanes_.size());
      impl_.encodeBatchVerifyInput(graph, width);
      impl_.encodeBatchEmbedding(graph, DecodeTensor::InputTokens,
                                 DecodeTensor::Hidden0, width);
      impl_.encodeTargetVerifyBatchForward(graph, entries(), items_, stats_);
    }

    // Policy (reads the masks), acceptance and commits.
    void encodeCommit(CommandGraph &graph) {
      std::array<uint32_t, kLaneCount> maximumRetained{};
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane)
        maximumRetained[lane] = lanes_[lane].maximumRetained;
      if (lanes_[0].wideLookup) {
        impl_.encodeLookup16Commit(graph, *lanes_[0].request, items_[0],
                                   maximumRetained[0], stats_);
        return;
      }
      impl_.encodeTargetVerifyBatchPolicy(graph, entries());
      impl_.encodeBatchAcceptance(graph, entries(),
                                  {maximumRetained.data(), lanes_.size()});
      impl_.encodeBatchGdnCommit(graph, entries());
      impl_.encodeDraftStateCommitBatch(graph, entries(), items_, stats_);
    }

    bool masksReady() const {
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (!abandoned_[lane] && lanes_[lane].request->maskWords.empty())
          return false;
      }
      return true;
    }

    void finishMaskWait() {
      maskWaitSeconds_ += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - *maskWaitStarted_)
                              .count();
      maskWaitStarted_.reset();
    }

    // Masks and uniforms for the policy; an abandoned lane gets none.
    void loadMasks() {
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        Request &entry = *lanes_[lane].request;
        const auto masks = abandoned_[lane] ? std::span<const uint32_t>{}
                                            : std::span<const uint32_t>{entry.maskWords};
        if (lanes_[lane].wideLookup)
          impl_.loadLookup16PolicyBuffers(entry, masks);
        else
          impl_.loadPolicyBuffers(entry, lane, masks);
      }
    }

    // SPLASH_DRAFT_AHEAD_GRAMMAR: launch the next cycle's block behind the commit.
    bool launchAhead() {
      return impl_.grammarAhead &&
             std::none_of(abandoned_.begin(), abandoned_.begin() + lanes_.size(),
                          [](bool abandoned) { return abandoned; }) &&
             impl_.aheadLaunchAllowed(entries(), items_, true, lanes_[0].promptLookup);
    }

    void submit(const CommandGraph &graph,
                std::span<Request *const> aheadEntries = {}, bool launch = false,
                const metal::MetalBackend::ChainGates *gates = nullptr) {
      command_ = impl_.submitWithAhead(
          graph.dispatches(),
          [wake = wake_](uint64_t) {
            if (*wake)
              (*wake)();
          },
          aheadEntries, items_, launch, gates);
    }

    void addTiming(CommandTiming value) noexcept {
      timing_.gpuSeconds += value.gpuSeconds;
      timing_.wallSeconds += value.wallSeconds;
    }

    Impl &impl_;
    std::vector<DecodeLaneResult> lanes_;
    std::vector<ModelStepResult> results_;
    std::vector<ModelBatchItem> items_;
    ops::Q4DispatchStats stats_;
    uint32_t planWidth_ = 0;
    bool adopted_ = false;
    std::optional<metal::MetalBackend::ChainGates> chain_;
    bool proposalsKnown_ = false;
    std::array<Request *, kLaneCount> entries_{};
    Stage stage_ = Stage::Draft;
    CommandTicket command_;
    CommandTiming timing_;
    std::array<bool, kLaneCount> abandoned_{};
    double targetForwardGpuSeconds_ = 0.0;
    double maskWaitSeconds_ = 0.0;
    std::optional<std::chrono::steady_clock::time_point> maskWaitStarted_;
    std::shared_ptr<std::function<void()>> wake_;
  };
};

Runtime::Runtime(RuntimeContext context)
    : impl_(std::make_unique<Impl>(context)) {}

Runtime::~Runtime() = default;

void Runtime::checkHealth() { impl_->backend.checkHealth(); }

bool Runtime::needsHealthCheck() const noexcept {
  return impl_->backend.needsHealthCheck();
}

void Runtime::beginColdRequest(const ModelRequest &request,
                               uint32_t stateSlot) {
  if (auto admission = beginAt(request, stateSlot); !admission) {
    throw metal::MetalAllocationError(
        std::string("unable to allocate sequence state cell: ") +
            metal::allocationFailureName(admission.failure), admission.failure);
  }
  try {
    setDraftContextPlan(
        request.id,
        planDraftContext(0, static_cast<uint32_t>(request.prompt.size()),
                         std::nullopt, {}));
  } catch (...) {
    end(request.id);
    throw;
  }
}

StateAdmission Runtime::begin(const ModelRequest &request) {
  Impl::ImageAdmission images(*impl_, request.id);
  StateAdmission admission = admitIdleSlot(impl_->states, [&](uint32_t slot) {
    if (auto imageAdmission = impl_->stageImages(request); !imageAdmission)
      return imageAdmission;
    return beginAt(request, slot);
  });
  images.committed = admission.granted();
  return admission;
}

void Runtime::suspend(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || entry.verifyMaskInFlight) {
    throw std::logic_error("Qwen request cannot be suspended");
  }
  impl_->retireAhead();  // SPLASH_DRAFT_AHEAD: see retireAheadFor
  impl_->states.releaseSlot(entry.slot, requestId);
  static_cast<void>(impl_->states.releaseIdle(0, 0));
  impl_->pageTableBindings[entry.slot] = {};
  entry.images.clear();
  entry.draftContextPlan.reset();
  entry.draftContextValid = false;
  entry.draftContextThrough = 0;
  entry.replayingGeneration |= entry.promptComplete;
  entry.promptComplete = false;
  entry.resident = false;
}

StateAdmission Runtime::resume(const ModelRequest &request) {
  Impl::Request &entry = impl_->request(request.id);
  if (entry.resident) {
    throw std::logic_error("Qwen request is not suspended");
  }
  if (request.prompt.size() < entry.promptTokens) {
    throw std::invalid_argument("recomputed history cannot shorten the prompt");
  }
  Impl::ImageAdmission images(*impl_, request.id);
  StateAdmission admission =
      admitIdleSlot(impl_->states, [&](uint32_t slot) {
        if (auto imageAdmission = impl_->stageImages(request); !imageAdmission)
          return imageAdmission;
        return impl_->states.tryActivateSlot(slot, request.id);
      });
  if (admission.granted()) {
    entry.slot = *admission.cell;
    entry.resident = true;
    entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
    if (auto staged = impl_->stagedImages.find(request.id);
        staged != impl_->stagedImages.end()) {
      entry.images = std::move(staged->second);
      impl_->stagedImages.erase(staged);
    }
  }
  images.committed = admission.granted();
  return admission;
}

metal::AllocationResult Runtime::beginAt(const ModelRequest &request, uint32_t stateSlot) {
  if (!request.id || stateSlot >= kLaneCount || request.prompt.empty()) {
    throw std::invalid_argument("invalid executor request activation");
  }
  if (impl_->requests.contains(request.id)) {
    throw std::logic_error("request is already active");
  }
  Impl::Request entry;
  entry.id = request.id;
  entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
  entry.maxNewTokens = request.maxNewTokens;
  entry.cohort = request.cohort;
  entry.sampling = request.sampling;
  entry.constraint = request.constraint;
  entry.draftHeadSegment =
      impl_->draftModel.promptSegment(request.prompt, &entry.draftHeadSegmentUsed);
  if (impl_->promptLookupEnabled && request.scoreTokens.empty() && request.images.empty() &&
      request.imagePixels.empty()) {
    const size_t count = std::min(request.prompt.size(), kPromptLookupHistory);
    entry.lookupHistory.reserve(kPromptLookupHistory + kDecodeRows);
    entry.lookupHistory.assign(request.prompt.end() - count, request.prompt.end());
  }
  const BatchCohort expected =
      entry.constraint == ConstraintMode::TokenMask
          ? BatchCohort::Constrained
          : (Impl::samplingEnabled(entry) ? BatchCohort::Sampling
                                          : BatchCohort::Greedy);
  if (entry.cohort != expected || !std::isfinite(entry.sampling.temperature) ||
      entry.sampling.temperature < 0.0F ||
      !std::isfinite(entry.sampling.topP) || entry.sampling.topP <= 0.0F ||
      entry.sampling.topP > 1.0F ||
      entry.sampling.topK > ops::kTargetSamplingCandidates ||
      (Impl::samplingEnabled(entry) && !entry.sampling.topK)) {
    throw std::invalid_argument("request sampling/cohort contract is invalid");
  }
  if (!request.scoreTokens.empty()) {
    if (request.maxNewTokens != 0 ||
        request.constraint != ConstraintMode::None ||
        request.cohort != BatchCohort::Greedy || !request.images.empty() ||
        !request.imagePixels.empty() ||
        request.scoreTokens.size() < ExecutionLimits::minimumScoreOptions ||
        request.scoreTokens.size() > ExecutionLimits::maximumScoreOptions) {
      throw std::invalid_argument("invalid score request");
    }
    std::vector<uint32_t> distinct(request.scoreTokens.begin(),
                                   request.scoreTokens.end());
    std::sort(distinct.begin(), distinct.end());
    if (std::adjacent_find(distinct.begin(), distinct.end()) !=
            distinct.end() ||
        std::any_of(distinct.begin(), distinct.end(), [&](uint32_t token) {
          return token >= impl_->geometry.target.vocabularySize;
        })) {
      throw std::invalid_argument("score token is out of vocabulary");
    }
    entry.scoreTokens.assign(request.scoreTokens.begin(),
                             request.scoreTokens.end());
  }
  entry.decodeStage = entry.cohort == BatchCohort::Constrained
                          ? DecodeStage::RequestInitialMask
                          : DecodeStage::Regular;
  if (auto admission = impl_->states.tryActivateSlot(stateSlot, request.id);
      !admission)
    return admission;
  entry.slot = stateSlot;
  entry.resident = true;
  if (auto staged = impl_->stagedImages.find(request.id);
      staged != impl_->stagedImages.end()) {
    entry.images = std::move(staged->second);
    impl_->stagedImages.erase(staged);
  }
  // SPLASH_WIDE_LOOKUP32=alt: every other admitted request (in-run A/B).
  entry.wide32 = impl_->wide32Mode &&
                 (std::strcmp(impl_->wide32Mode, "1") == 0 || impl_->wide32Admitted++ % 2 == 1);
  entry.wideGdn = !impl_->wideGdnAlternate || impl_->wideGdnAdmitted++ % 2 == 1
                     ? impl_->wideGdnRoute
                     : ops::WideGdn::Chain;
  try {
    auto [_, inserted] = impl_->requests.emplace(request.id, std::move(entry));
    if (!inserted) {
      throw std::logic_error("request insertion lost uniqueness");
    }
  } catch (...) {
    impl_->states.releaseSlot(stateSlot, request.id);
    throw;
  }
  return true;
}

void Runtime::restore(uint64_t requestId, uint32_t restoredPrefixLength,
                      std::shared_ptr<const CompositeState> restoredState,
                      bool restoreDraftState) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || !restoredState) {
    throw std::invalid_argument("cannot restore a nonresident request");
  }
  if (restoredPrefixLength >= entry.promptTokens) {
    throw std::invalid_argument(
        "reusable Qwen prefix must leave an input token to replay");
  }
  impl_->retireAhead();  // SPLASH_DRAFT_AHEAD: its rings may be rewritten
  impl_->states.restore(entry.slot, *restoredState, restoreDraftState);
  if (!restoreDraftState)
    ++impl_->counters.draftStateRestoreSkipped;
  const QwenLogicalLengths &lengths =
      impl_->states.metadata(entry.slot).lengths;
  if (lengths.targetTokens != restoredPrefixLength ||
      (restoreDraftState &&
       !lengths.hasCompleteDraftWindow(kDraftCacheStride)) ||
      (!restoreDraftState && lengths.draftLength != 0)) {
    throw std::invalid_argument("prefix logical length does not match state");
  }
  // Images fully inside the restored prefix are never encoded; their spans
  // stay because rotary positions after them depend on their grids.
  for (Impl::ImageState &image : entry.images) {
    if (image.span.end() <= restoredPrefixLength) {
      image.data.reset();
    }
  }
  entry.promptComplete = false;
  if (!entry.replayingGeneration) {
    entry.finalTargetHidden.clear();
    entry.pendingToken.reset();
  }
  entry.draftContextValid = restoreDraftState;
  entry.draftContextThrough = restoreDraftState ? restoredPrefixLength : 0;
  entry.draftContextPlan.reset();
}

void Runtime::setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || plan.replayEnd != entry.promptTokens) {
    throw std::invalid_argument("draft context plan does not match request");
  }
  const uint64_t current =
      impl_->states.metadata(entry.slot).lengths.targetTokens;
  if (plan.replayBegin != current ||
      plan.restoredDraftBoundary !=
          (current ? std::optional<uint32_t>(static_cast<uint32_t>(current))
                   : std::nullopt)) {
    throw std::invalid_argument("draft context plan restore boundary is stale");
  }
  entry.draftContextPlan = std::move(plan);
}

std::vector<ModelStepResult>
Runtime::prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return prefillAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                std::function<void()> completion) {
  switch (plan.kind) {
  case WorkKind::Prefill:
    return prefillAsync(plan, items, std::move(completion));
  case WorkKind::Decode:
    return decodeAsync(plan, items, std::move(completion));
  }
  throw std::logic_error("unknown model work kind");
}

std::unique_ptr<ModelBatchTicket>
Runtime::prefillAsync(const BatchPlan &plan,
                      std::span<const ModelBatchItem> items,
                      std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Prefill);
  if (plan.decodeStage != DecodeStage::Regular) {
    throw std::invalid_argument("Qwen prefill cannot resume a mask plan");
  }
  impl_->retireAhead();  // SPLASH_DRAFT_AHEAD: prefill writes lane buffers

  std::array<Impl::Request *, kLaneCount> entries{};
  CommandGraph graph;
  impl_->encodePackedPrefillGraph(graph, items, entries);
  const bool encodesImages = std::any_of(
      entries.begin(), entries.begin() + items.size(), [](const auto *entry) {
        return std::any_of(entry->images.begin(), entry->images.end(),
                           [](const auto &image) {
                             return image.data && image.data->encoding;
                           });
      });
  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  auto notify = [completion = std::move(completion)](uint64_t) {
    if (completion)
      completion();
  };
  CommandTicket command =
      impl_->backend.submitCommandAsync(graph.dispatches(), std::move(notify));
  Impl *impl = impl_.get();
  auto finish = [impl, entries,
                 items = std::move(copiedItems)](CommandTiming timing) mutable {
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      for (Impl::ImageState &image : entries[lane]->images) {
        if (!image.data || !image.data->encoding)
          continue;
        image.data->encoding = false;
        image.data->encoded = true;
        image.data->pixels = MetalBuffer{};
      }
    }

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      Impl::Request &entry = *entries[lane];
      const ModelBatchItem &item = items[lane];
      const auto captures = Impl::activeDraftCaptures(entry, item);
      const uint32_t capturedRows = Impl::captureRows(captures);
      uint32_t activeRows = 0;
      uint32_t materializationRows = 0;
      for (const auto &capture : captures) {
        activeRows += capture.activeRows;
        materializationRows += capture.materializationRows;
      }
      if (activeRows + materializationRows != capturedRows) {
        throw std::logic_error("draft capture telemetry is inconsistent");
      }
      impl->counters.targetPrefillRows += item.tokenCount;
      impl->counters.draftContextRowsActive += activeRows;
      impl->counters.draftContextRowsMaterialization += materializationRows;
      impl->counters.draftContextRowsAvoided += item.tokenCount - capturedRows;
      for (const auto &capture : captures) {
        const bool continues =
            !capture.resetDraftState && entry.draftContextValid &&
            entry.draftContextThrough == capture.absoluteBegin;
        if (!continues) {
          ++impl->counters.draftStateResets;
        }
        entry.draftContextValid = true;
        entry.draftContextThrough = capture.absoluteEnd;
      }
      impl->states.swapParity(entry.slot);
      uint64_t nextLength = item.logicalPosition + item.tokenCount;
      QwenLogicalLengths lengths = impl->states.metadata(entry.slot).lengths;
      lengths.targetTokens = nextLength;
      for (const auto &capture : captures) {
        lengths = Impl::advanceDraftContext(lengths, nextLength, capture);
      }
      impl->states.updateLengths(entry.slot, lengths);
      entry.promptComplete = nextLength == entry.promptTokens;
      ModelStepResult result{entry.id, item.tokenCount, {}, false,
                             entry.decodeStage, 0, 0};
      if (entry.promptComplete && !entry.replayingGeneration) {
        entry.pendingToken.reset();
        if (!entry.scoreTokens.empty()) {
          // Score-only: read the raw bf16 logits at the final prompt position
          // (row lastRows-1 of the gathered head input) in requested order.
          const uint32_t lastRows = std::min(item.tokenCount, kDecodeRows);
          const uint16_t *logits = contents<uint16_t>(
              impl->decodeArena->get(lane, DecodeTensor::Logits),
              "score logits");
          const uint16_t *row =
              logits + uint64_t{lastRows - 1} *
                           impl->geometry.target.vocabularySize;
          result.scoreLogits.reserve(entry.scoreTokens.size());
          for (uint32_t token : entry.scoreTokens) {
            const float logit =
                std::bit_cast<float>(uint32_t{row[token]} << 16);
            if (!std::isfinite(logit)) {
              // A numerical outcome for this request, not a broken invariant:
              // report it as a lane failure so the engine drops this request
              // before cache publication or output and the batch survives.
              result.scoreLogits.clear();
              result.failure = "score logit is not finite";
              break;
            }
            result.scoreLogits.push_back(logit);
          }
          result.finished = true;
        } else if (entry.constraint == ConstraintMode::None) {
          entry.pendingToken = *contents<uint32_t>(
              impl->decodeArena->get(lane, DecodeTensor::OutputTokens),
              "prefill next token");
          if (!entry.pendingToken ||
              *entry.pendingToken >=
                  impl->geometry.target.vocabularySize) {
            throw std::runtime_error(
                "prefill policy selected an invalid token");
          }
          impl->emitTerminalAnchor(entry, result);
        } else {
          const uint32_t lastRows = std::min(item.tokenCount, kDecodeRows);
          impl->captureFinalHidden(
              entry, impl->decodeArena->get(lane, DecodeTensor::Hidden0),
              lastRows - 1);
        }
      }
      if (entry.promptComplete)
        entry.replayingGeneration = false;
      results.push_back(std::move(result));
    }
    impl->counters.lastPrefillWallSeconds = timing.wallSeconds;
    impl->counters.totalPrefillWallSeconds += timing.wallSeconds;
    impl->counters.lastPrefillGpuSeconds = timing.gpuSeconds;
    impl->counters.totalPrefillGpuSeconds += timing.gpuSeconds;
    return results;
  };
  return std::make_unique<DeferredMetalTicket>(std::move(command),
                                               std::move(finish), 0.0,
                                               !encodesImages);
}

std::vector<ModelStepResult>
Runtime::decode(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return decodeAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::decodeAsync(const BatchPlan &plan,
                     std::span<const ModelBatchItem> items,
                     std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Decode);
  const bool constrained = plan.cohort == BatchCohort::Constrained;
  if (plan.decodeStage != DecodeStage::Regular && !constrained) {
    throw std::invalid_argument(
        "only constrained decode uses a specialized decode stage");
  }
  // SPLASH_DRAFT_AHEAD: a block built for other lanes, positions, stages or
  // head rows finishes before any lane buffer is written.
  if (impl_->ahead && !impl_->aheadMayMatch(plan, items))
    impl_->retireAhead();

  std::vector<Impl::DecodeLaneResult> lanes(items.size());
  std::vector<ModelStepResult> results(items.size());
  ops::Q4DispatchStats batchStats;
  CommandTiming priorTiming;
  for (uint32_t lane = 0; lane < items.size(); ++lane) {
    const ModelBatchItem &item = items[lane];
    Impl::Request &entry = impl_->request(item.requestId);
    if ((entry.cohort == BatchCohort::Constrained) != constrained) {
      throw std::invalid_argument("request does not belong to batch cohort");
    }
    if (entry.decodeStage != plan.decodeStage) {
      throw std::logic_error("request decode stage does not match decode plan");
    }
    Impl::DecodeLaneResult &laneResult = lanes[lane];
    laneResult.request = &entry;
    results[lane].requestId = entry.id;

    if (constrained && plan.decodeStage == DecodeStage::RequestInitialMask) {
      if (entry.pendingToken || !entry.maskWords.empty()) {
        throw std::logic_error("initial mask request has stale decode state");
      }
      entry.decodeStage = DecodeStage::ApplyInitialMask;
      results[lane].nextDecodeStage = DecodeStage::ApplyInitialMask;
      continue;
    }

    if (constrained && plan.decodeStage == DecodeStage::ApplyInitialMask) {
      if (entry.maskWords.size() != impl_->geometry.maskWords() ||
          entry.pendingToken) {
        throw std::logic_error("initial anchor mask state is invalid");
      }
      const CommandTiming selection =
          impl_->selectPendingFromFinalHidden(entry, lane, entry.maskWords);
      priorTiming.gpuSeconds += selection.gpuSeconds;
      priorTiming.wallSeconds += selection.wallSeconds;
      entry.maskWords.clear();
      entry.decodeStage = DecodeStage::Regular;
      results[lane].nextDecodeStage = DecodeStage::Regular;
      if (impl_->emitTerminalAnchor(entry, results[lane]))
        continue;
    }

    if (!entry.pendingToken)
      throw std::logic_error("decode request has no current anchor");
    const uint32_t remaining = entry.maxNewTokens - entry.generatedTokens;
    if (!remaining)
      throw std::logic_error("completed request was decoded");
    if (isStopToken(impl_->geometry, *entry.pendingToken) || remaining == 1) {
      throw std::logic_error("terminal anchor was not emitted on selection");
    }

    if (constrained) {
      if (!entry.maskWords.empty()) {
        throw std::logic_error("constrained request has stale mask state");
      }
      laneResult.draftForMask = true;
      if (Impl::samplingEnabled(entry))
        Impl::stageSamplingCycle(entry);
      laneResult.draftComputed = true;
      continue;  // lane buffers are written after the draft-ahead choice below
    }

    if (Impl::samplingEnabled(entry))
      Impl::stageSamplingCycle(entry);

    // DFlash has one physical graph: anchor + seven proposal rows. A shorter
    // output budget only lowers the token-exact commit count; it never
    // changes the Metal graph shape.
    laneResult.currentAnchor = *entry.pendingToken;
    laneResult.maximumRetained = std::min(remaining, kDecodeRows);
    laneResult.draftComputed = true;
    laneResult.verify = true;
  }

  Impl::DecodeGraphLease graphLease(*impl_);
  CommandGraph &commandGraph = graphLease.graph;
  if (impl_->streamChunk && !constrained)
    commandGraph.streamAt(impl_->streamChunk,
                          [impl = impl_.get()](std::span<const metal::ComputeDispatch> head) {
                            impl->backend.streamHead(head);
                          });
  uint32_t verified = 0;
  uint32_t draftComputed = 0;
  for (const Impl::DecodeLaneResult &lane : lanes)
    verified += lane.verify ? 1U : 0U;
  for (const Impl::DecodeLaneResult &lane : lanes)
    draftComputed += lane.draftComputed ? 1U : 0U;
  if (verified && verified != lanes.size()) {
    throw std::logic_error("decode batch mixed mask and verify phases");
  }
  const uint32_t width = static_cast<uint32_t>(lanes.size());
  if (draftComputed && draftComputed != lanes.size()) {
    throw std::logic_error("decode batch mixed draft execution phases");
  }
  // Lookup choice first, buffer writes after: an adoptable draft-ahead
  // block must not see a host write to the lane buffers it uses.
  std::optional<Impl::WideProposal> wideProposal;
  std::optional<std::array<uint32_t, 7>> narrowProposal;
  if (draftComputed && width == 1 && (verified || lanes[0].draftForMask)) {
    metal::hostPhase("lk0");
    wideProposal = impl_->lookup16Proposal(*lanes[0].request, items[0]);
    if (!wideProposal)
      narrowProposal = impl_->promptLookupProposal(*lanes[0].request);
    // kind: 3 wide 32-row lookup, 2 wide lookup, 1 narrow lookup, 0 none (drafter)
    metal::hostPhase("lk1", wideProposal ? (wideProposal->tiles == 4 ? 3 : 2)
                            : narrowProposal ? 1 : 0,
                     static_cast<long long>(lanes[0].request->lookupHistory.size()));
  }
  const bool wideLookup = wideProposal.has_value();
  const bool lookup = wideLookup || narrowProposal.has_value();
  const bool maskDraft =
      draftComputed && !verified &&
      std::all_of(lanes.begin(), lanes.end(),
                  [](const auto &lane) { return lane.draftForMask; });
  bool adopt = false;
  if (impl_->ahead) {
    adopt = (verified || (maskDraft && impl_->grammarAhead)) && !lookup;
    for (uint32_t lane = 0; adopt && lane < width; ++lane)
      adopt = lanes[lane].request->cycleUniforms == impl_->ahead->uniforms[lane];
    for (const Impl::DecodeLaneResult &lane : lanes) {
      lane.request->aheadAdopted += adopt ? 1 : 0;
      lane.request->aheadLookupDrops += lookup ? 1 : 0;
    }
    if (adopt) {
      // Its dispatches count toward this cycle, as when encoded inline.
      batchStats = impl_->ahead->stats;
      impl_->ahead.reset();
    } else {
      impl_->retireAhead();
    }
  }
  metal::hostPhase("lanes", adopt ? 1 : 0);
  for (uint32_t lane = 0; lane < width; ++lane) {
    if (!lanes[lane].verify && !lanes[lane].draftForMask)
      continue;
    impl_->prepareDecodeLane(*lanes[lane].request, items[lane], lane, !adopt);
    if (!adopt)
      impl_->loadPolicyBuffers(*lanes[lane].request, lane, {});
  }
  if (wideLookup) {
    impl_->prepareLookup16(*lanes[0].request, items[0], *wideProposal);
  } else if (narrowProposal)
    impl_->preparePromptLookup(*narrowProposal);
  if (lookup)
    lanes[0].promptLookup = true;
  if (wideLookup) {
    lanes[0].wideLookup = true;
    lanes[0].wideTiles = wideProposal->tiles;
    const auto &entry = *lanes[0].request;
    lanes[0].maximumRetained = std::min(entry.maxNewTokens - entry.generatedTokens,
                                        wideProposal->tiles * kDecodeRows);
  }
  const uint32_t physicalWidth = wideLookup ? wideProposal->tiles : width;
  if (draftComputed || verified) {
    const uint32_t ropeRows = physicalWidth * kDecodeRows;
    impl_->addRopeTables(
        commandGraph,
        impl_->decodeArena->packed(DecodeTensor::Positions, physicalWidth), ropeRows,
        impl_->decodeArena->packed(DecodeTensor::DraftPositions, physicalWidth),
        ropeRows, impl_->decodeArena->packed(DecodeTensor::RopeCos, physicalWidth),
        impl_->decodeArena->packed(DecodeTensor::RopeSin, physicalWidth),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeCos, physicalWidth),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeSin, physicalWidth));
  }
  if (draftComputed && !lookup && !adopt) {
    std::array<Impl::Request *, kLaneCount> requests{};
    std::array<uint64_t, kLaneCount> logicalPositions{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      requests[lane] = lanes[lane].request;
      logicalPositions[lane] = items[lane].logicalPosition;
    }
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::DraftInputTokens,
                                DecodeTensor::DraftHidden0, width);
    impl_->encodeDraftBatchGraph(commandGraph, {requests.data(), lanes.size()},
                                 {logicalPositions.data(), lanes.size()},
                                 batchStats);
  }
  if (verified && wideLookup) {
    impl_->encodeLookup16Forward(commandGraph, *lanes[0].request, items[0], batchStats);
    impl_->encodeLookup16Commit(commandGraph, *lanes[0].request, items[0],
                                lanes[0].maximumRetained, batchStats);
  } else if (verified) {
    impl_->encodeBatchVerifyInput(commandGraph, width);
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::InputTokens,
                                DecodeTensor::Hidden0, width);
    std::array<Impl::Request *, kLaneCount> requests{};
    std::array<uint32_t, kLaneCount> maximumRetained{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      requests[lane] = lanes[lane].request;
      maximumRetained[lane] = lanes[lane].maximumRetained;
    }
    impl_->encodeTargetVerifyBatchForward(
        commandGraph, {requests.data(), lanes.size()}, items, batchStats);
    impl_->encodeTargetVerifyBatchPolicy(commandGraph,
                                         {requests.data(), lanes.size()});
    impl_->encodeBatchAcceptance(commandGraph, {requests.data(), lanes.size()},
                                 {maximumRetained.data(), lanes.size()});
    impl_->encodeBatchGdnCommit(commandGraph, {requests.data(), lanes.size()});
    const bool seamSibling =
        Impl::seamSiblingEnabled() && lanes.size() == 1 && !constrained;
    // B2 only for now (lead 11:1x): B3/B4 keep the stock projections until they are timed in the engine.
    const bool groupedLanes =
        Impl::groupedContextKvMultiLane() && lanes.size() == 2 && !constrained;
    const size_t seamBegin = commandGraph.dispatches().size();
    impl_->encodeDraftStateCommitBatch(
        commandGraph, {requests.data(), lanes.size()}, items, batchStats, {},
        seamSibling, groupedLanes);
    if (seamSibling) impl_->placeSeamSiblings(commandGraph, seamBegin);
  }

  const bool overlapConstraintMask =
      constrained && !lanes.empty() &&
      std::all_of(lanes.begin(), lanes.end(),
                  [](const auto &lane) { return lane.draftForMask; });
  if (overlapConstraintMask) {
    return std::make_unique<Impl::ConstrainedDecodeTicket>(
        *impl_, std::move(lanes), std::move(results), items, batchStats,
        plan.width(), priorTiming, commandGraph, adopt,
        impl_->grammarChain && lanes.size() == 1, std::move(completion));
  }

  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  const uint32_t planWidth = plan.width();
  Impl *impl = impl_.get();
  auto finish = [impl, lanes = std::move(lanes), results = std::move(results),
                 items = std::move(copiedItems), batchStats, planWidth,
                 priorTiming](CommandTiming timing) mutable {
    timing.gpuSeconds += priorTiming.gpuSeconds;
    timing.wallSeconds += priorTiming.wallSeconds;
    return impl->finalizeDecode(lanes, std::move(results), items, batchStats,
                                planWidth, timing);
  };

  if (commandGraph.empty()) {
    std::vector<ModelStepResult> ready = finish(CommandTiming{});
    return std::make_unique<ReadyModelTicket>(std::move(ready),
                                              priorTiming.wallSeconds * 1000.0);
  }

  auto notify = [completion = std::move(completion)](uint64_t) {
    if (completion)
      completion();
  };
  // SPLASH_DRAFT_AHEAD: after a drafter cycle, the next cycle's draft block
  // is built once this command is committed and runs right behind it.
  std::array<Impl::Request *, kLaneCount> aheadRequests{};
  for (uint32_t lane = 0; lane < width; ++lane)
    aheadRequests[lane] = &impl->request(items[lane].requestId);
  const std::span<Impl::Request *const> aheadEntries(aheadRequests.data(), width);
  CommandTicket command = impl_->submitWithAhead(
      commandGraph.dispatches(), std::move(notify), aheadEntries, items,
      impl_->aheadLaunchAllowed(aheadEntries, items, verified != 0, lookup));
  return std::make_unique<DeferredMetalTicket>(
      std::move(command), std::move(finish), priorTiming.wallSeconds * 1000.0);
}

std::shared_ptr<const CompositeState> Runtime::snapshot(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident)
    throw std::logic_error("request is not resident");
  const QwenSlotMetadata &metadata = impl_->states.metadata(entry.slot);
  if (!metadata.lengths.hasCompleteDraftWindow(kDraftCacheStride) ||
      metadata.lengths.targetTokens % kv::kPageTokens) {
    throw std::logic_error("cannot snapshot uncommitted draft state");
  }
  return impl_->states.snapshot(entry.slot);
}

uint64_t Runtime::reclaimIdleState() noexcept {
  // One idle buffer per call, so a denied allocation frees only what it
  // needs; rebuildable caches go once the pool is empty.
  const uint32_t cells = impl_->states.idleCells();
  const uint32_t rings = impl_->states.idleRings();
  if (cells)
    return impl_->states.releaseIdle(cells - 1, rings);
  if (rings)
    return impl_->states.releaseIdle(0, rings - 1);
  uint64_t released = 0;
  released += impl_->dropEmbeddingCache();
  if (impl_->vision && impl_->visionIdle()) {
    released += impl_->vision->arenaBytes();
    impl_->vision.reset();
  }
  return released;
}

void Runtime::provideMask(uint64_t requestId, std::span<const uint32_t> words) {
  Impl::Request &entry = impl_->request(requestId);
  const bool acceptsMask =
      waitsForMask(entry.decodeStage) || entry.verifyMaskInFlight;
  // Initial-mask replies can race resource preemption. They belong to the
  // host continuation, not the released device state.
  if (entry.constraint != ConstraintMode::TokenMask || !acceptsMask ||
      !entry.maskWords.empty()) {
    throw std::logic_error("request is not waiting for a token mask");
  }
  const uint32_t maskWords = impl_->geometry.maskWords();
  uint64_t expected = entry.verifyMaskInFlight
                          ? uint64_t{entry.verifyRows + 1} * maskWords
                          : maskWords;
  if (words.size() != expected) {
    throw std::invalid_argument("token mask has the wrong word count");
  }
  const uint32_t rows = static_cast<uint32_t>(words.size() / maskWords);
  for (uint32_t row = 0; row < rows; ++row) {
    auto begin = words.begin() + uint64_t{row} * maskWords;
    if (std::none_of(begin, begin + maskWords,
                     [](uint32_t word) { return word != 0; })) {
      throw std::invalid_argument("token mask row permits no vocabulary token");
    }
  }
  if (entry.verifyMaskInFlight) {
    if (!entry.pendingToken || (words[*entry.pendingToken / 32] &
                                (1U << (*entry.pendingToken % 32))) == 0) {
      throw std::invalid_argument(
          "verify mask is not synchronized to the pending anchor");
    }
  }
  entry.maskWords.assign(words.begin(), words.end());
}

void Runtime::end(uint64_t requestId) {
  impl_->stagedImages.erase(requestId);
  auto found = impl_->requests.find(requestId);
  if (found == impl_->requests.end())
    return;
  for (const Impl::ImageState &image : found->second.images)
    impl_->retainEmbeddings(image);
  impl_->retireAheadFor(requestId);  // SPLASH_DRAFT_AHEAD
  if (found->second.resident) {
    impl_->states.releaseSlot(found->second.slot, requestId);
  }
  if (!found->second.lookupHistory.empty()) {
    std::fprintf(stderr,
                 "prompt_lookup request=%llu cycles=%llu hits=%llu retained=%llu\n",
                 static_cast<unsigned long long>(requestId),
                 static_cast<unsigned long long>(found->second.lookupCycles),
                 static_cast<unsigned long long>(found->second.lookupHits),
                 static_cast<unsigned long long>(found->second.lookupRetained));
    if (impl_->wideLookupEnabled) {
      std::fprintf(stderr, "wide_lookup request=%llu hits=%llu retained=%llu wide32=%d hits32=%llu gdn_single=%d gdn_cycles=%llu\n",
                   static_cast<unsigned long long>(requestId),
                   static_cast<unsigned long long>(found->second.wideLookupHits),
                   static_cast<unsigned long long>(found->second.wideLookupRetained),
                   found->second.wide32 ? 1 : 0,
                   static_cast<unsigned long long>(found->second.wide32Hits),
                   static_cast<int>(found->second.wideGdn),
                   static_cast<unsigned long long>(found->second.wideGdnCycles));
    }
  }
  if (impl_->draftAhead) {
    std::fprintf(stderr, "draft_ahead request=%llu launched=%llu adopted=%llu lookup_drops=%llu\n",
                 static_cast<unsigned long long>(requestId),
                 static_cast<unsigned long long>(found->second.aheadLaunched),
                 static_cast<unsigned long long>(found->second.aheadAdopted),
                 static_cast<unsigned long long>(found->second.aheadLookupDrops));
  }
  impl_->requests.erase(found);
}

namespace {

WarmupStepResult warmupResult(uint64_t estimatedPeakBytes, double wallSeconds,
                              std::string detail) {
  if (!estimatedPeakBytes) {
    throw std::logic_error("warmup peak estimate must be nonzero");
  }
  if (!(wallSeconds > 0.0) || !std::isfinite(wallSeconds)) {
    throw std::logic_error("warmup wall time must be finite and positive");
  }
  return {true, estimatedPeakBytes, std::move(detail), wallSeconds, {}};
}

std::vector<uint32_t> warmupPages(kv::PageStorage &storage, uint32_t first,
                                  uint32_t count) {
  if (!count || uint64_t{first} + count > storage.pageCount()) {
    throw std::invalid_argument("warmup KV page range is unavailable");
  }
  std::vector<uint32_t> result(count);
  for (uint32_t index = 0; index < count; ++index) {
    const uint32_t page = first + index;
    if (auto admission = storage.ensureResident(page); !admission) {
      throw metal::MetalAllocationError(
          std::string("warmup could not reserve KV page backing: ") +
              metal::allocationFailureName(admission.failure), admission.failure);
    }
    result[index] = page;
  }
  return result;
}

} // namespace

void Runtime::prepareWarmupDecode(uint64_t requestId, uint32_t anchor) {
  // Teacher-force a valid input so EOS selected by synthetic prefill cannot
  // prevent the warmup from exercising the real draft/verify/commit graph.
  while (anchor < impl_->geometry.target.vocabularySize &&
         isStopToken(impl_->geometry, anchor))
    ++anchor;
  if (anchor >= impl_->geometry.target.vocabularySize)
    throw std::logic_error("decode warmup has no non-terminal input token");
  auto &entry = impl_->request(requestId);
  entry.pendingToken = anchor;
  entry.generatedTokens = 0;
}

WarmupStepResult Runtime::warmupPrefill(uint32_t rows) {
  using Clock = std::chrono::steady_clock;
  if (!rows || rows > kPrefillRows)
    throw std::invalid_argument("invalid prefill warmup row count");
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 100;
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::vector<uint32_t> warmupPrompt(rows, 0);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 16;
  beginColdRequest(request, 0);
  try {
    std::vector<uint32_t> pages = warmupPages(
        impl_->kvPages, 0, (rows + kv::kPageTokens - 1) / kv::kPageTokens);
    BatchPlan plan{WorkKind::Prefill,
                   BatchCohort::Greedy,
                   {{id, rows}},
                   DecodeStage::Regular};
    ModelBatchItem item{id, 0, 0, 0, rows, pages};
    item.inputTokens = request.prompt;
    const auto phaseStart = Clock::now();
    auto result = prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    wallSeconds = std::chrono::duration<double>(Clock::now() - phaseStart).count();
    if (result.size() != 1 || result[0].consumedPromptTokens != rows) {
      throw std::runtime_error("prefill warmup result mismatch");
    }
    lanes.push_back({std::move(result[0]), impl_->request(id).pendingToken,
                     impl_->states.metadata(0).lengths.targetTokens});
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  auto result = warmupResult(impl_->estimatedWarmupPeak(), wallSeconds,
                            "real " + std::to_string(rows) +
                                "-row packed KV target+draft prefill [M32]");
  result.lanes = std::move(lanes);
  return result;
}

WarmupStepResult Runtime::warmupDecodeBatch(uint32_t width) {
  using Clock = std::chrono::steady_clock;
  if (!width || width > kLaneCount) {
    throw std::invalid_argument("invalid decode warmup width");
  }
  constexpr uint64_t firstId = std::numeric_limits<uint64_t>::max() - 110;
  // Plan order is deliberately unrelated to physical slot order. DecodeArena
  // lanes belong to the explicit BatchPlan, while recurrent/KV state remains
  // addressed by each item.stateSlot; batching must never assume slot 0..3.
  constexpr std::array<uint32_t, kLaneCount> slotOrder{2, 0, 3, 1};
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::array<std::vector<uint32_t>, kLaneCount> pages;
  try {
    for (uint32_t lane = 0; lane < width; ++lane) {
      std::vector<uint32_t> warmupPrompt{lane};
      ModelRequest request;
      request.id = firstId + lane;
      request.prompt = warmupPrompt;
      request.maxNewTokens = 16;
      beginColdRequest(request, slotOrder[lane]);
      pages[lane] = warmupPages(impl_->kvPages, 5 + lane, 1);
      BatchPlan prefillPlan{WorkKind::Prefill,
                            BatchCohort::Greedy,
                            {{request.id, 1}},
                            DecodeStage::Regular};
      ModelBatchItem item{request.id, slotOrder[lane], 0, 0, 1, pages[lane]};
      item.inputTokens = request.prompt;
      static_cast<void>(
          prefill(prefillPlan, std::span<const ModelBatchItem>(&item, 1)));
      prepareWarmupDecode(request.id, warmupPrompt.back());
    }
    BatchPlan plan;
    plan.kind = WorkKind::Decode;
    plan.cohort = BatchCohort::Greedy;
    std::vector<ModelBatchItem> items;
    for (uint32_t lane = 0; lane < width; ++lane) {
      plan.items.push_back({firstId + lane, 0});
      items.push_back({firstId + lane, slotOrder[lane], 1, 0, 0, pages[lane]});
    }
    const auto phaseStart = Clock::now();
    auto decoded = decode(plan, items);
    wallSeconds = std::chrono::duration<double>(Clock::now() - phaseStart).count();
    bool committedEveryLane = decoded.size() == width;
    for (uint32_t lane = 0; committedEveryLane && lane < width; ++lane) {
      const auto &lengths = impl_->states.metadata(slotOrder[lane]).lengths;
      committedEveryLane = !decoded[lane].outputTokens.empty() &&
                           lengths.targetTokens > 1 &&
                           lengths.targetTokens ==
                               1 + decoded[lane].outputTokens.size() -
                                   decoded[lane].outputTokensWithoutKv &&
                           lengths.hasCompleteDraftWindow(kDraftCacheStride);
    }
    const bool fusedWidth =
        width == 1 ||
        (width == 2 && impl_->counters.lastDecodeFusedOperations &&
         impl_->counters.lastDecodeM16Dispatches) ||
        (width == 3 && impl_->counters.lastDecodeFusedOperations &&
         impl_->counters.lastDecodeM24Dispatches) ||
        (width == 4 && impl_->counters.lastDecodeFusedOperations &&
         impl_->counters.lastDecodeM32Dispatches);
    const bool fusedMaximum =
        width != kLaneCount || (impl_->counters.lastDecodeM32Dispatches > 0 &&
                                impl_->counters.lastDecodeM16Dispatches == 0);
    if (!committedEveryLane || !fusedWidth || !fusedMaximum ||
        impl_->counters.lastDecodeWidth != width) {
      throw std::runtime_error(
          "decode warmup B" + std::to_string(width) +
          " mismatch [committed=" + std::to_string(committedEveryLane) +
          ",fused=" + std::to_string(fusedWidth) +
          ",maximum=" + std::to_string(fusedMaximum) +
          ",m16=" + std::to_string(impl_->counters.lastDecodeM16Dispatches) +
          ",m24=" + std::to_string(impl_->counters.lastDecodeM24Dispatches) +
          ",m32=" + std::to_string(impl_->counters.lastDecodeM32Dispatches) +
          "]");
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
      lanes.push_back({std::move(decoded[lane]),
                       impl_->request(firstId + lane).pendingToken,
                       impl_->states.metadata(slotOrder[lane]).lengths.targetTokens});
      end(firstId + lane);
    }
  } catch (...) {
    for (uint32_t lane = 0; lane < width; ++lane)
      end(firstId + lane);
    throw;
  }
  auto result = warmupResult(impl_->estimatedWarmupPeak(), wallSeconds,
                            "real B" + std::to_string(width) +
                                " draft/verify/commit decode");
  result.lanes = std::move(lanes);
  return result;
}

WarmupStepResult Runtime::warmupDraftVerifyCommit() {
  // Verify that a further commit preserves equal target and draft lengths.
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 120;
  double wallSeconds = 0.0;
  std::vector<uint32_t> warmupPrompt{1};
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 16;
  beginColdRequest(request, 0);
  try {
    std::vector<uint32_t> pages = warmupPages(impl_->kvPages, 9, 1);
    BatchPlan prefillPlan{WorkKind::Prefill,
                          BatchCohort::Greedy,
                          {{id, 1}},
                          DecodeStage::Regular};
    ModelBatchItem prefillItem{id, 0, 0, 0, 1, pages};
    prefillItem.inputTokens = request.prompt;
    static_cast<void>(
        prefill(prefillPlan, std::span<const ModelBatchItem>(&prefillItem, 1)));
    prepareWarmupDecode(id, warmupPrompt.back());
    BatchPlan decodePlan{
        WorkKind::Decode, BatchCohort::Greedy, {{id, 0}}, DecodeStage::Regular};
    ModelBatchItem decodeItem{id, 0, 1, 0, 0, pages};
    auto result =
        decode(decodePlan, std::span<const ModelBatchItem>(&decodeItem, 1));
    const auto &lengths = impl_->states.metadata(0).lengths;
    if (result.size() != 1 || result[0].outputTokens.empty() ||
        !lengths.hasCompleteDraftWindow(kDraftCacheStride) ||
        lengths.targetTokens <= 1 ||
        lengths.targetTokens != 1 + result[0].outputTokens.size() -
                                    result[0].outputTokensWithoutKv) {
      throw std::runtime_error("draft/target commit length mismatch");
    }
    wallSeconds = impl_->counters.lastDecodeWallSeconds;
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return warmupResult(impl_->estimatedWarmupPeak(), wallSeconds,
                      "real draft verify acceptance and exact commit");
}

WarmupStepResult Runtime::warmupCompositeStateRestore() {
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 121;
  constexpr uint32_t prefixTokens = 2 * kv::kPageTokens;
  constexpr uint32_t suffixTokens = kDecodeRows;
  constexpr uint32_t promptTokens = prefixTokens + suffixTokens;
  std::vector<uint32_t> warmupPrompt(promptTokens, 2);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 8;
  std::shared_ptr<const CompositeState> cachedState;
  uint64_t estimatedPeakBytes = impl_->estimatedWarmupPeak();
  double wallSeconds = 0.0;
  beginColdRequest(request, 0);
  try {
    if (impl_->kvPages.pageCount() <= 12) {
      throw std::runtime_error(
          "historical prefix warmup requires at least 13 KV pages");
    }
    // Deliberately non-contiguous physical ids exercise page-table lookup.
    const std::vector<uint32_t> pages{12, 10, 11};
    BatchPlan plan{WorkKind::Prefill,
                   BatchCohort::Greedy,
                   {{id, prefixTokens}},
                   DecodeStage::Regular};
    ModelBatchItem item{id, 0, 0, 0, prefixTokens, pages};
    item.inputTokens =
        std::span<const uint32_t>(request.prompt).first(prefixTokens);
    static_cast<void>(prefill(plan, std::span<const ModelBatchItem>(&item, 1)));
    wallSeconds = impl_->counters.lastPrefillWallSeconds;
    cachedState = snapshot(id);
    if (!cachedState)
      throw metal::MetalAllocationError("prefix warmup state allocation failed");
    // The snapshot remains live across restore; its cache slot is allocated
    // through the state storage, so the state term of the estimate already
    // covers it.
    estimatedPeakBytes = impl_->estimatedWarmupPeak();
    end(id);
    beginColdRequest(request, 1);
    restore(id, prefixTokens, cachedState, true);
    setDraftContextPlan(
        id, planDraftContext(prefixTokens, promptTokens, prefixTokens, {}));
    const auto &restored = impl_->states.metadata(1).lengths;
    if (restored.targetTokens != prefixTokens ||
        !restored.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::runtime_error("prefix restore length mismatch");
    }

    // Continue from committed KV history. This M8 command teacher-forces a
    // new chunk, then the real speculative cycle overwrites its speculative
    // page suffix and advances only the accepted commit length.
    BatchPlan suffixPlan{WorkKind::Prefill,
                         BatchCohort::Greedy,
                         {{id, suffixTokens}},
                         DecodeStage::Regular};
    ModelBatchItem suffix{id,           1,    prefixTokens, prefixTokens,
                          suffixTokens, pages};
    suffix.inputTokens = std::span<const uint32_t>(request.prompt)
                             .subspan(prefixTokens, suffixTokens);
    static_cast<void>(
        prefill(suffixPlan, std::span<const ModelBatchItem>(&suffix, 1)));
    prepareWarmupDecode(id, warmupPrompt.back());
    const double continuationWallSeconds =
        impl_->counters.lastPrefillWallSeconds;
    wallSeconds += continuationWallSeconds;
    BatchPlan decodePlan{
        WorkKind::Decode, BatchCohort::Greedy, {{id, 0}}, DecodeStage::Regular};
    ModelBatchItem decodeItem{id, 1, promptTokens, 0, 0, pages};
    auto decoded =
        decode(decodePlan, std::span<const ModelBatchItem>(&decodeItem, 1));
    const double historicalDecodeWallSeconds =
        impl_->counters.lastDecodeWallSeconds;
    wallSeconds += historicalDecodeWallSeconds;
    const auto &continued = impl_->states.metadata(1).lengths;
    if (decoded.size() != 1 || decoded[0].outputTokens.empty() ||
        !continued.hasCompleteDraftWindow(kDraftCacheStride) ||
        continued.targetTokens <= promptTokens ||
        continued.targetTokens !=
            promptTokens + decoded[0].outputTokens.size() -
                decoded[0].outputTokensWithoutKv) {
      throw std::runtime_error(
          "restored historical prefix did not continue exactly");
    }
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return warmupResult(
      estimatedPeakBytes, wallSeconds,
      "real paged-KV state restore, arbitrary page table, slot move, "
      "bounded restore continuation, and decode");
}

ModelMemoryActual Runtime::actualRuntimeMemory() const {
  return {impl_->states.actualAllocatedBytes(), impl_->prefillArena->bytes(),
          impl_->decodeArena->bytes(), impl_->draftModel.restrictedHeadAllocatedBytes()};
}

ModelTelemetry Runtime::telemetry() const noexcept {
  ModelTelemetry result = impl_->counters;
  result.stateResidentBytes = impl_->states.actualAllocatedBytes();
  result.warmIdleStateCells = impl_->states.idleCells();
  return result;
}

std::array<float, 16> Runtime::debugSamplingUniforms() const {
  std::array<float, 16> result;
  static_assert(kSamplingUniformCount == result.size());
  std::copy_n(contents<float>(
                  impl_->decodeArena->get(0, DecodeTensor::SamplingUniforms),
                  "sampling uniform snapshot"),
              result.size(), result.begin());
  return result;
}

ModelMemoryPlan plannedRuntimeMemory(const DeviceCapabilities &device,
                                     const ModelPackage &package,
                                     const ops::ExecutionPlans &operators,
                                     kv::Format format) {
  requireCompatibleModelPackage(package);
  if (device.appleGpuFamily < DeviceCapabilities::kMinimumAppleGpuFamily) {
    throw std::invalid_argument("model runtime requires Apple tensor BF16");
  }
  const RuntimeGeometry geometry = RuntimeGeometry::from(package, format);
  const uint64_t workingSet = device.recommendedMaxWorkingSetBytes;
  return {package.stateLayout().activeCellBytes(),
          plannedPrefillBytes(geometry, operators),
          plannedDecodeBytes(geometry, operators),
          std::min(kPipelineReserveBytes, workingSet / 100),
          std::min(kRuntimeOverheadReserveBytes, workingSet / 50)};
}

std::unique_ptr<StateStorage>
createStateStorage(metal::MetalBackend &backend,
                   metal::AllocationAdmission admitAllocation,
                   const ModelPackage &package) {
  requireCompatibleModelPackage(package);
  return std::make_unique<QwenStateStorage>(
      backend, std::move(admitAllocation), package.stateLayout());
}

std::unique_ptr<RuntimeModel> createRuntime(RuntimeContext context) {
  return std::make_unique<Runtime>(std::move(context));
}

} // namespace splash::model

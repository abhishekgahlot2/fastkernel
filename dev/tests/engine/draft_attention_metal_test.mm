// Draft sliding-window attention against a direct CPU reference. The kernel
// streams the 2048-slot ring in ring-aligned 128-token tiles over a fixed
// number of splits and combines their partials in split order; these cases
// cover an unaligned wrap, a wrap that starts exactly one slot before the ring
// end, the short prefix before any wrap, aligned wraps, windows that fill
// exactly one or several splits, and the last slot of the ring, so that tile
// bounds, the window mask, empty splits and the empty-row softmax guard are
// all exercised.
#include "metal/MetalBackend.hpp"
#include "ops/DraftAttention.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::metal::CommandGraph;
using namespace splash::ops;

constexpr uint32_t kRows = 8;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQueryHeadsPerKv = 4;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kWindow = 2048;
constexpr uint32_t kLanes = 4;
// The shipped split count and the fp32 partial one split leaves per (lane,
// head) behind the grouped queries: 32 rows x (128 + max + sum).
constexpr uint32_t kSplits = 4;
constexpr uint64_t kPartialBytes = uint64_t{32} * 130 * sizeof(float);
constexpr uint32_t kAttention = kKvHeads * kQueryHeadsPerKv * kHeadDim;
constexpr uint32_t kGroupRows = kQueryHeadsPerKv * kRows;
constexpr float kScale = 0.08838834765F;

constexpr std::array kShapes{
    DraftAttentionShape{5120, 1280, 6144, 4096, 32, 8, 128},
    DraftAttentionShape{2048, 512, 6144, 4096, 32, 8, 128}};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid draft attention request was accepted");
}

uint16_t toBfloat(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

float fromBfloat(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         const char *label) {
  MetalBuffer buffer =
      backend.allocateBuffer(count * sizeof(uint16_t), BufferStorage::Shared,
                             label);
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = toBfloat(random.unit());
  return buffer;
}

// Reference for one lane and kv head: every row attends to the ring window it
// can see plus all eight current rows, exactly as the kernel's mask defines.
void referenceRows(const uint16_t *queries, const uint16_t *keys,
                   const uint16_t *values, const uint16_t *queryKeys,
                   const uint16_t *queryValues, uint32_t cacheLength,
                   std::vector<float> &output) {
  const uint32_t commonStart =
      cacheLength >= kWindow - 1 ? cacheLength - (kWindow - 1) : 0;
  const uint32_t oldCount = cacheLength - commonStart;
  std::vector<double> scores(oldCount + kRows);
  std::vector<double> accumulated(kHeadDim);
  for (uint32_t row = 0; row < kGroupRows; ++row) {
    const uint32_t proposal = row % kRows;
    const uint32_t queryPosition = cacheLength + proposal;
    const uint32_t rowStart =
        queryPosition >= kWindow - 1 ? queryPosition - (kWindow - 1) : 0;
    const uint32_t hiddenPrefix = rowStart - commonStart;
    const uint16_t *query = queries + uint64_t{row} * kHeadDim;
    double best = -INFINITY;
    for (uint32_t key = 0; key < oldCount + kRows; ++key) {
      double score = -INFINITY;
      if (key >= oldCount) {
        const uint32_t current = key - oldCount;
        double dot = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          dot += double(fromBfloat(query[d])) *
                 fromBfloat(queryKeys[current * kHeadDim + d]);
        }
        score = dot * kScale;
      } else if (key >= hiddenPrefix) {
        const uint32_t slot = (commonStart + key) % kWindow;
        double dot = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          dot += double(fromBfloat(query[d])) *
                 fromBfloat(keys[uint64_t{slot} * kHeadDim + d]);
        }
        score = dot * kScale;
      }
      scores[key] = score;
      best = std::max(best, score);
    }
    double sum = 0.0;
    std::fill(accumulated.begin(), accumulated.end(), 0.0);
    for (uint32_t key = 0; key < oldCount + kRows; ++key) {
      if (scores[key] == -INFINITY)
        continue;
      const double probability = std::exp(scores[key] - best);
      sum += probability;
      for (uint32_t d = 0; d < kHeadDim; ++d) {
        const float value =
            key >= oldCount
                ? fromBfloat(queryValues[uint64_t{d} * kRows + key - oldCount])
                : fromBfloat(values[uint64_t{d} * kWindow +
                                    (commonStart + key) % kWindow]);
        accumulated[d] += probability * value;
      }
    }
    for (uint32_t d = 0; d < kHeadDim; ++d)
      output[uint64_t{row} * kHeadDim + d] =
          static_cast<float>(accumulated[d] / sum);
  }
}

void runCase(MetalBackend &backend, uint32_t lanes, DraftAttentionShape shape,
             const std::array<uint32_t, kLanes> &cacheLengths) {
  Random random(0x5eed0000ULL + cacheLengths[0]);
  const uint64_t ringElements = uint64_t{kKvHeads} * kWindow * kHeadDim;
  // The grouped-queries tensor is sized by the plan so the split partials
  // behind the query rows end exactly at the allocation under validation.
  MetalBuffer queries = randomBfloat(
      backend, DraftAttention::plan(shape, lanes).workspace().groupedQueriesBytes / 2,
      random, "draft queries");
  std::vector<MetalBuffer> keys;
  std::vector<MetalBuffer> values;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    // Each tensor ends exactly at the last head's ring, so a tile that reads
    // past the ring reads past the allocation.
    keys.push_back(randomBfloat(backend, ringElements, random, "draft keys"));
    values.push_back(
        randomBfloat(backend, ringElements, random, "draft values"));
  }
  MetalBuffer queryKeys = randomBfloat(
      backend, uint64_t{lanes} * kKvHeads * kRows * kHeadDim, random,
      "draft query keys");
  MetalBuffer queryValues = randomBfloat(
      backend, uint64_t{lanes} * kKvHeads * kHeadDim * kRows, random,
      "draft query values");
  std::vector<uint16_t> input(
      static_cast<const uint16_t *>(queries.contents()),
      static_cast<const uint16_t *>(queries.contents()) +
          uint64_t{lanes} * kRows * kAttention);

  std::vector<uint16_t> baseline;
  for (const auto configuration : DraftAttention::candidates(shape)) {
    std::memcpy(queries.contents(), input.data(), input.size() * 2);
    CommandGraph graph;
    DraftAttention::addDecode(graph,
        {queries, keys, values, queryKeys, queryValues}, cacheLengths, kWindow,
        DraftAttention::plan(shape, lanes, configuration));
    const auto dispatches = graph.dispatches();
    require(dispatches.size() == 2 &&
                dispatches[0].threadgroups.x == kKvHeads &&
                dispatches[0].threadgroups.y == lanes &&
                dispatches[0].threadgroups.z == kSplits &&
                dispatches[1].threadgroups.x == kKvHeads &&
                dispatches[1].threadgroups.y == lanes &&
                dispatches[1].threadgroups.z == 1,
            "draft attention core dispatch changed with configuration");
    static_cast<void>(backend.submitCommand(dispatches));
    const auto *output = static_cast<const uint16_t *>(queries.contents());
    if (baseline.empty())
      baseline.assign(output, output + input.size());
    else
      require(std::equal(baseline.begin(), baseline.end(), output),
              "draft attention candidate changed core output");
  }

  const auto *output = static_cast<const uint16_t *>(queries.contents());
  std::vector<float> reference(uint64_t{kGroupRows} * kHeadDim);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    // Reference the first and final head; all eight execute above, ending
    // exactly at each lane's allocation boundary under shader validation.
    for (const uint32_t head : {0U, kKvHeads - 1}) {
      const uint64_t queryOffset =
          uint64_t{lane} * kRows * kAttention + uint64_t{head} * kGroupRows *
                                                    kHeadDim;
      referenceRows(
          input.data() + queryOffset,
          static_cast<const uint16_t *>(keys[lane].contents()) +
              uint64_t{head} * kWindow * kHeadDim,
          static_cast<const uint16_t *>(values[lane].contents()) +
              uint64_t{head} * kWindow * kHeadDim,
          static_cast<const uint16_t *>(queryKeys.contents()) +
              (uint64_t{lane} * kKvHeads + head) * kRows * kHeadDim,
          static_cast<const uint16_t *>(queryValues.contents()) +
              (uint64_t{lane} * kKvHeads + head) * kHeadDim * kRows,
          cacheLengths[lane], reference);
      for (uint64_t index = 0; index < reference.size(); ++index) {
        const float actual = fromBfloat(output[queryOffset + index]);
        const float expected = reference[index];
        if (!std::isfinite(actual) ||
            std::fabs(actual - expected) >
                0.02F + 0.02F * std::fabs(expected)) {
          std::cerr << "lane " << lane << " head " << head << " length "
                    << cacheLengths[lane] << " row " << index / kHeadDim
                    << " dim " << index % kHeadDim << ": " << actual
                    << " vs " << expected << '\n';
          throw std::runtime_error("draft attention diverged from reference");
        }
      }
    }
  }
}

void planGeometry() {
  for (const auto shape : kShapes) {
    const auto candidates = DraftAttention::candidates(shape);
    require(candidates.size() == 4 && candidates.front() == DraftAttentionConfiguration{},
            "draft candidate list lost the full-grid baseline");
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
      const uint64_t rows = uint64_t{lanes} * kRows;
      for (const auto configuration : candidates) {
        const auto plan = DraftAttention::plan(shape, lanes, configuration);
        const auto workspace = plan.workspace();
        require(plan.lanes() == lanes && plan.shape() == shape &&
                    plan.configuration() == configuration &&
                    workspace.convolutionBytes == rows * shape.hiddenSize * 2 &&
                    workspace.qkvBytes == rows * 6144 * 2 &&
                    workspace.groupedQueriesBytes ==
                        rows * 4096 * 2 +
                            uint64_t{lanes} * kKvHeads * kSplits * kPartialBytes &&
                    workspace.queryKeysBytes == rows * 8 * 128 * 2 &&
                    workspace.queryValuesBytes == rows * 8 * 128 * 2,
                "draft plan padded lanes or changed tensor storage");
      }
    }
    rejects([&] { (void)DraftAttention::plan(shape, 0); });
    rejects([&] { (void)DraftAttention::plan(shape, 5); });
    rejects([&] { (void)DraftAttention::plan(shape, 1, {1}); });
    rejects([&] { (void)DraftAttention::plan(shape, 1, {61}); });
  }
  auto unsupported = kShapes[0];
  unsupported.queryHeads = 16;
  rejects([&] { (void)DraftAttention::candidates(unsupported); });
  rejects([&] { (void)DraftAttention::plan({}, 1); });
}

void fillDyadic(const MetalBuffer &buffer, uint32_t multiplier, uint32_t modulus) {
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
    values[i] = toBfloat((int((i * multiplier) % modulus) - int(modulus / 2)) /
                        8.0F);
}

void surroundingPhases(MetalBackend &backend, DraftAttentionShape shape,
                       uint32_t lanes) {
  const auto baselinePlan = DraftAttention::plan(shape, lanes);
  const auto workspace = baselinePlan.workspace();
  const uint64_t rows = uint64_t{lanes} * kRows;
  auto allocate = [&](uint64_t bytes) {
    return backend.allocateBuffer(bytes, BufferStorage::Shared, "draft phases");
  };
  const auto input = allocate(workspace.convolutionBytes);
  const auto dynamic = allocate(rows * shape.dynamicSize * 2);
  const auto weights = allocate(uint64_t{4} * shape.hiddenSize * 2);
  const auto residual = allocate(workspace.convolutionBytes);
  const auto output = allocate(workspace.convolutionBytes);
  fillDyadic(input, 7, 23);
  fillDyadic(dynamic, 5, 17);
  fillDyadic(weights, 3, 13);
  fillDyadic(residual, 11, 19);
  const auto *in = static_cast<const uint16_t *>(input.contents());
  const auto *dyn = static_cast<const uint16_t *>(dynamic.contents());
  const auto *base = static_cast<const uint16_t *>(weights.contents());
  const auto *res = static_cast<const uint16_t *>(residual.contents());
  const uint32_t channelsPerGroup = 16;
  const uint32_t convolutionGroups = shape.hiddenSize / channelsPerGroup;

  Random random(0x5eed1234 + shape.hiddenSize + lanes);
  const auto qkv = randomBfloat(backend, workspace.qkvBytes / 2, random, "QKV");
  const auto queries = allocate(workspace.groupedQueriesBytes);
  const auto queryKeys = allocate(workspace.queryKeysBytes);
  const auto queryValues = allocate(workspace.queryValuesBytes);
  const auto packed = allocate(workspace.groupedQueriesBytes);
  const auto queryNorm = allocate(kHeadDim * 2);
  const auto keyNorm = allocate(kHeadDim * 2);
  std::fill_n(static_cast<uint16_t *>(queryNorm.contents()), kHeadDim,
              toBfloat(1));
  std::fill_n(static_cast<uint16_t *>(keyNorm.contents()), kHeadDim,
              toBfloat(1));
  const auto ropeCos = allocate(rows * kHeadDim / 2 * sizeof(float));
  const auto ropeSin = allocate(rows * kHeadDim / 2 * sizeof(float));
  for (uint64_t i = 0; i < rows * kHeadDim / 2; ++i) {
    static_cast<float *>(ropeCos.contents())[i] = std::cos(float(i) * 0.01F);
    static_cast<float *>(ropeSin.contents())[i] = std::sin(float(i) * 0.01F);
  }
  std::vector<uint16_t> originalQkv(
      static_cast<const uint16_t *>(qkv.contents()),
      static_cast<const uint16_t *>(qkv.contents()) + workspace.qkvBytes / 2);
  const std::array preparedBuffers{qkv, queries, queryKeys, queryValues};
  std::array<std::vector<uint16_t>, 4> preparedBaseline;
  for (const auto configuration : DraftAttention::candidates(shape)) {
    const auto plan = DraftAttention::plan(shape, lanes, configuration);
    for (const auto stage : {DraftConvolutionStage::Prepare,
                            DraftConvolutionStage::Residual}) {
      std::memset(output.contents(), 0xFF, output.sizeBytes());
      CommandGraph graph;
      DraftAttention::addConvolution(graph,
          {input, dynamic, weights, residual, output}, plan, stage);
      std::array<uint32_t, 3> params{};
      std::memcpy(params.data(), graph.dispatches()[0].bytes[0].data,
                  sizeof(params));
      const uint32_t expectedGroups = configuration.groups
          ? configuration.groups
          : (kRows * shape.hiddenSize + 255) / 256;
      require(graph.dispatches()[0].threadgroups.x == params[0] &&
                  params[0] == expectedGroups && params[2] == lanes,
              "convolution dispatch and loop stride permit overlapping writes");
      static_cast<void>(backend.submitCommand(graph.dispatches()));
      const auto *actual = static_cast<const uint16_t *>(output.contents());
      const bool finish = stage == DraftConvolutionStage::Residual;
      const uint32_t kind = finish ? 1 : 0;
      for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t channel = 0; channel < shape.hiddenSize; ++channel) {
          const uint64_t index = row * shape.hiddenSize + channel;
          const uint32_t group = channel / channelsPerGroup;
          float value = fromBfloat(in[index]) *
              (fromBfloat(base[(kind * 2) * shape.hiddenSize + channel]) +
               fromBfloat(dyn[row * shape.dynamicSize +
                              (kind * 2) * convolutionGroups + group]));
          if (row % kRows != 0)
            value += fromBfloat(in[index - shape.hiddenSize]) *
                (fromBfloat(base[(kind * 2 + 1) * shape.hiddenSize + channel]) +
                 fromBfloat(dyn[row * shape.dynamicSize +
                                (kind * 2 + 1) * convolutionGroups + group]));
          if (finish)
            value += fromBfloat(res[index]);
          require(actual[index] == toBfloat(value),
                  "draft convolution differed from exact CPU arithmetic");
        }
      }
    }

    std::memcpy(qkv.contents(), originalQkv.data(), qkv.sizeBytes());
    CommandGraph graph;
    DraftAttention::addPrepare(graph,
        {qkv, queries, queryNorm, keyNorm, ropeCos, ropeSin, queryKeys,
         queryValues}, plan);
    DraftAttention::addReorder(graph, queries, packed, plan);
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    for (size_t i = 0; i < preparedBuffers.size(); ++i) {
      const auto *values =
          static_cast<const uint16_t *>(preparedBuffers[i].contents());
      if (configuration == DraftAttentionConfiguration{})
        preparedBaseline[i].assign(values,
                                   values + preparedBuffers[i].sizeBytes() / 2);
      else
        require(std::equal(preparedBaseline[i].begin(), preparedBaseline[i].end(),
                           values),
                "draft prepare candidate changed normalization/RoPE/storage");
    }
    const auto *grouped = static_cast<const uint16_t *>(queries.contents());
    const auto *reordered = static_cast<const uint16_t *>(packed.contents());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint64_t laneOffset = uint64_t{lane} * kRows * kAttention;
      for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t head = 0; head < shape.queryHeads; ++head) {
          for (uint32_t dim = 0; dim < kHeadDim; ++dim)
            require(reordered[laneOffset + row * kAttention + head * kHeadDim +
                               dim] ==
                        grouped[laneOffset + (head * kRows + row) * kHeadDim +
                                dim],
                    "draft reorder differed from CPU layout reference");
        }
      }
    }
  }

  CommandGraph invalid;
  rejects([&] {
    DraftAttention::addConvolution(invalid,
        {input, dynamic, weights, residual, output}, baselinePlan,
        static_cast<DraftConvolutionStage>(2));
  });
  const auto shortBuffer = backend.view(output, 0, output.sizeBytes() - 2);
  rejects([&] {
    DraftAttention::addConvolution(invalid,
        {input, dynamic, weights, residual, shortBuffer}, baselinePlan,
        DraftConvolutionStage::Prepare);
  });
  rejects([&] {
    DraftAttention::addPrepare(invalid,
        {qkv, queries, queryNorm, keyNorm, ropeCos, ropeSin, {}, queryValues},
        baselinePlan);
  });
  rejects([&] {
    DraftAttention::addReorder(invalid, queries, {}, baselinePlan);
  });
  const std::array<MetalBuffer, kLanes> emptyRings{};
  const std::array<uint32_t, kLanes> lengths{};
  rejects([&] {
    DraftAttention::addDecode(invalid,
        {queries, emptyRings, emptyRings, queryKeys, queryValues}, lengths,
        kWindow, baselinePlan);
  });
  rejects([&] {
    DraftAttention::addDecode(invalid,
        {queries, emptyRings, emptyRings, queryKeys, queryValues}, lengths,
        kWindow - 1, baselinePlan);
  });
  require(invalid.empty(), "invalid draft request partially encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-attention METALLIB");
    planGeometry();
    MetalBackend backend(argv[1]);
    for (const auto shape : kShapes) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        surroundingPhases(backend, shape, lanes);
      runCase(backend, 1, shape, {0, 0, 0, 0});
      runCase(backend, 2, shape, {2048, 2047, 0, 0});
      runCase(backend, 3, shape, {2100, 4094, 6143, 0});
      runCase(backend, 4, shape, {262137, 4094, 500, 6143});
      runCase(backend, 4, shape, {512, 513, 1024, 1536});
      runCase(backend, 4, shape, {2046, 2049, 4095, 4096});
      runCase(backend, 2, shape, {8191, 262144, 0, 0});
    }
    std::cout << "draft_attention_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_attention_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

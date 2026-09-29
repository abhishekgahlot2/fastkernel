// Modified by meowkernels.
#include "model/QwenTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "metal/EnvSwitch.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace splash::model {
namespace {

template <class Layout>
QwenTargetGeometry commonGeometry(const Layout &layout) {
  QwenTargetGeometry result;
  result.maximumContextTokens = layout.maximumContextTokens;
  result.layers = layout.layers;
  result.hiddenSize = layout.hiddenSize;
  result.vocabularySize = layout.vocabularySize;
  result.packedGdnWidth = layout.packedGdnWidth;
  result.packedAttentionWidth = layout.packedFullWidth;
  result.convolutionDimension = layout.convolutionDimension;
  result.attentionWidth = layout.attentionWidth;
  result.attentionQueryHeads = layout.attentionQueryHeads;
  result.attentionKvHeads = layout.attentionKvHeads;
  result.attentionHeadDimension = layout.attentionHeadDimension;
  result.rotaryPairs = layout.rotaryPairs;
  result.rotaryTheta = layout.rotaryTheta;
  result.gdnKeyHeads = layout.gdnKeyHeads;
  result.gdnValueHeads = layout.gdnValueHeads;
  result.gdnHeadDimension = layout.gdnHeadDimension;
  result.maskToken = layout.maskToken;
  result.stopTokens = layout.stopTokens;
  result.kvLayout = layout.kvLayout();
  result.stateLayout = layout.gdnStateLayout();
  result.captureLayerCount =
      static_cast<uint32_t>(layout.hiddenCaptureLayers.size());
  std::copy(layout.hiddenCaptureLayers.begin(),
            layout.hiddenCaptureLayers.end(),
            result.captureLayerValues.begin());
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_8Layout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.ffnKind = QwenFfnKind::Dense;
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_6MoeLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.moe = {layout.hiddenSize, layout.experts, layout.expertsPerToken,
                layout.expertIntermediateSize};
  result.ffnKind = QwenFfnKind::SparseMoe;
  return result;
}

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

template <class Layer>
constexpr bool hasDenseFfn = requires(const Layer &layer) {
  layer.gateProjection;
  layer.upProjection;
  layer.downProjection;
};

template <class Mixer>
constexpr bool isGdnMixer =
    std::is_same_v<std::remove_cvref_t<Mixer>, QwenGdnWeights>;

} // namespace

QwenMixerWeights readQwenMixer(WeightFile &file, metal::MetalBackend &backend,
                               const QwenMixerGeometry &geometry,
                               bool fullAttention) {
  constexpr uint64_t kFloat32Bytes = 4;
  if (fullAttention) {
    QwenAttentionWeights attention;
    attention.inputProjection =
        readQ4Projection(file, backend, geometry.packedAttentionWidth,
                         geometry.hiddenSize, "attention-input");
    const uint64_t headNormBytes = checkedWeightMultiply(
        geometry.attentionHeadDimension, kBFloat16Bytes, "head norm bytes");
    attention.queryNorm = file.section(headNormBytes, "query-norm");
    attention.keyNorm = file.section(headNormBytes, "key-norm");
    attention.outputProjection =
        readQ4Projection(file, backend, geometry.hiddenSize,
                         geometry.attentionWidth, "attention-output");
    return attention;
  }
  QwenGdnWeights gdn;
  gdn.inputProjection = readQ4Projection(
      file, backend, geometry.packedGdnWidth, geometry.hiddenSize, "gdn-input");
  gdn.convolutionWeights = file.section(
      checkedWeightMultiply(
          checkedWeightMultiply(geometry.convolutionDimension, kGdnConvolutionTaps,
                                "convolution elements"),
          kBFloat16Bytes, "convolution bytes"),
      "gdn-convolution");
  gdn.decay = file.section(checkedWeightMultiply(geometry.gdnValueHeads,
                                                 kFloat32Bytes,
                                                 "GDN decay bytes"),
                           "gdn-decay");
  gdn.timeBias = file.section(
      checkedWeightMultiply(geometry.gdnValueHeads, kBFloat16Bytes,
                            "GDN time bias bytes"),
      "gdn-time-bias");
  gdn.mixerNorm = file.section(
      checkedWeightMultiply(geometry.gdnHeadDimension, kBFloat16Bytes,
                            "GDN norm bytes"),
      "gdn-norm");
  gdn.outputProjection = readQ4Projection(
      file, backend, geometry.hiddenSize, geometry.attentionWidth, "gdn-output");
  return gdn;
}

QwenTarget::QwenTarget(const Qwen3_8Weights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators, kv::Format format)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  geometry_.kvLayout.format = format;
  requireWeights(weights, geometry_);
}

QwenTarget::QwenTarget(const Qwen3_6MoeWeights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators, kv::Format format)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  geometry_.kvLayout.format = format;
  requireWeights(weights, geometry_);
}

QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &weights) {
  return geometryFor(weights.layout);
}

QwenTargetGeometry qwenTargetGeometry(const Qwen3_6MoeWeights &weights) {
  return geometryFor(weights.layout);
}

const ops::Q4Projection &QwenTarget::vocabularyProjection() const noexcept {
  return std::visit([](const auto *weights) -> const ops::Q4Projection & {
    return weights->logitsProjection;
  }, weights_);
}

void QwenTarget::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::LayerStorage> kvLayers) const {
  std::visit(
      [&](const auto *weights) {
        addPrefillImpl(*weights, graph, std::move(buffers), sequences, rows,
                       kvLayers);
      },
      weights_);
}

template <class Weights>
void QwenTarget::addPrefillImpl(
    const Weights &weights, metal::CommandGraph &graph,
    QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::LayerStorage> kvLayers) const {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }
  const ops::LinearMatrix gdnInput{geometry_.packedGdnWidth,
                                     geometry_.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry_.packedAttentionWidth,
                                           geometry_.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry_.hiddenSize,
                                        geometry_.attentionWidth};
  const auto moePlan = [&]() -> std::optional<ops::MoePlan> {
    if constexpr (!hasDenseFfn<typename std::remove_cvref_t<decltype(weights.layers)>::value_type>)
      return operators_.moePrefill(geometry_.moe, rows);
    return std::nullopt;
  }();

  auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(uint16_t),
                         uint64_t{count} * width * sizeof(uint16_t));
  };
  auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(float),
                         uint64_t{count} * width * sizeof(float));
  };

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;
  for (uint32_t layerIndex = 0; layerIndex < geometry_.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    ops::Normalization::addRmsWithQ4Sums(
        graph, input, layer.inputNorm, buffers.normalized,
        buffers.projectionSums, geometry_.hiddenSize, rows);

    metal::MetalBuffer residual;
    std::visit(
        [&](const auto &mixer) {
          if constexpr (isGdnMixer<decltype(mixer)>) {
            operators_.linear().addPrefill(graph, buffers.normalized,
                           mixer.inputProjection, buffers.gdnPacked,
                           buffers.projectionSums, gdnInput, rows);
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              ops::GDN::addPrefill(
                  graph,
                  {u16(buffers.gdnPacked, sequence.rowBegin, sequence.rows,
                       geometry_.packedGdnWidth),
                   mixer.convolutionWeights, sequence.convolutionIn[gdnIndex],
                   sequence.convolutionOut[gdnIndex],
                   u16(buffers.gdnQueries, sequence.rowBegin, sequence.rows,
                       geometry_.gdnKeyWidth()),
                   u16(buffers.gdnKeys, sequence.rowBegin, sequence.rows,
                       geometry_.gdnKeyWidth()),
                   u16(buffers.gdnValues, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth),
                   mixer.decay, mixer.timeBias,
                   f32(buffers.gdnDecay, sequence.rowBegin, sequence.rows,
                       geometry_.gdnValueHeads),
                   u16(buffers.gdnBeta, sequence.rowBegin, sequence.rows,
                       geometry_.gdnValueHeads),
                   sequence.recurrentIn[gdnIndex],
                   sequence.recurrentOut[gdnIndex],
                   u16(buffers.recurrent, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth),
                   mixer.mixerNorm,
                   u16(buffers.gdnHidden, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth)},
                  geometry_.gdnShape(), sequence.rows);
            }
            operators_.linear().addPrefillSums(graph, buffers.gdnHidden,
                               buffers.projectionSums, mixerOutput, rows);
            operators_.linear().addPrefillResidual(
                graph, buffers.gdnHidden, mixer.outputProjection, input,
                buffers.gdnOutput, buffers.projectionSums, mixerOutput,
                rows);
            residual = buffers.gdnOutput;
            ++gdnIndex;
          } else {
            operators_.linear().addPrefill(graph, buffers.normalized,
                           mixer.inputProjection, buffers.fullPacked,
                           buffers.projectionSums, attentionInput, rows);
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              const uint64_t queryBytes =
                  uint64_t{geometry_.attentionQueryHeads} *
                  sequence.attentionStride * geometry_.attentionHeadDimension *
                  sizeof(uint16_t);
              const uint64_t kvBytes =
                  uint64_t{geometry_.attentionKvHeads} *
                  sequence.attentionStride * geometry_.attentionHeadDimension *
                  sizeof(uint16_t);
              metal::MetalBuffer queries = backend_.view(
                  buffers.fullQueries, sequence.queryOffset, queryBytes);
              metal::MetalBuffer attentionRows = backend_.view(
                  buffers.fullAttention, sequence.queryOffset, queryBytes);
              metal::MetalBuffer keys = backend_.view(
                  buffers.chunkKeys, sequence.kvOffset, kvBytes);
              metal::MetalBuffer values = backend_.view(
                  buffers.chunkValues, sequence.kvOffset, kvBytes);
              ops::PagedAttention::addPrefillProjection(
                  graph,
                  u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                      geometry_.packedAttentionWidth),
                  mixer.queryNorm, mixer.keyNorm,
                  f32(buffers.ropeCos, sequence.rowBegin, sequence.rows,
                      geometry_.rotaryPairs),
                  f32(buffers.ropeSin, sequence.rowBegin, sequence.rows,
                      geometry_.rotaryPairs),
                  queries, keys, values, sequence.rows,
                  sequence.attentionStride, sequence.attentionStride,
                  geometry_.attentionQueryHeads, geometry_.kvLayout);
              ops::PagedAttention::addPrefillStore(
                  graph, kvLayers[attentionIndex], keys, values,
                  sequence.pageTable, sequence.q8, geometry_.kvLayout);
              ops::PagedAttention::addPrefill(
                  graph, kvLayers[attentionIndex], queries, attentionRows,
                  buffers.attentionPartials, buffers.attentionStatistics,
                  sequence.pageTable, sequence.q8,
                  operators_.prefillAttention(
                      sequence.rows, geometry_.attentionQueryHeads,
                      geometry_.kvLayout, sequence.q8.committed_tokens));
              ops::PagedAttention::addPrefillGate(
                  graph,
                  u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                      geometry_.packedAttentionWidth),
                  attentionRows,
                  u16(buffers.attentionHidden, sequence.rowBegin,
                      sequence.rows, geometry_.attentionWidth),
                  sequence.rows, sequence.attentionStride,
                  sequence.attentionStride, geometry_.attentionQueryHeads,
                  geometry_.kvLayout);
            }
            operators_.linear().addPrefillSums(graph, buffers.attentionHidden,
                               buffers.projectionSums, mixerOutput, rows);
            operators_.linear().addPrefillResidual(
                graph, buffers.attentionHidden, mixer.outputProjection, input,
                buffers.attentionOutput, buffers.projectionSums, mixerOutput,
                rows);
            residual = buffers.attentionOutput;
            ++attentionIndex;
          }
        },
        layer.mixer);

    ops::Normalization::addRmsWithQ4Sums(
        graph, residual, layer.postAttentionNorm, buffers.normalized,
        buffers.projectionSums, geometry_.hiddenSize, rows);
    if constexpr (hasDenseFfn<std::remove_cvref_t<decltype(layer)>>) {
      const ops::LinearMatrix up{geometry_.denseIntermediateSize,
                                   geometry_.hiddenSize};
      const ops::LinearMatrix down{geometry_.hiddenSize,
                                     geometry_.denseIntermediateSize};
      operators_.linear().addPrefill(graph, buffers.normalized, layer.gateProjection,
                     buffers.denseGateScratch, buffers.projectionSums, up,
                     rows);
      operators_.linear().addPrefillUpWithGate(
          graph, buffers.normalized, layer.upProjection,
          buffers.denseGateScratch, buffers.denseIntermediate,
          buffers.projectionSums, buffers.downProjectionSums, up, rows);
      operators_.linear().addPrefillResidual(
          graph, buffers.denseIntermediate, layer.downProjection, residual,
          output, buffers.downProjectionSums, down, rows);
    } else {
      ops::MoE::add(
          graph,
          {buffers.normalized, residual, output, buffers.selectedExperts,
           buffers.routingWeights, buffers.tileDescriptors, buffers.tileCount,
           buffers.groupedRoutes, buffers.routeRows, buffers.groupedInput,
           buffers.expertIntermediate, buffers.expertOutput},
          layer.ffn, *moePlan);
    }

    const auto captureLayers = geometry_.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      const uint32_t slot =
          static_cast<uint32_t>(captured - captureLayers.begin());
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        for (uint32_t index = 0; index < sequence.captureCount; ++index) {
          const QwenTargetPrefillCapture &capture = sequence.captures[index];
          ops::DraftAttention::captureTargetHidden(
              graph, output, buffers.captured, capture.rows, slot,
              capture.sourceStart, capture.destinationStart,
              geometry_.hiddenSize, geometry_.capturedHiddenSize());
        }
      }
    }
  }
  if (gdnIndex != geometry_.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const kv::LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  std::visit(
      [&](const auto *weights) {
        addVerifyImpl(*weights, graph, std::move(buffers), kvLayers, q8,
                      verify, lanes, stats, false, {});
      },
      weights_);
}

void QwenTarget::addVerify16(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const kv::LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify,
    ops::Q4DispatchStats &stats,
    metal::MetalBuffer convolutionScratch, uint32_t tiles,
    ops::WideGdn gdnRoute) const {
  static_assert(ExecutionLimits::targetVerifyRows == 8);
  static_assert(ExecutionLimits::maximumBatchWidth >= 4);
  std::visit(
      [&](const auto *weights) {
        addVerifyImpl(*weights, graph, std::move(buffers), kvLayers, q8,
                      verify, tiles, stats, true, convolutionScratch,
                      gdnRoute);
      },
      weights_);
}

bool preparesInputSums(const ops::Q4Linear &linear, ops::LinearMatrix input, uint32_t lanes,
                       bool sumsScratch) {
  // SPLASH_INPUT_FUSED_SUMS (default on): the input RMS also writes the input
  // projection's group sums, and the projection runs as the split-K kernel
  // that reads them (Linear.cpp; reassociates K). Serving ms/step -2.1%,
  // quality-gated.
  static const bool fusedInputSums = metal::envSwitch("SPLASH_INPUT_FUSED_SUMS");
  // SPLASH_M16_INPUT_SUMS (default on) extends it to 16-row (two-lane or wide)
  // verifies, whose N128 M16 projections become the split-K M16 consumer.
  // Chosen for M-invariance, not speed (cost ~0): a row's bytes must not depend
  // on how many rows share the pass. Hidden+logits 3788/3788 vs 8-row verifies
  // (row-invariance audit).
  static const bool m16InputSums = metal::envSwitch("SPLASH_M16_INPUT_SUMS");
  // SPLASH_M24_INPUT_SUMS (default on) does the same for 24/32-row verifies (the
  // split-K M24/M32 consumer), so each row matches the 8-row verify's bytes
  // (32-row gate 1: 4000/4000). Saves 4.2 ms per 32-row cycle; the 24-row
  // (three-request) cycle pads to M32 below (m24).
  static const bool m24InputSums = metal::envSwitch("SPLASH_M24_INPUT_SUMS");
  const auto tile = [&](uint32_t tileLanes) {
    return linear.plan({input, tileLanes * ExecutionLimits::targetVerifyRows,
        ops::LinearPhase::Decode, ops::LinearEpilogue::None}).configuration().tile;
  };
  // Wider verifies take the sums only when the 8-row verify does, so a row's K
  // reduction doesn't depend on how many rows share the pass (all-Macs G0: an
  // 8-core Apple10 plans the 8-row GDN input as Paired256, which reads none).
  if (!sumsScratch || input.inputSize > 5120 || input.inputSize % 1024 || !fusedInputSums ||
      tile(1) != ops::LinearTile::Paired128)
    return false;
  if (lanes == 1) return true;
  const ops::LinearTile wider = tile(lanes);
  return lanes == 2 ? m16InputSums && wider == ops::LinearTile::N128
                    : m24InputSums && (wider == ops::LinearTile::N128 || wider == ops::LinearTile::N256);
}

uint32_t rowStableVerifyRows(const ops::Q4Linear &linear, const QwenTargetGeometry &geometry,
                             bool sumsScratch) {
  struct Projection {
    ops::LinearMatrix matrix;
    ops::LinearEpilogue epilogue;
    bool input;
  };
  const Projection projections[] = {
      {{geometry.packedGdnWidth, geometry.hiddenSize}, ops::LinearEpilogue::None, true},
      {{geometry.packedAttentionWidth, geometry.hiddenSize}, ops::LinearEpilogue::None, true},
      {{geometry.hiddenSize, geometry.attentionWidth}, ops::LinearEpilogue::Residual, false},
      {{geometry.denseIntermediateSize, geometry.hiddenSize}, ops::LinearEpilogue::GateUp, false},
      {{geometry.hiddenSize, geometry.denseIntermediateSize}, ops::LinearEpilogue::Residual, false},
      {{geometry.vocabularySize, geometry.hiddenSize}, ops::LinearEpilogue::None, false}};
  // The K reduction a row gets: sequential tiles share one full-K order, the
  // split4 tiles and the consumer of RMS-written input sums another, and a
  // simdgroup plan's order depends on its K splits.
  const auto reduction = [&](const Projection &p, uint32_t lanes) {
    if (p.input && preparesInputSums(linear, p.matrix, lanes, sumsScratch))
      return std::pair{ops::LinearTile::Split32PrecomputedSums, 1U};
    const auto config = linear.plan({p.matrix, lanes * ExecutionLimits::targetVerifyRows,
        ops::LinearPhase::Decode, p.epilogue}).configuration();
    switch (config.tile) {
    case ops::LinearTile::N128:
    case ops::LinearTile::N256:
    case ops::LinearTile::Paired128:
    case ops::LinearTile::Paired256: return std::pair{ops::LinearTile::N128, 1U};
    default: return std::pair{config.tile, config.splits};
    }
  };
  uint32_t rows = ExecutionLimits::targetVerifyRows;
  for (const uint32_t lanes : {2U, 4U}) {  // the wide lookup widths: 16 and 32 rows
    for (const Projection &p : projections)  // (no dense FFN: MoE has no wide verify)
      if (p.matrix.outputSize && p.matrix.inputSize && reduction(p, lanes) != reduction(p, 1))
        return rows;
    rows = lanes * ExecutionLimits::targetVerifyRows;
  }
  return rows;
}

template <class Weights>
void QwenTarget::addVerifyImpl(
    const Weights &weights, metal::CommandGraph &graph,
    QwenTargetVerifyBuffers buffers,
    std::span<const kv::LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats, bool wide,
    metal::MetalBuffer convolutionScratch, ops::WideGdn gdnRoute) const {
  using Layer =
      typename std::remove_cvref_t<decltype(weights.layers)>::value_type;
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      q8.size() != ExecutionLimits::maximumBatchWidth ||
      verify.size() != ExecutionLimits::maximumBatchWidth ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers ||
      (wide &&
       ((lanes != 2 && lanes != 4) || !hasDenseFfn<Layer> ||
        !geometry_.gdnShape().valid() ||
        buffers.arrived.sizeBytes() < lanes * sizeof(uint32_t) ||
        buffers.generation.sizeBytes() < lanes * sizeof(uint32_t) ||
        convolutionScratch.sizeBytes() <
            (lanes > 2 ? 2 : 1) * geometry_.stateLayout.convolutionBytes()))) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  // SPLASH_SPLIT4_M16 (default on; read at every encode so one binary serves both
  // arms of an A/B): the two-request (B2) 16-row split4 residual bodies (mixer out and
  // FFN down) take the M8 footer and metadata hoist. Wide lookup cycles keep them.
  // B2 lockstep -0.216 ms per cycle; byte-identical (exact-dot fixtures, random
  // bytes, oracle incl. the B2 witness).
  const bool m16HoistFooter = metal::envSwitch("SPLASH_SPLIT4_M16") && !wide && lanes == 2;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = verify[lane].committed_tokens;
  const auto attentionPlan = operators_.verifyAttention(
      lanes, geometry_.attentionQueryHeads, geometry_.kvLayout, histories);
  const ops::LinearMatrix gdnInput{geometry_.packedGdnWidth, geometry_.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry_.packedAttentionWidth, geometry_.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry_.hiddenSize, geometry_.attentionWidth};
  const auto moePlan = [&]() -> std::optional<ops::MoePlan> {
    if constexpr (!hasDenseFfn<typename std::remove_cvref_t<decltype(weights.layers)>::value_type>)
      return operators_.moeDecode(geometry_.moe, lanes);
    return std::nullopt;
  }();
  constexpr uint32_t tileRows = kv::kPageTokens;

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;
  for (uint32_t layerIndex = 0; layerIndex < geometry_.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    const bool gdnLayer = std::holds_alternative<QwenGdnWeights>(layer.mixer);
    const bool prepareInputSums = preparesInputSums(operators_.linear(),
        gdnLayer ? gdnInput : attentionInput, lanes,
        buffers.linearScratch.sums && !buffers.linearScratch.input);
    // SPLASH_M24_PAD3 (read per encode, default on): B3's input RMS and
    // input projections run over the idle fourth lane too (32 rows), so the
    // projection takes the split-K M32 consumer (cheaper than M24; each real
    // row's bytes are unchanged, the fourth lane's rows are discarded). The
    // RMS publishes the sums as [group][rows], so it pads as well. B3 -4.58 ms
    // per cycle; exact per row (row-invariance PAD3 gate).
    bool pad3 = false;
    if (prepareInputSums && lanes == 3 && !wide && buffers.normalizedPadded &&
        buffers.gdnPackedPadded.size() == buffers.gdnPacked.size())
      pad3 = metal::envSwitch("SPLASH_M24_PAD3");  // only on eligible B3 layers (Codex 06:58)
    const uint32_t inputLanes = pad3 ? lanes + 1 : lanes;
    const metal::MetalBuffer inputNormalized = pad3 ? buffers.normalizedPadded : buffers.normalized;
    // The RMS stays its own dispatch: folding it into the previous layer's down
    // epilogue moves the bf16 rounding points (texts changed 15/18) and saved
    // nothing (megakernel-recipe/RECIPE.md rule 15).
    if (prepareInputSums)
      ops::Normalization::addRmsStagedSplitSums(
          graph, pad3 ? buffers.hiddenPadded[layerIndex & 1] : input, layer.inputNorm,
          inputNormalized, buffers.linearScratch.sums, geometry_.hiddenSize,
          inputLanes * ExecutionLimits::targetVerifyRows);
    else
      ops::Normalization::addRms(graph, input, layer.inputNorm,
                                 buffers.normalized, geometry_.hiddenSize, rows, buffers.linearScratch);

    metal::MetalBuffer residual;
    std::visit(
        [&](const auto &mixer) {
          if constexpr (isGdnMixer<decltype(mixer)>) {
            operators_.linear().addDecodeBatch(graph,
                               inputNormalized, mixer.inputProjection,
                               pad3 ? buffers.gdnPackedPadded[gdnIndex] : buffers.gdnPacked[gdnIndex],
                               gdnInput, inputLanes,
                               stats, buffers.linearScratch, true, prepareInputSums);
            ops::GdnDecodeBuffers gdn{
                buffers.gdnPacked[gdnIndex], mixer.convolutionWeights,
                buffers.currentGdnStates, buffers.nextGdnStates,
                buffers.gdnMixed[gdnIndex], mixer.decay, mixer.timeBias,
                buffers.gdnDecay[gdnIndex], buffers.gdnBeta[gdnIndex],
                buffers.recurrent, mixer.mixerNorm, buffers.gdnHidden,
                buffers.arrived, buffers.generation,
                wide ? ops::LinearScratch{} : buffers.linearScratch};
            // SPLASH_GDN_FUSED_SUMS (default on): the GDN output kernel writes the
            // Split32 sums the separate dispatch wrote (byte-exact gate); texts
            // 18/18 identical, ms/cycle -0.083.
            static const bool fusedSplitSums = metal::envSwitch("SPLASH_GDN_FUSED_SUMS");
            const bool prepareSplitSums = fusedSplitSums && !wide && lanes == 1 &&
                geometry_.gdnValueHeads == 48 && !buffers.linearScratch.input &&
                operators_.linear().plan({mixerOutput, rows, ops::LinearPhase::Decode,
                    ops::LinearEpilogue::Residual}).configuration().tile ==
                    ops::LinearTile::Split32PrecomputedSums;
            gdn.precomputeSplitSums = prepareSplitSums;
            const ops::GdnStateStrides state{
                geometry_.stateLayout.convolutionLayerBytes(),
                geometry_.stateLayout.recurrentLayerBytes(),
                geometry_.stateLayout.convolutionBytes()};
            if (wide)
              ops::GDN::addDecode16(graph, std::move(gdn),
                                    convolutionScratch, geometry_.gdnShape(),
                                    gdnIndex, state, lanes, gdnRoute);
            else
              ops::GDN::addDecode(graph, std::move(gdn), geometry_.gdnShape(),
                                  lanes, gdnIndex, state);
            operators_.linear().addResidualBatch(
                graph, buffers.gdnHidden,
                mixer.outputProjection, input, buffers.gdnOutput, mixerOutput,
                lanes, stats, buffers.linearScratch, !wide, prepareSplitSums,
                m16HoistFooter);
            residual = buffers.gdnOutput;
            ++gdnIndex;
          } else {
            operators_.linear().addDecodeBatch(graph,
                               inputNormalized, mixer.inputProjection,
                               pad3 ? buffers.fullPackedPadded : buffers.fullPacked,
                               attentionInput, inputLanes,
                               stats, buffers.linearScratch, true, prepareInputSums);
            ops::PagedAttention::addVerifyProjection(
                graph, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm,
                buffers.ropeCos, buffers.ropeSin, buffers.fullQueries,
                buffers.chunkKeys[attentionIndex],
                buffers.chunkValues[attentionIndex],
                ExecutionLimits::targetVerifyRows, tileRows, tileRows,
                geometry_.attentionQueryHeads, geometry_.kvLayout, lanes);
            ops::PagedAttention::addVerify(
                graph, kvLayers[attentionIndex],
                {buffers.chunkKeys[attentionIndex],
                 buffers.chunkValues[attentionIndex], buffers.fullQueries,
                 buffers.attentionPartials, buffers.attentionStatistics,
                 buffers.fullAttention, buffers.pageTables},
                q8, verify, attentionPlan);
            ops::PagedAttention::addVerifyGate(
                graph, buffers.fullPacked, buffers.fullAttention,
                buffers.attentionHidden, ExecutionLimits::targetVerifyRows,
                tileRows, tileRows, geometry_.attentionQueryHeads,
                geometry_.kvLayout, lanes, buffers.linearScratch);
            operators_.linear().addResidualBatch(
                graph, buffers.attentionHidden,
                mixer.outputProjection, input, buffers.attentionOutput,
                mixerOutput, lanes, stats, buffers.linearScratch, true, false,
                m16HoistFooter);
            residual = buffers.attentionOutput;
            ++attentionIndex;
          }
        },
        layer.mixer);

    ops::Normalization::addRms(graph, residual, layer.postAttentionNorm,
                               buffers.normalized, geometry_.hiddenSize, rows, buffers.linearScratch);
    if constexpr (hasDenseFfn<std::remove_cvref_t<decltype(layer)>>) {
      const ops::LinearMatrix up{geometry_.denseIntermediateSize, geometry_.hiddenSize};
      const ops::LinearMatrix down{geometry_.hiddenSize, geometry_.denseIntermediateSize};
      // SPLASH_FFN_FUSED_SUMS (default on): the gate/up kernel writes the down
      // projection's sums; byte-exact outputs and sums, oracle PASS
      // (stack12).
      static const bool fusedFfnSums = metal::envSwitch("SPLASH_FFN_FUSED_SUMS");
      // SPLASH_M16_FFN_SUMS (default on; read at every encode so one binary
      // serves both arms of an A/B): the two-request (B2) 16-row gate/up writes the
      // down input sums. Wide single-request lookup cycles keep the separate sums.
      // B2 lockstep -0.117 ms/cycle, oracle identical.
      const bool m16FfnSums = metal::envSwitch("SPLASH_M16_FFN_SUMS");
      const bool downSplit = operators_.linear().plan({down, rows, ops::LinearPhase::Decode,
              ops::LinearEpilogue::Residual}).configuration().tile ==
              ops::LinearTile::Split32PrecomputedSums;
      const std::string gateUpPipeline = std::string(operators_.linear().plan({up, rows,
              ops::LinearPhase::Decode, ops::LinearEpilogue::GateUp}).pipeline());
      const bool prepareDownSums = downSplit &&
          ((fusedFfnSums && !wide && lanes == 1 &&
            gateUpPipeline == "decode_linear_q4_n256_gate_up") ||
           (m16FfnSums && !wide && lanes == 2 &&
            gateUpPipeline == "decode_linear_q4_n256_gate_up_m16"));
      operators_.linear().addGateUpBatch(graph, buffers.normalized, layer.gateProjection,
                         layer.upProjection, buffers.denseGateScratch,
                         buffers.denseIntermediate, up, lanes, stats,
                         buffers.linearScratch, true, prepareDownSums);
      operators_.linear().addResidualBatch(
          graph, buffers.denseIntermediate,
          layer.downProjection, residual, output, down, lanes, stats,
          buffers.linearScratch, false, prepareDownSums, m16HoistFooter);
    } else {
      ops::MoE::add(
          graph,
          {buffers.normalized, residual, output, buffers.selectedExperts,
           buffers.routingWeights, buffers.tileDescriptors, buffers.tileCount,
           buffers.groupedRoutes, buffers.routeRows, buffers.groupedInput,
           buffers.expertIntermediate, buffers.expertOutput},
          layer.ffn, *moePlan);
    }

    const auto captureLayers = geometry_.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      ops::DraftAttention::captureTargetHidden(
          graph, output, buffers.capturedTargetHidden, rows,
          static_cast<uint32_t>(captured - captureLayers.begin()), 0, 0,
          geometry_.hiddenSize, geometry_.capturedHiddenSize());
    }
  }
  if (gdnIndex != geometry_.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  ops::Normalization::addRms(graph, buffers.hidden[geometry_.layers & 1],
                             std::visit([](const auto *value) {
                               return value->finalNorm;
                             }, weights_),
                             buffers.finalHidden, geometry_.hiddenSize, rows, buffers.linearScratch);
  const ops::LinearMatrix head{geometry_.vocabularySize, geometry_.hiddenSize};
  operators_.linear().addDecodeBatch(graph, buffers.finalHidden,
                     vocabularyProjection(), buffers.logits, head, lanes,
                     stats, buffers.linearScratch, true);
}

void QwenTarget::addHead(metal::CommandGraph &graph,
                         metal::MetalBuffer hidden,
                         metal::MetalBuffer finalHidden,
                         metal::MetalBuffer logits,
                         uint32_t normalizedRows, ops::LinearScratch scratch) const {
  if (!normalizedRows ||
      normalizedRows > ExecutionLimits::targetVerifyRows) {
    throw std::invalid_argument("invalid Qwen head row count");
  }
  const metal::MetalBuffer norm = std::visit(
      [](const auto *weights) { return weights->finalNorm; }, weights_);
  ops::Normalization::addRms(graph, std::move(hidden), norm, finalHidden,
                             geometry_.hiddenSize, normalizedRows);
  const ops::LinearMatrix head{geometry_.vocabularySize, geometry_.hiddenSize};
  operators_.linear().addDecode(graph,
                std::move(finalHidden), vocabularyProjection(),
                std::move(logits), head, scratch);
}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  const ops::Q4Projection &embedding = std::visit(
      [](const auto *weights) -> const ops::Q4Projection & {
        return weights->tokenEmbedding;
      },
      weights_);
  ops::Embedding::add(graph, std::move(tokens), embedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &graph,
                                QwenTargetCommitBuffers buffers,
                                uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  ops::GDN::addCommit(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      geometry_.gdnShape(), geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

void QwenTarget::addStateCommit16(
    metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
    metal::MetalBuffer convolutionScratch, uint32_t tiles) const {
  if (geometry_.ffnKind != QwenFfnKind::Dense)
    throw std::invalid_argument("wide Qwen commit requires a dense target");
  ops::GDN::addCommit16(
      graph,
      {buffers.packed, buffers.mixed, buffers.decay, buffers.beta,
       buffers.currentStates, buffers.nextStates, buffers.retainedCounts},
      std::move(convolutionScratch),
      geometry_.gdnShape(), geometry_.stateLayout.layers,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()}, tiles);
}

} // namespace splash::model

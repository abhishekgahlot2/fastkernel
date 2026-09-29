#include "tuning/TuningWorkloads.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace {

using namespace splash::model;
using namespace splash::ops;

static_assert(!tuning::kPrefillProbeRows.empty() &&
              tuning::kPrefillProbeRows.back() == ExecutionLimits::prefillTokenBudget);
static_assert(tuning::kDecodeProbeWidths == std::array<uint32_t, 4>{1, 2, 3, 4});

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
  throw std::runtime_error("invalid tuning inventory was accepted");
}

// The collector only borrows projection metadata. Empty immutable Metal
// handles make any accidental allocation, weight access or dispatch fail;
// these tests never construct a MetalBackend or load a model package.
Q4Projection projection(uint32_t output, uint32_t input) {
  return {{}, {}, {}, output, input};
}

template <class Weights, class Layout> Weights targetWeights(Layout layout) {
  Weights target;
  target.layout = layout;
  target.logitsProjection = projection(layout.vocabularySize, layout.hiddenSize);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    auto &layer = target.layers.emplace_back();
    if (layout.isFullAttentionLayer(index)) {
      QwenAttentionWeights mixer;
      mixer.inputProjection = projection(layout.packedFullWidth, layout.hiddenSize);
      mixer.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = mixer;
    } else {
      QwenGdnWeights mixer;
      mixer.inputProjection = projection(layout.packedGdnWidth, layout.hiddenSize);
      mixer.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = mixer;
    }
    if constexpr (std::is_same_v<Weights, Qwen3_8Weights>) {
      layer.gateProjection = projection(layout.intermediateSize, layout.hiddenSize);
      layer.upProjection = layer.gateProjection;
      layer.downProjection = projection(layout.hiddenSize, layout.intermediateSize);
    } else {
      const Q8Projection router{{}, {}, {}, 256, layout.hiddenSize};
      const ExpertQ4Projection up{{}, layout.experts, layout.expertIntermediateSize,
                                  layout.hiddenSize, 16'384};
      const ExpertQ4Projection down{{}, layout.experts, layout.hiddenSize,
                                    layout.expertIntermediateSize, 16'384};
      auto sharedUp = up;
      auto sharedDown = down;
      sharedUp.experts = sharedDown.experts = 1;
      layer.ffn = {router, up, up, down, sharedUp, sharedUp, sharedDown, router};
    }
  }
  return target;
}

DFlashDraftWeights draftWeights(DFlashDraftLayout layout) {
  DFlashDraftWeights draft;
  draft.layout = layout;
  draft.contextProjection = projection(layout.hiddenSize, layout.targetHiddenSize);
  draft.selectorProjection = projection(layout.selectorRank, layout.hiddenSize);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    auto &layer = draft.layers.emplace_back();
    layer.attentionDynamic = projection(layout.dynamicSize, layout.hiddenSize);
    layer.mlpDynamic = layer.attentionDynamic;
    layer.qkvProjection = projection(layout.qkvSize, layout.hiddenSize);
    layer.outputProjection = projection(layout.hiddenSize, layout.attentionSize);
    layer.gateProjection = projection(layout.intermediateSize, layout.hiddenSize);
    layer.upProjection = layer.gateProjection;
    layer.downProjection = projection(layout.hiddenSize, layout.intermediateSize);
  }
  return draft;
}

std::set<LinearWorkload> linearKeys(const TuningWorkloads &inventory) {
  std::set<LinearWorkload> result;
  for (const auto &input : inventory.linear) {
    require(result.insert(input.workload).second, "duplicate linear workload");
    require(!input.weights.empty() &&
                input.weights.size() <= tuning::kMaximumLinearTuningRepresentatives,
            "linear representative count is empty or unbounded");
    for (const auto &weights : input.weights) {
      require(weights.projection.inputSize == input.workload.matrix.inputSize &&
                  weights.projection.outputSize == input.workload.matrix.outputSize,
              "linear representative has a different projection geometry");
      require(weights.gate.has_value() ==
                  (input.workload.epilogue == LinearEpilogue::GateUp),
              "gate weight view attached to wrong semantic epilogue");
      if (weights.gate)
        require(weights.gate->inputSize == weights.projection.inputSize &&
                    weights.gate->outputSize == weights.projection.outputSize,
                "fused gate/up lost its matching gate projection");
    }
  }
  require(std::is_sorted(inventory.linear.begin(), inventory.linear.end(),
                        [](const auto &a, const auto &b) {
                          return a.workload < b.workload;
                        }),
          "linear inventory order is nondeterministic");
  return result;
}

// This operation list follows the semantic production calls, independently
// of the collector's layer visitor. In particular, draft residuals are in
// convolution, not linear; target down/mixer residuals are fused linear ops.
std::set<LinearWorkload> expectedLinear(
    const ModelPackage &package, std::span<const uint32_t> prefill,
    std::span<const uint32_t> decode) {
  const auto target = std::visit([](const auto &weights) {
    return qwenTargetGeometry(weights);
  }, package.target);
  const auto &draft = package.draft.layout;
  std::set<LinearWorkload> result;
  auto add = [&](LinearMatrix matrix, LinearPhase phase, LinearEpilogue epilogue) {
    for (uint32_t size : phase == LinearPhase::Prefill ? prefill : decode)
      result.insert({matrix, phase == LinearPhase::Prefill ? size : size * 8,
                      phase, epilogue});
  };
  for (auto phase : {LinearPhase::Prefill, LinearPhase::Decode}) {
    for (auto matrix : {LinearMatrix{target.packedGdnWidth, target.hiddenSize},
                         LinearMatrix{target.packedAttentionWidth, target.hiddenSize},
                         LinearMatrix{draft.hiddenSize, draft.targetHiddenSize},
                         LinearMatrix{draft.qkvSize, draft.hiddenSize}})
      add(matrix, phase, LinearEpilogue::None);
    add({target.hiddenSize, target.attentionWidth}, phase, LinearEpilogue::Residual);
    if (target.ffnKind == QwenFfnKind::Dense)
      add({target.hiddenSize, target.denseIntermediateSize}, phase,
           LinearEpilogue::Residual);
  }
  if (target.ffnKind == QwenFfnKind::Dense) {
    const LinearMatrix up{target.denseIntermediateSize, target.hiddenSize};
    add(up, LinearPhase::Prefill, LinearEpilogue::None);
    add(up, LinearPhase::Prefill, LinearEpilogue::UpWithGate);
    add(up, LinearPhase::Decode, LinearEpilogue::GateUp);
  }
  for (auto matrix : {LinearMatrix{target.vocabularySize, target.hiddenSize},
                       LinearMatrix{draft.dynamicSize, draft.hiddenSize},
                       LinearMatrix{draft.hiddenSize, draft.attentionSize},
                       LinearMatrix{draft.hiddenSize, draft.intermediateSize},
                       LinearMatrix{draft.selectorRank, draft.hiddenSize}})
    add(matrix, LinearPhase::Decode, LinearEpilogue::None);
  add({draft.intermediateSize, draft.hiddenSize}, LinearPhase::Decode,
       LinearEpilogue::GateUp);
  return result;
}

void checkPair(ModelPackage package, bool sparse) {
  const auto startup = collectTuningWorkloads(
      package, tuning::kPrefillProbeRows, tuning::kDecodeProbeWidths);
  require(linearKeys(startup) == expectedLinear(
              package, tuning::kPrefillProbeRows, tuning::kDecodeProbeWidths),
          "startup inventory differs from fixed 2048 prefill and B1-B4 decode");
  require(startup.moe.size() ==
              (sparse ? tuning::kPrefillProbeRows.size() +
                            tuning::kDecodeProbeWidths.size()
                      : 0U),
          "startup MoE inventory contains extra prefill shapes");
  for (const auto &input : startup.moe)
    require(input.workload.phase == MoePhase::Prefill
                ? std::find(tuning::kPrefillProbeRows.begin(),
                            tuning::kPrefillProbeRows.end(),
                            input.workload.rows) !=
                      tuning::kPrefillProbeRows.end()
                : input.workload.rows >= 8 && input.workload.rows <= 32 &&
                      input.workload.rows % 8 == 0,
            "startup MoE inventory contains an unsupported row count");
  // The metadata collector still describes exact ragged rows for correctness
  // and dependency checks; these are not additional calibration workloads.
  constexpr std::array prefill{2048U, 17U, 2048U};
  constexpr std::array decode{4U, 1U, 3U, 2U, 4U};
  const auto inventory = collectTuningWorkloads(package, prefill, decode);
  const auto keys = linearKeys(inventory);
  for (const auto &input : inventory.linear) {
    require(input.weights.size() == 1, "tied empty views were not deduplicated");
    const auto &weight = input.weights.front().projection;
    require(!weight.weights && !weight.scales && !weight.biases,
            "inventory created weight backing");
  }
  require(keys == expectedLinear(package, prefill, decode),
          "inventory differs from production operation/phase/epilogue set");
  const auto geometry = std::visit([](const auto &weights) {
    return qwenTargetGeometry(weights);
  }, package.target);
  require(inventory.targetAttention == AttentionShape{
              geometry.attentionQueryHeads, geometry.attentionKvHeads,
              geometry.attentionHeadDimension} &&
              inventory.draftAttention == package.draft.layout.attentionShape(),
          "attention geometry was inferred from a model-name preset");
  std::set<MoeWorkload> actualMoe;
  for (const auto &input : inventory.moe) {
    require(actualMoe.insert(input.workload).second, "duplicate MoE workload");
    require(input.weights.size() == 1, "tied empty MoE views were not deduplicated");
    for (const auto &weights : input.weights)
      require(weights.expertGate.inputSize == input.workload.shape.hiddenSize &&
                  weights.expertDown.outputSize == input.workload.shape.hiddenSize &&
                  weights.expertGate.experts == input.workload.shape.experts &&
                  weights.sharedGate.experts == 1,
              "MoE representative lost real router/expert geometry");
  }
  std::set<MoeWorkload> expectedMoe;
  if (sparse) {
    for (uint32_t rows : prefill)
      expectedMoe.insert({geometry.moe, rows, MoePhase::Prefill});
    for (uint32_t width : decode)
      expectedMoe.insert({geometry.moe, width * 8, MoePhase::Decode});
  }
  require(actualMoe == expectedMoe, "dense/sparse FFN inventory is incorrect");

  package.descriptor.name = "unseen-paired-model-with-identical-operators";
  package.manifestFingerprintSha256 = "different-weight-identity";
  // An extra same-shape layer and duplicate probe sizes must not multiply the
  // offline sweep. Neither sessions nor manifest names are workload keys.
  std::visit([](auto &target) {
    target.layers.push_back(target.layers.front());
  }, package.target);
  package.draft.layers.push_back(package.draft.layers.front());
  const auto renamed = collectTuningWorkloads(package, prefill, decode);
  require(linearKeys(renamed) == keys &&
              renamed.targetAttention == inventory.targetAttention &&
              renamed.draftAttention == inventory.draftAttention,
          "dedup depends on layer count or model/weight names");
  std::set<MoeWorkload> renamedMoe;
  for (const auto &input : renamed.moe)
    renamedMoe.insert(input.workload);
  require(renamedMoe == actualMoe && renamed.moe.size() == inventory.moe.size(),
          "MoE dedup depends on layer count or model/weight names");
  const auto noDecode = collectTuningWorkloads(package, prefill, {});
  require(linearKeys(noDecode) == expectedLinear(package, prefill, {}),
          "prefill-only sweep introduced decode work");
  const auto empty = collectTuningWorkloads(package, {}, {});
  require(empty.linear.empty() && empty.moe.empty(), "empty probe sets created work");
  rejects([&] { (void)collectTuningWorkloads(package, std::array{0U}, decode); });
  rejects([&] { (void)collectTuningWorkloads(package, std::array{2049U}, decode); });
  rejects([&] { (void)collectTuningWorkloads(package, prefill, std::array{0U}); });
  rejects([&] { (void)collectTuningWorkloads(package, prefill, std::array{5U}); });
  package.draft.contextProjection.inputSize = 0;
  rejects([&] { (void)collectTuningWorkloads(package, prefill, decode); });
}

void run() {
  ModelPackage dense;
  dense.target = targetWeights<Qwen3_8Weights>(Qwen3_8Layout{});
  dense.draft = draftWeights(DFlashDraftLayout{});
  checkPair(dense, false);

  ModelPackage sparse;
  sparse.target = targetWeights<Qwen3_6MoeWeights>(Qwen3_6MoeLayout{});
  DFlashDraftLayout smallerDraft;
  smallerDraft.layers = 6;
  smallerDraft.hiddenSize = 2048;
  smallerDraft.dynamicSize = 512;
  smallerDraft.intermediateSize = 6144;
  smallerDraft.targetHiddenSize = 16384;
  sparse.draft = draftWeights(smallerDraft);
  checkPair(sparse, true);

  Qwen3_8Layout alternateDense;
  alternateDense.hiddenSize = 4096;
  alternateDense.intermediateSize = 14336;
  DFlashDraftLayout alternateDraft;
  alternateDraft.hiddenSize = 4096;
  alternateDraft.dynamicSize = 1024;
  alternateDraft.intermediateSize = 12288;
  alternateDraft.targetHiddenSize = 20480;
  dense.target = targetWeights<Qwen3_8Weights>(alternateDense);
  dense.draft = draftWeights(alternateDraft);
  checkPair(dense, false);

  Qwen3_6MoeLayout alternateSparse;
  alternateSparse.hiddenSize = 1024;
  alternateSparse.gdnKeyHeads = 8;
  alternateSparse.gdnValueHeads = 16;
  alternateSparse.convolutionDimension = 4096;
  alternateSparse.attentionWidth = 2048;
  alternateSparse.attentionQueryHeads = 8;
  alternateSparse.packedGdnWidth = 6400;
  alternateSparse.packedFullWidth = 5120;
  alternateSparse.experts = 32;
  alternateSparse.expertsPerToken = 4;
  alternateSparse.expertIntermediateSize = 1024;
  smallerDraft.hiddenSize = 1024;
  smallerDraft.dynamicSize = 256;
  smallerDraft.intermediateSize = 4096;
  smallerDraft.targetHiddenSize = 8192;
  sparse.target = targetWeights<Qwen3_6MoeWeights>(alternateSparse);
  sparse.draft = draftWeights(smallerDraft);
  checkPair(sparse, true);

  dense.draft.layers.clear();
  rejects([&] { (void)collectTuningWorkloads(dense, std::array{32U}, std::array{1U}); });
  std::get<Qwen3_6MoeWeights>(sparse.target).layers.clear();
  rejects([&] { (void)collectTuningWorkloads(sparse, std::array{32U}, std::array{1U}); });
}

void metadataViews(const char *metallib) {
  using namespace splash::metal;
  MetalBackend backend(metallib);
  const uint64_t submissions = backend.submissionCount();
  // Metadata-only test: one small allocation, no data access or GPU command.
  // Components are intentionally tiny because no projection is executed.
  const auto backing = backend.allocateBuffer(32 * 1024, BufferStorage::Shared,
                                               "tuning-inventory-view-test");
  ModelPackage package;
  package.target = targetWeights<Qwen3_8Weights>(Qwen3_8Layout{});
  package.draft = draftWeights(DFlashDraftLayout{});
  auto &target = std::get<Qwen3_8Weights>(package.target);
  const LinearWorkload key{{target.layout.intermediateSize, target.layout.hiddenSize},
                           8, LinearPhase::Decode, LinearEpilogue::GateUp};
  auto view = [&](uint32_t index, bool gate) {
    const uint64_t offset = uint64_t{index * 6 + (gate ? 3U : 0U)} * 128;
    return Q4Projection{
        backend.view(backing, offset, 128),
        backend.view(backing, offset + 128, 128),
        backend.view(backing, offset + 256, 128),
        key.matrix.outputSize, key.matrix.inputSize};
  };
  enum class Variation { Distinct, GateOnly, ScaleOnly, Tied };
  for (const auto variation : {Variation::Distinct, Variation::GateOnly,
                                Variation::ScaleOnly, Variation::Tied}) {
    for (uint32_t index = 0; index < target.layers.size(); ++index) {
      // Repeat each pair through freshly created view objects. Metadata, not
      // wrapper-object identity, must determine whether a tensor is tied.
      const uint32_t representative = index / 2;
      auto &layer = target.layers[index];
      layer.upProjection = view(variation == Variation::Distinct ? representative : 0, false);
      layer.gateProjection = view(variation == Variation::Distinct ||
                                     variation == Variation::GateOnly ? representative : 0, true);
      if (variation == Variation::ScaleOnly)
        layer.upProjection.scales = view(representative, false).scales;
    }
    // Target and draft execute the same GateUp shape in this pair. They also
    // share a representative here, so this must not add another measurement.
    for (auto &layer : package.draft.layers) {
      layer.upProjection = target.layers.front().upProjection;
      layer.gateProjection = target.layers.front().gateProjection;
    }
    const auto bytes = backend.memoryStats().allocatedBytes;
    const auto inventory = collectTuningWorkloads(package, std::array{32U}, std::array{1U});
    (void)linearKeys(inventory);
    const auto found = std::find_if(inventory.linear.begin(), inventory.linear.end(),
                                    [&](const auto &input) { return input.workload == key; });
    require(found != inventory.linear.end(), "GateUp inventory disappeared");
    const size_t expected = variation == Variation::Tied
        ? 1 : tuning::kMaximumLinearTuningRepresentatives;
    require(found->weights.size() == expected,
            "distinct/tied weight view bound or dedup is wrong");
    // 64 layers have 32 distinct paired views: keep the endpoints and six
    // evenly spaced interior bundles, rather than the first eight bundles.
    constexpr std::array<size_t, 8> sourceLayers{0, 8, 16, 26, 34, 44, 52, 62};
    for (size_t index = 0; index < expected; ++index) {
      const auto &actual = found->weights[index];
      const auto &source = target.layers[sourceLayers[index]];
      require(actual.projection.weights.sameView(source.upProjection.weights) &&
                  actual.projection.scales.sameView(source.upProjection.scales) &&
                  actual.projection.biases.sameView(source.upProjection.biases) &&
                  actual.gate &&
                  actual.gate->weights.sameView(source.gateProjection.weights) &&
                  actual.gate->scales.sameView(source.gateProjection.scales) &&
                  actual.gate->biases.sameView(source.gateProjection.biases),
              "representatives missed layer depth or lost gate/up pairing");
    }
    require(backend.memoryStats().allocatedBytes == bytes,
            "collecting representative views copied weight backing");
  }

  package.target = targetWeights<Qwen3_6MoeWeights>(Qwen3_6MoeLayout{});
  auto &sparse = std::get<Qwen3_6MoeWeights>(package.target);
  DFlashDraftLayout sparseDraft;
  sparseDraft.layers = 6;
  sparseDraft.hiddenSize = 2048;
  sparseDraft.dynamicSize = 512;
  sparseDraft.intermediateSize = 6144;
  sparseDraft.targetHiddenSize = 16384;
  package.draft = draftWeights(sparseDraft);
  auto moeView = [&](uint32_t index) {
    auto buffer = [&](uint32_t component) {
      return backend.view(backing, uint64_t{index * 12 + component} * 128, 128);
    };
    const auto &layout = sparse.layout;
    const Q8Projection router{buffer(0), buffer(1), buffer(2), 256, layout.hiddenSize};
    const Q8Projection sharedRouter{buffer(3), buffer(4), buffer(5), 256, layout.hiddenSize};
    auto expert = [&](uint32_t component, uint32_t count, bool down) {
      return ExpertQ4Projection{buffer(component), count,
          down ? layout.hiddenSize : layout.expertIntermediateSize,
          down ? layout.expertIntermediateSize : layout.hiddenSize, 16'384};
    };
    return MoeWeights{router, expert(6, layout.experts, false),
        expert(7, layout.experts, false), expert(8, layout.experts, true),
        expert(9, 1, false), expert(10, 1, false), expert(11, 1, true), sharedRouter};
  };
  enum class MoeVariation { Distinct, RouterOnly, SharedOnly, StrideOnly, Tied };
  for (const auto variation : {MoeVariation::Distinct, MoeVariation::RouterOnly,
                                MoeVariation::SharedOnly, MoeVariation::StrideOnly,
                                MoeVariation::Tied}) {
    for (uint32_t index = 0; index < sparse.layers.size(); ++index) {
      const uint32_t representative = index / 2;
      auto &weights = sparse.layers[index].ffn;
      weights = moeView(variation == MoeVariation::Distinct ? representative : 0);
      if (variation == MoeVariation::RouterOnly)
        weights.router.scales = moeView(representative).router.scales;
      if (variation == MoeVariation::SharedOnly)
        weights.sharedDown.packed = moeView(representative).sharedDown.packed;
      if (variation == MoeVariation::StrideOnly)
        weights.expertGate.expertStrideBytes += representative * 16'384;
    }
    const auto bytes = backend.memoryStats().allocatedBytes;
    const auto inventory = collectTuningWorkloads(package, std::array{32U}, std::array{1U});
    require(inventory.moe.size() == 2, "MoE metadata changed workload geometry");
    for (const auto &input : inventory.moe) {
      const size_t expected = variation == MoeVariation::Tied
          ? 1 : tuning::kMaximumMoeTuningRepresentatives;
      require(input.weights.size() == expected,
              "MoE representative bound ignored routed/shared weight identities");
      // The 40-layer sparse fixture has 20 distinct router/expert bundles.
      constexpr std::array<size_t, 8> sourceLayers{0, 4, 10, 16, 20, 26, 32, 38};
      for (size_t index = 0; index < expected; ++index) {
        const auto &actual = input.weights[index];
        const auto &source = sparse.layers[sourceLayers[index]].ffn;
        for (auto field : {&MoeWeights::router, &MoeWeights::sharedExpertGate}) {
          const auto &left = actual.*field;
          const auto &right = source.*field;
          require(left.weights.sameView(right.weights) &&
                      left.scales.sameView(right.scales) && left.biases.sameView(right.biases),
                  "MoE representatives missed router layer depth");
        }
        for (auto field : {&MoeWeights::expertGate, &MoeWeights::expertUp,
                            &MoeWeights::expertDown, &MoeWeights::sharedGate,
                            &MoeWeights::sharedUp, &MoeWeights::sharedDown}) {
          const auto &left = actual.*field;
          const auto &right = source.*field;
          require(left.packed.sameView(right.packed) &&
                      left.expertStrideBytes == right.expertStrideBytes,
                  "MoE representative lost router/expert/shared pairing");
        }
      }
    }
    require(backend.memoryStats().allocatedBytes == bytes,
            "collecting MoE representatives copied weight backing");
  }
  require(backend.submissionCount() == submissions,
          "collecting representative metadata submitted GPU work");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 2)
      throw std::invalid_argument("usage: tuning-workloads [metallib]");
    run();
    if (argc == 2)
      metadataViews(argv[1]);
    std::cout << "Tuning workload inventory tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Tuning workload inventory failed: " << error.what() << '\n';
    return 1;
  }
}

// Modified by meowkernels.
#include "model/ModelFactory.hpp"
#include "model/RuntimeArenas.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

using namespace splash;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Exception, class Function>
void rejects(Function function, const char *message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

uint64_t aligned(uint64_t bytes) {
  constexpr uint64_t alignment = 16 * 1024;
  return (bytes + alignment - 1) & ~(alignment - 1);
}

template <class Weights>
model::ModelPackage package() {
  model::ModelPackage result;
  Weights target;
  model::DFlashDraftLayout draft;
  if constexpr (std::is_same_v<Weights, model::Qwen3_6MoeWeights>) {
    draft.layers = 6;
    draft.hiddenSize = 2048;
    draft.dynamicSize = 512;
    draft.intermediateSize = 6144;
    draft.targetHiddenSize = target.layout.capturedHiddenSize();
  }
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.layout.hiddenSize;
  result.descriptor = model::makeModelDescriptor(
      "operator workspace test", target.layout, draft, vision);
  result.target = std::move(target);
  result.draft.layout = draft;
  return result;
}

void checkPackage(const model::ModelPackage &package, uint32_t family) {
  DeviceCapabilities device;
  device.appleGpuFamily = family;
  ops::ExecutionPlans baseline(device);
  const auto before = model::plannedRuntimeMemory(device, package, baseline);
  const auto geometry = std::visit([](const auto &weights) {
    return model::qwenTargetGeometry(weights);
  }, package.target);
  const ops::AttentionShape attention{geometry.attentionQueryHeads,
                                      geometry.attentionKvHeads,
                                      geometry.attentionHeadDimension};
  ops::OperatorChoices choices;
  choices.prefillAttention.push_back(
      {{attention}, {ops::PrefillSplitMultiplier::Two}});
  choices.draftAttention.push_back(
      {{package.draft.layout.attentionShape(), 3}, {80}});
  if (geometry.ffnKind == model::QwenFfnKind::SparseMoe)
    choices.moe.push_back({{geometry.moe, 24, ops::MoePhase::Decode},
                           {ops::MoeExpertTile::M32}});
  ops::ExecutionPlans selected(device);
  selected.install(choices);
  const auto after = model::plannedRuntimeMemory(device, package, selected);
  const auto prefillBefore = baseline.prefillAttentionWorkspace(
      2048, attention.queryHeads, geometry.kvLayout);
  const auto prefillAfter = selected.prefillAttentionWorkspace(
      2048, attention.queryHeads, geometry.kvLayout);
  const uint64_t prefillGrowth =
      aligned(prefillAfter.partialsBytes) - aligned(prefillBefore.partialsBytes) +
      aligned(prefillAfter.statisticsBytes) - aligned(prefillBefore.statisticsBytes);
  // The selected split count and the fallback baseline share an arena whose
  // governed bound includes the larger candidate's exact scratch requirement.
  require(prefillGrowth > 0 &&
              after.sharedPrefillPlannedAllocatedBytes ==
                  before.sharedPrefillPlannedAllocatedBytes + prefillGrowth,
          "runtime prefill allocation lost the selected split workspace bound");
  const auto selectedPrefill = selected.prefillAttention(
      2048, attention.queryHeads, geometry.kvLayout, 131072);
  require(selectedPrefill.configuration.splitMultiplier == ops::PrefillSplitMultiplier::Two &&
              selectedPrefill.workspace.partialsBytes ==
                  2 * baseline.prefillAttention(2048, attention.queryHeads,
                                             geometry.kvLayout, 131072)
                      .workspace.partialsBytes,
          "runtime did not install the selected prefill split plan");

  uint64_t decodeGrowth = 0;
  if (geometry.ffnKind == model::QwenFfnKind::SparseMoe) {
    constexpr std::array fields{
        &ops::MoeWorkspace::selectedExpertsBytes,
        &ops::MoeWorkspace::routingWeightsBytes,
        &ops::MoeWorkspace::tileDescriptorsBytes,
        &ops::MoeWorkspace::tileCountBytes,
        &ops::MoeWorkspace::groupedRoutesBytes,
        &ops::MoeWorkspace::routeRowsBytes,
        &ops::MoeWorkspace::groupedInputBytes,
        &ops::MoeWorkspace::expertIntermediateBytes,
        &ops::MoeWorkspace::expertOutputBytes};
    const auto oldMoe = baseline.moeDecodeWorkspacePerLane(geometry.moe);
    const auto newMoe = selected.moeDecodeWorkspacePerLane(geometry.moe);
    for (auto field : fields)
      decodeGrowth += aligned(4 * (newMoe.*field)) - aligned(4 * (oldMoe.*field));
    require(decodeGrowth > 0, "M24 expert plan did not reserve larger scratch");
  }
  require(after.sharedDecodePlannedAllocatedBytes ==
              before.sharedDecodePlannedAllocatedBytes + decodeGrowth,
          "runtime decode allocation does not use all selected width bounds");
  require(after.activeStateCellPlannedAllocatedBytes ==
              before.activeStateCellPlannedAllocatedBytes &&
              after.pipelineReserveBytes == before.pipelineReserveBytes &&
              after.runtimeOverheadReserveBytes == before.runtimeOverheadReserveBytes,
          "kernel selection changed state or unrelated memory reserves");
  require(selected.draftAttention(package.draft.layout.attentionShape(), 3)
                  .configuration().groups == 80,
          "paired draft did not use the same selection owner");
  selected.install({});
  const auto reset = model::plannedRuntimeMemory(device, package, selected);
  require(reset.sharedPrefillPlannedAllocatedBytes ==
              before.sharedPrefillPlannedAllocatedBytes &&
              reset.sharedDecodePlannedAllocatedBytes ==
              before.sharedDecodePlannedAllocatedBytes,
          "reset left stale selected workspace");
}

void sumsOnlyLinearArena(const model::ModelPackage &package,
                         metal::MetalBackend *backend = nullptr) {
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  // Keep this fixture's baseline scratch-free, then install one sums-only
  // choice explicitly; serving defaults on the 40-core device also use sums.
  device.gpuCoreCount = 16;
  const auto geometry = model::RuntimeGeometry::from(package);
  const ops::LinearWorkload workload{
      {geometry.target.hiddenSize, geometry.target.denseIntermediateSize}, 8,
      ops::LinearPhase::Decode, ops::LinearEpilogue::Residual};
  ops::OperatorChoices choices;
  choices.linear.push_back({workload,
      {ops::LinearTile::Split32PrecomputedSums,
       workload.matrix.outputSize / 32, ops::LinearSimdgroups::Four}});
  // The default-on 16/24/32-row narrow split-K rules would give this 16-core
  // fixture the drafter's 32-row split sums too (larger than the chosen
  // workload's). Planners read the switches when built: pin them off for this
  // one, then restore whatever the caller exported.
  const std::array<const char *, 2> switches{"SPLASH_M16_NARROW_SPLIT", "SPLASH_M24_NARROW_SPLIT"};
  std::array<std::optional<std::string>, 2> saved;
  for (size_t i = 0; i < switches.size(); ++i) {
    if (const char *value = std::getenv(switches[i])) saved[i] = value;
    require(setenv(switches[i], "0", 1) == 0, "cannot pin the narrow split switches");
  }
  ops::ExecutionPlans plans(device);
  for (size_t i = 0; i < switches.size(); ++i) {
    if (saved[i]) setenv(switches[i], saved[i]->c_str(), 1);
    else unsetenv(switches[i]);
  }
  plans.install(choices);
  const auto size = model::DecodeArena::linearScratchSize(geometry, plans);
  require(!size.input &&
              size.sums == uint64_t{workload.rows} *
                  (workload.matrix.inputSize / 64) * sizeof(float) &&
              !size.partials && !size.counters,
          "sums-only Linear choice reserved unrelated arena fields");
  if (!backend) return;
  model::DecodeArena arena(*backend, geometry, plans);
  const auto scratch = arena.linearScratch();
  require(!scratch.input && scratch.sums &&
              scratch.sums.sizeBytes() == size.sums && !scratch.partials &&
              !scratch.counters,
          "sums-only Linear arena allocated unrelated scratch fields");
}

void decodeArenaViewCache(const model::ModelPackage &package,
                          metal::MetalBackend &backend) {
  const auto geometry = model::RuntimeGeometry::from(package);
  ops::ExecutionPlans plans(backend.capabilities());
  const uint64_t before = backend.memoryStats().allocatedBytes;
  {
    model::DecodeArena arena(backend, geometry, plans);
    const uint64_t allocated = backend.memoryStats().allocatedBytes;
    const auto lane0 = arena.get(0, model::DecodeTensor::Hidden0);
    const auto lane1 = arena.get(1, model::DecodeTensor::Hidden0);
    require(arena.packed(model::DecodeTensor::Hidden0, 1).sameView(lane0),
            "width-one packed view does not reuse lane zero");
    for (uint32_t lanes = 2; lanes <= model::kLaneCount; ++lanes) {
      const auto first = arena.packed(model::DecodeTensor::Hidden0, lanes);
      const auto second = arena.packed(model::DecodeTensor::Hidden0, lanes);
      require(first.sameView(second) &&
                  first.sizeBytes() == uint64_t{lanes} * lane0.sizeBytes() &&
                  first.contents() == lane0.contents(),
              "cached packed view identity, length or base offset is wrong");
    }
    require(static_cast<const uint8_t *>(lane1.contents()) ==
                static_cast<const uint8_t *>(lane0.contents()) +
                    lane0.sizeBytes(),
            "decode lane views are not adjacent");
    require(!arena.packed(model::DecodeTensor::MoeSelectedExperts,
                          model::kLaneCount),
            "inactive model-specific tensor acquired a cached view");

    constexpr std::array gdnBases{
        model::DecodeTensor::VerifyPackedBase,
        model::DecodeTensor::VerifyMixedBase,
        model::DecodeTensor::VerifyDecayBase,
        model::DecodeTensor::VerifyBetaBase};
    const uint32_t gdnLayers = geometry.target.stateLayout.layers;
    for (const auto base : gdnBases) {
      const auto storage = arena.gdnStorage(base);
      require(storage.sameView(arena.gdnStorage(base)),
              "cached GDN storage identity changed");
      const uint64_t stride = storage.sizeBytes() /
          (uint64_t{gdnLayers} * model::kLaneCount);
      for (uint32_t layer : {0U, gdnLayers - 1}) {
        for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes) {
          const auto first = arena.gdnBatchSlice(base, layer, lanes);
          const auto second = arena.gdnBatchSlice(base, layer, lanes);
          require(first.sameView(second) &&
                      first.sizeBytes() == uint64_t{lanes} * stride &&
                      static_cast<const uint8_t *>(first.contents()) ==
                          static_cast<const uint8_t *>(storage.contents()) +
                              uint64_t{layer} * model::kLaneCount * stride,
                  "cached GDN layer view identity, length or offset is wrong");
        }
      }
    }

    constexpr std::array attentionBases{model::DecodeTensor::ChunkKeysBase,
                                         model::DecodeTensor::ChunkValuesBase};
    const uint32_t attentionLayers = geometry.target.kvLayout.attentionLayers;
    const uint64_t attentionStride = model::decodeChunkLayerBytes(geometry);
    for (const auto base : attentionBases) {
      const auto origin = arena.attentionBatchSlice(base, 0, 1);
      for (uint32_t layer : {0U, attentionLayers - 1}) {
        for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes) {
          const auto first = arena.attentionBatchSlice(base, layer, lanes);
          const auto second = arena.attentionBatchSlice(base, layer, lanes);
          require(first.sameView(second) &&
                      first.sizeBytes() == uint64_t{lanes} * attentionStride &&
                      static_cast<const uint8_t *>(first.contents()) ==
                          static_cast<const uint8_t *>(origin.contents()) +
                              uint64_t{layer} * model::kLaneCount *
                                  attentionStride,
                  "cached attention layer identity, length or offset is wrong");
        }
      }
    }

    rejects<std::out_of_range>(
        [&] { (void)arena.packed(model::DecodeTensor::Hidden0, 0); },
        "zero packed width was accepted");
    rejects<std::out_of_range>(
        [&] {
          (void)arena.packed(model::DecodeTensor::Hidden0,
                             model::kLaneCount + 1);
        },
        "oversized packed width was accepted");
    rejects<std::logic_error>(
        [&] {
          (void)arena.packed(model::DecodeTensor::VerifyPackedBase, 1);
        },
        "layer-major tensor was accepted as packed scratch");
    rejects<std::out_of_range>(
        [&] {
          (void)arena.gdnBatchSlice(model::DecodeTensor::VerifyPackedBase,
                                    gdnLayers, 1);
        },
        "out-of-range GDN layer was accepted");
    rejects<std::invalid_argument>(
        [&] { (void)arena.gdnStorage(model::DecodeTensor::Hidden0); },
        "non-GDN tensor was accepted as GDN storage");
    rejects<std::out_of_range>(
        [&] {
          (void)arena.attentionBatchSlice(model::DecodeTensor::ChunkKeysBase,
                                          attentionLayers, 1);
        },
        "out-of-range attention layer was accepted");
    uint32_t guardCalls = 0;
    backend.setOperationGuard([&] { ++guardCalls; });
    (void)arena.packed(model::DecodeTensor::Hidden0, 4);
    (void)arena.gdnBatchSlice(model::DecodeTensor::VerifyPackedBase, 0, 4);
    (void)arena.gdnStorage(model::DecodeTensor::VerifyPackedBase);
    (void)arena.attentionBatchSlice(model::DecodeTensor::ChunkKeysBase, 0, 4);
    backend.setOperationGuard({});
    require(guardCalls == 0,
            "healthy cached view retrieval invoked the operation guard");
    require(backend.memoryStats().allocatedBytes == allocated,
            "cached view retrieval changed GPU allocation accounting");
  }
  require(backend.memoryStats().allocatedBytes == before,
          "cached arena views retained the allocation after arena destruction");
}

// All-Macs G0: wide lookups run only up to the rows whose K reductions match the
// 8-row verify's in every projection (the runtime checks this at startup).
void rowStableVerifyWidths(const model::ModelPackage &package) {
  const auto geometry = model::RuntimeGeometry::from(package);
  const auto widest = [&](uint32_t family, uint32_t cores, std::optional<bool> sums = {}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = cores;
    const ops::ExecutionPlans plans(device);
    const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
    return model::rowStableVerifyRows(plans.linear(), geometry.target,
                                      sums.value_or(scratch.sums && !scratch.input));
  };
  for (const uint32_t cores : {8U, 10U})
    require(widest(9, cores) == 16,
            "Apple9: 32-row input projections and head leave the simdgroup K splits");
  for (const uint32_t cores : {8U, 10U, 16U, 40U})
    require(widest(10, cores) == 32, "Apple10: 16- and 32-row verifies must stay row-stable");
  // With sums scratch, the 8-row GDN input (Paired256 at 8 cores) reads no sums,
  // so its 16/32-row verifies must not either.
  require(widest(10, 8, true) == 32, "8-core Apple10: M16/M24 input sums without 8-row sums");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc <= 2, "usage: model-execution-plans [METALLIB | --row-stable-verify]");
    const auto dense = package<model::Qwen3_8Weights>();
    if (argc == 2 && std::string_view(argv[1]) == "--row-stable-verify") {
      // Serving flags, set before any plan reads them (some are read once per process).
      for (const char *flag : {"SPLASH_INPUT_FUSED_SUMS", "SPLASH_M16_INPUT_SUMS", "SPLASH_M24_INPUT_SUMS",
                               "SPLASH_NARROW_SPLIT", "SPLASH_M16_NARROW_SPLIT", "SPLASH_M24_NARROW_SPLIT",
                               "SPLASH_M16_SPLIT_RESIDUAL", "SPLASH_WIDE_PROMPT_LOOKUP"})
        setenv(flag, "1", 1);
      rowStableVerifyWidths(dense);
      std::cout << "model execution plans: row-stable verify widths PASS "
                   "(Apple9 8/10c: 16 rows; Apple10 8/10/16/40c: 32)\n";
      return 0;
    }
    const auto sparse = package<model::Qwen3_6MoeWeights>();
    for (uint32_t family : {9U, 10U}) {
      checkPackage(dense, family);
      checkPackage(sparse, family);
    }
    sumsOnlyLinearArena(dense);
    if (argc == 2) {
      metal::MetalBackend backend(argv[1]);
      sumsOnlyLinearArena(dense, &backend);
      decodeArenaViewCache(dense, backend);
    }
    std::cout << "model execution plans: PASS (two paired geometries)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

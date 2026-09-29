// Modified by meowkernels.
// GDN verify decode and commit kernels against a direct CPU reference: the
// four-tap convolution with SiLU, the q/k RMS norms, the gates, the eight-row
// delta-rule recurrence over the fp32 state, the gated RMSNorm of the
// recurrent rows and the convolution carry, for both compiled geometries,
// every lane count, a two-layer state cell (so the layer offsets are
// exercised) and every retained count of the commit. The default-off wide
// compositions (16 rows = 2 M8 tiles, 32 rows = 4) are checked at every
// retained count against the same oracle.
// The recurrence and the
// gate are checked from the kernel's own q/k/v and gates after those were
// checked against the reference, so their tolerances stay at fp32 accuracy.
#include "metal/EnvSwitch.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/GDN.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kMaxLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kLayers = 2;
constexpr std::array kShapes{GdnShape{16, 48, 128, 10240, 16640},
                             GdnShape{16, 32, 128, 8192, 12544}};

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid GDN request was accepted");
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

double roundBfloat(double value) {
  return fromBfloat(toBfloat(static_cast<float>(value)));
}

// One bf16 unit in the last place at the reference's magnitude.
double bfloatUlp(double reference) {
  int exponent = 0;
  std::frexp(std::fabs(reference), &exponent);
  return std::ldexp(1.0, exponent - 8);
}

bool closeBfloat(uint16_t got, double reference, double ulps, double floor) {
  return std::fabs(double(fromBfloat(got)) - reference) <=
         ulps * bfloatUlp(reference) + floor;
}

bool closeFloat(float got, double reference, double tolerance) {
  return std::fabs(double(got) - reference) <=
         tolerance * (1.0 + std::fabs(reference));
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

uint64_t align16k(uint64_t bytes) { return (bytes + 16383) & ~uint64_t{16383}; }

double sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }

// The production state cell layout: every layer's conv rows, then every
// layer's recurrent state, both padded to 16 KiB.
struct Cell final {
  uint64_t convLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convBytes = 0;
  uint64_t bytes = 0;

  explicit Cell(const GdnShape &shape)
      : convLayerBytes(align16k(uint64_t{3} * shape.convolutionDimension * 2)),
        recurrentLayerBytes(align16k(uint64_t{shape.valueHeads} * kHeadDim *
                                     kHeadDim * 4)),
        convBytes(kLayers * convLayerBytes),
        bytes(convBytes + kLayers * recurrentLayerBytes) {}

  GdnStateStrides strides() const {
    return {convLayerBytes, recurrentLayerBytes, convBytes};
  }
  const uint16_t *conv(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const uint16_t *>(cell + layer * convLayerBytes);
  }
  const float *recurrent(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const float *>(cell + convBytes +
                                           layer * recurrentLayerBytes);
  }
};

struct Fixture final {
  const GdnShape &shape;
  Cell cell;
  uint32_t lanes;
  uint32_t tiles;  // > 1: one request's rows as that many M8 tiles
  MetalBuffer packed, convWeights, mixed, decayWeights, timeBias, decay, beta,
      recurrent, mixerNorm, hidden, arrived, generation, retained;
  std::array<MetalBuffer, kMaxLanes> current, next;
  std::vector<MetalBuffer> packedLayer, mixedLayer, decayLayer, betaLayer;
  uint64_t packedStride, mixedStride, gateStride;

  Fixture(MetalBackend &backend, const GdnShape &geometry, uint32_t laneCount,
          uint32_t wideTiles = 1)
      : shape(geometry), cell(geometry), lanes(laneCount), tiles(wideTiles),
        packedStride(uint64_t{kRows} * shape.packedWidth),
        mixedStride(uint64_t{kRows} * shape.convolutionDimension),
        gateStride(uint64_t{kRows} * shape.valueHeads) {
    Random random(0x6D4E1234ULL + laneCount);
    auto alloc = [&](uint64_t bytes, const char *label) {
      return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
    };
    auto fill = [&](MetalBuffer &buffer, float scale) {
      auto *values = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t index = 0; index < buffer.sizeBytes() / 2; ++index)
        values[index] = toBfloat(random.unit() * scale);
    };
    packed = alloc(kLayers * kMaxLanes * packedStride * 2, "gdn packed");
    fill(packed, 1.0F);
    convWeights = alloc(uint64_t{shape.convolutionDimension} * 4 * 2,
                        "gdn conv weights");
    fill(convWeights, 0.5F);
    mixed = alloc(kLayers * kMaxLanes * mixedStride * 2, "gdn mixed");
    decayWeights = alloc(uint64_t{shape.valueHeads} * 4, "gdn decay weights");
    for (uint32_t head = 0; head < shape.valueHeads; ++head)
      static_cast<float *>(decayWeights.contents())[head] =
          -std::exp(random.unit() * 1.5F);
    timeBias = alloc(uint64_t{shape.valueHeads} * 2, "gdn time bias");
    fill(timeBias, 0.5F);
    decay = alloc(kLayers * kMaxLanes * gateStride * 4, "gdn decay");
    beta = alloc(kLayers * kMaxLanes * gateStride * 2, "gdn beta");
    const uint64_t rowBytes =
        uint64_t{kMaxLanes} * kRows * shape.valueHeads * kHeadDim * 2;
    recurrent = alloc(rowBytes, "gdn recurrent rows");
    hidden = alloc(rowBytes, "gdn hidden");
    mixerNorm = alloc(uint64_t{kHeadDim} * 2, "gdn mixer norm");
    fill(mixerNorm, 1.0F);
    arrived = alloc(kMaxLanes * 4, "gdn arrived");
    generation = alloc(kMaxLanes * 4, "gdn generation");
    retained = alloc(kMaxLanes * 4, "gdn retained");
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      current[lane] = alloc(cell.bytes, "gdn current");
      next[lane] = alloc(cell.bytes, "gdn next");
      auto *bytes = static_cast<uint8_t *>(current[lane].contents());
      auto *conv = reinterpret_cast<uint16_t *>(bytes);
      for (uint64_t index = 0; index < cell.convBytes / 2; ++index)
        conv[index] = toBfloat(random.unit());
      auto *state = reinterpret_cast<float *>(bytes + cell.convBytes);
      for (uint64_t index = 0; index < (cell.bytes - cell.convBytes) / 4;
           ++index)
        state[index] = random.unit() * 0.5F;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      packedLayer.push_back(backend.view(packed,
                                         layer * kMaxLanes * packedStride * 2,
                                         kMaxLanes * packedStride * 2));
      mixedLayer.push_back(backend.view(mixed,
                                        layer * kMaxLanes * mixedStride * 2,
                                        kMaxLanes * mixedStride * 2));
      decayLayer.push_back(backend.view(decay,
                                        layer * kMaxLanes * gateStride * 4,
                                        kMaxLanes * gateStride * 4));
      betaLayer.push_back(backend.view(beta, layer * kMaxLanes * gateStride * 2,
                                       kMaxLanes * gateStride * 2));
    }
    clear();
  }

  void clear() {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      std::memset(next[lane].contents(), 0, cell.bytes);
    for (MetalBuffer *buffer :
         {&mixed, &decay, &beta, &recurrent, &hidden, &arrived, &generation})
      std::memset(buffer->contents(), 0, buffer->sizeBytes());
  }

  GdnDecodeBuffers decodeBuffers(uint32_t layer) const {
    return {packedLayer[layer], convWeights,     current,
            next,               mixedLayer[layer], decayWeights,
            timeBias,           decayLayer[layer], betaLayer[layer],
            recurrent,          mixerNorm,       hidden,
            arrived,            generation};
  }
  GdnCommitBuffers commitBuffers() const {
    return {packed, mixed, decay, beta, current, next, retained};
  }

  uint32_t rowCount() const { return tiles * kRows; }

  uint32_t physicalRow(uint32_t lane, uint32_t token) const {
    return tiles > 1 ? token : lane * kRows + token;
  }

  // Row `token` of lane `lane` in layer `layer` of the packed projection.
  const uint16_t *packedRow(uint32_t layer, uint32_t lane,
                            uint32_t token) const {
    return static_cast<const uint16_t *>(packed.contents()) +
           uint64_t{layer} * kMaxLanes * packedStride +
           uint64_t{physicalRow(lane, token)} * shape.packedWidth;
  }
  const uint16_t *mixedRow(uint32_t layer, uint32_t lane,
                           uint32_t token) const {
    return static_cast<const uint16_t *>(mixed.contents()) +
           uint64_t{layer} * kMaxLanes * mixedStride +
           uint64_t{physicalRow(lane, token)} * shape.convolutionDimension;
  }
  const float *decayRow(uint32_t layer, uint32_t lane, uint32_t token) const {
    return static_cast<const float *>(decay.contents()) +
           uint64_t{layer} * kMaxLanes * gateStride +
           uint64_t{physicalRow(lane, token)} * shape.valueHeads;
  }
  const uint16_t *betaRow(uint32_t layer, uint32_t lane,
                          uint32_t token) const {
    return static_cast<const uint16_t *>(beta.contents()) +
           uint64_t{layer} * kMaxLanes * gateStride +
           uint64_t{physicalRow(lane, token)} * shape.valueHeads;
  }
  const uint16_t *rowsOf(const MetalBuffer &buffer, uint32_t lane,
                         uint32_t token, uint32_t head) const {
    return static_cast<const uint16_t *>(buffer.contents()) +
           (uint64_t{physicalRow(lane, token)} * shape.valueHeads + head) *
               kHeadDim;
  }
  const uint8_t *cellBytes(const MetalBuffer &buffer) const {
    return static_cast<const uint8_t *>(buffer.contents());
  }
};

// Four-tap causal convolution of one channel at one token, three carried
// rows then the command's rows, rounded to bf16 and gated by SiLU.
double convolutionSilu(const Fixture &fixture, uint32_t layer, uint32_t lane,
                       uint32_t token, uint32_t channel) {
  const uint16_t *weights =
      static_cast<const uint16_t *>(fixture.convWeights.contents()) +
      uint64_t{channel} * 4;
  const uint16_t *carried =
      fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  double value = 0.0;
  for (uint32_t tap = 0; tap < 4; ++tap) {
    const uint32_t position = token + tap;
    const uint16_t input =
        position < 3
            ? carried[position * fixture.shape.convolutionDimension + channel]
            : fixture.packedRow(layer, lane, position - 3)[channel];
    value += double(fromBfloat(input)) * fromBfloat(weights[tap]);
  }
  value = roundBfloat(value);
  return roundBfloat(value * sigmoid(value));
}

uint16_t convolutionCarry(const Fixture &fixture, uint32_t layer,
                          uint32_t lane, uint32_t consumed, uint32_t row,
                          uint32_t channel) {
  const uint32_t source = consumed + row;
  const uint16_t *carried =
      fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  return source < 3
             ? carried[source * fixture.shape.convolutionDimension + channel]
             : fixture.packedRow(layer, lane, source - 3)[channel];
}

void checkConvolution(const Fixture &fixture, uint32_t layer, uint32_t lane,
                      const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  for (uint32_t token = 0; token < fixture.rowCount(); ++token) {
    const uint16_t *mixedRow = fixture.mixedRow(layer, lane, token);
    std::vector<double> conv(shape.convolutionDimension);
    for (uint32_t channel = 0; channel < shape.convolutionDimension; ++channel)
      conv[channel] = convolutionSilu(fixture, layer, lane, token, channel);
    // q and k: RMS-normalised per key head, rounded, then scaled and
    // rounded again.
    for (uint32_t part = 0; part < 2; ++part) {
      const double scale = part == 0 ? 0.0078125 : 0.08838834765;
      for (uint32_t head = 0; head < shape.keyHeads; ++head) {
        const uint32_t base = part * keyWidth + head * kHeadDim;
        double squares = 0.0;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim)
          squares += conv[base + dim] * conv[base + dim];
        const double inverse = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
          const double normalized = roundBfloat(conv[base + dim] * inverse);
          require(closeBfloat(mixedRow[base + dim], normalized * scale, 3.0,
                              1e-6),
                  where + ": mixed q/k row mismatch");
        }
      }
    }
    for (uint32_t channel = 2 * keyWidth; channel < shape.convolutionDimension;
         ++channel)
      require(closeBfloat(mixedRow[channel], conv[channel], 3.0, 1e-6),
              where + ": mixed v row mismatch");
  }
}

void checkGates(const Fixture &fixture, uint32_t layer, uint32_t lane,
                const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t bOffset = shape.convolutionDimension +
                           shape.valueHeads * kHeadDim;
  const uint32_t aOffset = bOffset + shape.valueHeads;
  const auto *decayWeights =
      static_cast<const float *>(fixture.decayWeights.contents());
  const auto *timeBias =
      static_cast<const uint16_t *>(fixture.timeBias.contents());
  for (uint32_t token = 0; token < fixture.rowCount(); ++token) {
    const uint16_t *packed = fixture.packedRow(layer, lane, token);
    const float *decay = fixture.decayRow(layer, lane, token);
    const uint16_t *beta = fixture.betaRow(layer, lane, token);
    for (uint32_t head = 0; head < shape.valueHeads; ++head) {
      const double b = fromBfloat(packed[bOffset + head]);
      require(closeBfloat(beta[head], sigmoid(b), 2.0, 1e-6),
              where + ": beta mismatch");
      const double x = roundBfloat(double(fromBfloat(packed[aOffset + head])) +
                                   fromBfloat(timeBias[head]));
      const double softplus =
          std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
      // The kernel rounds softplus to bf16 with fast transcendentals, so a
      // value near a rounding boundary may land one bf16 step away.
      bool matched = false;
      const uint16_t rounded = toBfloat(static_cast<float>(softplus));
      for (int step = -1; step <= 1 && !matched; ++step) {
        const double candidate =
            fromBfloat(static_cast<uint16_t>(rounded + step));
        matched = closeFloat(decay[head],
                             std::exp(double(decayWeights[head]) * candidate),
                             1e-4);
      }
      require(matched, where + ": decay mismatch");
    }
  }
}

// The delta rule over `tokens` rows of one value head from the lane's
// incoming state, driven by the kernel's own q/k/v rows and gates. Returns
// the final state; the recurrent output rows go to `rows` when requested.
std::vector<double> recurrence(const Fixture &fixture, uint32_t layer,
                               uint32_t lane, uint32_t head, uint32_t tokens,
                               std::vector<double> *rows) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  const uint32_t keyHead = head / (shape.valueHeads / shape.keyHeads);
  const float *stateIn =
      fixture.cell.recurrent(fixture.cellBytes(fixture.current[lane]), layer) +
      uint64_t{head} * kHeadDim * kHeadDim;
  std::vector<double> state(stateIn, stateIn + kHeadDim * kHeadDim);
  if (rows)
    rows->assign(uint64_t{tokens} * kHeadDim, 0.0);
  for (uint32_t token = 0; token < tokens; ++token) {
    const uint16_t *mixed = fixture.mixedRow(layer, lane, token);
    const uint16_t *query = mixed + keyHead * kHeadDim;
    const uint16_t *key = mixed + keyWidth + keyHead * kHeadDim;
    const uint16_t *value = mixed + 2 * keyWidth + head * kHeadDim;
    const double decay = fixture.decayRow(layer, lane, token)[head];
    const double beta = fromBfloat(fixture.betaRow(layer, lane, token)[head]);
    for (uint32_t valueDim = 0; valueDim < kHeadDim; ++valueDim) {
      double *row = state.data() + uint64_t{valueDim} * kHeadDim;
      double memory = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim) {
        row[keyDim] *= decay;
        memory += row[keyDim] * fromBfloat(key[keyDim]);
      }
      const double delta = (fromBfloat(value[valueDim]) - memory) * beta;
      double output = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim) {
        row[keyDim] += fromBfloat(key[keyDim]) * delta;
        output += row[keyDim] * fromBfloat(query[keyDim]);
      }
      if (rows)
        (*rows)[uint64_t{token} * kHeadDim + valueDim] = output;
    }
  }
  return state;
}

void checkState(const Fixture &fixture, uint32_t layer, uint32_t lane,
                uint32_t head, const std::vector<double> &expected,
                const std::string &where) {
  const float *state =
      fixture.cell.recurrent(fixture.cellBytes(fixture.next[lane]), layer) +
      uint64_t{head} * kHeadDim * kHeadDim;
  for (uint32_t index = 0; index < kHeadDim * kHeadDim; ++index)
    require(closeFloat(state[index], expected[index], 1e-4),
            where + ": recurrent state mismatch");
}

void checkCarry(const Fixture &fixture, uint32_t layer, uint32_t lane,
                uint32_t consumed, const std::string &where) {
  const uint16_t *carry =
      fixture.cell.conv(fixture.cellBytes(fixture.next[lane]), layer);
  for (uint32_t row = 0; row < 3; ++row)
    for (uint32_t channel = 0; channel < fixture.shape.convolutionDimension;
         ++channel)
      require(carry[row * fixture.shape.convolutionDimension + channel] ==
                  convolutionCarry(fixture, layer, lane, consumed, row,
                                   channel),
              where + ": convolution carry mismatch");
}

void checkDecode(const Fixture &fixture, uint32_t layer, uint32_t lane) {
  const GdnShape &shape = fixture.shape;
  const std::string where = "decode layer " + std::to_string(layer) +
                            " lane " + std::to_string(lane);
  checkConvolution(fixture, layer, lane, where);
  checkGates(fixture, layer, lane, where);
  checkCarry(fixture, layer, lane, fixture.rowCount(), where);
  // Every layer writes the recurrent rows and the hidden rows into the same
  // scratch, as the model graph does, so those hold the last layer's values.
  const bool lastLayer = layer + 1 == kLayers;
  std::vector<double> rows;
  for (uint32_t head = 0; head < shape.valueHeads; ++head) {
    const std::vector<double> state =
        recurrence(fixture, layer, lane, head, fixture.rowCount(), &rows);
    checkState(fixture, layer, lane, head, state, where);
    if (!lastLayer)
      continue;
    for (uint32_t token = 0; token < fixture.rowCount(); ++token) {
      const uint16_t *recurrent =
          fixture.rowsOf(fixture.recurrent, lane, token, head);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        require(closeBfloat(recurrent[dim], rows[token * kHeadDim + dim], 2.0,
                            1e-6),
                where + ": recurrent row mismatch");
    }
  }
  if (!lastLayer)
    return;
  // The gated RMSNorm, from the kernel's own recurrent rows.
  const auto *norm =
      static_cast<const uint16_t *>(fixture.mixerNorm.contents());
  const uint32_t zOffset = shape.convolutionDimension;
  for (uint32_t token = 0; token < fixture.rowCount(); ++token) {
    const uint16_t *packed = fixture.packedRow(layer, lane, token);
    for (uint32_t head = 0; head < shape.valueHeads; ++head) {
      const uint16_t *recurrent =
          fixture.rowsOf(fixture.recurrent, lane, token, head);
      const uint16_t *hidden =
          fixture.rowsOf(fixture.hidden, lane, token, head);
      double squares = 0.0;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double row = fromBfloat(recurrent[dim]);
        squares += row * row;
      }
      const double inverse = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double normalized = roundBfloat(
            fromBfloat(recurrent[dim]) * inverse * fromBfloat(norm[dim]));
        const double gate = fromBfloat(packed[zOffset + head * kHeadDim + dim]);
        require(closeBfloat(hidden[dim], normalized * gate * sigmoid(gate), 2.0,
                            1e-6),
                where + ": hidden mismatch");
      }
    }
  }
}

void requireUntouched(const Fixture &fixture, uint32_t lane,
                      const std::string &where) {
  const uint8_t *bytes = fixture.cellBytes(fixture.next[lane]);
  for (uint64_t index = 0; index < fixture.cell.bytes; ++index)
    require(bytes[index] == 0, where + ": idle lane cell was written");
}

void runDecode(MetalBackend &backend, const GdnShape &shape, uint32_t lanes) {
  Fixture fixture(backend, shape, lanes);
  CommandGraph graph;
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    GDN::addDecode(graph, fixture.decodeBuffers(layer), shape, lanes, layer,
                   fixture.cell.strides());
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const std::string where = "lanes " + std::to_string(lanes);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    const auto *generation =
        static_cast<const uint32_t *>(fixture.generation.contents());
    const auto *arrived =
        static_cast<const uint32_t *>(fixture.arrived.contents());
    require(arrived[lane] == 0, where + ": completion counter not reset");
    if (lane >= lanes) {
      require(generation[lane] == 0, where + ": idle lane completed");
      requireUntouched(fixture, lane, where);
      continue;
    }
    require(generation[lane] == kLayers, where + ": layer completion count");
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      checkDecode(fixture, layer, lane);
  }

  // The commit replays the retained rows from the incoming cell over the
  // decoded q/k/v and gates; eight retained rows leave the decoded cell.
  std::vector<std::vector<uint8_t>> decoded;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    decoded.emplace_back(fixture.cellBytes(fixture.next[lane]),
                         fixture.cellBytes(fixture.next[lane]) +
                             fixture.cell.bytes);
  CommandGraph commit;
  GDN::addCommit(commit, fixture.commitBuffers(), shape, kLayers, lanes,
                 fixture.cell.strides());
  for (uint32_t base = 1; base <= kRows; ++base) {
    auto *retained = static_cast<uint32_t *>(fixture.retained.contents());
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      retained[lane] = 1 + (base + lane - 1) % kRows;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      std::memcpy(fixture.next[lane].contents(), decoded[lane].data(),
                  fixture.cell.bytes);
    static_cast<void>(backend.submitCommand(commit.dispatches()));
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t count = retained[lane];
      const std::string commitWhere =
          where + " commit retained " + std::to_string(count) + " lane " +
          std::to_string(lane);
      if (count == kRows) {
        require(std::memcmp(fixture.next[lane].contents(),
                            decoded[lane].data(), fixture.cell.bytes) == 0,
                commitWhere + ": full acceptance rewrote the cell");
        continue;
      }
      // Every head at the shortest, a middle and the longest partial replay;
      // a spread of heads for the other counts keeps the double-precision
      // replay short under shader validation.
      std::vector<uint32_t> heads{0, shape.valueHeads / 2,
                                  shape.valueHeads - 1};
      if (count == 1 || count == 4 || count == 7) {
        heads.clear();
        for (uint32_t head = 0; head < shape.valueHeads; ++head)
          heads.push_back(head);
      }
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        checkCarry(fixture, layer, lane, count, commitWhere);
        for (uint32_t head : heads)
          checkState(fixture, layer, lane, head,
                     recurrence(fixture, layer, lane, head, count, nullptr),
                     commitWhere);
      }
    }
  }
}

// The wide decode against `tiles` chained M8 decodes, each continuing from
// the previous tile's fully accepted state.
void checkDecodeWideGpuOracle(MetalBackend &backend, const Fixture &fixture,
                              uint32_t layer) {
  const GdnShape &shape = fixture.shape;
  const uint32_t tiles = fixture.tiles;
  const std::string label = std::to_string(tiles * kRows) + "-row";
  auto alloc = [&](uint64_t bytes, const char *name) {
    return backend.allocateBuffer(bytes, BufferStorage::Shared, name);
  };
  std::vector<std::array<MetalBuffer, kMaxLanes>> states(tiles + 1);
  for (auto &cells : states) {
    for (MetalBuffer &cell : cells) {
      cell = alloc(fixture.cell.bytes, "wide reference state");
      std::memset(cell.contents(), 0, fixture.cell.bytes);
    }
  }
  std::memcpy(states[0][0].contents(), fixture.current[0].contents(),
              fixture.cell.bytes);
  auto mixed = alloc(tiles * fixture.mixedStride * 2, "wide reference mixed");
  auto decay = alloc(tiles * fixture.gateStride * 4, "wide reference decay");
  auto beta = alloc(tiles * fixture.gateStride * 2, "wide reference beta");
  auto recurrent =
      alloc(fixture.recurrent.sizeBytes(), "wide reference recurrent");
  auto hidden = alloc(fixture.hidden.sizeBytes(), "wide reference hidden");
  auto arrived = alloc(sizeof(uint32_t), "wide reference arrived");
  auto generation = alloc(sizeof(uint32_t), "wide reference generation");
  std::memset(arrived.contents(), 0, arrived.sizeBytes());
  std::memset(generation.contents(), 0, generation.sizeBytes());
  const uint64_t rowBytes = uint64_t{shape.valueHeads} * kHeadDim * 2;
  CommandGraph graph;
  for (uint32_t tile = 0; tile < tiles; ++tile) {
    GdnDecodeBuffers buffers = fixture.decodeBuffers(layer);
    buffers.packed = backend.view(
        fixture.packedLayer[layer], uint64_t{tile} * fixture.packedStride * 2,
        fixture.packedStride * 2);
    buffers.currentStates = std::span(states[tile]);
    buffers.nextStates = std::span(states[tile + 1]);
    buffers.mixed = backend.view(mixed, uint64_t{tile} * fixture.mixedStride * 2,
                                 fixture.mixedStride * 2);
    buffers.decay = backend.view(decay, uint64_t{tile} * fixture.gateStride * 4,
                                 fixture.gateStride * 4);
    buffers.beta = backend.view(beta, uint64_t{tile} * fixture.gateStride * 2,
                                fixture.gateStride * 2);
    buffers.recurrent = backend.view(recurrent, uint64_t{tile} * kRows * rowBytes,
                                     uint64_t{kRows} * rowBytes);
    buffers.hidden = backend.view(hidden, uint64_t{tile} * kRows * rowBytes,
                                  uint64_t{kRows} * rowBytes);
    buffers.arrived = arrived;
    buffers.generation = generation;
    GDN::addDecode(graph, buffers, shape, 1, layer,
                   fixture.cell.strides());
  }
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const uint64_t outputBytes = uint64_t{tiles * kRows} * rowBytes;
  require(std::memcmp(fixture.recurrent.contents(), recurrent.contents(),
                      outputBytes) == 0,
          label + " recurrent outputs differ from chained accepted M8 scans");
  require(std::memcmp(fixture.hidden.contents(), hidden.contents(),
                      outputBytes) == 0,
          label + " hidden outputs differ from chained accepted M8 scans");
  const uint8_t *wideState = fixture.cellBytes(fixture.next[0]);
  const uint8_t *referenceState = fixture.cellBytes(states[tiles][0]);
  require(std::memcmp(wideState + layer * fixture.cell.convLayerBytes,
                      referenceState + layer * fixture.cell.convLayerBytes,
                      fixture.cell.convLayerBytes) == 0,
          label + " convolution state differs from chained accepted M8 scans");
  const uint64_t recurrentOffset =
      fixture.cell.convBytes + layer * fixture.cell.recurrentLayerBytes;
  require(std::memcmp(wideState + recurrentOffset,
                      referenceState + recurrentOffset,
                      fixture.cell.recurrentLayerBytes) == 0,
          label + " recurrent state differs from chained accepted M8 scans");
}

// The wide commit at `count` retained rows against the 8-row kernels on the
// same inputs: the fully accepted tiles decoded one after another, then the
// partial tile decoded and committed at its own retained count. The whole
// state cell (every layer's conv carry and recurrent state) must match byte
// for byte. `wideNext` is the wide commit's result.
void checkCommitWideGpuOracle(MetalBackend &backend, const Fixture &fixture,
                              const MetalBuffer &wideNext, uint32_t count,
                              const std::string &where) {
  const GdnShape &shape = fixture.shape;
  auto alloc = [&](uint64_t bytes, const char *name) {
    auto buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, name);
    std::memset(buffer.contents(), 0, bytes);
    return buffer;
  };
  auto cells = [&](const char *name) {
    std::array<MetalBuffer, kMaxLanes> lanes;
    for (MetalBuffer &cell : lanes)
      cell = alloc(fixture.cell.bytes, name);
    return lanes;
  };
  auto state = cells("m8 reference state");
  std::memcpy(state[0].contents(), fixture.current[0].contents(),
              fixture.cell.bytes);
  // The 8-row tape: [layer][lane][rows], lane 0 used.
  auto packed = alloc(fixture.packed.sizeBytes(), "m8 reference packed");
  auto mixed = alloc(fixture.mixed.sizeBytes(), "m8 reference mixed");
  auto decay = alloc(fixture.decay.sizeBytes(), "m8 reference decay");
  auto beta = alloc(fixture.beta.sizeBytes(), "m8 reference beta");
  auto recurrent = alloc(fixture.recurrent.sizeBytes(), "m8 reference recurrent");
  auto hidden = alloc(fixture.hidden.sizeBytes(), "m8 reference hidden");
  auto arrived = alloc(kMaxLanes * 4, "m8 reference arrived");
  auto generation = alloc(kMaxLanes * 4, "m8 reference generation");
  const uint64_t packedLayer = kMaxLanes * fixture.packedStride * 2;
  const uint64_t mixedLayer = kMaxLanes * fixture.mixedStride * 2;
  const uint64_t decayLayer = kMaxLanes * fixture.gateStride * 4;
  const uint64_t betaLayer = kMaxLanes * fixture.gateStride * 2;
  // Tile `tile` decoded by the 8-row kernel from `from` into `to`, every layer.
  auto decodeTile = [&](uint32_t tile, std::array<MetalBuffer, kMaxLanes> &from,
                        std::array<MetalBuffer, kMaxLanes> &to) {
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      std::memcpy(static_cast<uint8_t *>(packed.contents()) + layer * packedLayer,
                  static_cast<const uint8_t *>(fixture.packed.contents()) +
                      layer * packedLayer + uint64_t{tile} * fixture.packedStride * 2,
                  fixture.packedStride * 2);
    std::memset(arrived.contents(), 0, arrived.sizeBytes());
    std::memset(generation.contents(), 0, generation.sizeBytes());
    CommandGraph graph;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      GdnDecodeBuffers buffers = fixture.decodeBuffers(layer);
      buffers.packed = backend.view(packed, layer * packedLayer, packedLayer);
      buffers.mixed = backend.view(mixed, layer * mixedLayer, mixedLayer);
      buffers.decay = backend.view(decay, layer * decayLayer, decayLayer);
      buffers.beta = backend.view(beta, layer * betaLayer, betaLayer);
      buffers.recurrent = recurrent;
      buffers.hidden = hidden;
      buffers.currentStates = std::span(from);
      buffers.nextStates = std::span(to);
      buffers.arrived = arrived;
      buffers.generation = generation;
      GDN::addDecode(graph, buffers, shape, 1, layer, fixture.cell.strides());
    }
    static_cast<void>(backend.submitCommand(graph.dispatches()));
  };
  const uint32_t full = count / kRows;
  const uint32_t rest = count % kRows;
  for (uint32_t tile = 0; tile < full; ++tile) {
    auto next = cells("m8 reference next");
    decodeTile(tile, state, next);
    state = next;
  }
  if (rest) {
    auto next = cells("m8 reference partial");
    decodeTile(full, state, next);
    auto retained = alloc(kMaxLanes * 4, "m8 reference retained");
    static_cast<uint32_t *>(retained.contents())[0] = rest;
    CommandGraph commit;
    GDN::addCommit(commit, {packed, mixed, decay, beta, state, next, retained},
                   shape, kLayers, 1, fixture.cell.strides());
    static_cast<void>(backend.submitCommand(commit.dispatches()));
    state = next;
  }
  // Every layer's live bytes: the 3 carried conv rows and the recurrent state
  // (the cells' 16 KiB padding is never read or written by either path).
  const auto *wide = fixture.cellBytes(wideNext);
  const auto *reference = fixture.cellBytes(state[0]);
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    const uint64_t conv = uint64_t{layer} * fixture.cell.convLayerBytes;
    const uint64_t recurrentAt =
        fixture.cell.convBytes + uint64_t{layer} * fixture.cell.recurrentLayerBytes;
    require(std::memcmp(wide + conv, reference + conv,
                        uint64_t{3} * shape.convolutionDimension * 2) == 0,
            where + ": conv carry differs from the 8-row decode + commit, layer " +
                std::to_string(layer));
    require(std::memcmp(wide + recurrentAt, reference + recurrentAt,
                        uint64_t{shape.valueHeads} * kHeadDim * kHeadDim * 4) == 0,
            where + ": recurrent state differs from the 8-row decode + commit, layer " +
                std::to_string(layer));
  }
}

void runDecodeWide(MetalBackend &backend, const GdnShape &shape,
                   uint32_t tiles, WideGdn route = WideGdn::Chain) {
  Fixture fixture(backend, shape, 1, tiles);
  const std::string label =
      std::to_string(tiles * kRows) + "-row" +
      (route == WideGdn::Single ? " single-pass" : route == WideGdn::SingleParts ? " single-pass parts" : "");
  const uint64_t slots = tiles > 2 ? 2 : 1;
  require(gdnDecode16ConvolutionScratchBytes(fixture.cell.strides(), tiles) ==
              slots * fixture.cell.convLayerBytes &&
              gdnCommit16ConvolutionScratchBytes(fixture.cell.strides(), tiles) ==
                  slots * fixture.cell.convBytes,
          label + " GDN scratch contract changed");
  auto forwardScratch = backend.allocateBuffer(
      gdnDecode16ConvolutionScratchBytes(fixture.cell.strides(), tiles));
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    CommandGraph graph;
    GDN::addDecode16(graph, fixture.decodeBuffers(layer), forwardScratch,
                     shape, layer, fixture.cell.strides(), tiles, route);
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    if (shape.valueHeads == 48)
      checkDecodeWideGpuOracle(backend, fixture, layer);
  }
  const auto *generation =
      static_cast<const uint32_t *>(fixture.generation.contents());
  const auto *arrived =
      static_cast<const uint32_t *>(fixture.arrived.contents());
  for (uint32_t tile = 0; tile < tiles; ++tile) {
    require(arrived[tile] == 0, label + " completion counters did not reset");
    require(generation[tile] == kLayers,
            label + " tiles did not each complete every layer");
  }
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    checkDecode(fixture, layer, 0);
  for (uint32_t lane = 1; lane < kMaxLanes; ++lane)
    requireUntouched(fixture, lane, label + " idle physical lane");

  std::vector<uint8_t> decoded(
      fixture.cellBytes(fixture.next[0]),
      fixture.cellBytes(fixture.next[0]) + fixture.cell.bytes);
  auto commitScratch = backend.allocateBuffer(
      gdnCommit16ConvolutionScratchBytes(fixture.cell.strides(), tiles));
  CommandGraph commit;
  GDN::addCommit16(commit, fixture.commitBuffers(), commitScratch, shape,
                   kLayers, fixture.cell.strides(), tiles);
  auto *retained = static_cast<uint32_t *>(fixture.retained.contents());
  for (uint32_t count = 0; count <= tiles * kRows; ++count) {
    retained[0] = count;
    std::memcpy(fixture.next[0].contents(), decoded.data(), fixture.cell.bytes);
    static_cast<void>(backend.submitCommand(commit.dispatches()));
    const std::string where =
        label + " commit retained " + std::to_string(count);
    checkCommitWideGpuOracle(backend, fixture, fixture.next[0], count, where);
    if (count == tiles * kRows) {
      require(std::memcmp(fixture.next[0].contents(), decoded.data(),
                          fixture.cell.bytes) == 0,
              where + ": full acceptance rewrote the cell");
      continue;
    }
    std::vector<uint32_t> heads{0, shape.valueHeads / 2,
                                shape.valueHeads - 1};
    // Every head at the tile edges: none, a tile's last row, the next tile's first.
    if (count % kRows <= 1) {
      heads.clear();
      for (uint32_t head = 0; head < shape.valueHeads; ++head)
        heads.push_back(head);
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      checkCarry(fixture, layer, 0, count, where);
      for (uint32_t head : heads)
        checkState(fixture, layer, 0, head,
                   recurrence(fixture, layer, 0, head, count, nullptr),
                   where);
    }
  }
}

void splitSumsPreparation(MetalBackend &backend) {
  const auto &shape=kShapes[0];
  Fixture fixture(backend,shape,1);
  const uint32_t width=shape.valueHeads*shape.headDimension;
  const uint64_t bytes=uint64_t{width}/64*8*sizeof(float);
  auto referenceSums=backend.allocateBuffer(bytes);
  auto guarded=backend.allocateBuffer(bytes+128);
  std::memset(guarded.contents(),0xcd,guarded.sizeBytes());
  auto sums=backend.view(guarded,64,bytes);
  const bool parts4=splash::metal::envSwitch("SPLASH_GDN_VALUE_PARTS","4");
  for (uint32_t layer : {0u,kLayers-1}) {
    fixture.clear();
    CommandGraph reference;
    GDN::addDecode(reference,fixture.decodeBuffers(layer),shape,1,layer,
                   fixture.cell.strides());
    reference.add("decode_linear_q4_split_sums",{fixture.hidden,referenceSums},
                  width,{width/256,1,1},{128,1,1});
    require(reference.dispatches().size()==2 &&
            reference.dispatches()[0].pipelineName=="verify_gdn_fused" &&
            reference.dispatches()[1].pipelineName=="decode_linear_q4_split_sums",
            "compiled reference route changed under value-parts flag");
    (void)backend.submitCommand(reference.dispatches());
    const auto *hidden=static_cast<const uint8_t *>(fixture.hidden.contents());
    const auto *state=static_cast<const uint8_t *>(fixture.next[0].contents());
    const std::vector<uint8_t> expectedHidden(hidden,hidden+fixture.hidden.sizeBytes());
    const std::vector<uint8_t> expectedState(state,state+fixture.next[0].sizeBytes());
    for (uint32_t repeat=0;repeat<2;++repeat) {
      fixture.clear();
      auto buffers=fixture.decodeBuffers(layer);
      buffers.linearScratch.sums=sums;
      buffers.precomputeSplitSums=true;
      CommandGraph fused;
      GDN::addDecode(fused,buffers,shape,1,layer,fixture.cell.strides());
      const auto dispatches=fused.dispatches();
      if (parts4) {
        require(dispatches.size()==2 &&
                dispatches[0].pipelineName=="verify_gdn_value_parts4_scan" &&
                dispatches[0].threadgroups.x==192 &&
                dispatches[0].threadsPerThreadgroup.x==256 &&
                dispatches[1].pipelineName=="verify_gdn_value_parts4_finalize" &&
                dispatches[1].threadgroups.x==48 &&
                dispatches[1].threadsPerThreadgroup.x==256,
                "Parts 4 split-sums route shape mismatch");
      } else {
        require(dispatches.size()==1 &&
                dispatches[0].pipelineName=="verify_gdn_fused_split_sums" &&
                dispatches[0].threadgroups.x==48 &&
                dispatches[0].threadsPerThreadgroup.x==256,
                "fused sums route not selected");
      }
      (void)backend.submitCommand(dispatches);
      require(!std::memcmp(expectedHidden.data(),fixture.hidden.contents(),expectedHidden.size()),
              "split sums changed GDN output");
      require(!std::memcmp(expectedState.data(),fixture.next[0].contents(),expectedState.size()),
              "split sums changed GDN state");
      require(!std::memcmp(sums.contents(),referenceSums.contents(),bytes),
              "GDN split sums differ from original reduction order");
      const auto *guard=static_cast<const uint8_t *>(guarded.contents());
      for(uint32_t n=0;n<64;++n)
        require(guard[n]==0xcd && guard[64+bytes+n]==0xcd,"sum guard overwritten");
    }
  }
  std::cout << "GDN split sums: layers 0/last route/bytes/guard/repeat PASS\n";
}

void fusedPreparation(MetalBackend &backend, const GdnShape &shape, uint32_t lanes) {
  Fixture fixture(backend, shape, lanes);
  const uint32_t width = shape.valueHeads * shape.headDimension;
  auto table = backend.allocateBuffer(width * 16 * lanes);
  auto sums = backend.allocateBuffer(width / 2 * lanes);
  auto referenceTable = backend.allocateBuffer(width * 16 * lanes);
  auto referenceSums = backend.allocateBuffer(width / 2 * lanes);
  CommandGraph reference;
  GDN::addDecode(reference, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides());
  reference.add("decode_linear_q4_prepare", {fixture.hidden, referenceTable, referenceSums},
                width, {width / 32, lanes, 1}, {128, 1, 1});
  (void)backend.submitCommand(reference.dispatches());
  std::vector<uint8_t> expected(width * 16 * lanes);
  std::memcpy(expected.data(), fixture.hidden.contents(), expected.size());
  fixture.clear();
  auto buffers = fixture.decodeBuffers(0);
  buffers.linearScratch = {table, sums, {}, {}};
  CommandGraph fused;
  GDN::addDecode(fused, buffers, shape, lanes, 0, fixture.cell.strides());
  (void)backend.submitCommand(fused.dispatches());
  require(!std::memcmp(expected.data(), fixture.hidden.contents(), expected.size()),
          "fused GDN changed output");
  require(!std::memcmp(table.contents(), referenceTable.contents(), width * 16 * lanes),
          "fused GDN table mismatch");
  require(!std::memcmp(sums.contents(), referenceSums.contents(), width / 2 * lanes),
          "fused GDN sums mismatch");
  for (uint32_t lane=0;lane<lanes;++lane) checkDecode(fixture, 0, lane);
}

void rejectsInvalid(MetalBackend &backend) {
  const GdnShape &shape = kShapes[1];
  Fixture fixture(backend, shape, 1);
  CommandGraph graph;
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0), shape, 0, 0,
                   fixture.cell.strides());
  });
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0), shape, kMaxLanes + 1, 0,
                   fixture.cell.strides());
  });
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0),
                   GdnShape{16, 40, 128, 9216, 14400}, 1, 0,
                   fixture.cell.strides());
  });
  rejects([&] {
    GDN::addCommit(graph, fixture.commitBuffers(), shape, 0, 1,
                   fixture.cell.strides());
  });
  rejects([&] {
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch.input = backend.allocateBuffer(16);
    buffers.linearScratch.sums = backend.allocateBuffer(4);
    GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides());
  });
  rejects([&] {
    GDN::addDecode16(graph, fixture.decodeBuffers(0),
                     backend.allocateBuffer(16), shape, 0,
                     fixture.cell.strides());
  });
  rejects([&] {
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch.input = backend.allocateBuffer(16);
    buffers.linearScratch.sums = backend.allocateBuffer(4);
    GDN::addDecode16(
        graph, buffers,
        backend.allocateBuffer(
            gdnDecode16ConvolutionScratchBytes(fixture.cell.strides())),
        shape, 0, fixture.cell.strides());
  });
  rejects([&] {
    GDN::addCommit16(graph, fixture.commitBuffers(),
                     backend.allocateBuffer(16), shape, kLayers,
                     fixture.cell.strides());
  });
  require(graph.empty(), "invalid GDN request partially encoded a graph");
}

// `gdn-decode METALLIB --bench`: GPU time of one decode cycle's GDN work at
// model shape (48 layers, 48 value heads), serial encoder as in production:
// the 8-row kernel, 2/4 parallel lanes, and the wide 16/32-row tile chains.
// Timing only (median of 10 commands after 3 warm-ups); no correctness check.
void bench(MetalBackend &backend) {
  constexpr uint32_t layers = 48, reps = 10, warm = 3;
  const GdnShape &shape = kShapes[0];
  const uint64_t convLayer = align16k(uint64_t{3} * shape.convolutionDimension * 2);
  const uint64_t recurrentLayer =
      align16k(uint64_t{shape.valueHeads} * kHeadDim * kHeadDim * 4);
  const GdnStateStrides strides{convLayer, recurrentLayer, layers * convLayer};
  const uint64_t cellBytes = layers * (convLayer + recurrentLayer);
  Random random(0xB3C4ULL);
  auto alloc = [&](uint64_t bytes, float scale, bool bf16) {
    auto buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, "gdn bench");
    if (bf16) {
      auto *values = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t i = 0; i < bytes / 2; ++i) values[i] = toBfloat(random.unit() * scale);
    } else {
      auto *values = static_cast<float *>(buffer.contents());
      for (uint64_t i = 0; i < bytes / 4; ++i) values[i] = random.unit() * scale;
    }
    return buffer;
  };
  const uint64_t packedStride = uint64_t{kRows} * shape.packedWidth * 2;
  const uint64_t mixedStride = uint64_t{kRows} * shape.convolutionDimension * 2;
  const uint64_t gateStride = uint64_t{kRows} * shape.valueHeads;
  auto packed = alloc(layers * kMaxLanes * packedStride, 1.0F, true);
  auto mixed = alloc(layers * kMaxLanes * mixedStride, 0.0F, true);
  auto decay = alloc(layers * kMaxLanes * gateStride * 4, 0.0F, false);
  auto beta = alloc(layers * kMaxLanes * gateStride * 2, 0.0F, true);
  auto convWeights = alloc(uint64_t{shape.convolutionDimension} * 4 * 2, 0.5F, true);
  auto decayWeights = alloc(uint64_t{shape.valueHeads} * 4, 0.5F, false);
  auto timeBias = alloc(uint64_t{shape.valueHeads} * 2, 0.5F, true);
  auto mixerNorm = alloc(kHeadDim * 2, 1.0F, true);
  const uint64_t rowBytes = uint64_t{kMaxLanes} * kRows * shape.valueHeads * kHeadDim * 2;
  auto recurrent = alloc(rowBytes, 0.0F, true);
  auto hidden = alloc(rowBytes, 0.0F, true);
  auto arrived = alloc(kMaxLanes * 4, 0.0F, false);
  auto generation = alloc(kMaxLanes * 4, 0.0F, false);
  auto scratch = alloc(gdnDecode16ConvolutionScratchBytes(strides, 4), 0.0F, true);
  std::array<MetalBuffer, kMaxLanes> current, next;
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    current[lane] = alloc(cellBytes, 0.25F, false);
    next[lane] = alloc(cellBytes, 0.0F, false);
  }
  auto buffers = [&](uint32_t layer) {
    return GdnDecodeBuffers{
        backend.view(packed, layer * kMaxLanes * packedStride, kMaxLanes * packedStride),
        convWeights, current, next,
        backend.view(mixed, layer * kMaxLanes * mixedStride, kMaxLanes * mixedStride),
        decayWeights, timeBias,
        backend.view(decay, layer * kMaxLanes * gateStride * 4, kMaxLanes * gateStride * 4),
        backend.view(beta, layer * kMaxLanes * gateStride * 2, kMaxLanes * gateStride * 2),
        recurrent, mixerNorm, hidden, arrived, generation};
  };
  struct Variant { const char *name; uint32_t lanes, tiles, rows; WideGdn route; };
  for (const Variant variant : {Variant{"wide 32 single pass parts", 1, 4, 32, WideGdn::SingleParts},
                                Variant{"wide 16 single pass parts", 1, 2, 16, WideGdn::SingleParts},
                                Variant{"wide 32 single pass", 1, 4, 32, WideGdn::Single},
                                Variant{"wide 16 single pass", 1, 2, 16, WideGdn::Single},
                                Variant{"wide 32 (4 tiles)", 1, 4, 32, WideGdn::Chain},
                                Variant{"wide 16 (2 tiles)", 1, 2, 16, WideGdn::Chain},
                                Variant{"m8 (1 lane)", 1, 0, 8, WideGdn::Chain},
                                Variant{"2 lanes", 2, 0, 16, WideGdn::Chain},
                                Variant{"4 lanes", 4, 0, 32, WideGdn::Chain}}) {
    CommandGraph graph;
    for (uint32_t layer = 0; layer < layers; ++layer) {
      if (variant.tiles)
        GDN::addDecode16(graph, buffers(layer), scratch, shape, layer, strides, variant.tiles,
                         variant.route);
      else
        GDN::addDecode(graph, buffers(layer), shape, variant.lanes, layer, strides);
    }
    std::vector<double> ms;
    for (uint32_t rep = 0; rep < warm + reps; ++rep) {
      const auto timing = backend.submitCommand(graph.dispatches());
      if (rep >= warm) ms.push_back(timing.gpuSeconds * 1e3);
    }
    std::sort(ms.begin(), ms.end());
    const double median = ms[ms.size() / 2];
    std::cout << "gdn bench " << variant.name << ": " << graph.dispatches().size()
              << " dispatches, GPU " << median << " ms per cycle (min " << ms.front()
              << ", max " << ms.back() << "), " << median / variant.rows << " ms per row\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2 && !(argc == 3 && std::string_view(argv[2]) == "--bench"))
      throw std::invalid_argument("usage: gdn-decode METALLIB [--bench]");
    MetalBackend backend(argv[1]);
    if (argc == 3) {
      bench(backend);
      return 0;
    }
    splitSumsPreparation(backend);
    rejectsInvalid(backend);
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes=1;lanes<=kMaxLanes;++lanes) fusedPreparation(backend, shape, lanes);
    for (const GdnShape &shape : kShapes)
      for (uint32_t tiles : {2u, 4u})
        for (WideGdn route : {WideGdn::Chain, WideGdn::Single, WideGdn::SingleParts})
          runDecodeWide(backend, shape, tiles, route);
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        runDecode(backend, shape, lanes);
    std::cout << "gdn_decode_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gdn_decode_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

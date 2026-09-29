// Row-invariance audit (diagnostic, lead 2026-09-29 02:0x): for the Q4 projection families the 8-row verify
// and the 16-row wide-lookup verify use at the live flags (decode-profile B1/B2 kernel lists), run the M8 kernel
// on rows 0-7 and the M16 kernel on rows 0-15 whose rows 0-7 are the same input, and report whether rows 0-7 of
// the two outputs are byte-identical. Reports only; exits 0 unless a dispatch fails.
#include "../../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Linear.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kRows = 8;
constexpr uint32_t kInput = 5120;
constexpr uint32_t kOutput = 16640;
constexpr uint32_t kQuantGroup = 64;

struct Buffers {
  MetalBuffer input, weights, scales, biases, residual, sums8, sums16;
};

ComputeDispatch dispatch(std::string pipeline, std::vector<MetalBuffer> buffers, const void *params,
                         uint32_t paramsSize, uint32_t groups, uint32_t threads) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  for (uint32_t index = 0; index < buffers.size(); ++index)
    result.buffers.push_back({index, std::move(buffers[index])});
  result.bytes = {{static_cast<uint32_t>(buffers.size()), params, paramsSize}};
  result.threadgroups = {groups, 1, 1};
  result.threadsPerThreadgroup = {threads, 1, 1};
  return result;
}

// Prints the comparison and returns how many elements differ (a diagnostic for most rows; the PAD3 rows gate on it).
uint64_t report(const std::string &what, const MetalBuffer &rows8, const MetalBuffer &rows16,
                uint32_t rows = kRows) {
  const uint64_t elements = uint64_t{rows} * kOutput;
  const auto *a = static_cast<const uint16_t *>(rows8.contents());
  const auto *b = static_cast<const uint16_t *>(rows16.contents());
  uint64_t differ = 0;
  float maxDiff = 0;
  for (uint64_t index = 0; index < elements; ++index) {
    if (a[index] == b[index]) continue;
    ++differ;
    const float x = std::bit_cast<float>(uint32_t{a[index]} << 16);
    const float y = std::bit_cast<float>(uint32_t{b[index]} << 16);
    maxDiff = std::max(maxDiff, std::fabs(x - y));
  }
  std::cout << "row_invariance " << what << ": rows 0-" << rows - 1 << " " << (differ ? "DIFFER" : "identical") << " ("
            << differ << "/" << elements << " elements, max |diff| " << maxDiff << ")\n";
  return differ;
}

void run(const std::string &metallib) {
  MetalBackend backend(metallib);
  const auto shared = [&](uint64_t bytes, const char *label) {
    return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  };
  const uint64_t weightElements = uint64_t{kInput} * kOutput;
  const uint64_t parameters = weightElements / kQuantGroup;
  Buffers b{shared(4 * kRows * kInput * 2, "input"), shared(weightElements / 2, "weights"),
            shared(parameters * 2, "scales"), shared(parameters * 2, "biases"),
            shared(4 * kRows * kOutput * 2, "residual"),
            shared(uint64_t{kRows} * (kInput / kQuantGroup) * 4, "sums8"),
            shared(uint64_t{2 * kRows} * (kInput / kQuantGroup) * 4, "sums16")};
  std::mt19937 random(7319);
  std::uniform_real_distribution<float> values(-1.0f, 1.0f), small(-0.02f, 0.02f);
  auto *input = static_cast<__bf16 *>(b.input.contents());
  for (uint64_t i = 0; i < 4ull * kRows * kInput; ++i) input[i] = __bf16(values(random));
  auto *weights = static_cast<uint8_t *>(b.weights.contents());
  for (uint64_t i = 0; i < weightElements / 2; ++i) weights[i] = static_cast<uint8_t>(random());
  auto *scales = static_cast<__bf16 *>(b.scales.contents());
  auto *biases = static_cast<__bf16 *>(b.biases.contents());
  for (uint64_t i = 0; i < parameters; ++i) {
    scales[i] = __bf16(small(random));
    biases[i] = __bf16(small(random));
  }
  auto *residual = static_cast<__bf16 *>(b.residual.contents());
  for (uint64_t i = 0; i < 4ull * kRows * kOutput; ++i) residual[i] = __bf16(values(random));
  const MetalBuffer input8 = backend.view(b.input, 0, uint64_t{kRows} * kInput * 2);
  const MetalBuffer residual8 = backend.view(b.residual, 0, uint64_t{kRows} * kOutput * 2);
  const uint32_t inputSize = kInput;
  // The exact Split32 input sums each consumer family reads (M8: [group][8 rows], M16: [group][16 rows]).
  (void)backend.submit(dispatch("decode_linear_q4_split_sums", {input8, b.sums8}, &inputSize,
                                sizeof(inputSize), kInput / 256, 128));
  (void)backend.submit(dispatch("decode_linear_q4_split_sums_m16", {b.input, b.sums16}, &inputSize,
                                sizeof(inputSize), kInput / 256, 256));

  const Q4Params split{kOutput, kInput, kOutput / 32};
  const auto out = [&](uint32_t rows, const char *label) {
    return shared(uint64_t{rows} * kOutput * 2, label);
  };
  // Input projections: the M8 split4 precomputed-sums bodies (plain, footer, hoist, hoist+footer) vs M16's.
  MetalBuffer m16 = out(16, "m16");
  (void)backend.submit(dispatch("decode_linear_q4_n32_split4_precomputed_sums_m16",
                                {b.input, b.weights, b.scales, b.biases, m16, b.sums16}, &split,
                                sizeof(split), kOutput / 32, 256));
  for (const char *suffix : {"", "_ftr", "_hoist", "_hoist_ftr"}) {
    MetalBuffer m8 = out(8, "m8");
    (void)backend.submit(dispatch(std::string("decode_linear_q4_n32_split4_precomputed_sums") + suffix,
                                  {input8, b.weights, b.scales, b.biases, m8, b.sums8}, &split, sizeof(split),
                                  kOutput / 32, 128));
    report(std::string("split4_precomputed_sums") + suffix + " (M8) vs _m16", m8, m16);
  }
  // Residual projections, same pairing.
  MetalBuffer r16 = out(16, "r16");
  (void)backend.submit(dispatch("decode_linear_q4_n32_split4_precomputed_sums_residual_m16",
                                {b.input, b.weights, b.scales, b.biases, b.residual, r16, b.sums16}, &split,
                                sizeof(split), kOutput / 32, 256));
  for (const char *suffix : {"", "_ftr", "_hoist", "_hoist_ftr"}) {
    MetalBuffer r8 = out(8, "r8");
    (void)backend.submit(dispatch(std::string("decode_linear_q4_n32_split4_precomputed_sums_residual") + suffix,
                                  {input8, b.weights, b.scales, b.biases, residual8, r8, b.sums8}, &split,
                                  sizeof(split), kOutput / 32, 128));
    report(std::string("split4_precomputed_sums_residual") + suffix + " (M8) vs _m16", r8, r16);
  }
  // Sequential N128 tiles: the M8 paired tile vs the M16 tile (the 16-row verify's input projections).
  const Q4Params n128{kOutput, kInput, 60};
  MetalBuffer n16 = out(16, "n16");
  (void)backend.submit(dispatch("decode_linear_q4_n128_m16", {b.input, b.weights, b.scales, b.biases, n16},
                                &n128, sizeof(n128), 60, 256));
  MetalBuffer n8 = out(8, "n8");
  (void)backend.submit(dispatch("decode_linear_q4_n128_paired", {input8, b.weights, b.scales, b.biases, n8},
                                &n128, sizeof(n128), 60, 256));
  report("n128_paired (M8) vs n128_m16", n8, n16);
  // The 8-row verify's input projections run split4 precomputed sums; the 16-row one runs N128 M16 tiles.
  MetalBuffer p8 = out(8, "p8");
  (void)backend.submit(dispatch("decode_linear_q4_n32_split4_precomputed_sums_hoist_ftr",
                                {input8, b.weights, b.scales, b.biases, p8, b.sums8}, &split, sizeof(split),
                                kOutput / 32, 128));
  report("split4_precomputed_sums_hoist_ftr (M8) vs n128_m16", p8, n16);
  // Gate/up: both streams read the same weights here.
  const Q4Params n256{kOutput, kInput, 60};
  MetalBuffer g16 = out(16, "g16");
  (void)backend.submit(dispatch("decode_linear_q4_n256_gate_up_m16",
                                {b.input, b.weights, b.scales, b.biases, g16, b.weights, b.scales, b.biases},
                                &n256, sizeof(n256), 60, 256));
  MetalBuffer g8 = out(8, "g8");
  (void)backend.submit(dispatch("decode_linear_q4_n256_gate_up",
                                {input8, b.weights, b.scales, b.biases, g8, b.weights, b.scales, b.biases},
                                &n256, sizeof(n256), 60, 256));
  report("n256_gate_up (M8) vs n256_gate_up_m16", g8, g16);
  // The live M8 gate/up (SPLASH_FFN_FUSED_SUMS) also publishes the next projection's Split32 sums.
  MetalBuffer gs8 = out(8, "gs8");
  MetalBuffer publishedSums = shared(uint64_t{kRows} * (kOutput / kQuantGroup) * 4, "gate-up-sums");
  (void)backend.submit(dispatch("decode_linear_q4_n256_gate_up_sums",
                                {input8, b.weights, b.scales, b.biases, gs8, b.weights, b.scales, b.biases,
                                 publishedSums},
                                &n256, sizeof(n256), 60, 256));
  report("n256_gate_up_sums (M8) vs n256_gate_up_m16", gs8, g16);

  // SPLASH_M24_PAD3 gate (Codex 06:58): the three PAD3 comparisons must each run and match, and a corrupted-output
  // control must be caught; any failure makes the process fail. The other rows stay diagnostics.
  std::vector<std::string> padFailures;
  uint32_t padChecks = 0;
  // SPLASH_M24_INPUT_SUMS (24/32 rows).
  MetalBuffer wide24;
  for (const uint32_t rows : {24u, 32u}) {
    const std::string m = "_m" + std::to_string(rows);
    const MetalBuffer inputRows = backend.view(b.input, 0, uint64_t{rows} * kInput * 2);
    MetalBuffer sums = shared(uint64_t{rows} * (kInput / kQuantGroup) * 4, "sums-rows");
    (void)backend.submit(dispatch("decode_linear_q4_split_sums" + m, {inputRows, sums}, &inputSize,
                                  sizeof(inputSize), kInput / 256, 256));
    MetalBuffer wide = out(rows, "wide");
    (void)backend.submit(dispatch("decode_linear_q4_n32_split4_precomputed_sums" + m,
                                  {inputRows, b.weights, b.scales, b.biases, wide, sums}, &split,
                                  sizeof(split), kOutput / 32, 256));
    report("split4_precomputed_sums_hoist_ftr (M8) vs" + m, p8, wide);
    report("split4_precomputed_sums_m16 vs" + m, m16, wide, 2 * kRows);
    // SPLASH_M24_PAD3: B3's 24 rows through the M32 consumer, the fourth lane's
    // rows being other data, against the M24 consumer on the same 24 rows.
    if (rows == 24) {
      wide24 = wide;
    } else {
      ++padChecks;
      if (report("split4_precomputed_sums_m24 vs _m32 with 8 padded rows (B3 pad)", wide24, wide, 24))
        padFailures.push_back("B3 pad: M24 vs M32 rows 0-23 differ");
    }
    // Sequential-everywhere route: the 24/32-row plain input tiles vs the M8 paired N128 tile.
    for (const auto &[name, threads] : {std::pair{std::string("decode_linear_q4_n128") + m, 256u},
                                        std::pair{std::string("decode_linear_q4_n256") + m, 256u},
                                        std::pair{std::string("decode_linear_q4_n128_m24_sg4"), 128u}}) {
      if (rows == 32 && name == "decode_linear_q4_n128_m24_sg4") continue;
      MetalBuffer plain = out(rows, "plain");
      (void)backend.submit(dispatch(name, {inputRows, b.weights, b.scales, b.biases, plain}, &n128, sizeof(n128),
                                    60, threads));
      report("n128_paired (M8) vs " + name, n8, plain);
    }
  }

  // SPLASH_M24_PAD3 poison (lead 06:4x): the idle fourth lane's rows (24-31) as NaN/+Inf/-Inf through the production
  // pad path, the input RMS publishing split sums ([group][rows]) and then the split-K M32 consumer. Rows 0-23 must equal
  // the clean 24-row run (RMS [group][24] + M24 consumer) byte for byte: normalized rows, sums and outputs.
  {
    constexpr uint32_t realRows = 24, paddedRows = 32, groups = kInput / kQuantGroup;
    const uint32_t width = kInput;
    MetalBuffer hidden = shared(uint64_t{paddedRows} * kInput * 2, "pad-hidden");
    std::memcpy(hidden.contents(), b.input.contents(), uint64_t{paddedRows} * kInput * 2);
    auto *rowsBits = static_cast<uint16_t *>(hidden.contents());
    constexpr std::array<uint16_t, 3> poison{0x7FC0, 0x7F80, 0xFF80};  // NaN, +Inf, -Inf (bf16)
    for (uint64_t i = uint64_t{realRows} * kInput; i < uint64_t{paddedRows} * kInput; ++i)
      rowsBits[i] = poison[i % poison.size()];
    MetalBuffer normWeight = shared(kInput * 2, "norm-weight");
    auto *normValues = static_cast<__bf16 *>(normWeight.contents());
    for (uint32_t i = 0; i < kInput; ++i) normValues[i] = __bf16(1.0f + small(random));
    MetalBuffer norm24 = shared(uint64_t{realRows} * kInput * 2, "norm24"),
                norm32 = shared(uint64_t{paddedRows} * kInput * 2, "norm32"),
                sums24 = shared(uint64_t{groups} * realRows * 4, "rms-sums24"),
                sums32 = shared(uint64_t{groups} * paddedRows * 4, "rms-sums32");
    (void)backend.submit(dispatch("norm_rms_staged_split_sums",
                                  {backend.view(b.input, 0, uint64_t{realRows} * kInput * 2), normWeight, norm24, sums24},
                                  &width, sizeof(width), realRows, SPLASH_STAGED_NORM_THREADS));
    (void)backend.submit(dispatch("norm_rms_staged_split_sums", {hidden, normWeight, norm32, sums32}, &width,
                                  sizeof(width), paddedRows, SPLASH_STAGED_NORM_THREADS));
    const bool normEqual = std::memcmp(norm24.contents(), norm32.contents(), uint64_t{realRows} * kInput * 2) == 0;
    uint32_t sumsDiffer = 0;
    const auto *s24 = static_cast<const uint32_t *>(sums24.contents());
    const auto *s32 = static_cast<const uint32_t *>(sums32.contents());
    for (uint32_t g = 0; g < groups; ++g)
      for (uint32_t r = 0; r < realRows; ++r)
        sumsDiffer += s24[g * realRows + r] != s32[g * paddedRows + r];
    std::cout << "row_invariance PAD3 poison: RMS normalized rows 0-23 " << (normEqual ? "identical" : "DIFFER")
              << ", sums rows 0-23 " << (sumsDiffer ? "DIFFER" : "identical") << " (" << sumsDiffer << "/"
              << groups * realRows << ")\n";
    ++padChecks;
    if (!normEqual || sumsDiffer)
      padFailures.push_back("PAD3 poison: RMS normalized rows or sums differ");
    MetalBuffer clean = out(realRows, "pad-clean"), padded = out(paddedRows, "pad-poisoned");
    (void)backend.submit(dispatch("decode_linear_q4_n32_split4_precomputed_sums_m24",
                                  {norm24, b.weights, b.scales, b.biases, clean, sums24}, &split, sizeof(split),
                                  kOutput / 32, 256));
    (void)backend.submit(dispatch("decode_linear_q4_n32_split4_precomputed_sums_m32",
                                  {norm32, b.weights, b.scales, b.biases, padded, sums32}, &split, sizeof(split),
                                  kOutput / 32, 256));
    ++padChecks;
    if (report("PAD3 poison: _m24 clean vs _m32 with NaN/Inf rows 24-31", clean, padded, realRows))
      padFailures.push_back("PAD3 poison: M32 outputs differ");
    // Control: one flipped bit in a real row must be caught by the same comparison.
    MetalBuffer corrupted = out(paddedRows, "pad-corrupted");
    std::memcpy(corrupted.contents(), padded.contents(), uint64_t{paddedRows} * kOutput * 2);
    static_cast<uint16_t *>(corrupted.contents())[uint64_t{5} * kOutput + 17] ^= 1;
    if (!report("PAD3 control (one flipped bit in row 5, must DIFFER)", clean, corrupted, realRows))
      padFailures.push_back("PAD3 control: a corrupted element went undetected");
  }
  if (padChecks != 3)
    padFailures.push_back("PAD3: " + std::to_string(padChecks) + " of 3 comparisons ran");
  if (!padFailures.empty()) {
    std::string reasons;
    for (const std::string &failure : padFailures) reasons += (reasons.empty() ? "" : "; ") + failure;
    throw std::runtime_error("PAD3 gate: " + reasons);
  }
  std::cout << "row_invariance PAD3 gate PASS (3 comparisons identical, corrupted-output control caught)\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: row-invariance <metallib>\n";
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}

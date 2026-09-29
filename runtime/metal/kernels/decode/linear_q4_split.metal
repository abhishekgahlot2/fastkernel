// Modified by meowkernels.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Split-K decode projections for fixed row batches. A threadgroup holds
// Parts partitions of Simdgroups simdgroups each and launches with
// Parts * Simdgroups * 32 threads; the partitions stream equal K ranges of one
// Rows x TileN tile into threadgroup partials (q4_mpp_tile_split), then the whole
// threadgroup reduces the partials and applies the epilogue. A projection
// with fewer 256-wide tiles than the GPU has cores cannot fill the cores with
// the sequential kernels; splitting K multiplies the simdgroups per tile
// instead of narrowing the tile further. No device scratch, extra dispatch or
// weight-layout change. Requires input_size % 1024 == 0 (four 256-input
// ranges) and, as dispatched by ops::Q4Linear, one threadgroup per tile.
//
// The buffer contracts equal the decode kernels in linear_q4.metal: Affine
// (input, weights, scales, biases, output, params), Residual (..., residual,
// output, params) and GateUp (gate stream, output, up stream, params).
template <ushort TileN, ushort Simdgroups, bool Residual, bool GateUp = false,
          bool LocalInputSync = false, bool PrecomputedInputSums = false,
          ushort Rows = 8, bool FooterGuard = false, bool HoistMetadata = false>
inline void q4_split(device bfloat *input, device uchar *weights,
                     device bfloat *scales, device bfloat *biases,
                     device bfloat *residual, device bfloat *output,
                     device uchar *upWeights, device bfloat *upScales,
                     device bfloat *upBiases, constant Q4Params &p, uint group,
                     uint lane, uint simd, threadgroup float *sums,
                     threadgroup float *partials,
                     device const float *globalSums = nullptr) {
  static_assert(Rows == 8 || PrecomputedInputSums,
                "larger split row tiles require precomputed input sums");
  constexpr uint Parts = 4;
  uint partition = simd / Simdgroups;
  for (uint tile = group; tile < p.output_size / TileN;
       tile += p.persistent_groups) {
    q4_mpp_tile_split<TileN, GateUp, 256, true, Simdgroups, Parts,
                      LocalInputSync, PrecomputedInputSums, Rows, HoistMetadata>(
        input, weights, scales, biases, partials, upWeights, upScales,
        upBiases, p.input_size, sums + partition * (8 * Rows), tile * TileN, lane,
        simd % Simdgroups, partition, globalSums);
    // Same epilogue as q4_mpp_tile: one bf16 rounding of the projection,
    // then the residual add or the SiLU gate, then the output rounding.
    for (uint i = simd * 32 + lane; i < Rows * TileN;
         i += Parts * Simdgroups * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part)
        value += partials[part * Rows * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      value = float(bfloat(value));
      if constexpr (GateUp) {
        float up = 0;
        for (uint part = 0; part < Parts; ++part)
          up += partials[(Parts + part) * Rows * TileN + i];
        value = value / (1.0f + fast::exp2(-1.44269504089f * value)) *
                float(bfloat(up));
      }
      if constexpr (Residual)
        value += float(residual[index]);
      output[index] = bfloat(value);
    }
    // The next tile's partials overwrite this reduction's inputs. FooterGuard (SPLASH_SPLIT4_FOOTER, exact): on a
    // group's last tile nothing writes its scratch again, so that one reuse barrier is skipped; the test is uniform
    // across the threadgroup, and every non-final tile keeps its barrier.
    if (!FooterGuard || tile + p.persistent_groups < p.output_size / TileN)
      threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define Q4_SPLIT_AFFINE(Name, TileN, Simdgroups, LocalInputSync)               \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q4Params &params [[buffer(5)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[4 * 64], partials[4 * 8 * TileN];                   \
    q4_split<TileN, Simdgroups, false, false, LocalInputSync>(                 \
        input, weights, scales, biases, output, output, weights, scales,       \
        biases, params, group, lane, simd, sums, partials);                    \
  }

#define Q4_SPLIT_RESIDUAL(Name, TileN, Simdgroups, LocalInputSync)             \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[4 * 64], partials[4 * 8 * TileN];                   \
    q4_split<TileN, Simdgroups, true, false, LocalInputSync>(                  \
        input, weights, scales, biases, residual, output, weights, scales,     \
        biases, params, group, lane, simd, sums, partials);                    \
  }

// Threads per threadgroup = 4 partitions x Simdgroups x 32. Only N32 and
// N64 are instantiated: these are the tiles selected by the split policy.
// A wider tile would require separate register-pressure and timing evidence.
Q4_SPLIT_AFFINE(decode_linear_q4_n32_split4, 32, 1, false) // 128 threads
Q4_SPLIT_AFFINE(decode_linear_q4_n32_split4_local_sync, 32, 1, true)
Q4_SPLIT_AFFINE(decode_linear_q4_n64_split4, 64, 2, false) // 256 threads
Q4_SPLIT_RESIDUAL(decode_linear_q4_n32_split4_residual, 32, 1, false)
Q4_SPLIT_RESIDUAL(decode_linear_q4_n32_split4_local_sync_residual, 32, 1, true)
Q4_SPLIT_RESIDUAL(decode_linear_q4_n64_split4_residual, 64, 2, false)
#undef Q4_SPLIT_AFFINE
#undef Q4_SPLIT_RESIDUAL

kernel void decode_linear_q4_split_sums(
    device bfloat *input [[buffer(0)]], device float *sums [[buffer(1)]],
    constant uint &inputSize [[buffer(2)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  q4_store_input_sums<8, 4>(input, inputSize, block * 256, sums, block * 32,
                            lane, simd);
}

kernel void decode_linear_q4_split_sums_m16(
    device bfloat *input [[buffer(0)]], device float *sums [[buffer(1)]],
    constant uint &inputSize [[buffer(2)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  q4_store_input_sums<16, 8>(input, inputSize, block * 256, sums, block * 64,
                             lane, simd);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4Params &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, false, false, false, true>(
      input, weights, scales, biases, output, output, weights, scales, biases,
      params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4Params &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, true, false, false, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, localSums, partials, sums);
}

// SPLASH_SPLIT4_FOOTER (exact scheduling): the two M8 entries above with the last-tile
// scratch-reuse barrier skipped (q4_split FooterGuard); the backend selects them per submission.
kernel void decode_linear_q4_n32_split4_precomputed_sums_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4Params &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, false, false, false, true, 8, true>(
      input, weights, scales, biases, output, output, weights, scales, biases,
      params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4Params &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, true, false, false, true, 8, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, localSums, partials, sums);
}

// SPLASH_SPLIT4_HOIST: opt-in M8 metadata loads before each matmul pair,
// with explicit stock-order FMAs. Footer selection stays independent.
kernel void decode_linear_q4_n32_split4_precomputed_sums_hoist(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4Params &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, false, false, false, true, 8, false, true>(
      input, weights, scales, biases, output, output, weights, scales, biases,
      params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_hoist_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4Params &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, false, false, false, true, 8, true, true>(
      input, weights, scales, biases, output, output, weights, scales, biases,
      params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_hoist(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4Params &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, true, false, false, true, 8, false, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_hoist_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4Params &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, true, false, false, true, 8, true, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, localSums, partials, sums);
}

// SPLASH_M16_INPUT_SUMS: 16-row (two-lane) input projections reading the
// input RMS's [group * 16 + row] sums; same partitions as the residual form.
kernel void decode_linear_q4_n32_split4_precomputed_sums_m16(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4Params &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[4 * 16 * 32];
  q4_split<32, 2, false, false, false, true, 16>(
      input, weights, scales, biases, output, output, weights, scales,
      biases, params, group, lane, simd, partials, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_m16(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4Params &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[4 * 16 * 32];
  q4_split<32, 2, true, false, false, true, 16>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, partials, partials, sums);
}

// SPLASH_SPLIT4_M16=1 (B2 only, chosen per encode by QwenTarget): the 16-row residual entry above with the M8
// footer (last-tile scratch-reuse barrier skipped) and metadata hoist (explicit stock-order FMAs).
kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_m16_hoist_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4Params &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[4 * 16 * 32];
  q4_split<32, 2, true, false, false, true, 16, true, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, partials, partials, sums);
}

// SPLASH_M24_NARROW_SPLIT: three- and four-lane (24/32-row) narrow projections
// as four K partitions of two simdgroups (256 threads) reading precomputed sums.
#define Q4_SPLIT_MULTI_ROW(Rows)                                                \
  kernel void decode_linear_q4_n32_split4_precomputed_sums_m##Rows(             \
      device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],  \
      device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]], \
      device bfloat *output [[buffer(4)]],                                      \
      device const float *sums [[buffer(5)]],                                   \
      constant Q4Params &params [[buffer(6)]],                                  \
      uint group [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd [[simdgroup_index_in_threadgroup]]) {                           \
    threadgroup float partials[4 * Rows * 32];                                  \
    q4_split<32, 2, false, false, false, true, Rows>(                           \
        input, weights, scales, biases, output, output, weights, scales,        \
        biases, params, group, lane, simd, partials, partials, sums);           \
  }                                                                             \
  kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_m##Rows(    \
      device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],  \
      device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]], \
      device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]], \
      device const float *sums [[buffer(6)]],                                   \
      constant Q4Params &params [[buffer(7)]],                                  \
      uint group [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd [[simdgroup_index_in_threadgroup]]) {                           \
    threadgroup float partials[4 * Rows * 32];                                  \
    q4_split<32, 2, true, false, false, true, Rows>(                            \
        input, weights, scales, biases, residual, output, weights, scales,      \
        biases, params, group, lane, simd, partials, partials, sums);           \
  }                                                                             \
  kernel void decode_linear_q4_split_sums_m##Rows(                              \
      device bfloat *input [[buffer(0)]], device float *sums [[buffer(1)]],     \
      constant uint &inputSize [[buffer(2)]],                                   \
      uint block [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd [[simdgroup_index_in_threadgroup]]) {                           \
    q4_store_input_sums<Rows, 8>(input, inputSize, block * 256, sums,          \
                                 block * 4 * Rows, lane, simd);                 \
  }
Q4_SPLIT_MULTI_ROW(24)
Q4_SPLIT_MULTI_ROW(32)
#undef Q4_SPLIT_MULTI_ROW

// 128 threads: four single-simdgroup partitions, two weight streams.
kernel void decode_linear_q4_n32_split4_gate_up(
    device bfloat *input [[buffer(0)]], device uchar *weights_0 [[buffer(1)]],
    device bfloat *scales_0 [[buffer(2)]],
    device bfloat *biases_0 [[buffer(3)]], device bfloat *output [[buffer(4)]],
    device uchar *weights_1 [[buffer(5)]], device bfloat *scales_1 [[buffer(6)]],
    device bfloat *biases_1 [[buffer(7)]], constant Q4Params &params [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[4 * 64], partials[2 * 4 * 8 * 32];
  q4_split<32, 1, false, true>(input, weights_0, scales_0, biases_0, output,
                               output, weights_1, scales_1, biases_1, params,
                               group, lane, simd, sums, partials);
}

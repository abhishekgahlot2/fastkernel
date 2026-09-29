// Modified by meowkernels.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Decode projections: persistent threadgroups stride over TileN-wide output
// tiles, TileCall names the q4_mpp_tiles.h instantiation and Sums holds eight
// input sums per row. The auxiliary buffer is the residual the epilogue adds
// or the gate it applies SiLU to.
#define Q4_DECODE_AFFINE(Name, TileCall, Sums, TileN)                          \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q4Params &params [[buffer(5)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, output, params.output_size, params.input_size,          \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }

#define Q4_DECODE_AUXILIARY(Name, Auxiliary, TileCall, Sums, TileN)            \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }

#define Q4_DECODE_GATE_UP(Name, TileCall, Sums, TileN)                         \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights_0 [[buffer(1)]],                      \
                   device bfloat *scales_0 [[buffer(2)]],                      \
                   device bfloat *biases_0 [[buffer(3)]],                      \
                   device bfloat *output [[buffer(4)]],                        \
                   device uchar *weights_1 [[buffer(5)]],                      \
                   device bfloat *scales_1 [[buffer(6)]],                      \
                   device bfloat *biases_1 [[buffer(7)]],                      \
                   constant Q4Params &params [[buffer(8)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights_0, scales_0, biases_0, output, weights_1,        \
               scales_1, biases_1, output, params.output_size,                 \
               params.input_size, input_sums, tile * TileN, simd_lane,         \
               simd_group);                                                    \
    }                                                                          \
  }

Q4_DECODE_AFFINE(decode_linear_q4_n128, (q4_mpp_tile<128, false, false, 256>), 64,
                 128)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m16,
                 (q4_mpp_tile_batched<16, 128, false, false, 256>), 128, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m24,
                 (q4_mpp_tile_batched<24, 128, false, false, 256>), 192, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m24_sg4,
                 (q4_mpp_tile_batched<24, 128, false, false, 256, false, 4>), 192, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n256_m16,
                 (q4_mpp_tile_batched<16, 256, false, false>), 128, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n256_m24,
                 (q4_mpp_tile_batched<24, 256, false, false>), 192, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n256, (q4_mpp_tile<256, false, false>), 64, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n128_paired,
                 (q4_mpp_tile<128, false, false, 256, true>), 64, 128)
// Same tile with four SIMD groups (128 threads). On M5 Max it halves the
// dependent-dispatch ramp/tail loss of the wide one-lane input projections
//. Selected by SPLASH_INPUT_SG4=1.
Q4_DECODE_AFFINE(decode_linear_q4_n128_paired_sg4,
                 (q4_mpp_tile<128, false, false, 256, true, 4>), 64, 128)
// 128 threads: four 8 x 256 tiles per core reach the occupancy knee for very
// wide one-lane projections, with half the input re-reads of N128 tiles.
Q4_DECODE_AFFINE(decode_linear_q4_n256_paired_sg4,
                 (q4_mpp_tile<256, false, false, 256, true, 4>), 64, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_paired, residual,
                    (q4_mpp_tile<128, false, true, 256, true>), 64, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual, residual,
                    (q4_mpp_tile<128, false, true, 256>), 64, 128)
Q4_DECODE_GATE_UP(decode_linear_q4_n256_gate_up, (q4_mpp_tile<256, true, false>), 64, 256)

// Publish the rounded gate/up values and their exact Split32 sums together.
kernel void decode_linear_q4_n256_gate_up_sums(
    device bfloat *input [[buffer(0)]], device uchar *gate_weights [[buffer(1)]],
    device bfloat *gate_scales [[buffer(2)]], device bfloat *gate_biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device uchar *up_weights [[buffer(5)]],
    device bfloat *up_scales [[buffer(6)]], device bfloat *up_biases [[buffer(7)]],
    device float *output_sums [[buffer(8)]],
    constant Q4Params &params [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[64];
  for (uint tile = group; tile < params.output_size / 256;
       tile += params.persistent_groups)
    q4_mpp_tile<256, true, false, 256, false, 8, true>(
        input, gate_weights, gate_scales, gate_biases, output, up_weights,
        up_scales, up_biases, output, params.output_size, params.input_size,
        input_sums, tile * 256, lane, simd, output_sums);
}

Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m16, residual,
                    (q4_mpp_tile_batched<16, 128, false, true, 256>), 128, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m24, residual,
                    (q4_mpp_tile_batched<24, 128, false, true, 256>), 192, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m24_sg4, residual,
                    (q4_mpp_tile_batched<24, 128, false, true, 256, false, 4>), 192, 128)
Q4_DECODE_GATE_UP(decode_linear_q4_n256_gate_up_m16,
                  (q4_mpp_tile_batched<16, 256, true, false>), 128, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m32,
                 (q4_mpp_tile_batched<32, 128, false, false, 256>), 256, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n256_m32,
                 (q4_mpp_tile_batched<32, 256, false, false>), 256, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m32, residual,
                    (q4_mpp_tile_batched<32, 128, false, true, 256>), 256, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n256_up_silu_m32, gate,
                    (q4_mpp_tile_batched<32, 256, false, false, 256, true>),
                    256, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n256_up_silu_m24, gate,
                    (q4_mpp_tile_batched<24, 256, false, false, 256, true>),
                    192, 256)
#undef Q4_DECODE_AFFINE
#undef Q4_DECODE_AUXILIARY
#undef Q4_DECODE_GATE_UP

// SPLASH_M16_FFN_SUMS=1 (default off): the 16-row gate/up tile unchanged, then this
// tile's down-projection input sums with the same q4_store_input_sums<16, 8> call that
// decode_linear_q4_split_sums_m16 makes over the same BF16 outputs, so the down projection
// needs no separate sums dispatch.
kernel void decode_linear_q4_n256_gate_up_m16_sums(
    device bfloat *input [[buffer(0)]], device uchar *gate_weights [[buffer(1)]],
    device bfloat *gate_scales [[buffer(2)]], device bfloat *gate_biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device uchar *up_weights [[buffer(5)]],
    device bfloat *up_scales [[buffer(6)]], device bfloat *up_biases [[buffer(7)]],
    device float *output_sums [[buffer(8)]], constant Q4Params &params [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[128];
  for (uint tile = group; tile < params.output_size / 256; tile += params.persistent_groups) {
    q4_mpp_tile_batched<16, 256, true, false>(
        input, gate_weights, gate_scales, gate_biases, output, up_weights, up_scales,
        up_biases, output, params.output_size, params.input_size, input_sums, tile * 256,
        lane, simd);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    q4_store_input_sums<16, 8>(output, params.output_size, tile * 256, output_sums,
                               tile * 64, lane, simd);
  }
}

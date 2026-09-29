// Grouped drafter context K/V projection (M16/M24/M32 entries for B2-B4; engine arm SPLASH_GROUPED_CONTEXT_KV,
// default on). The full projection's own tile call, no new arithmetic: only the K/V suffix tiles (from kv_first_tile,
// the part draft_context_kv_commit reads: columns 4096..6143 of both drafters' Rows x 6144 qkv) of each layer's output
// are computed; the Q prefix is left unwritten. Any input width K and any layer count up to 7 (the buffer slots): the
// host (Q4Linear::addContextKv) passes the full projection's N and K and its one-tile-per-group grid N/128.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

static_assert(SPLASH_CONTEXT_KV_MAX_LAYERS == 7, "the grouped entries below bind seven layer slots");

// Per-layer control: invoke once per drafter layer, grid {kv_tiles,1,1}, threads {256,1,1}.
kernel void decode_linear_q4_n128_paired_context_tail(
    device bfloat *input [[buffer(0)]],
    device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]],
    device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant Q4ContextKvParams &params [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  if (group >= params.kv_tiles || params.persistent_groups != params.output_size / 128 ||
      params.kv_first_tile + params.kv_tiles != params.persistent_groups)
    return;
  threadgroup float input_sums[64];
  q4_mpp_tile<128, false, false, 256, true>(
      input, weights, scales, biases, output, weights, scales, biases, output,
      params.output_size, params.input_size, input_sums, (params.kv_first_tile + group) * 128,
      lane, simd);
}

// Grouped: layers x kv_tiles groups, one tile each (27B drafter: 5 x 16 = 80; 35B: 6 x 16 = 96). Layer choice is
// uniform; scratch stays group-private. Seven layer slots: (weights, scales, biases) of layer i at buffers 1+3i..3+3i,
// outputs (disjoint Rows x N BF16 buffers, the unused Q prefix kept) at 22..28, params at 29; the host binds its last
// layer into unused slots, which no group reads. One entry per row count, each making the same tile call as the full
// projection it replaces (its Q4Params and threads per group): M8 decode_linear_q4_n128_paired, M16 _n128_m16, M24
// _n128_m24_sg4 (128 threads), M32 _n128_m32 (linear_q4.metal).
#define Q4_CONTEXT_KV_PICK(prefix, layer)                                                                   \
  (layer == 0 ? prefix##0 : layer == 1 ? prefix##1 : layer == 2 ? prefix##2 : layer == 3 ? prefix##3       \
   : layer == 4 ? prefix##4 : layer == 5 ? prefix##5 : prefix##6)
#define Q4_CONTEXT_KV_GROUPED(Name, TileCall, Sums)                                                        \
  kernel void Name(device bfloat *input [[buffer(0)]],                                                      \
                   device uchar *weights0 [[buffer(1)]], device bfloat *scales0 [[buffer(2)]],              \
                   device bfloat *biases0 [[buffer(3)]], device uchar *weights1 [[buffer(4)]],              \
                   device bfloat *scales1 [[buffer(5)]], device bfloat *biases1 [[buffer(6)]],              \
                   device uchar *weights2 [[buffer(7)]], device bfloat *scales2 [[buffer(8)]],              \
                   device bfloat *biases2 [[buffer(9)]], device uchar *weights3 [[buffer(10)]],             \
                   device bfloat *scales3 [[buffer(11)]], device bfloat *biases3 [[buffer(12)]],            \
                   device uchar *weights4 [[buffer(13)]], device bfloat *scales4 [[buffer(14)]],            \
                   device bfloat *biases4 [[buffer(15)]], device uchar *weights5 [[buffer(16)]],            \
                   device bfloat *scales5 [[buffer(17)]], device bfloat *biases5 [[buffer(18)]],            \
                   device uchar *weights6 [[buffer(19)]], device bfloat *scales6 [[buffer(20)]],            \
                   device bfloat *biases6 [[buffer(21)]], device bfloat *output0 [[buffer(22)]],            \
                   device bfloat *output1 [[buffer(23)]], device bfloat *output2 [[buffer(24)]],            \
                   device bfloat *output3 [[buffer(25)]], device bfloat *output4 [[buffer(26)]],            \
                   device bfloat *output5 [[buffer(27)]], device bfloat *output6 [[buffer(28)]],            \
                   constant Q4ContextKvParams &params [[buffer(29)]],                                       \
                   uint group [[threadgroup_position_in_grid]],                                             \
                   uint lane [[thread_index_in_simdgroup]],                                                 \
                   uint simd [[simdgroup_index_in_threadgroup]]) {                                          \
    if (params.layers == 0 || params.layers > SPLASH_CONTEXT_KV_MAX_LAYERS || !params.kv_tiles ||          \
        group >= params.layers * params.kv_tiles || params.persistent_groups != params.output_size / 128 ||  \
        params.kv_first_tile + params.kv_tiles != params.persistent_groups)                                 \
      return;                                                                                               \
    const uint layer = group / params.kv_tiles;                                                             \
    device uchar *weights = Q4_CONTEXT_KV_PICK(weights, layer);                                              \
    device bfloat *scales = Q4_CONTEXT_KV_PICK(scales, layer);                                               \
    device bfloat *biases = Q4_CONTEXT_KV_PICK(biases, layer);                                               \
    device bfloat *output = Q4_CONTEXT_KV_PICK(output, layer);                                               \
    threadgroup float input_sums[Sums];                                                                     \
    TileCall(input, weights, scales, biases, output, weights, scales, biases, output, params.output_size,  \
             params.input_size, input_sums, (params.kv_first_tile + group % params.kv_tiles) * 128, lane,   \
             simd);                                                                                         \
  }

Q4_CONTEXT_KV_GROUPED(decode_linear_q4_n128_paired_context_grouped,
                      (q4_mpp_tile<128, false, false, 256, true>), 64)
Q4_CONTEXT_KV_GROUPED(decode_linear_q4_n128_m16_context_grouped,
                      (q4_mpp_tile_batched<16, 128, false, false, 256>), 128)
Q4_CONTEXT_KV_GROUPED(decode_linear_q4_n128_m24_sg4_context_grouped,
                      (q4_mpp_tile_batched<24, 128, false, false, 256, false, 4>), 192)
Q4_CONTEXT_KV_GROUPED(decode_linear_q4_n128_m32_context_grouped,
                      (q4_mpp_tile_batched<32, 128, false, false, 256>), 256)

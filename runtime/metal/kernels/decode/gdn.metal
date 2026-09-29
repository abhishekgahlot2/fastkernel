// Modified by meowkernels.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/q4_sgmatrix.h"

// Decode threadgroups are 256 threads: one simdgroup per verify row in the
// prologue, and in the scan the head's 128 state rows strided over the eight
// simdgroups (each advances two of its sixteen rows at a time).
constant uint kDecodeSimdgroups = 8;

inline void grid_completion(device atomic_uint &arrived,
                            device atomic_uint &generation, uint group_count,
                            uint thread_index) {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    uint ticket = atomic_fetch_add_explicit(&arrived, 1, memory_order_relaxed);
    if (ticket + 1 == group_count) {
      atomic_store_explicit(&arrived, 0, memory_order_relaxed);
      atomic_fetch_add_explicit(&generation, 1, memory_order_relaxed);
    }
  }
}

// Eight verify rows' conv+SiLU, q/k RMS norms and gates for one value head.
// One simdgroup per row holds channels 32g + lane (g = 0..3). RMS reduction
// sums each 32-channel group, then adds the four partials in channel order.
// q/k go to threadgroup memory for the scan; v/gates go to device memory for
// both scan and commit.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline void gdn_decode_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device bfloat *mixed_qkv, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    uint packed_width, threadgroup bfloat *queries, threadgroup bfloat *keys,
    uint value_head, uint lane, uint simd_group) {
  constexpr uint Tokens = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  constexpr uint Groups = HeadDim / 32;
  static_assert(Tokens == kDecodeSimdgroups && HeadDim == 128,
                "one simdgroup per verify row, four channels per lane");
  const uint key_head = value_head / HeadsPerKey;
  // The key head's q/k rows and conv carry are shared by HeadsPerKey value
  // heads; the first of them writes the shared copies.
  const bool shared_writer = value_head % HeadsPerKey == 0;
  const uint token = simd_group;
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;

  float q[Groups], k[Groups];
  for (uint g = 0; g < Groups; ++g) {
    q[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                               packed_width, ConvDim, token,
                               q_channel + 32 * g));
    k[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                               packed_width, ConvDim, token,
                               k_channel + 32 * g));
  }
  float q_sum = 0.0f, k_sum = 0.0f;
  for (uint g = 0; g < Groups; ++g) {
    q_sum += simd_sum(q[g] * q[g]);
    k_sum += simd_sum(k[g] * k[g]);
  }
  const float q_scale = rsqrt(q_sum / HeadDim + 1e-6f);
  const float k_scale = rsqrt(k_sum / HeadDim + 1e-6f);
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
    const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
    queries[token * HeadDim + dim] = query;
    keys[token * HeadDim + dim] = key;
    if (shared_writer) {
      mixed_qkv[token * ConvDim + q_channel + 32 * g] = query;
      mixed_qkv[token * ConvDim + k_channel + 32 * g] = key;
    }
    mixed_qkv[token * ConvDim + v_channel + 32 * g] =
        gdn_conv_silu(packed, conv_state_in, conv_weights, packed_width,
                      ConvDim, token, v_channel + 32 * g);
  }
  if (lane == 0) {
    const uint gate_index = token * ValueHeads + value_head;
    gdn_write_gates(packed + token * packed_width, dt_bias, a_scale, BOffset,
                    AOffset, value_head, beta[gate_index], decay[gate_index]);
  }
  if (simd_group < 3) {
    const uint row = simd_group;
    for (uint g = 0; g < Groups; ++g) {
      conv_state_out[row * ConvDim + v_channel + 32 * g] =
          gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Tokens,
                         row, v_channel + 32 * g);
      if (shared_writer) {
        conv_state_out[row * ConvDim + q_channel + 32 * g] =
            gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim,
                           Tokens, row, q_channel + 32 * g);
        conv_state_out[row * ConvDim + k_channel + 32 * g] =
            gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim,
                           Tokens, row, k_channel + 32 * g);
      }
    }
  }
}

// Delta-rule recurrence over 128 state rows, four fp32 columns per lane.
// RowsInFlight rows advance together to overlap their reductions and arithmetic.
// Each row preserves the decay, memory, delta, update, output operation order.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight>
inline void gdn_decode_scan(device const bfloat *mixed_qkv,
                            device const float *decay,
                            device const bfloat *beta,
                            device const float *state_in,
                            device float *state_out, device bfloat *output,
                            threadgroup const bfloat *queries,
                            threadgroup const bfloat *keys, uint value_head,
                            uint lane, uint simd_group) {
  constexpr uint Tokens = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint Batches = HeadDim / kDecodeSimdgroups;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the head");
  const uint value_channel = 2 * KeyWidth + value_head * HeadDim;
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    float state[RowsInFlight][4];
    uint value_dim[RowsInFlight];
    ulong state_base[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r) {
      value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
      state_base[r] =
          (ulong(value_head) * HeadDim + value_dim[r]) * HeadDim + lane * 4;
      for (uint i = 0; i < 4; ++i)
        state[r][i] = state_in[state_base[r] + i];
    }
    for (uint token = 0; token < Tokens; ++token) {
      const float d = decay[token * ValueHeads + value_head];
      const float b = float(beta[token * ValueHeads + value_head]);
      threadgroup const bfloat *key = keys + token * HeadDim + lane * 4;
      threadgroup const bfloat *query = queries + token * HeadDim + lane * 4;
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] *= d;
          memory[r] += state[r][i] * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(mixed_qkv[token * ConvDim + value_channel + value_dim[r]]) -
             memory[r]) *
            b;
        result[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] += float(key[i]) * delta;
          result[r] += state[r][i] * float(query[i]);
        }
        result[r] = simd_sum(result[r]);
      }
      if (lane == 0) {
        for (uint r = 0; r < RowsInFlight; ++r) {
          output[(ulong(token) * ValueHeads + value_head) * HeadDim +
                 value_dim[r]] = bfloat(result[r]);
        }
      }
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state_out[state_base[r] + i] = state[r][i];
  }
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          bool SkipFull = true>
inline void
gdn_commit_phase(device const bfloat *packed, device const bfloat *mixed_qkv,
                 device const float *decay, device const bfloat *beta,
                 device const bfloat *conv_state_in,
                 device bfloat *conv_state_out, device const float *state_in,
                 device float *state_out, uint retained, uint groups,
                 uint packed_width, uint group, uint thread_index, uint lane,
                 uint simd_group) {
  constexpr uint KeyDim = HeadDim, ValueDim = HeadDim;
  constexpr uint ValueBatches = ValueDim / 8;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;

  uint count = retained;
  if (SkipFull && count == SPLASH_TARGET_VERIFY_ROWS)
    return;
  for (uint element = group * 256 + thread_index; element < 3 * ConvDim;
       element += groups * 256) {
    uint row = element / ConvDim;
    uint channel = element % ConvDim;
    conv_state_out[element] = gdn_conv_carry(
        packed, conv_state_in, packed_width, ConvDim, count, row, channel);
  }

  for (uint task = group; task < ValueHeads * ValueBatches; task += groups) {
    uint value_head = task / ValueBatches;
    uint value_dim = (task % ValueBatches) * 8 + simd_group;
    uint key_head = value_head / HeadsPerKey;
    ulong state_base = (ulong(value_head) * ValueDim + value_dim) * KeyDim;
    float local_state[4];
    for (uint i = 0; i < 4; ++i) {
      local_state[i] = state_in[state_base + lane * 4 + i];
    }
    for (uint token = 0; token < count; ++token) {
      ulong key_base = ulong(token) * ConvDim + key_head * KeyDim;
      float memory = 0.0f;
      float d = decay[token * ValueHeads + value_head];
      for (uint i = 0; i < 4; ++i) {
        uint dim = lane * 4 + i;
        local_state[i] *= d;
        memory +=
            local_state[i] * float(mixed_qkv[key_base + dim + KeyWidth]);
      }
      memory = simd_sum(memory);
      ulong value_index =
          ulong(token) * ConvDim + value_head * ValueDim + value_dim +
          2 * KeyWidth;
      float delta = (float(mixed_qkv[value_index]) - memory) *
                    float(beta[token * ValueHeads + value_head]);
      for (uint i = 0; i < 4; ++i) {
        uint dim = lane * 4 + i;
        local_state[i] += float(mixed_qkv[key_base + dim + KeyWidth]) * delta;
      }
    }
    for (uint i = 0; i < 4; ++i) {
      state_out[state_base + lane * 4 + i] = local_state[i];
    }
  }
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline void gdn_commit_prefix_batch_phase(
    device const bfloat *packed, device const bfloat *mixed_qkv,
    device const float *decay, device const bfloat *beta,
    device const uchar *current_0, device const uchar *current_1,
    device const uchar *current_2, device const uchar *current_3,
    device uchar *next_0, device uchar *next_1, device uchar *next_2,
    device uchar *next_3, device const uint *retained,
    constant GDNBatchCommitParams &params, uint3 group, uint thread_index,
    uint simd_lane, uint simd_group) {
  if (group.z >= params.lanes)
    return;
  uint batch = group.z;
  uint layer = group.y;
  device const uchar *current = batch == 0   ? current_0
                                : batch == 1 ? current_1
                                : batch == 2 ? current_2
                                             : current_3;
  device uchar *next = batch == 0   ? next_0
                       : batch == 1 ? next_1
                       : batch == 2 ? next_2
                                    : next_3;
  packed += (ulong(layer) * SPLASH_MAXIMUM_BATCH_WIDTH + batch) * params.packed_stride;
  mixed_qkv += (ulong(layer) * SPLASH_MAXIMUM_BATCH_WIDTH + batch) * params.mixed_stride;
  decay += (ulong(layer) * SPLASH_MAXIMUM_BATCH_WIDTH + batch) * params.decay_stride;
  beta += (ulong(layer) * SPLASH_MAXIMUM_BATCH_WIDTH + batch) * params.beta_stride;
  device const bfloat *conv_state_in = reinterpret_cast<device const bfloat *>(
      current + ulong(layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  gdn_commit_phase<KeyHeads, ValueHeads, HeadDim, ConvDim>(
      packed, mixed_qkv, decay, beta, conv_state_in, conv_state_out, state_in,
      state_out, retained[batch], params.groups, params.packed_width, group.x,
      thread_index, simd_lane, simd_group);
}

#define GDN_COMMIT_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim)         \
  kernel void Name(                                                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *mixed_qkv [[buffer(1)]],                           \
      device const float *decay [[buffer(2)]],                                \
      device const bfloat *beta [[buffer(3)]],                                \
      device const uchar *current_0 [[buffer(4)]],                            \
      device const uchar *current_1 [[buffer(5)]],                            \
      device const uchar *current_2 [[buffer(6)]],                            \
      device const uchar *current_3 [[buffer(7)]],                            \
      device uchar *next_0 [[buffer(8)]], device uchar *next_1 [[buffer(9)]], \
      device uchar *next_2 [[buffer(10)]],                                    \
      device uchar *next_3 [[buffer(11)]],                                    \
      device const uint *retained [[buffer(12)]],                             \
      constant GDNBatchCommitParams &params [[buffer(13)]],                   \
      uint3 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint simd_lane [[thread_index_in_simdgroup]],                           \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    gdn_commit_prefix_batch_phase<KeyHeads, ValueHeads, HeadDim, ConvDim>(    \
        packed, mixed_qkv, decay, beta, current_0, current_1, current_2,       \
        current_3, next_0, next_1, next_2, next_3, retained, params, group,    \
        thread_index, simd_lane, simd_group);                                 \
  }

GDN_COMMIT_ENTRY(verify_gdn_commit, 16, 48, 128, 10240)
GDN_COMMIT_ENTRY(verify_gdn_commit_vh32, 16, 32, 128, 8192)
#undef GDN_COMMIT_ENTRY

// One logical 16- or 32-row request is stored in physical M8 lanes 0..Tiles-1.
// Tile 0 starts from the committed state; later tiles continue both
// transitions from the previous tile's: the recurrent state in place in next,
// the convolution carry through non-aliasing scratch (two slots, alternating),
// and the last tile writes the carry to next. With Tiles = 2 this is the
// original 16-row pair (first: current -> scratch, second: scratch -> next).
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint Tile, uint Tiles>
inline void gdn_commit_tile_phase(
    device const bfloat *packed, device const bfloat *mixed_qkv,
    device const float *decay, device const bfloat *beta,
    device const uchar *current, device uchar *next, device uchar *scratch,
    device const uint *retained, constant GDNBatchCommitParams &params,
    uint2 group, uint thread_index, uint simd_lane, uint simd_group) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint WideRows = Tiles * Rows;
  if (retained[0] == WideRows)
    return;
  const uint layer = group.y;
  constexpr uint half_index = Tile;
  packed += ulong(layer * SPLASH_MAXIMUM_BATCH_WIDTH + half_index) *
            params.packed_stride;
  mixed_qkv += ulong(layer * SPLASH_MAXIMUM_BATCH_WIDTH + half_index) *
               params.mixed_stride;
  decay += ulong(layer * SPLASH_MAXIMUM_BATCH_WIDTH + half_index) *
           params.decay_stride;
  beta += ulong(layer * SPLASH_MAXIMUM_BATCH_WIDTH + half_index) *
          params.beta_stride;
  device const bfloat *current_conv = reinterpret_cast<device const bfloat *>(
      current + ulong(layer) * params.conv_layer_bytes);
  device bfloat *next_conv = reinterpret_cast<device bfloat *>(
      next + ulong(layer) * params.conv_layer_bytes);
  // Scratch slot k of a layer: the tile-(k mod 2) carry (slot 1 only for Tiles > 2).
  device bfloat *scratch_in = reinterpret_cast<device bfloat *>(
      scratch + (ulong(layer) * (Tiles > 2 ? 2 : 1) + (Tile + 1) % 2) * params.conv_layer_bytes);
  device bfloat *scratch_out = reinterpret_cast<device bfloat *>(
      scratch + (ulong(layer) * (Tiles > 2 ? 2 : 1) + Tile % 2) * params.conv_layer_bytes);
  device const float *current_state = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  device float *next_state = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  const uint total = retained[0];
  const uint count = min(total > Tile * Rows ? total - Tile * Rows : 0, Rows);
  gdn_commit_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, false>(
      packed, mixed_qkv, decay, beta,
      Tile == 0 ? current_conv : scratch_in,
      Tile == Tiles - 1 ? next_conv : scratch_out,
      Tile == 0 ? current_state : next_state, next_state, count, params.groups,
      params.packed_width, group.x, thread_index, simd_lane, simd_group);
}

#define GDN_COMMIT16_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, Tile, Tiles) \
  kernel void Name(                                                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *mixed_qkv [[buffer(1)]],                           \
      device const float *decay [[buffer(2)]],                                \
      device const bfloat *beta [[buffer(3)]],                                \
      device const uchar *current [[buffer(4)]],                              \
      device uchar *next [[buffer(5)]], device uchar *scratch [[buffer(6)]],  \
      device const uint *retained [[buffer(7)]],                              \
      constant GDNBatchCommitParams &params [[buffer(8)]],                    \
      uint2 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint simd_lane [[thread_index_in_simdgroup]],                           \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    gdn_commit_tile_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, Tile, Tiles>(\
        packed, mixed_qkv, decay, beta, current, next, scratch, retained,     \
        params, group, thread_index, simd_lane, simd_group);                  \
  }

GDN_COMMIT16_ENTRY(verify_gdn_commit16_first, 16, 48, 128, 10240, 0, 2)
GDN_COMMIT16_ENTRY(verify_gdn_commit16_second, 16, 48, 128, 10240, 1, 2)
GDN_COMMIT16_ENTRY(verify_gdn_commit16_first_vh32, 16, 32, 128, 8192, 0, 2)
GDN_COMMIT16_ENTRY(verify_gdn_commit16_second_vh32, 16, 32, 128, 8192, 1, 2)
// SPLASH_WIDE_LOOKUP32: the 32-row request's four tiles.
GDN_COMMIT16_ENTRY(verify_gdn_commit32_t0, 16, 48, 128, 10240, 0, 4)
GDN_COMMIT16_ENTRY(verify_gdn_commit32_t1, 16, 48, 128, 10240, 1, 4)
GDN_COMMIT16_ENTRY(verify_gdn_commit32_t2, 16, 48, 128, 10240, 2, 4)
GDN_COMMIT16_ENTRY(verify_gdn_commit32_t3, 16, 48, 128, 10240, 3, 4)
GDN_COMMIT16_ENTRY(verify_gdn_commit32_t0_vh32, 16, 32, 128, 8192, 0, 4)
GDN_COMMIT16_ENTRY(verify_gdn_commit32_t1_vh32, 16, 32, 128, 8192, 1, 4)
GDN_COMMIT16_ENTRY(verify_gdn_commit32_t2_vh32, 16, 32, 128, 8192, 2, 4)
GDN_COMMIT16_ENTRY(verify_gdn_commit32_t3_vh32, 16, 32, 128, 8192, 3, 4)
#undef GDN_COMMIT16_ENTRY

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight>
inline void gdn_decode_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device bfloat *mixed, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    device bfloat *recurrent, device const bfloat *gdn_norm_weight,
    device bfloat *gdn_hidden, device atomic_uint *arrived,
    device atomic_uint *generation, constant GDNDecodeBatchParams &params,
    uint2 group, uint thread_index, uint lane, uint simd_group,
    threadgroup float *scratch, threadgroup bfloat *prepared,
    device bfloat *q4_table = nullptr, device float *q4_sums = nullptr) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  uint batch = group.y;
  if (batch >= params.lanes || group.x >= ValueHeads)
    return;
  device const uchar *current = batch == 0
      ? current0
      : (batch == 1 ? current1 : (batch == 2 ? current2 : current3));
  device uchar *next = batch == 0
      ? next0
      : (batch == 1 ? next1 : (batch == 2 ? next2 : next3));
  packed += ulong(batch) * Rows * params.packed_width;
  mixed += ulong(batch) * Rows * ConvDim;
  decay += ulong(batch) * Rows * ValueHeads;
  beta += ulong(batch) * Rows * ValueHeads;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);

  device bfloat *lane_recurrent =
      recurrent + ulong(batch) * Rows * ValueWidth;
  device bfloat *lane_hidden = gdn_hidden + ulong(batch) * Rows * ValueWidth;
  threadgroup bfloat *queries = prepared;
  threadgroup bfloat *keys = prepared + Rows * HeadDim;
  gdn_decode_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim>(
      packed, conv_weights, conv_state_in, conv_state_out, mixed, a_scale,
      dt_bias, decay, beta, params.packed_width, queries, keys, group.x, lane,
      simd_group);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  gdn_decode_scan<KeyHeads, ValueHeads, HeadDim, ConvDim, RowsInFlight>(
      mixed, decay, beta, state_in, state_out, lane_recurrent, queries, keys,
      group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  gdn_gate_phase<ValueHeads, HeadDim, ConvDim, kDecodeSimdgroups>(
      lane_recurrent, packed, gdn_norm_weight, lane_hidden, Rows * ValueHeads,
      ValueHeads, params.packed_width, scratch, group.x, thread_index, lane,
      simd_group);
  if (q4_table) {
    // Each group owns this head for all eight rows. Publish its rounded
    // outputs before the eight SIMD groups transpose one row each.
    threadgroup_barrier(mem_flags::mem_device);
    for (uint g = 0; g < HeadDim / 64; ++g) {
      const uint column = group.x * HeadDim + g * 64 + 2 * lane;
      const uint index = simd_group * ValueWidth + column;
      q4sg::write_input(q4_table + ulong(batch) * ValueWidth * Rows,
                        q4_sums + ulong(batch) * ValueWidth / 8,
                        column / 64, simd_group, lane,
                        lane_hidden[index], lane_hidden[index + 1]);
    }
  }
  if (!q4_table && q4_sums) {
    // Match decode_linear_q4_split_sums exactly: lane and lane+32, then
    // simd_sum. Each head owns two groups and each SIMD group owns one row.
    threadgroup_barrier(mem_flags::mem_device);
    for (uint g = 0; g < HeadDim / 64; ++g) {
      const uint column = group.x * HeadDim + g * 64 + lane;
      const uint index = simd_group * ValueWidth + column;
      const float sum = simd_sum(float(lane_hidden[index]) +
                                 float(lane_hidden[index + 32]));
      if (lane == 0)
        q4_sums[(group.x * (HeadDim / 64) + g) * Rows + simd_group] = sum;
    }
  }
  grid_completion(arrived[batch], generation[batch], ValueHeads,
                  thread_index);
}

#define GDN_DECODE_BUFFERS \
    device const bfloat *packed [[buffer(0)]], \
    device const bfloat *conv_weights [[buffer(1)]], \
    device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]], \
    device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]], \
    device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]], \
    device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]], \
    device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]], \
    device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]], \
    device bfloat *beta [[buffer(14)]], device bfloat *recurrent [[buffer(15)]], \
    device const bfloat *gdn_norm_weight [[buffer(16)]], device bfloat *gdn_hidden [[buffer(17)]], \
    device atomic_uint *arrived [[buffer(18)]], device atomic_uint *generation [[buffer(19)]]
#define GDN_DECODE_THREADS \
    uint2 group [[threadgroup_position_in_grid]], \
    uint thread_index [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
#define GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, Table, Sums) \
    threadgroup float scratch[kDecodeSimdgroups]; \
    threadgroup bfloat prepared[2 * SPLASH_TARGET_VERIFY_ROWS * HeadDim]; \
    gdn_decode_batch_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, 2>( \
        packed, conv_weights, current0, current1, current2, current3, next0, \
        next1, next2, next3, mixed, a_scale, dt_bias, decay, beta, recurrent, \
        gdn_norm_weight, gdn_hidden, arrived, generation, params, group, \
        thread_index, lane, simd_group, scratch, prepared, Table, Sums);
#define GDN_DECODE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim) \
  kernel void Name(GDN_DECODE_BUFFERS, \
      constant GDNDecodeBatchParams &params [[buffer(20)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, nullptr, nullptr) \
  }
#define GDN_DECODE_Q4_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim) \
  kernel void Name(GDN_DECODE_BUFFERS, \
      device bfloat *q4_table [[buffer(20)]], device float *q4_sums [[buffer(21)]], \
      constant GDNDecodeBatchParams &params [[buffer(22)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, q4_table, q4_sums) \
  }

kernel void verify_gdn_fused_split_sums(GDN_DECODE_BUFFERS,
    device float *split_sums [[buffer(20)]],
    constant GDNDecodeBatchParams &params [[buffer(21)]], GDN_DECODE_THREADS) {
  GDN_DECODE_BODY(16, 48, 128, 10240, nullptr, split_sums)
}

#include "metal/kernels/decode/gdn_value_parts.h"

// Two rows overlap reductions and arithmetic without the register cost of four.
GDN_DECODE_ENTRY(verify_gdn_fused, 16, 48, 128, 10240)
GDN_DECODE_ENTRY(verify_gdn_fused_vh32, 16, 32, 128, 8192)
GDN_DECODE_Q4_ENTRY(verify_gdn_fused_q4, 16, 48, 128, 10240)
GDN_DECODE_Q4_ENTRY(verify_gdn_fused_q4_vh32, 16, 32, 128, 8192)

// Tile of a Tiles x 8-row request (Tiles = 2: the original first/second pair;
// Tiles = 4: SPLASH_WIDE_LOOKUP32), chained as in gdn_commit_tile_phase: the
// recurrent state in place in next, the convolution carry through two
// alternating scratch slots, the last tile's carry to next.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight, uint Tile, uint Tiles>
inline void gdn_decode_tile_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current, device uchar *next, device bfloat *mixed,
    device const float *a_scale, device const bfloat *dt_bias,
    device float *decay, device bfloat *beta, device bfloat *recurrent,
    device const bfloat *gdn_norm_weight, device bfloat *gdn_hidden,
    device atomic_uint *arrived, device atomic_uint *generation,
    device uchar *conv_scratch, constant GDNDecodeBatchParams &params,
    uint value_head, uint thread_index, uint lane, uint simd_group,
    threadgroup float *scratch, threadgroup bfloat *prepared) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint half_index = Tile;
  packed += ulong(half_index * Rows) * params.packed_width;
  mixed += ulong(half_index * Rows) * ConvDim;
  decay += ulong(half_index * Rows) * ValueHeads;
  beta += ulong(half_index * Rows) * ValueHeads;
  recurrent += ulong(half_index * Rows) * ValueWidth;
  gdn_hidden += ulong(half_index * Rows) * ValueWidth;
  device const bfloat *current_conv = reinterpret_cast<device const bfloat *>(
      current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *next_conv = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *scratch_in = reinterpret_cast<device bfloat *>(
      conv_scratch + ((Tile + 1) % 2) * params.conv_layer_bytes);
  device bfloat *scratch_out = reinterpret_cast<device bfloat *>(
      conv_scratch + (Tile % 2) * params.conv_layer_bytes);
  device const float *current_state = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device float *next_state = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  threadgroup bfloat *queries = prepared;
  threadgroup bfloat *keys = prepared + Rows * HeadDim;
  gdn_decode_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim>(
      packed, conv_weights, Tile == 0 ? current_conv : scratch_in,
      Tile == Tiles - 1 ? next_conv : scratch_out, mixed, a_scale, dt_bias, decay,
      beta, params.packed_width, queries, keys, value_head, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  gdn_decode_scan<KeyHeads, ValueHeads, HeadDim, ConvDim, RowsInFlight>(
      mixed, decay, beta, Tile == 0 ? current_state : next_state, next_state,
      recurrent, queries, keys, value_head, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  gdn_gate_phase<ValueHeads, HeadDim, ConvDim, kDecodeSimdgroups>(
      recurrent, packed, gdn_norm_weight, gdn_hidden, Rows * ValueHeads,
      ValueHeads, params.packed_width, scratch, value_head, thread_index, lane,
      simd_group);
  // The last tile completes on counter 0 (the host's lane-0 check), tile t < last on t + 1.
  constexpr uint counter = Tile == Tiles - 1 ? 0 : Tile + 1;
  grid_completion(arrived[counter], generation[counter], ValueHeads,
                  thread_index);
}

#define GDN_DECODE16_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, Tile, Tiles) \
  kernel void Name(                                                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *conv_weights [[buffer(1)]],                        \
      device const uchar *current [[buffer(2)]],                              \
      device uchar *next [[buffer(3)]], device bfloat *mixed [[buffer(4)]],   \
      device const float *a_scale [[buffer(5)]],                              \
      device const bfloat *dt_bias [[buffer(6)]],                             \
      device float *decay [[buffer(7)]], device bfloat *beta [[buffer(8)]],   \
      device bfloat *recurrent [[buffer(9)]],                                 \
      device const bfloat *gdn_norm_weight [[buffer(10)]],                    \
      device bfloat *gdn_hidden [[buffer(11)]],                               \
      device atomic_uint *arrived [[buffer(12)]],                             \
      device atomic_uint *generation [[buffer(13)]],                          \
      device uchar *conv_scratch [[buffer(14)]],                              \
      constant GDNDecodeBatchParams &params [[buffer(15)]],                   \
      uint value_head [[threadgroup_position_in_grid]],                       \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float scratch[kDecodeSimdgroups];                             \
    threadgroup bfloat prepared[2 * SPLASH_TARGET_VERIFY_ROWS * HeadDim];     \
    gdn_decode_tile_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, 2, Tile, Tiles>(\
        packed, conv_weights, current, next, mixed, a_scale, dt_bias, decay,  \
        beta, recurrent, gdn_norm_weight, gdn_hidden, arrived, generation,    \
        conv_scratch, params, value_head, thread_index, lane, simd_group,     \
        scratch, prepared);                                                   \
  }

GDN_DECODE16_ENTRY(verify_gdn_fused16_first, 16, 48, 128, 10240, 0, 2)
GDN_DECODE16_ENTRY(verify_gdn_fused16_second, 16, 48, 128, 10240, 1, 2)
GDN_DECODE16_ENTRY(verify_gdn_fused16_first_vh32, 16, 32, 128, 8192, 0, 2)
GDN_DECODE16_ENTRY(verify_gdn_fused16_second_vh32, 16, 32, 128, 8192, 1, 2)
// SPLASH_WIDE_LOOKUP32: the 32-row request's four tiles.
GDN_DECODE16_ENTRY(verify_gdn_fused32_t0, 16, 48, 128, 10240, 0, 4)
GDN_DECODE16_ENTRY(verify_gdn_fused32_t1, 16, 48, 128, 10240, 1, 4)
GDN_DECODE16_ENTRY(verify_gdn_fused32_t2, 16, 48, 128, 10240, 2, 4)
GDN_DECODE16_ENTRY(verify_gdn_fused32_t3, 16, 48, 128, 10240, 3, 4)
GDN_DECODE16_ENTRY(verify_gdn_fused32_t0_vh32, 16, 32, 128, 8192, 0, 4)
GDN_DECODE16_ENTRY(verify_gdn_fused32_t1_vh32, 16, 32, 128, 8192, 1, 4)
GDN_DECODE16_ENTRY(verify_gdn_fused32_t2_vh32, 16, 32, 128, 8192, 2, 4)
GDN_DECODE16_ENTRY(verify_gdn_fused32_t3_vh32, 16, 32, 128, 8192, 3, 4)
#undef GDN_DECODE16_ENTRY

// SPLASH_WIDE_GDN_SINGLE (host flag, default off): a wide lookup's Tiles x 8
// rows in one pass per layer instead of Tiles chained tile dispatches. The
// per-row arithmetic is the tile chain's, i.e. Tiles chained M8 verifies:
// conv+SiLU and q/k norms per row on one simdgroup (rows simd_group + 8j), the
// delta rule over all rows with each state row kept in registers (the chain
// stores and reloads it as fp32, which is exact), the gated norm per row. These
// are copies of gdn_decode_prologue / gdn_decode_scan with a row loop, so the
// M8 kernels' source (the byte reference) is untouched.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim, uint Tokens>
inline void gdn_wide_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device bfloat *mixed_qkv, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    uint packed_width, threadgroup bfloat *queries, threadgroup bfloat *keys,
    uint value_head, uint lane, uint simd_group) {
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  constexpr uint Groups = HeadDim / 32;
  static_assert(Tokens % kDecodeSimdgroups == 0 && HeadDim == 128,
                "whole rows per simdgroup, four channels per lane");
  const uint key_head = value_head / HeadsPerKey;
  const bool shared_writer = value_head % HeadsPerKey == 0;
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;
  for (uint token = simd_group; token < Tokens; token += kDecodeSimdgroups) {
    float q[Groups], k[Groups];
    for (uint g = 0; g < Groups; ++g) {
      q[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                                 packed_width, ConvDim, token,
                                 q_channel + 32 * g));
      k[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                                 packed_width, ConvDim, token,
                                 k_channel + 32 * g));
    }
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint g = 0; g < Groups; ++g) {
      q_sum += simd_sum(q[g] * q[g]);
      k_sum += simd_sum(k[g] * k[g]);
    }
    const float q_scale = rsqrt(q_sum / HeadDim + 1e-6f);
    const float k_scale = rsqrt(k_sum / HeadDim + 1e-6f);
    for (uint g = 0; g < Groups; ++g) {
      const uint dim = 32 * g + lane;
      const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
      const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
      queries[token * HeadDim + dim] = query;
      keys[token * HeadDim + dim] = key;
      if (shared_writer) {
        mixed_qkv[token * ConvDim + q_channel + 32 * g] = query;
        mixed_qkv[token * ConvDim + k_channel + 32 * g] = key;
      }
      mixed_qkv[token * ConvDim + v_channel + 32 * g] =
          gdn_conv_silu(packed, conv_state_in, conv_weights, packed_width,
                        ConvDim, token, v_channel + 32 * g);
    }
    if (lane == 0) {
      const uint gate_index = token * ValueHeads + value_head;
      gdn_write_gates(packed + token * packed_width, dt_bias, a_scale, BOffset,
                      AOffset, value_head, beta[gate_index], decay[gate_index]);
    }
  }
  if (simd_group < 3) {
    const uint row = simd_group;
    for (uint g = 0; g < Groups; ++g) {
      conv_state_out[row * ConvDim + v_channel + 32 * g] =
          gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Tokens,
                         row, v_channel + 32 * g);
      if (shared_writer) {
        conv_state_out[row * ConvDim + q_channel + 32 * g] =
            gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim,
                           Tokens, row, q_channel + 32 * g);
        conv_state_out[row * ConvDim + k_channel + 32 * g] =
            gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim,
                           Tokens, row, k_channel + 32 * g);
      }
    }
  }
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight, uint Tokens>
inline void gdn_wide_scan(device const bfloat *mixed_qkv,
                          device const float *decay,
                          device const bfloat *beta,
                          device const float *state_in,
                          device float *state_out, device bfloat *output,
                          threadgroup const bfloat *queries,
                          threadgroup const bfloat *keys, uint value_head,
                          uint lane, uint simd_group) {
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint Batches = HeadDim / kDecodeSimdgroups;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the head");
  const uint value_channel = 2 * KeyWidth + value_head * HeadDim;
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    float state[RowsInFlight][4];
    uint value_dim[RowsInFlight];
    ulong state_base[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r) {
      value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
      state_base[r] =
          (ulong(value_head) * HeadDim + value_dim[r]) * HeadDim + lane * 4;
      for (uint i = 0; i < 4; ++i)
        state[r][i] = state_in[state_base[r] + i];
    }
    for (uint token = 0; token < Tokens; ++token) {
      const float d = decay[token * ValueHeads + value_head];
      const float b = float(beta[token * ValueHeads + value_head]);
      threadgroup const bfloat *key = keys + token * HeadDim + lane * 4;
      threadgroup const bfloat *query = queries + token * HeadDim + lane * 4;
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] *= d;
          memory[r] += state[r][i] * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(mixed_qkv[token * ConvDim + value_channel + value_dim[r]]) -
             memory[r]) *
            b;
        result[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] += float(key[i]) * delta;
          result[r] += state[r][i] * float(query[i]);
        }
        result[r] = simd_sum(result[r]);
      }
      if (lane == 0) {
        for (uint r = 0; r < RowsInFlight; ++r) {
          output[(ulong(token) * ValueHeads + value_head) * HeadDim +
                 value_dim[r]] = bfloat(result[r]);
        }
      }
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state_out[state_base[r] + i] = state[r][i];
  }
}

// One layer of a wide lookup in one dispatch: current -> next directly (the
// conv carry is the last three of the Tiles x 8 packed rows). Completes every
// tile's counter, as the chain does, so the host check is the same.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight, uint Tiles>
inline void gdn_wide_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current, device uchar *next, device bfloat *mixed,
    device const float *a_scale, device const bfloat *dt_bias,
    device float *decay, device bfloat *beta, device bfloat *recurrent,
    device const bfloat *gdn_norm_weight, device bfloat *gdn_hidden,
    device atomic_uint *arrived, device atomic_uint *generation,
    constant GDNDecodeBatchParams &params, uint value_head,
    uint thread_index, uint lane, uint simd_group, threadgroup float *scratch,
    threadgroup bfloat *prepared) {
  constexpr uint Tokens = Tiles * SPLASH_TARGET_VERIFY_ROWS;
  device const bfloat *current_conv = reinterpret_cast<device const bfloat *>(
      current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *next_conv = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device const float *current_state = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device float *next_state = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  threadgroup bfloat *queries = prepared;
  threadgroup bfloat *keys = prepared + Tokens * HeadDim;
  gdn_wide_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim, Tokens>(
      packed, conv_weights, current_conv, next_conv, mixed, a_scale, dt_bias,
      decay, beta, params.packed_width, queries, keys, value_head, lane,
      simd_group);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  gdn_wide_scan<KeyHeads, ValueHeads, HeadDim, ConvDim, RowsInFlight, Tokens>(
      mixed, decay, beta, current_state, next_state, recurrent, queries, keys,
      value_head, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  gdn_gate_phase<ValueHeads, HeadDim, ConvDim, kDecodeSimdgroups>(
      recurrent, packed, gdn_norm_weight, gdn_hidden, Tokens * ValueHeads,
      ValueHeads, params.packed_width, scratch, value_head, thread_index, lane,
      simd_group);
  for (uint counter = 0; counter < Tiles; ++counter)
    grid_completion(arrived[counter], generation[counter], ValueHeads,
                    thread_index);
}

#define GDN_WIDE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, Tiles)   \
  kernel void Name(                                                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *conv_weights [[buffer(1)]],                        \
      device const uchar *current [[buffer(2)]],                              \
      device uchar *next [[buffer(3)]], device bfloat *mixed [[buffer(4)]],   \
      device const float *a_scale [[buffer(5)]],                              \
      device const bfloat *dt_bias [[buffer(6)]],                             \
      device float *decay [[buffer(7)]], device bfloat *beta [[buffer(8)]],   \
      device bfloat *recurrent [[buffer(9)]],                                 \
      device const bfloat *gdn_norm_weight [[buffer(10)]],                    \
      device bfloat *gdn_hidden [[buffer(11)]],                               \
      device atomic_uint *arrived [[buffer(12)]],                             \
      device atomic_uint *generation [[buffer(13)]],                          \
      constant GDNDecodeBatchParams &params [[buffer(15)]],                   \
      uint value_head [[threadgroup_position_in_grid]],                       \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float scratch[kDecodeSimdgroups];                             \
    threadgroup bfloat                                                        \
        prepared[2 * Tiles * SPLASH_TARGET_VERIFY_ROWS * HeadDim];            \
    gdn_wide_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, 2, Tiles>(         \
        packed, conv_weights, current, next, mixed, a_scale, dt_bias, decay,  \
        beta, recurrent, gdn_norm_weight, gdn_hidden, arrived, generation,    \
        params, value_head, thread_index, lane, simd_group, scratch,          \
        prepared);                                                            \
  }

GDN_WIDE_ENTRY(verify_gdn_wide16, 16, 48, 128, 10240, 2)
GDN_WIDE_ENTRY(verify_gdn_wide32, 16, 48, 128, 10240, 4)
GDN_WIDE_ENTRY(verify_gdn_wide16_vh32, 16, 32, 128, 8192, 2)
GDN_WIDE_ENTRY(verify_gdn_wide32_vh32, 16, 32, 128, 8192, 4)
#undef GDN_WIDE_ENTRY

// SPLASH_WIDE_GDN_SINGLE=parts (VH48 only): the single pass split the way
// verify_gdn_value_parts4_scan/finalize split the M8 verify. Four threadgroups
// own disjoint 32-value slices of each head for all Tiles x 8 rows (each state
// row loaded once from current, advanced over every token, stored once to
// next), and a finalize joins the complete head for the gated norm. Copies of
// the parts4 prologue/scan with a row loop (rows simd_group + 8j); the parts4
// sources are untouched, and the per value-dim arithmetic is theirs.
template <uint Tokens>
inline void gdn_wide_parts4_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device bfloat *mixed, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    uint packed_width, uint value_head, uint part, uint lane, uint simd_group,
    threadgroup bfloat *queries, threadgroup bfloat *keys,
    threadgroup bfloat *values, threadgroup float *local_decay,
    threadgroup bfloat *local_beta) {
  constexpr uint KeyHeads = 16, ValueHeads = 48, HeadDim = 128;
  constexpr uint ConvDim = 10240;
  constexpr uint KeyWidth = KeyHeads * HeadDim, Groups = HeadDim / 32;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads, PartWidth = HeadDim / 4;
  static_assert(Tokens % kDecodeSimdgroups == 0, "whole rows per simdgroup");
  const uint key_head = value_head / HeadsPerKey;
  const bool publish_shared = part == 0 && value_head % HeadsPerKey == 0;
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;
  const uint part_begin = part * PartWidth;
  const uint part_end = part_begin + PartWidth;
  for (uint token = simd_group; token < Tokens; token += kDecodeSimdgroups) {
    float q[Groups], k[Groups];
    for (uint g = 0; g < Groups; ++g) {
      q[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                                 packed_width, ConvDim, token,
                                 q_channel + 32 * g));
      k[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                                 packed_width, ConvDim, token,
                                 k_channel + 32 * g));
    }
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint g = 0; g < Groups; ++g) {
      q_sum += simd_sum(q[g] * q[g]);
      k_sum += simd_sum(k[g] * k[g]);
    }
    const float q_scale = rsqrt(q_sum / HeadDim + 1e-6f);
    const float k_scale = rsqrt(k_sum / HeadDim + 1e-6f);
    for (uint g = 0; g < Groups; ++g) {
      const uint dim = 32 * g + lane;
      const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
      const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
      queries[token * HeadDim + dim] = query;
      keys[token * HeadDim + dim] = key;
      if (publish_shared) {
        mixed[token * ConvDim + q_channel + 32 * g] = query;
        mixed[token * ConvDim + k_channel + 32 * g] = key;
      }
      if (part == 0 || (dim >= part_begin && dim < part_end)) {
        const bfloat value =
            gdn_conv_silu(packed, conv_state_in, conv_weights, packed_width,
                          ConvDim, token, v_channel + 32 * g);
        if (dim >= part_begin && dim < part_end)
          values[token * HeadDim + dim] = value;
        if (part == 0)
          mixed[token * ConvDim + v_channel + 32 * g] = value;
      }
    }
    if (lane == 0) {
      bfloat token_beta;
      float token_decay;
      gdn_value_parts4_gates(packed + token * packed_width, dt_bias, a_scale,
                             value_head, token_beta, token_decay);
      local_beta[token] = token_beta;
      local_decay[token] = token_decay;
      if (part == 0) {
        const uint gate_index = token * ValueHeads + value_head;
        beta[gate_index] = token_beta;
        decay[gate_index] = token_decay;
      }
    }
  }
  if (part == 0 && simd_group < 3) {
    const uint row = simd_group;
    for (uint g = 0; g < Groups; ++g) {
      conv_state_out[row * ConvDim + v_channel + 32 * g] =
          gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Tokens,
                         row, v_channel + 32 * g);
      if (publish_shared) {
        conv_state_out[row * ConvDim + q_channel + 32 * g] =
            gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Tokens,
                           row, q_channel + 32 * g);
        conv_state_out[row * ConvDim + k_channel + 32 * g] =
            gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Tokens,
                           row, k_channel + 32 * g);
      }
    }
  }
}

template <uint Tokens>
inline void gdn_wide_parts4_scan(
    device const float *state_in, device float *state_out,
    device bfloat *recurrent, threadgroup const bfloat *queries,
    threadgroup const bfloat *keys, threadgroup const bfloat *values,
    threadgroup const float *decay, threadgroup const bfloat *beta,
    uint value_head, uint part, uint lane, uint simd_group) {
  constexpr uint ValueHeads = 48, HeadDim = 128;
  constexpr uint RowsInFlight = 2, BatchesPerPart = 4;
  const uint first_batch = part * BatchesPerPart;
  for (uint batch = first_batch; batch < first_batch + BatchesPerPart;
       batch += RowsInFlight) {
    float state[RowsInFlight][4];
    uint value_dim[RowsInFlight];
    ulong state_base[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r) {
      value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
      state_base[r] =
          (ulong(value_head) * HeadDim + value_dim[r]) * HeadDim + lane * 4;
      for (uint i = 0; i < 4; ++i)
        state[r][i] = state_in[state_base[r] + i];
    }
    for (uint token = 0; token < Tokens; ++token) {
      const float d = decay[token];
      const float b = float(beta[token]);
      threadgroup const bfloat *key = keys + token * HeadDim + lane * 4;
      threadgroup const bfloat *query = queries + token * HeadDim + lane * 4;
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] *= d;
          memory[r] += state[r][i] * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(values[token * HeadDim + value_dim[r]]) - memory[r]) * b;
        result[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] += float(key[i]) * delta;
          result[r] += state[r][i] * float(query[i]);
        }
        result[r] = simd_sum(result[r]);
      }
      if (lane == 0)
        for (uint r = 0; r < RowsInFlight; ++r)
          recurrent[(ulong(token) * ValueHeads + value_head) * HeadDim +
                    value_dim[r]] = bfloat(result[r]);
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state_out[state_base[r] + i] = state[r][i];
  }
}

#define GDN_WIDE_PARTS4_ENTRIES(Rows)                                        \
  kernel void verify_gdn_wide##Rows##_parts4_scan(                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *conv_weights [[buffer(1)]],                        \
      device const uchar *current [[buffer(2)]],                              \
      device uchar *next [[buffer(3)]], device bfloat *mixed [[buffer(4)]],   \
      device const float *a_scale [[buffer(5)]],                              \
      device const bfloat *dt_bias [[buffer(6)]],                             \
      device float *decay [[buffer(7)]], device bfloat *beta [[buffer(8)]],   \
      device bfloat *recurrent [[buffer(9)]],                                 \
      constant GDNDecodeBatchParams &params [[buffer(10)]],                   \
      uint group [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    constexpr uint Tokens = Rows;          \
    const uint value_head = group / 4;                                        \
    const uint part = group % 4;                                              \
    device const bfloat *conv_in = reinterpret_cast<device const bfloat *>(   \
        current + ulong(params.layer) * params.conv_layer_bytes);             \
    device bfloat *conv_out = reinterpret_cast<device bfloat *>(              \
        next + ulong(params.layer) * params.conv_layer_bytes);                \
    device const float *state_in = reinterpret_cast<device const float *>(    \
        current + params.convolution_state_bytes +                            \
        ulong(params.layer) * params.recurrent_layer_bytes);                  \
    device float *state_out = reinterpret_cast<device float *>(               \
        next + params.convolution_state_bytes +                               \
        ulong(params.layer) * params.recurrent_layer_bytes);                  \
    threadgroup bfloat queries[Tokens * 128], keys[Tokens * 128],             \
        values[Tokens * 128];                                                 \
    threadgroup float local_decay[Tokens];                                    \
    threadgroup bfloat local_beta[Tokens];                                    \
    gdn_wide_parts4_prologue<Tokens>(                                         \
        packed, conv_weights, conv_in, conv_out, mixed, a_scale, dt_bias,     \
        decay, beta, params.packed_width, value_head, part, lane, simd_group, \
        queries, keys, values, local_decay, local_beta);                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    gdn_wide_parts4_scan<Tokens>(state_in, state_out, recurrent, queries,     \
                                 keys, values, local_decay, local_beta,       \
                                 value_head, part, lane, simd_group);         \
  }                                                                           \
  kernel void verify_gdn_wide##Rows##_parts4_finalize(                       \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *recurrent [[buffer(1)]],                           \
      device const bfloat *gdn_norm_weight [[buffer(2)]],                     \
      device bfloat *gdn_hidden [[buffer(3)]],                                \
      device atomic_uint *arrived [[buffer(4)]],                              \
      device atomic_uint *generation [[buffer(5)]],                           \
      constant GDNDecodeBatchParams &params [[buffer(6)]],                    \
      uint value_head [[threadgroup_position_in_grid]],                       \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    constexpr uint Tokens = Rows;          \
    threadgroup float scratch[kDecodeSimdgroups];                             \
    gdn_gate_phase<48, 128, 10240, kDecodeSimdgroups>(                        \
        recurrent, packed, gdn_norm_weight, gdn_hidden, Tokens * 48, 48,      \
        params.packed_width, scratch, value_head, thread_index, lane,         \
        simd_group);                                                          \
    for (uint counter = 0; counter < Tokens / SPLASH_TARGET_VERIFY_ROWS;      \
         ++counter)                                                           \
      grid_completion(arrived[counter], generation[counter], 48,              \
                      thread_index);                                          \
  }

GDN_WIDE_PARTS4_ENTRIES(16)
GDN_WIDE_PARTS4_ENTRIES(32)
#undef GDN_WIDE_PARTS4_ENTRIES
#undef GDN_DECODE_ENTRY
#undef GDN_DECODE_Q4_ENTRY
#undef GDN_DECODE_BODY
#undef GDN_DECODE_THREADS
#undef GDN_DECODE_BUFFERS

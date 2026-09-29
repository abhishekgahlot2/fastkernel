// Modified by meowkernels.
#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct Q4Params {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t persistent_groups;
};

static_assert(sizeof(Q4Params) == 12,
              "Q4 decode projection parameters are 12 bytes on both sides");

// Layer slots of the grouped context K/V kernels (linear_q4_context_kv.metal, which static_asserts it against its
// argument list); Q4Linear admits at most this many drafter layers.
#define SPLASH_CONTEXT_KV_MAX_LAYERS 7u

// The grouped drafter context K/V projection (linear_q4_context_kv.metal): the full projection's Q4Params, then
// layers x kv_tiles groups, one 128-column tile each, from tile kv_first_tile of each layer's qkv output.
struct Q4ContextKvParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t persistent_groups;  // output_size / 128: the full projection's one-tile-per-group grid
  uint32_t layers;
  uint32_t kv_first_tile;
  uint32_t kv_tiles;
};

static_assert(sizeof(Q4ContextKvParams) == 24,
              "Q4 context K/V parameters are 24 bytes on both sides");

// Separate from ops::LinearMatrix so host-only fields cannot change the ABI.
struct Q4PrefillParams {
  uint32_t output_size;
  uint32_t input_size;
};

static_assert(sizeof(Q4PrefillParams) == 8,
              "Q4 prefill projection parameters are 8 bytes on both sides");

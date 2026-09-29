# Device policy and performance qualification

The convergence pass starts at `2a43027`. It preserves the measured default
execution paths while making offline Q4 calibration independent of fixed
threadgroup counts from individual GPUs. It adds no kernels, weight formats,
production allocations, startup benchmarks or per-model/per-SKU tables.

## Policy ownership

`runtime/ops/Linear.cpp` owns Q4 selection. Apple9 one-lane decode uses
bfloat simdgroup matrices with 1/2/4/8 K partitions. Apple10 uses MPP tiles,
with shape and core count selecting grids and the narrow M24 variant.
Apple10 split-K tiles are offline candidates only. Their former one-lane
defaults have been withdrawn after reproducible speculative-acceptance
reductions on some M5 prompts. Those projections use the existing sequential
tiles again; paired N256, M24, and Apple9 simdgroup selection are unchanged.
Prefill and wider decode batches retain their existing rules. The obsolete
Apple9 one-lane MPP branches have been removed; those kernels remain useful
as qualification references and offline candidates.

Core count comes from the Metal device's IORegistry property. Missing metadata
uses one 32-core estimate across families, an intermediate value in the
16–40-core range of our reference machines. This is not a calibrated optimum
or a performance guarantee for unidentified GPUs. A nonzero reported count
always overrides it. Family 11 policy tests check extrapolation only: actual
validation here covers families 9 and 10. Core count alone cannot describe
memory bandwidth, cache capacity, power state or compiler behavior.

MoE routing scales its row threshold with core count. The expert tile's
four-SIMD-group Apple9 decode default remains a family rule, measured on the
40-core M3 Max. Smaller Apple9 devices need an expert-kernel comparison before
claiming that rule is optimal. Prefill and Apple10 retain eight SIMD groups.

## Bounded offline calibration

Q4 candidates always start with the shipped baseline. Persistent grids now
include two, three and four threadgroups per reported core plus the full grid,
instead of fixed counts 36/60/80. Apple9 additionally exposes every valid
simdgroup split in 1/2/4/8. The maximum candidate count is 20, derived beside
`Q4Linear::kMaximumCandidates`; deduplication handles small grids. Prefill
candidates are unchanged. New candidates do not automatically change serving.

The existing offline tuner qualifies numerical results, admits the maximum
candidate workspace, alternates baseline/candidate timing and requires both
GPU and wall-time evidence. Interrupted or inconclusive runs keep the default.
Its scratch admission already covered candidate maxima; the independent batch
reuse test was corrected to do the same after the extra K splits exposed its
baseline-only allocation assumption.

On a new device, build the existing tuning tool and use an installed model:

```sh
make -j8 all build/engine-tests/tune-kernels
build/engine-tests/tune-kernels build/splash.metallib MODEL_ROOT \
  --seconds 30 --pairs 31 --candidates --confirm 12
```

Record GPU family/core count, power mode, OS/toolchain and source identity.
Keep other GPU work idle. The tool prints measurements and confirms complete
prefill/decode graphs; it does not persist a serving profile. Operator timings
include standalone preparation and do not substitute for fused-producer or
cold full-model measurements. Promote a default change only after repeatable
whole-model A/B results, unchanged correctness/state-restoration behavior,
and acceptable speculative acceptance and memory use. Check both models,
short/long prompts and batch widths 1–4. Preserve the baseline when evidence
is mixed. For the family-only MoE rule, separately compare four/eight groups
on the missing hardware; the existing MoE tuner does not vary that rule.

## Convergence validation (2026-09-21)

- An independent before/after snapshot compares 1,591,200 default plans across
  families 9/10/11, every core count 1–128, unknown count, the IORegistry reader's
  upper bound of 4096, production-like shapes and dispatch boundaries. Configs,
  pipeline names and workspace sizes are byte-identical. These simulated core
  counts establish policy consistency, not measured performance on those GPUs.
- Permanent CPU tests cover 84,240 decode workload/device combinations: valid
  grids, bounded unique candidates, retained defaults and workspace admission
  after installing each candidate. `test-engine-cpu` now runs `linear-plan
  --cpu`, so these checks do not depend on a Metal test run.
- M3 Max 40, M5 Pro 16 and M5 Pro 20 pass full builds and CPU suites, Linear
  candidate numerical tests, tuning controls/batch reuse, and the 168-case
  independent fp64 simdgroup test with GPU shader validation. Both remote
  machines are on AC. All build the same production source identity:
  `src-a996ac63153606d2ab3534be64d2ca804dc396d44e06ab222cfc42922b2a3740`.
- On each machine, the production metallib is byte-identical to its pre-cleanup
  library. This pass establishes unchanged default GPU work and valid expanded
  calibration; it does not claim a new serving speedup or repeat the previous
  whole-model ABBA measurements.

Raw logs, the original failed test and its passing rerun, and the independent
policy snapshot harness are archived under `convergence/` in the review
evidence directory. Earlier serving results remain in
[remaining-decode-optimizations.md](remaining-decode-optimizations.md) and
[apple9-simdgroup.md](apple9-simdgroup.md).

## Unknown-core fallback follow-up

The unknown-core path now uses a single 32-core estimate. The convergence
snapshot reported above precedes this fallback change; unknown-core plans
intentionally differ. Existing CPU policy tests cover unknown counts in both
prefill and decode, including equivalence to an explicitly reported 32-core
GPU. Known-core plans remain byte-identical in a separate before/after
comparison. This fallback does not require startup or user-run calibration.

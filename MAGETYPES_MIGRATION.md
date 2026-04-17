# Magetypes migration plan

Branch: `magetypes-refactor` (forked from `apr16-unsafe-reduction` at `f501fa5`).

Goal: replace the hand-written `SseVector` / `AvxVector` / `AvxVectorSse` /
`NeonVector` / `*Q0_15*` vector wrappers — which each wrap individual
intrinsic calls in `unsafe { }` blocks — with `magetypes` SIMD primitive
types that produce the same codegen without `unsafe` at the arithmetic
sites. Eventual target: `#![forbid(unsafe_code)]` at the crate root.

## Starting point

- Baseline commit: `f501fa5` on `apr16-unsafe-reduction`.
- Unsafe tokens in `src/`: **150**.
- Audit CI green (asm per-symbol, unsafe per-file, coverage, zenbench
  baseline for `linux-x86_64` committed).

## Mapping

### Types

| current wrapper                | magetypes replacement          | notes                                  |
|--------------------------------|--------------------------------|----------------------------------------|
| `SseVector` (`__m128`, f32)    | `magetypes::simd::v3::f32x4`   | v3 = AVX2+FMA tier, 128-bit lane       |
| `AvxVector` (`__m256`, f32)    | `magetypes::simd::v3::f32x8`   | AVX2 256-bit                           |
| `AvxVectorSse` (`__m128`, f32) | `magetypes::simd::v3::f32x4`   | same as SseVector                      |
| `NeonVector` (`float32x4_t`)   | `magetypes::simd::neon::f32x4` |                                        |
| `NeonVectorDouble`             | two `f32x4`s or a custom pair  | no native 256-bit NEON                 |
| `SseVector` (Q0_15, `__m128i`) | `magetypes::simd::v3::i16x8`   |                                        |
| `AvxVectorQ0_15` (`__m256i`)   | `magetypes::simd::v3::i16x16`  |                                        |
| `NeonVectorQ0_15` (`int16x4_t`)| half of `magetypes::simd::neon::i16x8` | NEON has 64-bit halfs too     |

### Operators

| current                                  | magetypes                            |
|------------------------------------------|--------------------------------------|
| `impl From<f32> for SseVector`           | `f32x4::splat(token, v)`             |
| `impl Add/Sub/Mul for SseVector` (trait) | `+` / `-` / `*` on `f32x4` (natural) |
| `impl FusedMultiplyAdd for SseVector`    | `f32x4::mla(a, b, c)` (method)       |
| `impl Fetcher<SseVector>::fetch`         | `f32x4::load(token, &array)` (safe)  |
| raw `unsafe { _mm_load_ps(ptr) }`        | `f32x4::load(token, &array)` via safe_unaligned_simd |

### Entry points

| current                                    | magetypes                           |
|--------------------------------------------|-------------------------------------|
| `#[target_feature(enable = "avx2,fma")]`   | `#[arcane]` with `X64V3Token` arg   |
| `#[target_feature(enable = "sse4.1")]`     | (no SSE4.1 token directly — lives under `X64V3Token` as AVX2+FMA tier, which is a superset on modern hardware) |
| `#[target_feature(enable = "neon,rdm")]`   | `NeonToken` (baseline on aarch64) + `Arm64V2Token` for RDM    |
| `transform_chunk` calls dispatch           | `incant!(interpolate(...), [v3, neon, scalar])` for runtime dispatch |

## Sequencing

1. **Drop custom wrappers.** Delete `SseVector`, `AvxVector`, `AvxVectorSse`,
   `NeonVector`, `NeonVectorDouble`, `SseVectorQ0_15`, `AvxVectorQ0_15`,
   `AvxVectorQ0_15Sse`, `NeonVectorQ0_15`, `NeonVectorQ0_15Double` from
   the interpolator files. All of their `impl From/Sub/Add/Mul/mla` blocks
   go with them — that's the ~120 `unsafe { intrinsic }` blocks we couldn't
   touch under Rust's trait-impl target_feature restrictions.
2. **Rewrite interpolator bodies.** `TetrahedralSse::interpolate` etc.
   take a token parameter, use `f32x4` throughout, call arithmetic via
   natural operators. Body shape should be close to 1:1 with today's.
3. **Fetcher swap.** `TetrahedralSseFetchVector::fetch` becomes a
   `#[arcane] fn` returning `f32x4`, using `f32x4::load(token, array)`.
   Needs a typed cube for now — revisits the Section 4 problem (cube
   `get_unchecked(offset..)`) but `safe_unaligned_simd` (re-exported by
   `archmage`) takes `&[f32; 4]` so the typed-cube refactor may become
   necessary alongside.
4. **Trait dispatch.** `SseMdInterpolation` / `NeonMdInterpolation` etc.
   stay as traits but their method signatures take the token. `Box<dyn>`
   dispatch from `transform_chunk` still works because the token is just
   a zero-sized type — passing it is free.
5. **Scalar fallback tier.** `#[magetypes(v3, neon, wasm128, scalar)]` on
   the generic interpolator body generates the scalar variant
   automatically, replacing the hand-maintained `src/conversions/lut4.rs`
   scalar path.
6. **`forbid(unsafe_code)`.** Add the attribute once all three SIMD
   interpolator crates are clean. `Fetcher::fetch` should be the last
   holdout to clear.

## Risks to measure

- **Codegen parity.** `f32x4 + f32x4` must produce `_mm_add_ps`, not a
  scalar loop. Verified by asm-audit at each step.
- **`f32x4::load` vs `_mm_load_ps`.** The `safe_unaligned_simd` load is
  reference-based; the current code uses an aligned pointer. Verify
  LLVM still emits `movaps` not `movups` when the reference is proven
  aligned (SseAlignedF32 has `#[repr(align(16), C)]`).
- **Bench regression.** Every commit runs the zenbench transforms baseline
  against `benchmarks/transforms_linux-x86_64_2026-04-17.md` as the
  reference. Safety refactor should not regress > bench noise.

## What's not in scope

- Performance tuning beyond what magetypes gives for free.
- Wider restructuring (keep per-arch interpolator files even if
  `#[magetypes]` could merge them — that's a follow-up).
- Replacing `Fetcher` trait with a different dispatch shape.

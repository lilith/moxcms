# archmage / magetypes gaps hit during moxcms migration

Running log of places where the current archmage + magetypes API
forced moxcms to add workarounds (extra `unsafe`, scalar fallback,
adapter layers), or where a reasonable pattern simply didn't
compile. Upstream fixes to any of these would let moxcms shed more
`unsafe` and/or match the hand-written codegen more directly.

Each entry: what we hit → what we did as a workaround → what an
upstream fix could look like.

## 1. No `X64V2Token` tier for `F32x4Backend` / `I16x8Backend`

**Where:** `src/conversions/sse/interpolator{,_q0_15}.rs` —
SSE4.1-only hand-written code paths.

**What we hit:** magetypes' `F32x4Backend` is implemented for
`X64V3Token` (AVX2+FMA), `NeonToken`, `Wasm128Token`, `ScalarToken`.
There is no impl for `X64V2Token` (SSE4.1 only). So the SSE4.1
hand-written paths in moxcms can't be migrated to the probe — the
probe's `#[magetypes(v3, neon, wasm128, scalar)]` tier list has no
"v2" option, and adding `v2` to the list fails to compile because
no backend exists.

**Workaround:** left `sse/interpolator*.rs` hand-written. SSE-only
CPUs continue to use the existing unsafe intrinsic path. Modern
x86_64 CPUs hit the AVX tier via V3, and SSE4.1-only CPUs (~10+
years old) are a dying tier.

**Upstream fix:** implement `F32x4Backend for X64V2Token` +
`I16x8Backend for X64V2Token`. V2 already has SSE4.1, which includes
`_mm_mul_ps`, `_mm_fmadd_ps` (wait — no, FMA is V3-only), etc.
Without FMA, the f32 body would need `mul + add` pairs instead of
fused. Codegen parity would be fine on SSE4.1 hardware; different
on FMA-capable.

**Priority:** Low. The world is moving off SSE4.1-only x86_64.

---

## 2. NEON `rdm` not in `F32x4Backend` / `I16x8Backend` for `NeonToken`

**Where:** `src/conversions/neon/interpolator_q0_15.rs` — Q0.15 path
on aarch64.

**What we hit:** hand-written NEON Q0.15 uses `vqrdmulhq_s16`
("signed rounding doubling multiply high, saturating") for the
fixed-point Q0.15 multiply in mla. This requires the `rdm` ARM
target feature, which is in `Arm64V2Token` (Cortex-A55+, Apple M1+,
Graviton 2+) but not in baseline `NeonToken`.

Our probe's `_neon` variant is gated by `#[target_feature(enable =
"neon")]` only. The scalar `q15_mulhrs` round-trip in the probe
body *may* autovec to `vqrdmulhq_s16` under LLVM, but *may not*
without `rdm` enabled. We haven't verified the actual codegen on
aarch64 yet.

**Workaround:** integrated as-is, accept potential NEON Q0.15 perf
regression pending asm audit + bench measurement.

**Upstream fix:** two options —
  (a) implement `F32x4Backend + I16x8Backend` for `Arm64V2Token`,
      enable `#[magetypes(v3, arm_v2, neon, wasm128, scalar)]`.
      Probe generates an `_arm_v2` variant with `rdm` enabled, we
      dispatch via that token when available.
  (b) expose an `i16x8::mul_high_round_scale()` method on the
      generic type, backed by `vqrdmulhq_s16` on NEON, `pmulhrsw`
      on x86_v3, `i16x8.q15mulr_sat_s` on wasm128, scalar on
      `ScalarToken`. Probe calls the primitive directly — no
      autovec guess.

Option (b) is cleaner — adds a named primitive for the actual
Q0.15 operation we want, instead of hoping LLVM reverse-engineers
it from scalar arithmetic. Option (a) is more general.

**Priority:** High for NEON Q0.15 perf. Needs upstream.

---

## 3. No way to return `GenericF32x4<Token>` from `#[magetypes]`-generated variants

**Where:** all four probe modules.

**What we hit:** `#[magetypes]` substitutes `Token` with each
tier's concrete token type. A probe fn that returns `GenericF32x4<Token>`
expands to four fns each with a different return type:
`f32x4<X64V3Token>`, `f32x4<NeonToken>`, etc. `incant!` at the
dispatch site then fails because the arms have different types.

**Workaround:** probe fns take `out: &mut [f32; 4]` (or `[f32; 8]`,
`[i16; 8]`, `[i16; 16]` for the other widths) and `.store(out)` at
the end. Return type is `()`. Caller materializes the SIMD type
from the stack buffer via `_mm_loadu_ps` / `vld1q_f32`. LLVM fuses
the round-trip away in the common case (when the caller is inside
a matching-feature `#[target_feature]` region — which we arrange
with a small `#[archmage::arcane]` dispatch helper).

**Upstream fix:** possibly a way to say "all tiers in this tier
list have the same associated `Output`, return `Output`"?  Unclear
how to spell this without type erasure. Current workaround is
acceptable — the `#[arcane]` helper puts the caller in the right
tf region so LLVM folds the round-trip.

**Priority:** Low — workaround is ergonomic enough.

---

## 4. No dual-cube fetch primitive for Double variants

**Where:** AVX + NEON Double interpolator variants
(`AvxMdInterpolationDouble`, `NeonMdInterpolationDouble`) and their
Q0.15 counterparts — 8 interpolators total, still hand-written on
`magetypes-refactor`.

**What we hit:** the hand-written `TetrahedralAvxFetchVector {
cube0, cube1 }` fetches 4 floats from each cube per (x,y,z) and
packs them into a 256-bit `AvxVector`:
```rust
AvxVector::from_sse(
    _mm_load_ps(cube0_ptr),
    _mm_load_ps(cube1_ptr),
)
```

Our `simd_interp_double.rs` probe takes a single `&[Aligned8<f32>]`
where each entry is 8 consecutive floats. There is no way to build
an `f32x8` from two `f32x4` values inside the generic probe body
— `GenericF32x8<T>` doesn't expose a `from_halves(lo, hi)` method.

**Workaround:** none yet. The 8 Double variants remain hand-written.
This is the blocker for deleting the shared `AvxVector*Sse` /
`NeonVector*` arithmetic impls (~16 `unsafe` tokens each across the
AVX/NEON files).

**Upstream fix:** add a `from_halves` construction method to the
wider generic types, token-gated like every other construction
path on `GenericF32x8<T>` / `GenericI16x16<T>`:

```rust
impl<T: F32x8Backend> f32x8<T> {
    #[inline(always)]
    pub fn from_halves(token: T, lo: f32x4<T::Half>, hi: f32x4<T::Half>) -> Self
    where
        T: F32x4BackendHalf,  // associated half-width backend
    { … }
}
```

Or, more directly (token as safety witness, halves as same backend):

```rust
impl<T: F32x8Backend + F32x4Backend> f32x8<T> {
    #[inline(always)]
    pub fn from_halves(token: T, lo: f32x4<T>, hi: f32x4<T>) -> Self { … }
}
```

Per-tier lowering:

- `X64V3Token`: `_mm256_insertf128_ps(_mm256_castps128_ps256(lo_repr), hi_repr, 1)`
- `NeonToken`: construct the `[float32x4_t; 2]` Repr directly
  from `(lo_repr, hi_repr)` — zero-cost because Repr *is* the pair
- `Wasm128Token`: same — `f32x8<Wasm128Token>` polyfills to
  `[v128; 2]`, `from_halves` is `[lo_repr, hi_repr]`
- `ScalarToken`: `[..lo_array, ..hi_array]` — 8-element i / f array

The token isn't decorative — **raising on x86 needs real CPU
feature proof the narrower halves don't carry on their own**.
`f32x4<X64V3Token>` exists on any V3 CPU, but combining two of
them into `f32x8` emits `vinsertf128` (or `vmovaps` ymm-pair)
which is **AVX** — a *different* ISA tier than the SSE4.1 you'd
need for `f32x4` alone. The token passed to `from_halves` is what
proves AVX (and hence the 256-bit-capable ISA path) is available.

Lowering (`split` / `to_halves`) is the other direction and
doesn't need a token: extracting the low/high 128 bits of an
already-existing `ymm` register is just `vextractf128` which comes
with the same ISA tier the 256-bit vector was built under — and
on NEON / Wasm128 where the wider type is literally a Repr pair,
lowering is a tuple destructure.

On NEON/Wasm128, `GenericF32x8<T>` polyfills to a Repr pair, so
raising is also just a tuple construction — no wider ISA required.
The token is redundant in those tiers, but the API takes one
uniformly to match the construction-gated pattern of `splat` /
`load` / `zero` / `from_array`.

Same shape for `i16x16::from_halves(token: T, lo: i16x8<T>,
hi: i16x8<T>)`, `i32x8::from_halves`, etc.

With this primitive, a single `interpolate_*_dual_cube` probe fn
can fetch from two 4-wide cubes and pack into `f32x8` per-access.
The Double variants migrate cleanly and the ~16 shared `unsafe`
tokens in moxcms come out.

**Priority:** **High.** This is the largest remaining piece to
net the moxcms `unsafe` reduction.

---

## 5. `cargo fix` strips macro-provided imports

**Where:** edits to `simd_interp*.rs` test modules — `use
archmage::incant;` and `use archmage::SimdToken;` were flagged as
unused and auto-removed by `cargo fix --lib --allow-dirty`, even
though `incant!(…)` and `TOKEN::summon()` were live in the same
module.

**What we hit:** `cargo fix` can't see through proc-macro expansion
to tell the import is used inside `incant!(…)`.

**Workaround:** manually restore the imports after `cargo fix`.
Noticed at integration time; not systemic.

**Upstream fix:** probably needs a fix in rustc's unused-import
lint (or cargo fix's interaction with it), not archmage itself.
Could add `#[allow(unused_imports)]` to the imports as a hint, but
that's worse than just not running cargo fix on these files.

**Priority:** None — annoyance, not a blocker.

---

## 6. `incant!` default tier list includes `v4` under `avx512` feature

**Where:** `simd_interp*.rs` test modules with `cargo test
--all-features`.

**What we hit:** `cargo test --all-features` enables `avx512`. In
that configuration, `incant!(foo(x))` (no explicit tier list) uses
a default list that includes `v4`. Our `#[magetypes(v3, neon,
wasm128, scalar)]` tier list on the probe does *not* generate a
`_v4` variant (no `F32x4Backend for X64V4Token` in magetypes, I
think). So `incant!` fails with `no function or associated item
named interpolate_*_v4 in this scope`.

**Workaround:** always pass an explicit tier list to `incant!`
matching the probe's `#[magetypes]` list:
```rust
incant!(interpolate_tetra::<…>(…), [v3, neon, wasm128, scalar]);
```

**Upstream fix:** `incant!` should detect the surrounding
`#[magetypes]` tier list (or at least validate at expansion time
that each dispatched tier has a corresponding variant). Or the
docs should be louder about needing explicit lists when the
defaults extend past what the probe was generated for.

**Priority:** Low — workaround is a one-line change.

---

## 7. `#[deprecated(since = "0.5.0")]` on `forge_token_dangerously`

**Where:** considered as a perf shortcut when we're already inside
`#[target_feature]` and have proof the feature is present —
avoiding `summon()`'s atomic load + branch.

**What we hit:** `forge_token_dangerously()` is gated behind
`forge-token-api` Cargo feature *and* `#[deprecated]` with note
"Pass tokens through from summon() instead". The recommendation is
to pass tokens through the call chain, not forge them locally.

**Workaround:** use `summon()` + `.expect(…)` at the trait method
boundary, dispatch through an `#[archmage::arcane]` helper so the
`#[target_feature]` boundary optimizes correctly.

**Upstream fix:** not needed — the deprecation is reasonable. Note
is here for completeness so future moxcms sessions remember not to
reach for `forge_token_dangerously`.

**Priority:** None — documented for future readers.

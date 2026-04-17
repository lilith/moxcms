# archmage / magetypes gaps hit during moxcms migration

Running log of places where the current archmage + magetypes API
forced moxcms to add workarounds (extra `unsafe`, scalar fallback,
adapter layers), or where a reasonable pattern simply didn't
compile.

## Boundary of responsibility

Per upstream maintainers (2026-04-17): archmage / magetypes will
ship **primitives that can be made safe and sound** — that means
the SIMD type construction / lowering / raising operations in
particular (`from_halves`, `split` / `to_halves`, possibly extra
backend tiers like `Arm64V2` for `rdm`). **Architectural
redesigns on the moxcms side are moxcms' job**, regardless of how
invasive — including the dispatch-up refactor (#0), restructuring
trait signatures, etc.

So entries below are split by ownership:

  * **Upstream-trackable** (#2, #4): primitives or backend tiers
    that archmage/magetypes will provide.
  * **Moxcms-owned** (#0, #7): redesigns or workarounds we
    implement here.
  * **Notes / wontfix** (#1, #3, #5, #6, #8): documented but
    don't need action.

Each entry below: what we hit → what we did as a workaround →
what an upstream fix would look like (and which side will do it).

## 0. Per-pixel function-call boundary cost on Rust stable 1.89

**Where:** every `inter3_sse` / `inter3_neon` call in the probe-integrated
path.

**What we hit:** the probe-integrated dispatch goes through an
`#[archmage::arcane]` helper fn because the trait method can't be
`#[target_feature]` (E0053). Each `inter3_sse` call on the hot path
therefore crosses a function-call boundary → tf helper → inlined
probe → return via stack round-trip → out.

For `rgb_lut_to_srgb` in the moxcms benches — small 3-channel LUT
with many calls per transform — this per-call boundary cost (~10ns
of call + prologue + epilogue + token-check) dominates a per-pixel
interpolation budget of a few ns. Measured: **−17% to −43%
throughput vs the baseline** (all four interpolation methods,
both `_lo` and `_hi` weight scales).

`#[cmyk_to_srgb]` is much less affected (larger LUT, fewer calls
per transform amortizes the boundary cost; +2% to +8% actually
*faster* than baseline on the same bench run).

**Root cause confirmed by asm inspection:** the arcane-generated
inner fn is emitted as a separate symbol with `#[inline]` (not
`#[inline(always)]`). LLVM doesn't inline across the tf boundary
on stable because `target_feature_inline_always` is nightly-only
(rust-lang/rust#145574). A manual `#[inline(always)]` on the
source fn doesn't propagate through the arcane-generated wrapper.

**Workaround attempts:**
  * Added `#[inline(always)]` on the `#[arcane]`-attributed source
    fn — no effect (macro emits its own `#[inline]` on the inner).
  * Replaced `#[arcane]` with plain `#[inline(always)]` + an
    `unsafe { probe_v3(…) }` call — my dispatch helper inlined
    into the trait impl (no more `__arcane_*_dispatch` symbol),
    but **the probe's inner tf fn still became the call
    boundary** (`#[magetypes]` emits `#[inline]`, not
    `#[inline(always)]`, on its inner tf fn). Same bench numbers
    as the `#[arcane]` version — ±1% noise. Reverted because it
    adds +1 unsafe block for no perf win.
  * Removed the bounds-hoist asserts (−0.5% — within noise, so not
    the cost).
  * Pass the output via `&mut [f32; 4]` + `_mm_loadu_ps(&out)`
    round-trip — already in the code; confirmed that LLVM does fuse
    the round-trip within the tf region.

**What actually works** (per archmage's design):
  * `#[rite]` helper fns inline into `#[arcane]` callers — but the
    trait method can't be `#[arcane]` (E0053 on tf attribute), and
    lifting the `#[arcane]` upward means dispatch-up (see below).
  * Plain `#[inline(always)]` fns inline into `#[arcane]` callers.
    Same constraint.

So the tf boundary is the real cost, and it only disappears when
the *caller* is a tf region. Getting there from a trait method
requires restructuring the dispatch.

**Architectural fix:** move dispatch up from the `inter3_sse`
trait method to `transform_chunk` (or higher):

  1. `transform_chunk` calls `X64V3Token::summon().expect(…)` once
     per chunk (1000s of pixels), not once per pixel.
  2. Instead of `Box<dyn AvxMdInterpolation>`, dispatch to a
     concrete `#[target_feature]` inner fn that takes the token
     and iterates the pixel loop internally.
  3. Per-pixel, the fn is a direct intra-tf call — no boundary,
     LLVM inlines the interpolator body.

This is the "dispatch-up" refactor noted in
`benchmarks/transforms_linux-x86_64_2026-04-17_post-integration.md`.
Requires changes to `src/conversions/transform_lut3_to_3.rs`,
`transform_lut3_to_4.rs`, `transform_lut4_to_3.rs` (the chunk
dispatchers) plus the trait definitions themselves.

**Owned by:** **moxcms.** Per upstream, redesigns here are our
job. The plan:

> **Dispatch-up refactor** — hoist `#[arcane]` (and the
> `summon` + token acquisition) above the pixel loop so the loop
> itself runs inside a `#[target_feature]` region. Every
> interpolator call inside the loop is then a same-tf-region
> call which LLVM inlines for free. Mechanically:
>
>   * `transform_chunk` (or the per-row dispatcher) acquires the
>     token once via `summon`/`incant!`.
>   * Replace `Box<dyn AvxMdInterpolation>` with a concrete
>     monomorphized fn-ptr or generic dispatch that takes the
>     token and the cube + weights, runs the pixel loop
>     internally.
>   * The interpolator probe is called from inside that tf
>     region — fully inlined, zero call-boundary cost.
>
> Touches `transform_lut3_to_3.rs`, `transform_lut3_to_4.rs`,
> `transform_lut4_to_3.rs`, plus the `*MdInterpolation` traits.
> Non-trivial but doesn't depend on upstream.

(Stabilization of `#![feature(target_feature_inline_always)]`
would resolve this organically — but until that happens,
dispatch-up is moxcms' workaround.)

**Priority:** High — this is the root cause of the moxcms
`rgb_lut_to_srgb` 30% regression. Fixable on our end (dispatch-
up) but non-trivial.

---

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

**Owned by:** **archmage / magetypes** (upstream). User confirmed
this is on the upstream roadmap.

**Priority:** High for NEON Q0.15 perf.

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

**Owned by:** **archmage / magetypes** (upstream). Tracked as
[archmage#36](https://github.com/imazen/archmage/issues/36).
User confirmed `from_halves` + `lo`/`hi` are on the upstream
roadmap as safe + sound primitives.

**Priority:** **High.** Largest remaining piece to net the moxcms
`unsafe` reduction.

---

## 6. `cargo fix` strips macro-provided imports

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

## 5. `incant!` default tier list includes `v4` under `avx512` feature

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

## 7. Per-arch alignment wrappers can't share a `Pod`-derived type

**Where:** `src/conversions/{avx,sse,neon}/interpolator{,_q0_15}.rs` —
each arch defines its own `SseAlignedF32` / `NeonAlignedF32` /
`AvxAlignedI16` / `NeonAlignedI16x4`. The probe defines
`simd_interp::Aligned4<T>`. All are `#[repr(align(16 or 8), C)]`
wrappers of `[f32; 4]` or `[i16; 4]`.

**What we hit:** the adapter fns reinterpret `&[SseAlignedF32]`
(etc.) into `&[Aligned4<f32>]` so the probe can consume them. The
layouts are literally identical, but the reinterpret requires
`unsafe { core::slice::from_raw_parts(…) }` — one block per
adapter × 16 adapters = 16 `unsafe` tokens on the moxcms side.

Attempted fix: derive `bytemuck::Pod + Zeroable` on `Aligned4<T>`
and use `bytemuck::cast_slice`. Blocked: `Pod` can't be derived
for non-`packed` generic structs (padding can't be verified at
derive time when `T` is a generic parameter).

Also attempted: `pub(crate) type SseAlignedF32 =
crate::conversions::simd_interp::Aligned4<f32>;`. Blocked: type
aliases can't be used as tuple-struct constructors
(`SseAlignedF32([…])` fails with E0423), and there are six
construction sites across `{avx,sse,neon}/{lut4_to_3,
t_lut3_to_3}.rs` that use this idiom.

**Workaround:** kept per-arch structs + the 16 `unsafe` slice
reinterprets in the adapter helpers.

**Owned by:** **moxcms.** Per upstream, this kind of layout-
plumbing redesign is on us. Two paths:

  (a) Add concrete `Aligned4F32` / `Aligned4I16` /
      `Aligned8F32` / `Aligned16I16` types in `simd_interp.rs`
      with manual `unsafe impl Pod` / `Zeroable`. Replace
      generic `Aligned4<T>` callers. Aliases per-arch types to
      these concrete types via `pub(crate) type SseAlignedF32 =
      Aligned4F32` (the constructor sites already use the named
      structs, so they'd update from `SseAlignedF32([…])` to
      `Aligned4F32::new([…])` factory call — also fix the type
      alias E0423).
  (b) Add an `unsafe impl<T: Pod> Pod for Aligned4<T> {}` — one
      unsafe impl in the probe module, replaces 16 unsafe slice
      casts in the adapters. Net −15. Localized.

(b) is the smallest change and gets us to net negative on
unsafe. (a) is cleaner long-term.

**Priority:** Medium. 16 `unsafe` tokens sitting in the
adapter helpers. Until resolved, moxcms' net `unsafe` stays ~16
above the pre-migration baseline even after the Double variants
migrate (which removes the other 16 we plan to remove).

---

## 8. `#[deprecated(since = "0.5.0")]` on `forge_token_dangerously`

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

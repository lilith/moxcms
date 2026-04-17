# Post-integration benchmarks (AVX + NEON × f32 + Q0.15 single-variant through magetypes probe)

**git:** `8f29de1` (HEAD of magetypes-refactor at time of run)
**baseline:** `benchmarks/transforms_linux-x86_64_2026-04-17.md` (commit `97d44665`)
**system:** AMD Ryzen 9 7950X, load avg 1.97 at start (quiet)
**total:** 75.1s (355 noisy rounds)

## cmyk_to_srgb

(Top bars, partial capture.)

| Benchmark | Throughput (this run) | Baseline | Delta |
|-----------|-----------------------|----------|-------|
| prism_lo | 152 Mpixels/s | 141 | **+7.8%** |
| linear_lo | 148 Mpixels/s | 139 | **+6.5%** |
| tetra_hi | 133 | 130 | +2.3% |
| prism_hi | 125 | 121 | +3.3% |
| pyramid_hi | 124 | 115 | +7.8% |
| linear_hi | 114 | 108 | +5.6% |
| (tetra_lo, pyramid_lo not captured) | — | — | — |

**Mostly improved.** Probe path wins on cmyk_to_srgb (the headline
case) by 2–8%.

## srgb_to_cmyk

| Benchmark | Throughput | Baseline | Delta |
|-----------|-----------|----------|-------|
| tetra_lo | 151 | 154 | −1.9% |
| tetra_hi | 132 | 135 | −2.2% |
| pyramid_lo | 140 | 145 | −3.4% |
| pyramid_hi | 114 | 120 | −5.0% |
| prism_lo | 132 | 137 | −3.6% |
| prism_hi | 111 | 113 | −1.8% |
| linear_lo | 121 | 123 | −1.6% |
| linear_hi | 100 | 105 | −4.8% |

**Small regression (2–5%).**

## rgb_lut_to_srgb

| Benchmark | Throughput | Baseline | Delta |
|-----------|-----------|----------|-------|
| tetra_lo | 186 | 268 | **−30.6%** |
| tetra_hi | 156 | 205 | **−23.9%** |
| pyramid_lo | 169 | 209 | **−19.1%** |
| pyramid_hi | 133 | 160 | **−16.9%** |
| prism_lo | 144 | 212 | **−32.1%** |
| prism_hi | 114 | 166 | **−31.3%** |
| linear_lo | 131 | 228 | **−42.5%** |
| linear_hi | 107 | 166 | **−35.5%** |

**Significant regression (17–43%).**

Why this group specifically? rgb_lut_to_srgb does many more
interpolator calls per pixel (smaller 3-channel LUT, more calls per
unit time) so per-call overhead dominates. Candidates for the
regression:

1. **`X64V3Token::summon()` per `inter3_sse`** — cached atomic load
   + branch. Should be ~1 ns but might not be amortizing as
   expected.
2. **Bounds-hoist assertions** — 4 compares + branches per call.
   Predicted-taken; should be ~4 ns.
3. **Stack round-trip** (`.store(out)` → `_mm_loadu_ps(&out)`) not
   fully fused. Checked asm — direct register-to-register path
   present but probably not always.
4. **`#[arcane]` wrapper function-call overhead** — measured: each
   arcane fn has its own prologue/epilogue. In theory LLVM inlines
   across tf boundaries within the same crate, but actual asm for
   the adapters showed wrapper symbols.

Next step: move dispatch up. Instead of summon + assert + probe at
every `inter3_sse` call, pull those out to the enclosing
`transform_chunk` (once per transform) and pass the token +
probe-suitable cube reference through. Then the inner-loop fn call
doesn't pay the per-call token / assert cost.

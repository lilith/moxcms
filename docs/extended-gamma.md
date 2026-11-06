# Extended pure-gamma correction

With extended-range RGB/XYZ enabled, negative linear samples can reach an ICC
pure-gamma curve. Its intended odd extension is `sign(x) * |x|^gamma`.
The fast power's negative-base handling instead depends on the exponent's
integer part. The approximation also needs a bounded input/result range:
zero, subnormals and extremes can otherwise produce unrelated finite values.

The fix uses the existing approximation inside its valid range, caches input
bounds and uses exact power elsewhere. A shared RGB admission check keeps the
common path short; the mixed/out-of-range path is a separate cold function.
Zero shortcuts preserve ICC exponent-zero and CICP negative-zero behavior.
CICP gamma 2.2/2.8 and SMPTE 428 retain their existing negative-input clamp.
No public API, power coefficients, table, dependency or version changes.

## Reproduce the failures

```sh
cargo run --locked --release --features extended_range --example extended_gamma
```

The [39-line example](../examples/extended_gamma.rs) prints raw and clipped
outputs for constructed pixels and samples. [Captured TSV](extended-gamma-failures.tsv)
contains five cases from each of four runtime revisions, freshly rerun with the
same example, Rust 1.99.1, pxfm 0.1.30 and default CPU flags on Ryzen 9 7900X.
This is a targeted reproduction, not a real-image corpus survey.

| Stage | Runtime source | Negative gamma encode of -0.5 | Clipped ProPhoto (0.5,0.5,0) to sRGB |
|---|---|---:|---|
| Original | [84104443](https://github.com/lilith/moxcms/commit/841044439077ec1ec67a71b2db979ab51d304fcf) | +0.72965837 | (1, 1, 0) |
| Sign only | [45b011cd](https://github.com/lilith/moxcms/commit/45b011cdd045351eea689d250df0e7cc55af84dd) | -0.72965837 | (1, 1, 0) |
| Range guard | [4ab190f0](https://github.com/lilith/moxcms/commit/4ab190f0e76f95b332a04ce059b51c893e2f5d79) | -0.72965837 | (0.6461351, 0.57291955, 0) |
| Optimized fix | [64226a86](https://github.com/lilith/moxcms/commit/64226a8671af1827ff1d30300c8d0c24bf689211) | -0.72965837 | (0.6461351, 0.57291955, 0) |

Here gamma encode uses the ICC gamma 563/256 and its reciprocal exponent.
The original raw ProPhoto result is `(1712.983,245.75737,-2482075000)`;
the corrected raw result is `(0.6461351,0.57291955,-0.60066473)`.
Black originally produces huge negative float values, despite clipping to black.
The midgray control is identical across all four versions. The TSV also records
zero and the smallest F32 subnormal through pure-gamma encode/decode.

## Measured performance

The measurements below are from [3e098543](https://github.com/lilith/moxcms/commit/3e09854375e9d9fe54e0d4fc88e31013f5528a9e).
The clean fix has byte-identical runtime source before each file's test module;
only test feature guards and comments differ. These are retained measurements,
not a new timing run of the cleaned history.

Neoverse N1, Rust 1.99.1 / LLVM 23.1.1, default target CPU, 1,024 RGB pixels per
row. All 45 declared cells passed the unchanged resource checks, with five fresh
processes and 30 rounds each. Family confidence is 95% across the study, using
1-0.05/45 per cell plus zenbench's candidate-family adjustment; margin is 1%.
Construction/allocation are excluded. CPU frequency was not independently verified.

| Workload | Original us | Sign only us | Range guard us | Optimized us |
|---|---:|---:|---:|---:|
| Positive encode | 30.444 | 24.610 | 24.813 | 24.006 |
| Encode, 10% zero | 30.411 | 24.601 | 28.814 | 25.698 |
| Encode, all zero | 30.157 | 24.314 | 22.358 | 6.594 |
| CICP gamma 2.2 to BT.2020 | 76.480 | — | 80.624 | 69.423 |
| CICP gamma 2.8 to BT.2020 | 75.435 | — | 78.899 | 67.650 |
| ProPhoto to sRGB, 10% zero | 74.598 | 65.019 | 66.348 | 64.695 |

[All 45 optimized comparisons](extended-gamma-timings.csv) include baselines,
process-level percentage estimates, adjusted intervals and verdicts. ICC groups
compare with sign-only; CICP groups compare with original. Results: 22 improvements,
7 non-inferior, 1 equivalent, 4 regressions and 11 inconclusive. Mixed-zero
evaluator rows regress 4–11% against sign-only, which still mishandles zero.
The full mixed-zero ProPhoto comparison is inconclusive; its means do not
establish non-inferiority. Original/sign-only zero cases are cost controls, not
accuracy-equivalent conversions. Earlier rejected prototype populations are
retained separately in the full report and are not pooled with this study.

On x86 Zen 4, Callgrind collects only the row operation for 4,096 RGB pixels.
Positive encode executes 581,673 instructions versus 663,593 for the first range
guard and 557,097 for sign-only. [All 52 counts](extended-gamma-instructions.csv)
cover the four runtime stages; the experimental control is omitted.
The hot evaluator has 133 static instructions, plus 245 in the cold RGB fallback,
versus 345 in the first guard's combined body. Total static code grows.
The U8 table case executes 94,354 instructions in every version.

Full [methodology, source identities and data](https://github.com/imazen/zenpixels/blob/159b3fe16eab5a6c02d91ed138a39f1505684eb8/benchmarks/moxcms_optimized_2026-11-06.md),
[hot assembly](https://github.com/imazen/zenpixels/blob/159b3fe16eab5a6c02d91ed138a39f1505684eb8/benchmarks/moxcms-gamma/final-evaluator.asm)
and [cold assembly](https://github.com/imazen/zenpixels/blob/159b3fe16eab5a6c02d91ed138a39f1505684eb8/benchmarks/moxcms-gamma/final-fallback.asm).
ARM wrapper: 370 s, peak RSS 0.06 GiB, min available 14856 MiB, peak load 0.98.
These are harness resources, not per-image memory. No x86 wall-clock or Zen 5
speed claim, construction timing, or exhaustive F32 sweep is asserted.

## Validation

The clean source passes 89 default library tests, 84 without default features,
and 103 with `extended_range,in_place,options`; strict Clippy passes for the
minimal and extended configurations and the example. All-targets tests pass
with default and extended configurations; the doc-test command and strict
extended all-targets Clippy also pass. Formatting passes.
Regression tests cover RGB, gray and LUT routes, sign extension, zero/extremes,
old approximation bits in range, and RGB/scalar agreement. Existing upstream
test expectations are unchanged. The earlier sampled comparison against the
range guard checked 4,595,836 evaluations, 589,824 transformed channels and
262,144 signed-zero results across all ICC gamma encodings; it is not exhaustive.

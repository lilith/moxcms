// Magetypes migration probe: a tiny stand-in for `TetrahedralSse::interpolate`
// written with `magetypes::simd::v3::f32x4` instead of the hand-wrapped
// `SseVector(__m128)`. Purpose is to see the full pipeline compile end to
// end — token-gated construction, natural operators, `mul_add` FMA, safe
// LUT load — before scaling to the six interpolator files.
//
// Scope note: magetypes exposes `v3` (AVX2+FMA, 128+256-bit), `v4`
// (AVX-512), `neon`, `wasm128`, `scalar`. There's no dedicated `v2`
// (SSE4.1-only) namespace; an archmage-powered moxcms would pick `v3` on
// x86_64 and fall back to `scalar` for older CPUs. moxcms' current SSE4.1-
// only build target is effectively superseded by `v3 + scalar`. This
// probe compiles under x86_64 only.
#![cfg(feature = "lut")]
#![allow(dead_code)]

use crate::conversions::interpolator::{BarycentricWeight, load_bary_weights};
use archmage::{SimdToken, X64V3Token};
use magetypes::simd::v3::f32x4;
use num_traits::AsPrimitive;

/// Aligned storage for a 4-lane f32 cube entry — same shape as
/// `SseAlignedF32`, just expressed in a way `f32x4::load` can consume
/// directly via a `&[f32; 4]` reference.
#[repr(align(16), C)]
pub(crate) struct F32x4Aligned(pub(crate) [f32; 4]);

/// Stand-in for `TetrahedralSse`. All the GRID_SIZE + fetcher plumbing
/// is inlined into this one fn for the probe — the real rewrite will
/// keep the trait-dispatched shape.
pub(crate) struct TetrahedralV3<const GRID_SIZE: usize> {}

impl<const GRID_SIZE: usize> TetrahedralV3<GRID_SIZE> {
    /// Mirror of `TetrahedralSse::interpolate`.  Token-gated — the mere
    /// existence of `token: X64V3Token` guarantees AVX2+FMA is present
    /// at runtime, which is what lets us call the intrinsics safely.
    pub(crate) fn interpolate<U: AsPrimitive<usize>, const BINS: usize>(
        &self,
        token: X64V3Token,
        in_r: U,
        in_g: U,
        in_b: U,
        lut: &[BarycentricWeight<f32>; BINS],
        cube: &[F32x4Aligned],
    ) -> f32x4 {
        let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);

        let fetch = |x: i32, y: i32, z: i32| -> f32x4 {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            // `safe_unaligned_simd` (re-exported by magetypes via
            // archmage) lets `f32x4::load` take `&[f32; 4]` — reference
            // based, not pointer based. The `#[repr(align(16), C)]` on
            // `F32x4Aligned` keeps `movaps`-grade alignment regardless.
            f32x4::load(token, &cube[offset].0)
        };

        let c0 = fetch(x, y, z);

        let (c1, c2, c3) = if rx >= ry {
            if ry >= rz {
                // rx >= ry && ry >= rz
                let c1 = fetch(x_n, y, z) - c0;
                let c2 = fetch(x_n, y_n, z) - fetch(x_n, y, z);
                let c3 = fetch(x_n, y_n, z_n) - fetch(x_n, y_n, z);
                (c1, c2, c3)
            } else if rx >= rz {
                // rx >= rz && rz >= ry
                let c1 = fetch(x_n, y, z) - c0;
                let c2 = fetch(x_n, y_n, z_n) - fetch(x_n, y, z_n);
                let c3 = fetch(x_n, y, z_n) - fetch(x_n, y, z);
                (c1, c2, c3)
            } else {
                // rz > rx && rx >= ry
                let c1 = fetch(x_n, y, z_n) - fetch(x, y, z_n);
                let c2 = fetch(x_n, y_n, z_n) - fetch(x_n, y, z_n);
                let c3 = fetch(x, y, z_n) - c0;
                (c1, c2, c3)
            }
        } else if rx >= rz {
            // ry > rx && rx >= rz
            let c1 = fetch(x_n, y_n, z) - fetch(x, y_n, z);
            let c2 = fetch(x, y_n, z) - c0;
            let c3 = fetch(x_n, y_n, z_n) - fetch(x_n, y_n, z);
            (c1, c2, c3)
        } else if ry >= rz {
            // ry >= rz && rz > rx
            let c1 = fetch(x_n, y_n, z_n) - fetch(x, y_n, z_n);
            let c2 = fetch(x, y_n, z) - c0;
            let c3 = fetch(x, y_n, z_n) - fetch(x, y_n, z);
            (c1, c2, c3)
        } else {
            // rz > ry && ry > rx
            let c1 = fetch(x_n, y_n, z_n) - fetch(x, y_n, z_n);
            let c2 = fetch(x, y_n, z_n) - fetch(x, y, z_n);
            let c3 = fetch(x, y, z_n) - c0;
            (c1, c2, c3)
        };

        // moxcms `a.mla(b, c)` = a + b*c.  magetypes `b.mul_add(c, a)`
        // = b*c + a.  Same math, arg order shuffled.
        let rxv = f32x4::splat(token, rx);
        let ryv = f32x4::splat(token, ry);
        let rzv = f32x4::splat(token, rz);
        let s0 = c1.mul_add(rxv, c0);
        let s1 = c2.mul_add(ryv, s0);
        c3.mul_add(rzv, s1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversions::interpolator::BarycentricWeight;

    #[test]
    fn v3_probe_runs_if_avx2_available() {
        // Skip on CPUs without AVX2+FMA — the probe is a minimal compile
        // + runtime smoke test, not a correctness oracle.
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipped: AVX2+FMA not available at runtime");
            return;
        };

        // 2×2×2 grid, all identity-ish.
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;

        let cube: Vec<F32x4Aligned> = (0..GRID_SIZE.pow(3))
            .map(|i| F32x4Aligned([i as f32, i as f32, i as f32, 0.0]))
            .collect();
        let weights: Box<[BarycentricWeight<f32>; BINS]> = {
            let mut w = Box::new([BarycentricWeight::<f32>::default(); BINS]);
            for (i, entry) in w.iter_mut().enumerate() {
                let x = (i as f32 * (GRID_SIZE as i32 - 1) as f32 / 255.0).floor() as i32;
                entry.x = x;
                entry.x_n = (x + 1).min(GRID_SIZE as i32 - 1);
                entry.w = ((i as f32 * (GRID_SIZE as i32 - 1) as f32 / 255.0) - x as f32) as f32;
            }
            w
        };

        let interp = TetrahedralV3::<GRID_SIZE> {};
        let out = interp.interpolate(token, 0u8, 0u8, 0u8, &weights, &cube);
        let _unused = out; // Just proving it compiles + runs; value-check comes later.
    }
}

// 256-bit Double-variant SIMD interpolators on magetypes `f32x8` —
// same structure as `simd_interp.rs` but over 8-lane cube entries.
// On NEON `GenericF32x8<NeonToken>` auto-polyfills to a pair of
// `float32x4_t` (see magetypes `src/simd/impls/arm_neon.rs:205`), so
// one generic body covers AVX2, NEON, WASM128, and scalar with no
// separate NEON-Double type needed.
//
// The "Double" variant packs two adjacent 4-channel cube entries into
// a single 8-lane load and interpolates two output pixels in a single
// pass. The caller splits the 8-wide result into two 4-wide halves.
//
// Scope: f32 path only. Q0.15 `i16x16` Double is a TODO (uses the same
// shape — `GenericI16x16<Token>` — with a scalar `q15_mulhrs` helper).
#![cfg(feature = "lut")]
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unreachable_pub)]

use crate::conversions::interpolator::{BarycentricWeight, load_bary_weights};
use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use num_traits::AsPrimitive;

/// 8-lane aligned cube entry: two consecutive 4-channel packings.
/// `#[repr(align(32), C)]` matches the `movaps`-grade alignment the
/// hand-written `AvxAlignedF32x2` kept.
#[repr(align(32), C)]
pub(crate) struct Aligned8<T>(pub(crate) [T; 8]);

// --- Tetrahedral ---------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
pub(crate) fn interpolate_tetra_double<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned8<f32>],
    out: &mut [f32; 8],
) {
    type f32x8 = GenericF32x8<Token>;

    let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);

    let fetch = |x: i32, y: i32, z: i32| -> f32x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x8::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);

    let (c1, c2, c3) = if rx >= ry {
        if ry >= rz {
            (
                fetch(x_n, y, z) - c0,
                fetch(x_n, y_n, z) - fetch(x_n, y, z),
                fetch(x_n, y_n, z_n) - fetch(x_n, y_n, z),
            )
        } else if rx >= rz {
            (
                fetch(x_n, y, z) - c0,
                fetch(x_n, y_n, z_n) - fetch(x_n, y, z_n),
                fetch(x_n, y, z_n) - fetch(x_n, y, z),
            )
        } else {
            (
                fetch(x_n, y, z_n) - fetch(x, y, z_n),
                fetch(x_n, y_n, z_n) - fetch(x_n, y, z_n),
                fetch(x, y, z_n) - c0,
            )
        }
    } else if rx >= rz {
        (
            fetch(x_n, y_n, z) - fetch(x, y_n, z),
            fetch(x, y_n, z) - c0,
            fetch(x_n, y_n, z_n) - fetch(x_n, y_n, z),
        )
    } else if ry >= rz {
        (
            fetch(x_n, y_n, z_n) - fetch(x, y_n, z_n),
            fetch(x, y_n, z) - c0,
            fetch(x, y_n, z_n) - fetch(x, y_n, z),
        )
    } else {
        (
            fetch(x_n, y_n, z_n) - fetch(x, y_n, z_n),
            fetch(x, y_n, z_n) - fetch(x, y, z_n),
            fetch(x, y, z_n) - c0,
        )
    };

    let rxv = f32x8::splat(token, rx);
    let ryv = f32x8::splat(token, ry);
    let rzv = f32x8::splat(token, rz);
    let s0 = c1.mul_add(rxv, c0);
    let s1 = c2.mul_add(ryv, s0);
    c3.mul_add(rzv, s1).store(out);
}

// --- Trilinear ----------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
pub(crate) fn interpolate_trilinear_double<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned8<f32>],
    out: &mut [f32; 8],
) {
    type f32x8 = GenericF32x8<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let fetch = |x: i32, y: i32, z: i32| -> f32x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x8::load(token, &cube[offset].0)
    };

    let c000 = fetch(x, y, z);
    let c100 = fetch(x_n, y, z);
    let c010 = fetch(x, y_n, z);
    let c110 = fetch(x_n, y_n, z);
    let c001 = fetch(x, y, z_n);
    let c101 = fetch(x_n, y, z_n);
    let c011 = fetch(x, y_n, z_n);
    let c111 = fetch(x_n, y_n, z_n);

    let dx_v = f32x8::splat(token, 1.0 - dr);
    let dr_v = f32x8::splat(token, dr);
    let dy_v = f32x8::splat(token, 1.0 - dg);
    let dg_v = f32x8::splat(token, dg);
    let dz_v = f32x8::splat(token, 1.0 - db);
    let db_v = f32x8::splat(token, db);

    let c00 = c100.mul_add(dr_v, c000 * dx_v);
    let c10 = c110.mul_add(dr_v, c010 * dx_v);
    let c01 = c101.mul_add(dr_v, c001 * dx_v);
    let c11 = c111.mul_add(dr_v, c011 * dx_v);

    let c0 = c10.mul_add(dg_v, c00 * dy_v);
    let c1 = c11.mul_add(dg_v, c01 * dy_v);

    c1.mul_add(db_v, c0 * dz_v).store(out);
}

// --- Pyramidal (options-gated) ------------------------------------------

#[cfg(feature = "options")]
#[magetypes(v3, neon, wasm128, scalar)]
pub(crate) fn interpolate_pyramid_double<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned8<f32>],
    out: &mut [f32; 8],
) {
    type f32x8 = GenericF32x8<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let fetch = |x: i32, y: i32, z: i32| -> f32x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x8::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);

    let (c1, c2, c3) = if dr > db && dg > db {
        let x0 = fetch(x_n, y_n, z_n);
        let x1 = fetch(x_n, y_n, z);
        let x2 = fetch(x_n, y, z);
        let x3 = fetch(x, y_n, z);
        (x0 - x1, x2 - c0, x3 - c0)
    } else if db > dr && dg > dr {
        let x0 = fetch(x_n, y_n, z_n);
        let x1 = fetch(x, y_n, z_n);
        let x2 = fetch(x, y_n, z);
        let x3 = fetch(x, y, z_n);
        (x0 - x1, x2 - c0, x3 - c0)
    } else {
        let x0 = fetch(x_n, y_n, z_n);
        let x1 = fetch(x_n, y, z_n);
        let x2 = fetch(x_n, y, z);
        let x3 = fetch(x, y, z_n);
        (x0 - x1, x2 - c0, x3 - c0)
    };

    let dr_v = f32x8::splat(token, dr);
    let dg_v = f32x8::splat(token, dg);
    let db_v = f32x8::splat(token, db);
    let s0 = c1.mul_add(dr_v, c0);
    let s1 = c2.mul_add(dg_v, s0);
    c3.mul_add(db_v, s1).store(out);
}

// --- Prismatic (options-gated) ------------------------------------------

#[cfg(feature = "options")]
#[magetypes(v3, neon, wasm128, scalar)]
pub(crate) fn interpolate_prism_double<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned8<f32>],
    out: &mut [f32; 8],
) {
    type f32x8 = GenericF32x8<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let fetch = |x: i32, y: i32, z: i32| -> f32x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x8::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);

    let (c1, c2, c3) = if dr > dg {
        let x0 = fetch(x_n, y_n, z_n);
        let x1 = fetch(x_n, y, z_n);
        let x2 = fetch(x_n, y, z);
        let x3 = fetch(x, y, z_n);
        (x0 - x1, x2 - c0, x3 - c0)
    } else {
        let x0 = fetch(x_n, y_n, z_n);
        let x1 = fetch(x, y_n, z_n);
        let x2 = fetch(x, y_n, z);
        let x3 = fetch(x, y, z_n);
        (x0 - x1, x2 - c0, x3 - c0)
    };

    let dr_v = f32x8::splat(token, dr);
    let dg_v = f32x8::splat(token, dg);
    let db_v = f32x8::splat(token, db);
    let s0 = c1.mul_add(dr_v, c0);
    let s1 = c2.mul_add(dg_v, s0);
    c3.mul_add(db_v, s1).store(out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::incant;

    fn make_identity_ramp<const BINS: usize, const GRID_SIZE: usize>()
    -> Box<[BarycentricWeight<f32>; BINS]> {
        let mut w = Box::new([BarycentricWeight::<f32>::default(); BINS]);
        for (i, entry) in w.iter_mut().enumerate() {
            let x = (i as f32 * (GRID_SIZE as i32 - 1) as f32 / (BINS - 1) as f32).floor() as i32;
            entry.x = x;
            entry.x_n = (x + 1).min(GRID_SIZE as i32 - 1);
            entry.w = (i as f32 * (GRID_SIZE as i32 - 1) as f32 / (BINS - 1) as f32) - x as f32;
        }
        w
    }

    fn make_cube<const GRID_SIZE: usize>() -> Vec<Aligned8<f32>> {
        (0..GRID_SIZE.pow(3))
            .map(|i| {
                Aligned8([
                    i as f32, i as f32, i as f32, 0.0, i as f32, i as f32, i as f32, 0.0,
                ])
            })
            .collect()
    }

    /// Scalar reference for the 8-lane Double variant — identical
    /// algorithm to the probe, expressed on `[f32; 8]`. Two adjacent
    /// 4-channel pixels packed per call; `[0..4]` is pixel 0, `[4..8]`
    /// is pixel 1, and they are independent along the cube axis.
    fn scalar_tetra_double_ref<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<f32>; 256],
        cube: &[Aligned8<f32>],
    ) -> [f32; 8] {
        let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [f32; 8] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let sub = |a: [f32; 8], b: [f32; 8]| -> [f32; 8] {
            let mut r = [0f32; 8];
            for i in 0..8 {
                r[i] = a[i] - b[i];
            }
            r
        };
        let mla = |acc: [f32; 8], a: [f32; 8], k: f32| -> [f32; 8] {
            let mut r = [0f32; 8];
            for i in 0..8 {
                r[i] = a[i].mul_add(k, acc[i]);
            }
            r
        };
        let c0 = fetch(x, y, z);
        let (c1, c2, c3) = if rx >= ry {
            if ry >= rz {
                (
                    sub(fetch(x_n, y, z), c0),
                    sub(fetch(x_n, y_n, z), fetch(x_n, y, z)),
                    sub(fetch(x_n, y_n, z_n), fetch(x_n, y_n, z)),
                )
            } else if rx >= rz {
                (
                    sub(fetch(x_n, y, z), c0),
                    sub(fetch(x_n, y_n, z_n), fetch(x_n, y, z_n)),
                    sub(fetch(x_n, y, z_n), fetch(x_n, y, z)),
                )
            } else {
                (
                    sub(fetch(x_n, y, z_n), fetch(x, y, z_n)),
                    sub(fetch(x_n, y_n, z_n), fetch(x_n, y, z_n)),
                    sub(fetch(x, y, z_n), c0),
                )
            }
        } else if rx >= rz {
            (
                sub(fetch(x_n, y_n, z), fetch(x, y_n, z)),
                sub(fetch(x, y_n, z), c0),
                sub(fetch(x_n, y_n, z_n), fetch(x_n, y_n, z)),
            )
        } else if ry >= rz {
            (
                sub(fetch(x_n, y_n, z_n), fetch(x, y_n, z_n)),
                sub(fetch(x, y_n, z), c0),
                sub(fetch(x, y_n, z_n), fetch(x, y_n, z)),
            )
        } else {
            (
                sub(fetch(x_n, y_n, z_n), fetch(x, y_n, z_n)),
                sub(fetch(x, y_n, z_n), fetch(x, y, z_n)),
                sub(fetch(x, y, z_n), c0),
            )
        };
        let s0 = mla(c0, c1, rx);
        let s1 = mla(s0, c2, ry);
        mla(s1, c3, rz)
    }

    fn random_cube_double<const GRID_SIZE: usize>(seed: u32) -> Vec<Aligned8<f32>> {
        (0..GRID_SIZE.pow(3))
            .map(|i| {
                let k = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed);
                let kf = |shift: u32| -> f32 {
                    let bits = k.rotate_left(shift) & 0x7FFF_FFFF;
                    bits as f32 / 0x4000_0000u32 as f32
                };
                Aligned8([kf(0), kf(5), kf(11), kf(17), kf(23), kf(2), kf(29), kf(3)])
            })
            .collect()
    }

    fn random_weights_double<const GRID_SIZE: usize>(
        seed: u32,
    ) -> Box<[BarycentricWeight<f32>; 256]> {
        let mut w = Box::new([BarycentricWeight::<f32>::default(); 256]);
        for (i, entry) in w.iter_mut().enumerate() {
            let k = (i as u32).wrapping_mul(0xB503_2B33).wrapping_add(seed);
            let x = ((k & 0xFFFF) as usize % GRID_SIZE.max(2)) as i32;
            entry.x = x.min(GRID_SIZE as i32 - 1);
            entry.x_n = (entry.x + 1).min(GRID_SIZE as i32 - 1);
            let bits = (k >> 16) & 0x7FFF_FFFF;
            entry.w = bits as f32 / 0x8000_0000u32 as f32;
        }
        w
    }

    const EQUIV_INPUTS_DOUBLE: &[(u8, u8, u8)] = &[
        (0, 0, 0),
        (255, 255, 255),
        (128, 64, 200),
        (1, 254, 128),
        (80, 80, 80),
        (200, 100, 50),
        (10, 200, 100),
        (33, 77, 222),
    ];

    fn assert_f32x8_close(got: [f32; 8], expect: [f32; 8], ctx: &str) {
        for lane in 0..8 {
            let diff = (got[lane] - expect[lane]).abs();
            assert!(
                diff < 1e-5,
                "{ctx} lane {lane}: got={} expect={} diff={}",
                got[lane],
                expect[lane],
                diff
            );
        }
    }

    fn scalar_trilinear_double_ref<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<f32>; 256],
        cube: &[Aligned8<f32>],
    ) -> [f32; 8] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [f32; 8] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let lerp = |a: [f32; 8], b: [f32; 8], t: f32| -> [f32; 8] {
            let it = 1.0 - t;
            let mut r = [0f32; 8];
            for i in 0..8 {
                r[i] = b[i].mul_add(t, a[i] * it);
            }
            r
        };
        let c000 = fetch(x, y, z);
        let c100 = fetch(x_n, y, z);
        let c010 = fetch(x, y_n, z);
        let c110 = fetch(x_n, y_n, z);
        let c001 = fetch(x, y, z_n);
        let c101 = fetch(x_n, y, z_n);
        let c011 = fetch(x, y_n, z_n);
        let c111 = fetch(x_n, y_n, z_n);
        let c00 = lerp(c000, c100, dr);
        let c10 = lerp(c010, c110, dr);
        let c01 = lerp(c001, c101, dr);
        let c11 = lerp(c011, c111, dr);
        let c0 = lerp(c00, c10, dg);
        let c1 = lerp(c01, c11, dg);
        lerp(c0, c1, db)
    }

    fn scalar_pyramid_double_ref<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<f32>; 256],
        cube: &[Aligned8<f32>],
    ) -> [f32; 8] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [f32; 8] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let sub = |a: [f32; 8], b: [f32; 8]| -> [f32; 8] {
            let mut r = [0f32; 8];
            for i in 0..8 {
                r[i] = a[i] - b[i];
            }
            r
        };
        let mla = |acc: [f32; 8], a: [f32; 8], k: f32| -> [f32; 8] {
            let mut r = [0f32; 8];
            for i in 0..8 {
                r[i] = a[i].mul_add(k, acc[i]);
            }
            r
        };
        let c0 = fetch(x, y, z);
        let (c1, c2, c3) = if dr > db && dg > db {
            let x0 = fetch(x_n, y_n, z_n);
            let x1 = fetch(x_n, y_n, z);
            let x2 = fetch(x_n, y, z);
            let x3 = fetch(x, y_n, z);
            (sub(x0, x1), sub(x2, c0), sub(x3, c0))
        } else if db > dr && dg > dr {
            let x0 = fetch(x_n, y_n, z_n);
            let x1 = fetch(x, y_n, z_n);
            let x2 = fetch(x, y_n, z);
            let x3 = fetch(x, y, z_n);
            (sub(x0, x1), sub(x2, c0), sub(x3, c0))
        } else {
            let x0 = fetch(x_n, y_n, z_n);
            let x1 = fetch(x_n, y, z_n);
            let x2 = fetch(x_n, y, z);
            let x3 = fetch(x, y, z_n);
            (sub(x0, x1), sub(x2, c0), sub(x3, c0))
        };
        let s0 = mla(c0, c1, dr);
        let s1 = mla(s0, c2, dg);
        mla(s1, c3, db)
    }

    fn scalar_prism_double_ref<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<f32>; 256],
        cube: &[Aligned8<f32>],
    ) -> [f32; 8] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [f32; 8] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let sub = |a: [f32; 8], b: [f32; 8]| -> [f32; 8] {
            let mut r = [0f32; 8];
            for i in 0..8 {
                r[i] = a[i] - b[i];
            }
            r
        };
        let mla = |acc: [f32; 8], a: [f32; 8], k: f32| -> [f32; 8] {
            let mut r = [0f32; 8];
            for i in 0..8 {
                r[i] = a[i].mul_add(k, acc[i]);
            }
            r
        };
        let c0 = fetch(x, y, z);
        let (c1, c2, c3) = if dr > dg {
            let x0 = fetch(x_n, y_n, z_n);
            let x1 = fetch(x_n, y, z_n);
            let x2 = fetch(x_n, y, z);
            let x3 = fetch(x, y, z_n);
            (sub(x0, x1), sub(x2, c0), sub(x3, c0))
        } else {
            let x0 = fetch(x_n, y_n, z_n);
            let x1 = fetch(x, y_n, z_n);
            let x2 = fetch(x, y_n, z);
            let x3 = fetch(x, y, z_n);
            (sub(x0, x1), sub(x2, c0), sub(x3, c0))
        };
        let s0 = mla(c0, c1, dr);
        let s1 = mla(s0, c2, dg);
        mla(s1, c3, db)
    }

    #[test]
    fn tetra_double_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights_double::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube_double::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS_DOUBLE {
            let expect = scalar_tetra_double_ref::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0f32; 8];
            incant!(
                interpolate_tetra_double::<u8, GRID_SIZE, 256>(
                    ir, ig, ib, &*weights, &cube, &mut got,
                ),
                [v3, neon, wasm128, scalar]
            );
            assert_f32x8_close(got, expect, &format!("tetra_double (r={ir},g={ig},b={ib})"));
        }
    }

    #[test]
    fn trilinear_double_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights_double::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube_double::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS_DOUBLE {
            let expect = scalar_trilinear_double_ref::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0f32; 8];
            incant!(
                interpolate_trilinear_double::<u8, GRID_SIZE, 256>(
                    ir, ig, ib, &*weights, &cube, &mut got,
                ),
                [v3, neon, wasm128, scalar]
            );
            assert_f32x8_close(
                got,
                expect,
                &format!("trilinear_double (r={ir},g={ig},b={ib})"),
            );
        }
    }

    #[cfg(feature = "options")]
    #[test]
    fn pyramid_double_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights_double::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube_double::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS_DOUBLE {
            let expect = scalar_pyramid_double_ref::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0f32; 8];
            incant!(
                interpolate_pyramid_double::<u8, GRID_SIZE, 256>(
                    ir, ig, ib, &*weights, &cube, &mut got,
                ),
                [v3, neon, wasm128, scalar]
            );
            assert_f32x8_close(
                got,
                expect,
                &format!("pyramid_double (r={ir},g={ig},b={ib})"),
            );
        }
    }

    #[cfg(feature = "options")]
    #[test]
    fn prism_double_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights_double::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube_double::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS_DOUBLE {
            let expect = scalar_prism_double_ref::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0f32; 8];
            incant!(
                interpolate_prism_double::<u8, GRID_SIZE, 256>(
                    ir, ig, ib, &*weights, &cube, &mut got,
                ),
                [v3, neon, wasm128, scalar]
            );
            assert_f32x8_close(got, expect, &format!("prism_double (r={ir},g={ig},b={ib})"));
        }
    }

    #[test]
    fn tetra_double_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 8];
        incant!(
            interpolate_tetra_double::<u8, GRID_SIZE, BINS>(
                128u8, 64u8, 200u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[test]
    fn trilinear_double_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 8];
        incant!(
            interpolate_trilinear_double::<u8, GRID_SIZE, BINS>(
                30u8, 90u8, 150u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[test]
    fn pyramid_double_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 8];
        incant!(
            interpolate_pyramid_double::<u8, GRID_SIZE, BINS>(
                50u8, 100u8, 150u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[test]
    fn prism_double_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 8];
        incant!(
            interpolate_prism_double::<u8, GRID_SIZE, BINS>(
                25u8, 75u8, 175u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }
}

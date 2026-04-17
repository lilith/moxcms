// 256-bit Q0.15 fixed-point Double-variant interpolators on
// `GenericI16x16<Token>`. Polyfills on NEON to `[int16x8_t; 2]`
// (see `arm_neon.rs:2983` — same pattern as f32x8 Double).
//
// Processes two adjacent 4-channel pixels in one pass across
// 16 i16 lanes. The cube entry is `[i16; 16]` = two packed
// 4-channel Q0.15 corners.
#![cfg(feature = "lut")]
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unreachable_pub)]

use crate::conversions::interpolator::{BarycentricWeight, load_bary_weights};
use archmage::magetypes;
use magetypes::simd::backends::I16x16Backend;
use magetypes::simd::generic::i16x16 as GenericI16x16;
use num_traits::AsPrimitive;

/// 16-lane aligned Q0.15 cube entry. Two adjacent 4-channel corners
/// `[low=r,g,b,pad | high=r,g,b,pad]`; high and low are zero-padded
/// in the last lane of each 4-tuple.
#[repr(align(32), C)]
pub(crate) struct Aligned16I16(pub(crate) [i16; 16]);

/// Q0.15 fixed-point mulhrs scaled to 16 lanes. Same algorithm as
/// `simd_interp_q0_15::q15_mulhrs`, doubled — LLVM is expected to
/// auto-vectorize inside the per-tier `#[target_feature]` region.
#[inline(always)]
fn q15_mulhrs_16<T: I16x16Backend>(
    token: T,
    a: GenericI16x16<T>,
    b: GenericI16x16<T>,
) -> GenericI16x16<T> {
    let ar = a.to_array();
    let br = b.to_array();
    let mut out = [0i16; 16];
    for i in 0..16 {
        let prod = (ar[i] as i32) * (br[i] as i32);
        out[i] = ((prod + 0x4000) >> 15) as i16;
    }
    GenericI16x16::<T>::from_array(token, out)
}

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
    lut: &[BarycentricWeight<i16>; BINS],
    cube: &[Aligned16I16],
    out: &mut [i16; 16],
) {
    type i16x16 = GenericI16x16<Token>;

    let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);

    // Bounds-hoist — see `simd_interp.rs::interpolate_tetra`.
    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> i16x16 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        i16x16::load(token, &cube[offset].0)
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

    let rxv = i16x16::splat(token, rx);
    let ryv = i16x16::splat(token, ry);
    let rzv = i16x16::splat(token, rz);
    let s0 = c0 + q15_mulhrs_16(token, c1, rxv);
    let s1 = s0 + q15_mulhrs_16(token, c2, ryv);
    let s2 = s1 + q15_mulhrs_16(token, c3, rzv);
    s2.store(out);
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
    lut: &[BarycentricWeight<i16>; BINS],
    cube: &[Aligned16I16],
    out: &mut [i16; 16],
) {
    type i16x16 = GenericI16x16<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> i16x16 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        i16x16::load(token, &cube[offset].0)
    };

    let c000 = fetch(x, y, z);
    let c100 = fetch(x_n, y, z);
    let c010 = fetch(x, y_n, z);
    let c110 = fetch(x_n, y_n, z);
    let c001 = fetch(x, y, z_n);
    let c101 = fetch(x_n, y, z_n);
    let c011 = fetch(x, y_n, z_n);
    let c111 = fetch(x_n, y_n, z_n);

    let dr_v = i16x16::splat(token, dr);
    let dg_v = i16x16::splat(token, dg);
    let db_v = i16x16::splat(token, db);

    let c00 = c000 + q15_mulhrs_16(token, c100 - c000, dr_v);
    let c10 = c010 + q15_mulhrs_16(token, c110 - c010, dr_v);
    let c01 = c001 + q15_mulhrs_16(token, c101 - c001, dr_v);
    let c11 = c011 + q15_mulhrs_16(token, c111 - c011, dr_v);

    let c0 = c00 + q15_mulhrs_16(token, c10 - c00, dg_v);
    let c1 = c01 + q15_mulhrs_16(token, c11 - c01, dg_v);

    let res = c0 + q15_mulhrs_16(token, c1 - c0, db_v);
    res.store(out);
}

// --- Pyramidal (options-gated) ------------------------------------------

/// 256-bit Q0.15 four-term pyramidal Double — same shape as
/// `simd_interp_q0_15::interpolate_pyramid` over 16 lanes.
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
    lut: &[BarycentricWeight<i16>; BINS],
    cube: &[Aligned16I16],
    out: &mut [i16; 16],
) {
    type i16x16 = GenericI16x16<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> i16x16 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        i16x16::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);
    let dr_v = i16x16::splat(token, dr);
    let dg_v = i16x16::splat(token, dg);
    let db_v = i16x16::splat(token, db);

    if dr > db && dg > db {
        let x0 = fetch(x_n, y_n, z_n);
        let x1 = fetch(x_n, y_n, z);
        let x2 = fetch(x_n, y, z);
        let x3 = fetch(x, y_n, z);
        let c1 = x0 - x1;
        let c2 = x2 - c0;
        let c3 = x3 - c0;
        let c4 = c0 - x3 - x2 + x1;
        let k = q15_mulhrs_16(token, dr_v, dg_v);
        let s0 = c0 + q15_mulhrs_16(token, c1, db_v);
        let s1 = s0 + q15_mulhrs_16(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs_16(token, c3, dg_v);
        (s2 + q15_mulhrs_16(token, c4, k)).store(out);
    } else if db > dr && dg > dr {
        let x0 = fetch(x, y, z_n);
        let x1 = fetch(x_n, y_n, z_n);
        let x2 = fetch(x, y_n, z_n);
        let x3 = fetch(x, y_n, z);
        let c1 = x0 - c0;
        let c2 = x1 - x2;
        let c3 = x3 - c0;
        let c4 = c0 - x3 - x0 + x2;
        let k = q15_mulhrs_16(token, dg_v, db_v);
        let s0 = c0 + q15_mulhrs_16(token, c1, db_v);
        let s1 = s0 + q15_mulhrs_16(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs_16(token, c3, dg_v);
        (s2 + q15_mulhrs_16(token, c4, k)).store(out);
    } else {
        let x0 = fetch(x, y, z_n);
        let x1 = fetch(x_n, y, z);
        let x2 = fetch(x_n, y, z_n);
        let x3 = fetch(x_n, y_n, z_n);
        let c1 = x0 - c0;
        let c2 = x1 - c0;
        let c3 = x3 - x2;
        let c4 = c0 - x1 - x0 + x2;
        let k = q15_mulhrs_16(token, db_v, dr_v);
        let s0 = c0 + q15_mulhrs_16(token, c1, db_v);
        let s1 = s0 + q15_mulhrs_16(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs_16(token, c3, dg_v);
        (s2 + q15_mulhrs_16(token, c4, k)).store(out);
    }
}

// --- Prismatic (options-gated) ------------------------------------------

/// 256-bit Q0.15 five-term prismatic Double — matches the 8-lane
/// `simd_interp_q0_15::interpolate_prism` (`db > dr` branch, two
/// `q15_mulhrs` bilinear corrections).
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
    lut: &[BarycentricWeight<i16>; BINS],
    cube: &[Aligned16I16],
    out: &mut [i16; 16],
) {
    type i16x16 = GenericI16x16<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> i16x16 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        i16x16::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);
    let dr_v = i16x16::splat(token, dr);
    let dg_v = i16x16::splat(token, dg);
    let db_v = i16x16::splat(token, db);
    let k_dgdb = q15_mulhrs_16(token, dg_v, db_v);
    let k_drdg = q15_mulhrs_16(token, dr_v, dg_v);

    if db > dr {
        let x0 = fetch(x, y, z_n);
        let x1 = fetch(x_n, y, z_n);
        let x2 = fetch(x, y_n, z);
        let x3 = fetch(x, y_n, z_n);
        let x4 = fetch(x_n, y_n, z_n);
        let c1 = x0 - c0;
        let c2 = x1 - x0;
        let c3 = x2 - c0;
        let c4 = c0 - x2 - x0 + x3;
        let c5 = x0 - x3 - x1 + x4;
        let s0 = c0 + q15_mulhrs_16(token, c1, db_v);
        let s1 = s0 + q15_mulhrs_16(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs_16(token, c3, dg_v);
        let s3 = s2 + q15_mulhrs_16(token, c4, k_dgdb);
        (s3 + q15_mulhrs_16(token, c5, k_drdg)).store(out);
    } else {
        let x0 = fetch(x_n, y, z);
        let x1 = fetch(x_n, y, z_n);
        let x2 = fetch(x, y_n, z);
        let x3 = fetch(x_n, y_n, z);
        let x4 = fetch(x_n, y_n, z_n);
        let c1 = x1 - x0;
        let c2 = x0 - c0;
        let c3 = x2 - c0;
        let c4 = x0 - x3 - x1 + x4;
        let c5 = c0 - x2 - x0 + x3;
        let s0 = c0 + q15_mulhrs_16(token, c1, db_v);
        let s1 = s0 + q15_mulhrs_16(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs_16(token, c3, dg_v);
        let s3 = s2 + q15_mulhrs_16(token, c4, k_dgdb);
        (s3 + q15_mulhrs_16(token, c5, k_drdg)).store(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::incant;

    #[inline(always)]
    fn mulhrs_ref(a: i16, b: i16) -> i16 {
        let prod = (a as i32) * (b as i32);
        ((prod + 0x4000) >> 15) as i16
    }

    fn scalar_tetra_double_q15<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned16I16],
    ) -> [i16; 16] {
        let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [i16; 16] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let sub = |a: [i16; 16], b: [i16; 16]| -> [i16; 16] {
            let mut r = [0i16; 16];
            for i in 0..16 {
                r[i] = a[i].wrapping_sub(b[i]);
            }
            r
        };
        let mla = |acc: [i16; 16], a: [i16; 16], k: i16| -> [i16; 16] {
            let mut r = [0i16; 16];
            for i in 0..16 {
                r[i] = acc[i].wrapping_add(mulhrs_ref(a[i], k));
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

    fn random_cube<const GRID_SIZE: usize>(seed: u32) -> Vec<Aligned16I16> {
        (0..GRID_SIZE.pow(3))
            .map(|i| {
                let k = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed);
                let b16 = |shift: u32| -> i16 { (k.rotate_left(shift) & 0x3FFF) as i16 };
                Aligned16I16([
                    b16(0),
                    b16(2),
                    b16(4),
                    0,
                    b16(6),
                    b16(8),
                    b16(10),
                    0,
                    b16(12),
                    b16(14),
                    b16(16),
                    0,
                    b16(18),
                    b16(20),
                    b16(22),
                    0,
                ])
            })
            .collect()
    }

    fn random_weights<const GRID_SIZE: usize>(seed: u32) -> Box<[BarycentricWeight<i16>; 256]> {
        let mut w = Box::new([BarycentricWeight::<i16>::default(); 256]);
        for (i, entry) in w.iter_mut().enumerate() {
            let k = (i as u32).wrapping_mul(0xB503_2B33).wrapping_add(seed);
            let x = ((k & 0xFFFF) as usize % GRID_SIZE.max(2)) as i32;
            entry.x = x.min(GRID_SIZE as i32 - 1);
            entry.x_n = (entry.x + 1).min(GRID_SIZE as i32 - 1);
            entry.w = ((k >> 16) & 0x7FFF) as i16;
        }
        w
    }

    const INPUTS: &[(u8, u8, u8)] = &[
        (0, 0, 0),
        (255, 255, 255),
        (128, 64, 200),
        (1, 254, 128),
        (80, 80, 80),
        (200, 100, 50),
        (10, 200, 100),
        (33, 77, 222),
    ];

    #[test]
    fn tetra_double_q15_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in INPUTS {
            let expect = scalar_tetra_double_q15::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0i16; 16];
            incant!(
                interpolate_tetra_double::<u8, GRID_SIZE, 256>(
                    ir, ig, ib, &*weights, &cube, &mut got,
                ),
                [v3, neon, wasm128, scalar]
            );
            assert_eq!(got, expect, "tetra_double_q15 (r={ir},g={ig},b={ib})");
        }
    }

    #[test]
    fn trilinear_double_q15_smoke() {
        const GRID_SIZE: usize = 2;
        let weights = random_weights::<GRID_SIZE>(1);
        let cube = random_cube::<GRID_SIZE>(2);
        let mut out = [0i16; 16];
        incant!(
            interpolate_trilinear_double::<u8, GRID_SIZE, 256>(
                30u8, 90u8, 150u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[test]
    fn pyramid_double_q15_smoke() {
        const GRID_SIZE: usize = 2;
        let weights = random_weights::<GRID_SIZE>(1);
        let cube = random_cube::<GRID_SIZE>(2);
        let mut out = [0i16; 16];
        incant!(
            interpolate_pyramid_double::<u8, GRID_SIZE, 256>(
                50u8, 100u8, 150u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[test]
    fn prism_double_q15_smoke() {
        const GRID_SIZE: usize = 2;
        let weights = random_weights::<GRID_SIZE>(1);
        let cube = random_cube::<GRID_SIZE>(2);
        let mut out = [0i16; 16];
        incant!(
            interpolate_prism_double::<u8, GRID_SIZE, 256>(
                25u8, 75u8, 175u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }
}

// Q0.15 fixed-point SIMD interpolators on magetypes `i16x8` — fan-out
// via `#[magetypes(v3, neon, wasm128, scalar)]`, fan-in via `incant!`.
//
// Mirrors `simd_interp.rs` (the f32x4 variant) one-to-one — same four
// interpolators (Tetrahedral, Trilinear, Pyramidal, Prismatic), same
// structure — just against `GenericI16x8<Token>` + fixed-point
// "mulhrs" semantics for the weight multiplications.
//
// The upstream `SseVector` wrapper used `_mm_mulhrs_epi16` (Q0.15
// multiply-high-round-scale) directly. magetypes does not expose this
// intrinsic. We implement it as a scalar round-trip — inside a per-tier
// `#[target_feature]` region LLVM is expected to auto-vectorize back to
// `pmulhrsw` / `vqrdmulhq_s16` / `v128.i16x8.q15mulr_sat`. If the asm
// audit shows it didn't, the next step is per-tier `#[arcane]`
// overrides via the fan-out/fan-in pattern.
//
// Note: only the low 4 lanes carry real data — the cube entries are
// `[i16; 4]` (aligned 8 bytes). We widen to `[i16; 8]` for the load so
// we can use magetypes' `i16x8` generic (there is no `i16x4`). The
// high 4 lanes are zero-padded and never read.
#![cfg(feature = "lut")]
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unreachable_pub)]

use crate::conversions::interpolator::{BarycentricWeight, load_bary_weights};
use archmage::magetypes;
use magetypes::simd::backends::I16x8Backend;
use magetypes::simd::generic::i16x8 as GenericI16x8;
use num_traits::AsPrimitive;

/// 4-lane aligned i16 LUT storage — half an i16x8 worth. High 4 lanes
/// padded with zero, consumed as an `&[i16; 8]` load.
#[repr(align(8), C)]
pub(crate) struct Aligned4I16(pub(crate) [i16; 4]);

/// Fixed-point Q0.15 mulhrs: `((a * b + 0x4000) >> 15)` lane-wise.
///
/// Pure safe code via array round-trip. Inside a tier-specific
/// `#[target_feature]` context (which `#[magetypes]` emits at the
/// variant level) LLVM should recognize the pattern and emit the
/// native intrinsic.
#[inline(always)]
fn q15_mulhrs<T: I16x8Backend>(
    token: T,
    a: GenericI16x8<T>,
    b: GenericI16x8<T>,
) -> GenericI16x8<T> {
    let ar = a.to_array();
    let br = b.to_array();
    let mut out = [0i16; 8];
    for i in 0..8 {
        let prod = (ar[i] as i32) * (br[i] as i32);
        out[i] = ((prod + 0x4000) >> 15) as i16;
    }
    GenericI16x8::<T>::from_array(token, out)
}

/// Safe widened load: copy `[i16; 4]` → `[i16; 8]` (high half zeroed)
/// and feed it to `i16x8::load`.
#[inline(always)]
fn load_q15<T: I16x8Backend>(token: T, x4: &Aligned4I16) -> GenericI16x8<T> {
    let mut eight = [0i16; 8];
    eight[..4].copy_from_slice(&x4.0);
    GenericI16x8::<T>::from_array(token, eight)
}

// --- Tetrahedral ---------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
pub(crate) fn interpolate_tetra<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<i16>; BINS],
    cube: &[Aligned4I16],
    out: &mut [i16; 8],
) {
    type i16x8 = GenericI16x8<Token>;

    let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);

    // See `simd_interp.rs::interpolate_tetra` for rationale — hoists
    // the bounds facts so LLVM can elide per-fetch panic_bounds_check.
    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> i16x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        load_q15(token, &cube[offset])
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

    // s0 = c0 + q15_mulhrs(c1, splat(rx))
    let rxv = i16x8::splat(token, rx);
    let ryv = i16x8::splat(token, ry);
    let rzv = i16x8::splat(token, rz);
    let s0 = c0 + q15_mulhrs(token, c1, rxv);
    let s1 = s0 + q15_mulhrs(token, c2, ryv);
    let s2 = s1 + q15_mulhrs(token, c3, rzv);
    s2.store(out);
}

// --- Trilinear ----------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
pub(crate) fn interpolate_trilinear<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<i16>; BINS],
    cube: &[Aligned4I16],
    out: &mut [i16; 8],
) {
    type i16x8 = GenericI16x8<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> i16x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        load_q15(token, &cube[offset])
    };

    let c000 = fetch(x, y, z);
    let c100 = fetch(x_n, y, z);
    let c010 = fetch(x, y_n, z);
    let c110 = fetch(x_n, y_n, z);
    let c001 = fetch(x, y, z_n);
    let c101 = fetch(x_n, y, z_n);
    let c011 = fetch(x, y_n, z_n);
    let c111 = fetch(x_n, y_n, z_n);

    let dr_v = i16x8::splat(token, dr);
    let dg_v = i16x8::splat(token, dg);
    let db_v = i16x8::splat(token, db);

    // c00 = c000 + mulhrs(c100 - c000, dr)
    let c00 = c000 + q15_mulhrs(token, c100 - c000, dr_v);
    let c10 = c010 + q15_mulhrs(token, c110 - c010, dr_v);
    let c01 = c001 + q15_mulhrs(token, c101 - c001, dr_v);
    let c11 = c011 + q15_mulhrs(token, c111 - c011, dr_v);

    let c0 = c00 + q15_mulhrs(token, c10 - c00, dg_v);
    let c1 = c01 + q15_mulhrs(token, c11 - c01, dg_v);

    let res = c0 + q15_mulhrs(token, c1 - c0, db_v);
    res.store(out);
}

// --- Pyramidal (options-gated) ------------------------------------------

/// Four-term Q0.15 pyramidal decomposition with a bilinear correction.
/// Mirrors `PyramidalSseQ0_15::interpolate` (`sse/interpolator_q0_15.rs`).
/// The bilinear weight is computed as the Q0.15 product
/// `q15_mulhrs(splat(a), splat(b))` — i.e. mulhrs on the splatted
/// scalar weights, matching the hand-written
/// `SseVector::from(a) * SseVector::from(b)` path.
#[cfg(feature = "options")]
#[magetypes(v3, neon, wasm128, scalar)]
pub(crate) fn interpolate_pyramid<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<i16>; BINS],
    cube: &[Aligned4I16],
    out: &mut [i16; 8],
) {
    type i16x8 = GenericI16x8<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> i16x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        load_q15(token, &cube[offset])
    };

    let c0 = fetch(x, y, z);
    let dr_v = i16x8::splat(token, dr);
    let dg_v = i16x8::splat(token, dg);
    let db_v = i16x8::splat(token, db);

    if dr > db && dg > db {
        let x0 = fetch(x_n, y_n, z_n);
        let x1 = fetch(x_n, y_n, z);
        let x2 = fetch(x_n, y, z);
        let x3 = fetch(x, y_n, z);
        let c1 = x0 - x1;
        let c2 = x2 - c0;
        let c3 = x3 - c0;
        let c4 = c0 - x3 - x2 + x1;
        let k = q15_mulhrs(token, dr_v, dg_v);
        let s0 = c0 + q15_mulhrs(token, c1, db_v);
        let s1 = s0 + q15_mulhrs(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs(token, c3, dg_v);
        (s2 + q15_mulhrs(token, c4, k)).store(out);
    } else if db > dr && dg > dr {
        let x0 = fetch(x, y, z_n);
        let x1 = fetch(x_n, y_n, z_n);
        let x2 = fetch(x, y_n, z_n);
        let x3 = fetch(x, y_n, z);
        let c1 = x0 - c0;
        let c2 = x1 - x2;
        let c3 = x3 - c0;
        let c4 = c0 - x3 - x0 + x2;
        let k = q15_mulhrs(token, dg_v, db_v);
        let s0 = c0 + q15_mulhrs(token, c1, db_v);
        let s1 = s0 + q15_mulhrs(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs(token, c3, dg_v);
        (s2 + q15_mulhrs(token, c4, k)).store(out);
    } else {
        let x0 = fetch(x, y, z_n);
        let x1 = fetch(x_n, y, z);
        let x2 = fetch(x_n, y, z_n);
        let x3 = fetch(x_n, y_n, z_n);
        let c1 = x0 - c0;
        let c2 = x1 - c0;
        let c3 = x3 - x2;
        let c4 = c0 - x1 - x0 + x2;
        let k = q15_mulhrs(token, db_v, dr_v);
        let s0 = c0 + q15_mulhrs(token, c1, db_v);
        let s1 = s0 + q15_mulhrs(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs(token, c3, dg_v);
        (s2 + q15_mulhrs(token, c4, k)).store(out);
    }
}

// --- Prismatic (options-gated) ------------------------------------------

/// Five-term Q0.15 prismatic decomposition with two bilinear
/// corrections. Mirrors `PrismaticSseQ0_15::interpolate`
/// (`sse/interpolator_q0_15.rs`). Note the branch uses `db > dr`
/// (not `>=`) matching the hand-written path — f32 scalar uses `>=`
/// but the hand-written Q0.15 SSE/NEON paths use `>`.
#[cfg(feature = "options")]
#[magetypes(v3, neon, wasm128, scalar)]
pub(crate) fn interpolate_prism<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<i16>; BINS],
    cube: &[Aligned4I16],
    out: &mut [i16; 8],
) {
    type i16x8 = GenericI16x8<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> i16x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        load_q15(token, &cube[offset])
    };

    let c0 = fetch(x, y, z);
    let dr_v = i16x8::splat(token, dr);
    let dg_v = i16x8::splat(token, dg);
    let db_v = i16x8::splat(token, db);
    let k_dgdb = q15_mulhrs(token, dg_v, db_v);
    let k_drdg = q15_mulhrs(token, dr_v, dg_v);

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
        let s0 = c0 + q15_mulhrs(token, c1, db_v);
        let s1 = s0 + q15_mulhrs(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs(token, c3, dg_v);
        let s3 = s2 + q15_mulhrs(token, c4, k_dgdb);
        (s3 + q15_mulhrs(token, c5, k_drdg)).store(out);
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
        let s0 = c0 + q15_mulhrs(token, c1, db_v);
        let s1 = s0 + q15_mulhrs(token, c2, dr_v);
        let s2 = s1 + q15_mulhrs(token, c3, dg_v);
        let s3 = s2 + q15_mulhrs(token, c4, k_dgdb);
        (s3 + q15_mulhrs(token, c5, k_drdg)).store(out);
    }
}

// Monomorphic wrappers so the `cargo asm` audit can see concrete
// emitted variants. These are a single fixed (GRID_SIZE=33, BINS=256,
// U=u8) specialization — representative of the production LUT shape.
// `pub` + `#[doc(hidden)]` makes the symbols survive DCE so the asm
// audit can read them, without widening the crate's public surface.
#[doc(hidden)]
pub mod __asm_audit {
    use super::*;

    #[inline(never)]
    pub fn tetra_u8_g33_b256(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned4I16],
        out: &mut [i16; 8],
    ) {
        archmage::incant!(
            interpolate_tetra::<u8, 33, 256>(in_r, in_g, in_b, lut, cube, out,),
            [v3, neon, wasm128, scalar]
        );
    }

    #[inline(never)]
    pub fn trilinear_u8_g33_b256(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned4I16],
        out: &mut [i16; 8],
    ) {
        archmage::incant!(
            interpolate_trilinear::<u8, 33, 256>(in_r, in_g, in_b, lut, cube, out,),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[inline(never)]
    pub fn pyramid_u8_g33_b256(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned4I16],
        out: &mut [i16; 8],
    ) {
        archmage::incant!(
            interpolate_pyramid::<u8, 33, 256>(in_r, in_g, in_b, lut, cube, out,),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[inline(never)]
    pub fn prism_u8_g33_b256(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned4I16],
        out: &mut [i16; 8],
    ) {
        archmage::incant!(
            interpolate_prism::<u8, 33, 256>(in_r, in_g, in_b, lut, cube, out,),
            [v3, neon, wasm128, scalar]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversions::interpolator::BarycentricWeight;
    use archmage::incant;

    /// Scalar reference for `_mm_mulhrs_epi16` — bit-exact by spec:
    /// result = ((a * b + 0x4000) >> 15) as i16, saturating only at the
    /// single edge case (-32768 * -32768).
    #[inline(always)]
    fn mulhrs_ref(a: i16, b: i16) -> i16 {
        let prod = (a as i32) * (b as i32);
        ((prod + 0x4000) >> 15) as i16
    }

    /// Scalar reference tetrahedral interpolator against which the
    /// magetypes probe is compared. Ground truth for the Q0.15 mulhrs
    /// semantics — must match the probe bit-exactly.
    fn scalar_tetra_q15<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned4I16],
    ) -> [i16; 4] {
        let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [i16; 4] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let sub = |a: [i16; 4], b: [i16; 4]| -> [i16; 4] {
            // Wrapping — matches `_mm_sub_epi16`.
            [
                a[0].wrapping_sub(b[0]),
                a[1].wrapping_sub(b[1]),
                a[2].wrapping_sub(b[2]),
                a[3].wrapping_sub(b[3]),
            ]
        };
        let mla = |acc: [i16; 4], a: [i16; 4], k: i16| -> [i16; 4] {
            [
                acc[0].wrapping_add(mulhrs_ref(a[0], k)),
                acc[1].wrapping_add(mulhrs_ref(a[1], k)),
                acc[2].wrapping_add(mulhrs_ref(a[2], k)),
                acc[3].wrapping_add(mulhrs_ref(a[3], k)),
            ]
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

    fn random_cube_q15<const GRID_SIZE: usize>(seed: u32) -> Vec<Aligned4I16> {
        (0..GRID_SIZE.pow(3))
            .map(|i| {
                let k = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed);
                let b16 = |shift: u32| -> i16 {
                    // Small range so wrapping_sub in the reference
                    // matches the probe's subtraction behavior.
                    (k.rotate_left(shift) & 0x3FFF) as i16
                };
                Aligned4I16([b16(0), b16(7), b16(13), b16(19)])
            })
            .collect()
    }

    fn random_weights_q15<const GRID_SIZE: usize>(seed: u32) -> Box<[BarycentricWeight<i16>; 256]> {
        let mut w = Box::new([BarycentricWeight::<i16>::default(); 256]);
        for (i, entry) in w.iter_mut().enumerate() {
            let k = (i as u32).wrapping_mul(0xB503_2B33).wrapping_add(seed);
            let x = ((k & 0xFFFF) as usize % GRID_SIZE.max(2)) as i32;
            entry.x = x.min(GRID_SIZE as i32 - 1);
            entry.x_n = (entry.x + 1).min(GRID_SIZE as i32 - 1);
            entry.w = ((k >> 16) & 0x7FFF) as i16; // [0, 0x7FFF]
        }
        w
    }

    const EQUIV_INPUTS_Q15: &[(u8, u8, u8)] = &[
        (0, 0, 0),
        (255, 255, 255),
        (128, 64, 200),
        (1, 254, 128),
        (80, 80, 80),
        (200, 100, 50),
        (10, 200, 100),
        (33, 77, 222),
    ];

    fn scalar_trilinear_q15<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned4I16],
    ) -> [i16; 4] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [i16; 4] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let sub = |a: [i16; 4], b: [i16; 4]| -> [i16; 4] {
            [
                a[0].wrapping_sub(b[0]),
                a[1].wrapping_sub(b[1]),
                a[2].wrapping_sub(b[2]),
                a[3].wrapping_sub(b[3]),
            ]
        };
        let mla = |acc: [i16; 4], a: [i16; 4], k: i16| -> [i16; 4] {
            [
                acc[0].wrapping_add(mulhrs_ref(a[0], k)),
                acc[1].wrapping_add(mulhrs_ref(a[1], k)),
                acc[2].wrapping_add(mulhrs_ref(a[2], k)),
                acc[3].wrapping_add(mulhrs_ref(a[3], k)),
            ]
        };
        // Mirror the probe's trilinear structure: c = a + mulhrs(b - a, t)
        let lerp = |a: [i16; 4], b: [i16; 4], t: i16| -> [i16; 4] { mla(a, sub(b, a), t) };
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

    /// Q0.15 pyramid reference — mirrors `PyramidalSseQ0_15::interpolate`
    /// in `sse/interpolator_q0_15.rs`. Four-term with a `mulhrs`
    /// bilinear correction on the splatted weight product.
    fn scalar_pyramid_q15<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned4I16],
    ) -> [i16; 4] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [i16; 4] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let add = |a: [i16; 4], b: [i16; 4]| -> [i16; 4] {
            [
                a[0].wrapping_add(b[0]),
                a[1].wrapping_add(b[1]),
                a[2].wrapping_add(b[2]),
                a[3].wrapping_add(b[3]),
            ]
        };
        let sub = |a: [i16; 4], b: [i16; 4]| -> [i16; 4] {
            [
                a[0].wrapping_sub(b[0]),
                a[1].wrapping_sub(b[1]),
                a[2].wrapping_sub(b[2]),
                a[3].wrapping_sub(b[3]),
            ]
        };
        let mla = |acc: [i16; 4], a: [i16; 4], k: i16| -> [i16; 4] {
            [
                acc[0].wrapping_add(mulhrs_ref(a[0], k)),
                acc[1].wrapping_add(mulhrs_ref(a[1], k)),
                acc[2].wrapping_add(mulhrs_ref(a[2], k)),
                acc[3].wrapping_add(mulhrs_ref(a[3], k)),
            ]
        };
        let c0 = fetch(x, y, z);
        if dr > db && dg > db {
            let x0 = fetch(x_n, y_n, z_n);
            let x1 = fetch(x_n, y_n, z);
            let x2 = fetch(x_n, y, z);
            let x3 = fetch(x, y_n, z);
            let c1 = sub(x0, x1);
            let c2 = sub(x2, c0);
            let c3 = sub(x3, c0);
            let c4 = add(sub(sub(c0, x3), x2), x1);
            let k = mulhrs_ref(dr, dg);
            let s0 = mla(c0, c1, db);
            let s1 = mla(s0, c2, dr);
            let s2 = mla(s1, c3, dg);
            mla(s2, c4, k)
        } else if db > dr && dg > dr {
            let x0 = fetch(x, y, z_n);
            let x1 = fetch(x_n, y_n, z_n);
            let x2 = fetch(x, y_n, z_n);
            let x3 = fetch(x, y_n, z);
            let c1 = sub(x0, c0);
            let c2 = sub(x1, x2);
            let c3 = sub(x3, c0);
            let c4 = add(sub(sub(c0, x3), x0), x2);
            let k = mulhrs_ref(dg, db);
            let s0 = mla(c0, c1, db);
            let s1 = mla(s0, c2, dr);
            let s2 = mla(s1, c3, dg);
            mla(s2, c4, k)
        } else {
            let x0 = fetch(x, y, z_n);
            let x1 = fetch(x_n, y, z);
            let x2 = fetch(x_n, y, z_n);
            let x3 = fetch(x_n, y_n, z_n);
            let c1 = sub(x0, c0);
            let c2 = sub(x1, c0);
            let c3 = sub(x3, x2);
            let c4 = add(sub(sub(c0, x1), x0), x2);
            let k = mulhrs_ref(db, dr);
            let s0 = mla(c0, c1, db);
            let s1 = mla(s0, c2, dr);
            let s2 = mla(s1, c3, dg);
            mla(s2, c4, k)
        }
    }

    /// Q0.15 prism reference — mirrors `PrismaticSseQ0_15::interpolate`.
    /// Five-term with two `mulhrs` bilinear corrections. Branch uses
    /// `db > dr` (not `>=`) matching the hand-written path.
    fn scalar_prism_q15<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<i16>; 256],
        cube: &[Aligned4I16],
    ) -> [i16; 4] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [i16; 4] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let add = |a: [i16; 4], b: [i16; 4]| -> [i16; 4] {
            [
                a[0].wrapping_add(b[0]),
                a[1].wrapping_add(b[1]),
                a[2].wrapping_add(b[2]),
                a[3].wrapping_add(b[3]),
            ]
        };
        let sub = |a: [i16; 4], b: [i16; 4]| -> [i16; 4] {
            [
                a[0].wrapping_sub(b[0]),
                a[1].wrapping_sub(b[1]),
                a[2].wrapping_sub(b[2]),
                a[3].wrapping_sub(b[3]),
            ]
        };
        let mla = |acc: [i16; 4], a: [i16; 4], k: i16| -> [i16; 4] {
            [
                acc[0].wrapping_add(mulhrs_ref(a[0], k)),
                acc[1].wrapping_add(mulhrs_ref(a[1], k)),
                acc[2].wrapping_add(mulhrs_ref(a[2], k)),
                acc[3].wrapping_add(mulhrs_ref(a[3], k)),
            ]
        };
        let c0 = fetch(x, y, z);
        let k_dgdb = mulhrs_ref(dg, db);
        let k_drdg = mulhrs_ref(dr, dg);
        if db > dr {
            let x0 = fetch(x, y, z_n);
            let x1 = fetch(x_n, y, z_n);
            let x2 = fetch(x, y_n, z);
            let x3 = fetch(x, y_n, z_n);
            let x4 = fetch(x_n, y_n, z_n);
            let c1 = sub(x0, c0);
            let c2 = sub(x1, x0);
            let c3 = sub(x2, c0);
            let c4 = add(sub(sub(c0, x2), x0), x3);
            let c5 = add(sub(sub(x0, x3), x1), x4);
            let s0 = mla(c0, c1, db);
            let s1 = mla(s0, c2, dr);
            let s2 = mla(s1, c3, dg);
            let s3 = mla(s2, c4, k_dgdb);
            mla(s3, c5, k_drdg)
        } else {
            let x0 = fetch(x_n, y, z);
            let x1 = fetch(x_n, y, z_n);
            let x2 = fetch(x, y_n, z);
            let x3 = fetch(x_n, y_n, z);
            let x4 = fetch(x_n, y_n, z_n);
            let c1 = sub(x1, x0);
            let c2 = sub(x0, c0);
            let c3 = sub(x2, c0);
            let c4 = add(sub(sub(x0, x3), x1), x4);
            let c5 = add(sub(sub(c0, x2), x0), x3);
            let s0 = mla(c0, c1, db);
            let s1 = mla(s0, c2, dr);
            let s2 = mla(s1, c3, dg);
            let s3 = mla(s2, c4, k_dgdb);
            mla(s3, c5, k_drdg)
        }
    }

    fn assert_q15_bitexact(got: [i16; 8], expect: [i16; 4], ctx: &str) {
        let got4 = [got[0], got[1], got[2], got[3]];
        assert_eq!(
            got4, expect,
            "{ctx} low half: got={got4:?} expect={expect:?}"
        );
        assert_eq!(
            [got[4], got[5], got[6], got[7]],
            [0, 0, 0, 0],
            "{ctx} high half should remain zero-padded"
        );
    }

    #[test]
    fn tetra_q0_15_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights_q15::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube_q15::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS_Q15 {
            let expect = scalar_tetra_q15::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0i16; 8];
            incant!(
                interpolate_tetra::<u8, GRID_SIZE, 256>(ir, ig, ib, &*weights, &cube, &mut got,),
                [v3, neon, wasm128, scalar]
            );
            assert_q15_bitexact(got, expect, &format!("tetra_q0_15 (r={ir},g={ig},b={ib})"));
        }
    }

    #[test]
    fn trilinear_q0_15_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights_q15::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube_q15::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS_Q15 {
            let expect = scalar_trilinear_q15::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0i16; 8];
            incant!(
                interpolate_trilinear::<u8, GRID_SIZE, 256>(ir, ig, ib, &*weights, &cube, &mut got,),
                [v3, neon, wasm128, scalar]
            );
            assert_q15_bitexact(
                got,
                expect,
                &format!("trilinear_q0_15 (r={ir},g={ig},b={ib})"),
            );
        }
    }

    #[cfg(feature = "options")]
    #[test]
    fn pyramid_q0_15_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights_q15::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube_q15::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS_Q15 {
            let expect = scalar_pyramid_q15::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0i16; 8];
            incant!(
                interpolate_pyramid::<u8, GRID_SIZE, 256>(ir, ig, ib, &*weights, &cube, &mut got,),
                [v3, neon, wasm128, scalar]
            );
            assert_q15_bitexact(
                got,
                expect,
                &format!("pyramid_q0_15 (r={ir},g={ig},b={ib})"),
            );
        }
    }

    #[cfg(feature = "options")]
    #[test]
    fn prism_q0_15_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights_q15::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube_q15::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS_Q15 {
            let expect = scalar_prism_q15::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0i16; 8];
            incant!(
                interpolate_prism::<u8, GRID_SIZE, 256>(ir, ig, ib, &*weights, &cube, &mut got,),
                [v3, neon, wasm128, scalar]
            );
            assert_q15_bitexact(got, expect, &format!("prism_q0_15 (r={ir},g={ig},b={ib})"));
        }
    }

    fn make_identity_ramp<const BINS: usize, const GRID_SIZE: usize>()
    -> Box<[BarycentricWeight<i16>; BINS]> {
        let mut w = Box::new([BarycentricWeight::<i16>::default(); BINS]);
        let grid = (GRID_SIZE as i32 - 1) as i64;
        for (i, entry) in w.iter_mut().enumerate() {
            let idx_f = i as i64 * grid;
            let x = (idx_f / (BINS - 1) as i64) as i32;
            entry.x = x;
            entry.x_n = (x + 1).min(GRID_SIZE as i32 - 1);
            // Q0.15 weight: fractional part scaled into [0, 0x7FFF]
            let frac = (idx_f - x as i64 * (BINS - 1) as i64) as f64 / (BINS - 1) as f64;
            entry.w = (frac * 32767.0) as i16;
        }
        w
    }

    fn make_cube<const GRID_SIZE: usize>() -> Vec<Aligned4I16> {
        (0..GRID_SIZE.pow(3))
            .map(|i| Aligned4I16([i as i16, i as i16, i as i16, 0]))
            .collect()
    }

    #[test]
    fn tetra_q0_15_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0i16; 8];
        incant!(
            interpolate_tetra::<u8, GRID_SIZE, BINS>(
                128u8, 64u8, 200u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[test]
    fn trilinear_q0_15_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0i16; 8];
        incant!(
            interpolate_trilinear::<u8, GRID_SIZE, BINS>(
                30u8, 90u8, 150u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[test]
    fn pyramid_q0_15_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0i16; 8];
        incant!(
            interpolate_pyramid::<u8, GRID_SIZE, BINS>(
                50u8, 100u8, 150u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[test]
    fn prism_q0_15_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0i16; 8];
        incant!(
            interpolate_prism::<u8, GRID_SIZE, BINS>(25u8, 75u8, 175u8, &*weights, &cube, &mut out,),
            [v3, neon, wasm128, scalar]
        );
    }
}

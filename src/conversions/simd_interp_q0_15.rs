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

    let fetch = |x: i32, y: i32, z: i32| -> i16x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        load_q15(token, &cube[offset])
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

    let dr_v = i16x8::splat(token, dr);
    let dg_v = i16x8::splat(token, dg);
    let db_v = i16x8::splat(token, db);
    let s0 = c0 + q15_mulhrs(token, c1, dr_v);
    let s1 = s0 + q15_mulhrs(token, c2, dg_v);
    let s2 = s1 + q15_mulhrs(token, c3, db_v);
    s2.store(out);
}

// --- Prismatic (options-gated) ------------------------------------------

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

    let fetch = |x: i32, y: i32, z: i32| -> i16x8 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        load_q15(token, &cube[offset])
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

    let dr_v = i16x8::splat(token, dr);
    let dg_v = i16x8::splat(token, dg);
    let db_v = i16x8::splat(token, db);
    let s0 = c0 + q15_mulhrs(token, c1, dr_v);
    let s1 = s0 + q15_mulhrs(token, c2, dg_v);
    let s2 = s1 + q15_mulhrs(token, c3, db_v);
    s2.store(out);
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

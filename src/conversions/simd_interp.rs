// Generic SIMD interpolators built on magetypes, fanned out per tier
// by `#[magetypes]` and fanned back in by `incant!` at the dispatch
// site. Each of the four interpolators (Tetrahedral, Pyramidal,
// Prismatic, Trilinear) is written exactly once here against the
// generic `GenericF32x4<T>` type; the macro generates `_v3`, `_neon`,
// `_wasm128`, `_scalar` suffixed specializations.
//
// Compared with the hand-written `SseVector` / `AvxVector` /
// `NeonVector` wrappers each of which carried ~12 `unsafe { intrinsic }`
// blocks, this module contains **zero** `unsafe` keywords.
//
// Scope: f32 (floating-point) tetrahedral/pyramidal/prismatic/trilinear
// over a 4-lane LUT entry. The Q0_15 (i16) variants and the 256-bit
// Double variants are not yet ported.
#![cfg(feature = "lut")]
#![allow(dead_code)]
#![allow(non_camel_case_types)]

use crate::conversions::interpolator::{BarycentricWeight, load_bary_weights};
use archmage::magetypes;
use magetypes::simd::generic::f32x4 as GenericF32x4;
use num_traits::AsPrimitive;

/// 4-lane aligned LUT storage. `#[repr(align(16), C)]` keeps `movaps`
/// alignment on x86 and 16-byte alignment on NEON/wasm128 — same
/// contract the hand-written `SseAlignedF32` / `NeonAlignedF32` kept.
#[repr(align(16), C)]
pub(crate) struct Aligned4<T>(pub(crate) [T; 4]);

// --- Tetrahedral ---------------------------------------------------------

/// Branch interpolation: six-simplex decomposition of the unit cube,
/// picks one tetrahedron based on the `(dr, dg, db)` ordering and
/// folds three edge differences via `mul_add`. The body reads identical
/// to the hand-written `TetrahedralSse::interpolate` — only the types
/// differ.
#[magetypes(v3, neon, wasm128, scalar)]
fn interpolate_tetra<U: AsPrimitive<usize>, const GRID_SIZE: usize, const BINS: usize>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned4<f32>],
    out: &mut [f32; 4],
) {
    type f32x4 = GenericF32x4<Token>;

    let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);

    let fetch = |x: i32, y: i32, z: i32| -> f32x4 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x4::load(token, &cube[offset].0)
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

    // moxcms `a.mla(b, c) = a + b*c`.  magetypes `b.mul_add(c, a) = b*c + a`.
    let rxv = f32x4::splat(token, rx);
    let ryv = f32x4::splat(token, ry);
    let rzv = f32x4::splat(token, rz);
    let s0 = c1.mul_add(rxv, c0);
    let s1 = c2.mul_add(ryv, s0);
    c3.mul_add(rzv, s1).store(out);
}

// --- Trilinear ----------------------------------------------------------

/// Eight-corner barycentric trilinear interpolation. No branching on
/// the weight order — always fetches the eight cube corners and weights
/// by `(1-dr, dr) × (1-dg, dg) × (1-db, db)`.
#[magetypes(v3, neon, wasm128, scalar)]
fn interpolate_trilinear<U: AsPrimitive<usize>, const GRID_SIZE: usize, const BINS: usize>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned4<f32>],
    out: &mut [f32; 4],
) {
    type f32x4 = GenericF32x4<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let fetch = |x: i32, y: i32, z: i32| -> f32x4 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x4::load(token, &cube[offset].0)
    };

    let c000 = fetch(x, y, z);
    let c100 = fetch(x_n, y, z);
    let c010 = fetch(x, y_n, z);
    let c110 = fetch(x_n, y_n, z);
    let c001 = fetch(x, y, z_n);
    let c101 = fetch(x_n, y, z_n);
    let c011 = fetch(x, y_n, z_n);
    let c111 = fetch(x_n, y_n, z_n);

    let dx_v = f32x4::splat(token, 1.0 - dr);
    let dr_v = f32x4::splat(token, dr);
    let dy_v = f32x4::splat(token, 1.0 - dg);
    let dg_v = f32x4::splat(token, dg);
    let dz_v = f32x4::splat(token, 1.0 - db);
    let db_v = f32x4::splat(token, db);

    // c00 = c000*(1-dr) + c100*dr
    let c00 = c100.mul_add(dr_v, c000 * dx_v);
    let c10 = c110.mul_add(dr_v, c010 * dx_v);
    let c01 = c101.mul_add(dr_v, c001 * dx_v);
    let c11 = c111.mul_add(dr_v, c011 * dx_v);

    // c0 = c00*(1-dg) + c10*dg
    let c0 = c10.mul_add(dg_v, c00 * dy_v);
    let c1 = c11.mul_add(dg_v, c01 * dy_v);

    // result = c0*(1-db) + c1*db
    c1.mul_add(db_v, c0 * dz_v).store(out);
}

// --- Pyramidal (options-gated) ------------------------------------------

#[cfg(feature = "options")]
#[magetypes(v3, neon, wasm128, scalar)]
fn interpolate_pyramid<U: AsPrimitive<usize>, const GRID_SIZE: usize, const BINS: usize>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned4<f32>],
    out: &mut [f32; 4],
) {
    type f32x4 = GenericF32x4<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let fetch = |x: i32, y: i32, z: i32| -> f32x4 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x4::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);

    // Pyramidal three-tetrahedron split by which coord is largest.
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

    let dr_v = f32x4::splat(token, dr);
    let dg_v = f32x4::splat(token, dg);
    let db_v = f32x4::splat(token, db);
    let s0 = c1.mul_add(dr_v, c0);
    let s1 = c2.mul_add(dg_v, s0);
    c3.mul_add(db_v, s1).store(out);
}

// --- Prismatic (options-gated) ------------------------------------------

#[cfg(feature = "options")]
#[magetypes(v3, neon, wasm128, scalar)]
fn interpolate_prism<U: AsPrimitive<usize>, const GRID_SIZE: usize, const BINS: usize>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned4<f32>],
    out: &mut [f32; 4],
) {
    type f32x4 = GenericF32x4<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    let fetch = |x: i32, y: i32, z: i32| -> f32x4 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x4::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);

    // Prismatic two-tetrahedron split on dr vs dg.
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

    let dr_v = f32x4::splat(token, dr);
    let dg_v = f32x4::splat(token, dg);
    let db_v = f32x4::splat(token, db);
    let s0 = c1.mul_add(dr_v, c0);
    let s1 = c2.mul_add(dg_v, s0);
    c3.mul_add(db_v, s1).store(out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversions::interpolator::BarycentricWeight;
    use archmage::incant;

    fn make_identity_ramp<const BINS: usize, const GRID_SIZE: usize>()
    -> Box<[BarycentricWeight<f32>; BINS]> {
        let mut w = Box::new([BarycentricWeight::<f32>::default(); BINS]);
        for (i, entry) in w.iter_mut().enumerate() {
            let x = (i as f32 * (GRID_SIZE as i32 - 1) as f32 / (BINS - 1) as f32).floor() as i32;
            entry.x = x;
            entry.x_n = (x + 1).min(GRID_SIZE as i32 - 1);
            entry.w =
                (i as f32 * (GRID_SIZE as i32 - 1) as f32 / (BINS - 1) as f32) - x as f32;
        }
        w
    }

    fn make_cube<const GRID_SIZE: usize>() -> Vec<Aligned4<f32>> {
        (0..GRID_SIZE.pow(3))
            .map(|i| Aligned4([i as f32, i as f32, i as f32, 0.0]))
            .collect()
    }

    #[test]
    fn tetra_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 4];
        incant!(interpolate_tetra::<u8, GRID_SIZE, BINS>(
            0u8, 0u8, 0u8, &*weights, &cube, &mut out,
        ));
    }

    #[test]
    fn trilinear_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 4];
        incant!(interpolate_trilinear::<u8, GRID_SIZE, BINS>(
            128u8, 64u8, 200u8, &*weights, &cube, &mut out,
        ));
    }

    #[cfg(feature = "options")]
    #[test]
    fn pyramid_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 4];
        incant!(interpolate_pyramid::<u8, GRID_SIZE, BINS>(
            50u8, 100u8, 150u8, &*weights, &cube, &mut out,
        ));
    }

    #[cfg(feature = "options")]
    #[test]
    fn prism_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 4];
        incant!(interpolate_prism::<u8, GRID_SIZE, BINS>(
            25u8, 75u8, 175u8, &*weights, &cube, &mut out,
        ));
    }
}

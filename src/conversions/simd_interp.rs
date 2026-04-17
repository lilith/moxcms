// Generic SIMD interpolators built on magetypes, fanned out per tier
// by `#[magetypes]` and fanned back in by `incant!` at the dispatch
// site. Each of the four interpolators (Tetrahedral, Pyramidal,
// Prismatic, Trilinear) is written exactly once here against the
// generic `GenericF32x4<T>` type; the macro generates `_v3`, `_neon`,
// `_wasm128`, `_scalar` suffixed specializations.
//
// Compared with the hand-written `SseVector` / `AvxVector` /
// `NeonVector` wrappers each of which gated ~12 intrinsic calls
// behind raw-pointer loads, this module contains no raw pointers and
// is composed entirely of safe operators on `GenericF32x4<Token>`.
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

/// 4-lane f32 aligned cube entry. `#[repr(align(16), C)]` keeps
/// `movaps`-grade alignment on x86 and matches the hand-written
/// `SseAlignedF32` / `NeonAlignedF32` layout. Concrete (not
/// generic) so `bytemuck::Pod` derive can verify padding statically
/// — adapter helpers cast `&[SseAlignedF32]` → `&[Aligned4F32]`
/// via `bytemuck::cast_slice` — safe, no raw pointer reinterpret.
#[repr(align(16), C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Aligned4F32(pub(crate) [f32; 4]);

impl Aligned4F32 {
    #[inline(always)]
    pub(crate) const fn new(v: [f32; 4]) -> Self {
        Self(v)
    }
}

// --- Tetrahedral ---------------------------------------------------------

/// Branch interpolation: six-simplex decomposition of the unit cube,
/// picks one tetrahedron based on the `(dr, dg, db)` ordering and
/// folds three edge differences via `mul_add`. The body reads identical
/// to the hand-written `TetrahedralSse::interpolate` — only the types
/// differ.
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
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned4F32],
    out: &mut [f32; 4],
) {
    type f32x4 = GenericF32x4<Token>;

    let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);

    // Bound all six indices and the cube length so LLVM can prove
    // every `cube[offset]` below is in-bounds (the hand-written path
    // reaches this by `get_unchecked`). `x,…,z_n ∈ [0, G)` implies
    // `offset ∈ [0, G³)`, and `cube.len() >= G³` closes the gap.
    // Four runtime asserts hoist out of the inner math; LLVM folds
    // them into a single branch-predicted-taken check.
    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

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
pub(crate) fn interpolate_trilinear<
    U: AsPrimitive<usize>,
    const GRID_SIZE: usize,
    const BINS: usize,
>(
    token: Token,
    in_r: U,
    in_g: U,
    in_b: U,
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned4F32],
    out: &mut [f32; 4],
) {
    type f32x4 = GenericF32x4<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    // See `interpolate_tetra` for rationale — hoists four runtime
    // asserts so LLVM can prove every `cube[offset]` in-bounds.
    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

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

/// Four-term pyramidal decomposition with bilinear correction.
/// Lifted verbatim from `Pyramidal::interpolate` in
/// `interpolator.rs`. Three branches on the `(dr, dg, db)` ordering;
/// each picks four cube corners + four partial differences. The last
/// term `c4` multiplies a bilinear product of two of the weights
/// (e.g. `dr * dg`) to recover the saddle the three linear terms
/// cannot represent.
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
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned4F32],
    out: &mut [f32; 4],
) {
    type f32x4 = GenericF32x4<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    // See `interpolate_tetra` for rationale — hoists four runtime
    // asserts so LLVM can prove every `cube[offset]` in-bounds.
    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> f32x4 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x4::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);

    let dr_v = f32x4::splat(token, dr);
    let dg_v = f32x4::splat(token, dg);
    let db_v = f32x4::splat(token, db);

    if dr > db && dg > db {
        let x0 = fetch(x_n, y_n, z_n);
        let x1 = fetch(x_n, y_n, z);
        let x2 = fetch(x_n, y, z);
        let x3 = fetch(x, y_n, z);
        let c1 = x0 - x1;
        let c2 = x2 - c0;
        let c3 = x3 - c0;
        let c4 = c0 - x3 - x2 + x1;
        let k = f32x4::splat(token, dr * dg);
        let s0 = c1.mul_add(db_v, c0);
        let s1 = c2.mul_add(dr_v, s0);
        let s2 = c3.mul_add(dg_v, s1);
        c4.mul_add(k, s2).store(out);
    } else if db > dr && dg > dr {
        let x0 = fetch(x, y, z_n);
        let x1 = fetch(x_n, y_n, z_n);
        let x2 = fetch(x, y_n, z_n);
        let x3 = fetch(x, y_n, z);
        let c1 = x0 - c0;
        let c2 = x1 - x2;
        let c3 = x3 - c0;
        let c4 = c0 - x3 - x0 + x2;
        let k = f32x4::splat(token, dg * db);
        let s0 = c1.mul_add(db_v, c0);
        let s1 = c2.mul_add(dr_v, s0);
        let s2 = c3.mul_add(dg_v, s1);
        c4.mul_add(k, s2).store(out);
    } else {
        let x0 = fetch(x, y, z_n);
        let x1 = fetch(x_n, y, z);
        let x2 = fetch(x_n, y, z_n);
        let x3 = fetch(x_n, y_n, z_n);
        let c1 = x0 - c0;
        let c2 = x1 - c0;
        let c3 = x3 - x2;
        let c4 = c0 - x1 - x0 + x2;
        let k = f32x4::splat(token, db * dr);
        let s0 = c1.mul_add(db_v, c0);
        let s1 = c2.mul_add(dr_v, s0);
        let s2 = c3.mul_add(dg_v, s1);
        c4.mul_add(k, s2).store(out);
    }
}

// --- Prismatic (options-gated) ------------------------------------------

/// Five-term prismatic decomposition with two bilinear corrections.
/// Lifted verbatim from `Prismatic::interpolate` in `interpolator.rs`.
/// Two branches on `db >= dr`. Five cube corners + five partial
/// differences; the last two terms multiply bilinear products of
/// the weights to recover the two saddle surfaces.
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
    lut: &[BarycentricWeight<f32>; BINS],
    cube: &[Aligned4F32],
    out: &mut [f32; 4],
) {
    type f32x4 = GenericF32x4<Token>;

    let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);

    // See `interpolate_tetra` for rationale — hoists four runtime
    // asserts so LLVM can prove every `cube[offset]` in-bounds.
    let g = GRID_SIZE as u32;
    assert!((x as u32) < g && (x_n as u32) < g);
    assert!((y as u32) < g && (y_n as u32) < g);
    assert!((z as u32) < g && (z_n as u32) < g);
    assert!(cube.len() >= GRID_SIZE * GRID_SIZE * GRID_SIZE);

    let fetch = |x: i32, y: i32, z: i32| -> f32x4 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        f32x4::load(token, &cube[offset].0)
    };

    let c0 = fetch(x, y, z);

    let dr_v = f32x4::splat(token, dr);
    let dg_v = f32x4::splat(token, dg);
    let db_v = f32x4::splat(token, db);

    if db >= dr {
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
        let k_dgdb = f32x4::splat(token, dg * db);
        let k_drdg = f32x4::splat(token, dr * dg);
        let s0 = c1.mul_add(db_v, c0);
        let s1 = c2.mul_add(dr_v, s0);
        let s2 = c3.mul_add(dg_v, s1);
        let s3 = c4.mul_add(k_dgdb, s2);
        c5.mul_add(k_drdg, s3).store(out);
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
        let k_dgdb = f32x4::splat(token, dg * db);
        let k_drdg = f32x4::splat(token, dr * dg);
        let s0 = c1.mul_add(db_v, c0);
        let s1 = c2.mul_add(dr_v, s0);
        let s2 = c3.mul_add(dg_v, s1);
        let s3 = c4.mul_add(k_dgdb, s2);
        c5.mul_add(k_drdg, s3).store(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversions::interpolator::BarycentricWeight;
    use archmage::incant;

    /// Scalar reference implementation of the six-tetrahedron
    /// barycentric decomposition, lifted verbatim from the body of
    /// `Tetrahedral::interpolate` in `interpolator.rs`. Kept inline in
    /// the test so the equivalence check does not depend on the
    /// `MultidimensionalInterpolation` trait, which works on `&[f32]`
    /// with a different strided fetch shape.
    fn scalar_tetra_ref<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<f32>; 256],
        cube: &[Aligned4F32],
    ) -> [f32; 4] {
        let (x, y, z, x_n, y_n, z_n, rx, ry, rz) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [f32; 4] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
        };
        let mla = |acc: [f32; 4], a: [f32; 4], k: f32| -> [f32; 4] {
            // Match the probe's mul_add ordering: acc + a * k.
            [
                a[0].mul_add(k, acc[0]),
                a[1].mul_add(k, acc[1]),
                a[2].mul_add(k, acc[2]),
                a[3].mul_add(k, acc[3]),
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

    /// Build a deterministic non-trivial cube so every branch of the
    /// tetra decomposition actually sees distinct values.
    fn random_cube<const GRID_SIZE: usize>(seed: u32) -> Vec<Aligned4F32> {
        (0..GRID_SIZE.pow(3))
            .map(|i| {
                let k = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed);
                let kf = |shift: u32| -> f32 {
                    let bits = k.rotate_left(shift) & 0x7FFF_FFFF;
                    bits as f32 / 0x4000_0000u32 as f32 // ~[0, 2)
                };
                Aligned4F32([kf(0), kf(7), kf(13), kf(19)])
            })
            .collect()
    }

    /// Deterministic barycentric weights with well-distributed x and w.
    fn random_weights<const GRID_SIZE: usize>(seed: u32) -> Box<[BarycentricWeight<f32>; 256]> {
        let mut w = Box::new([BarycentricWeight::<f32>::default(); 256]);
        for (i, entry) in w.iter_mut().enumerate() {
            let k = (i as u32).wrapping_mul(0xB503_2B33).wrapping_add(seed);
            let x = ((k & 0xFFFF) as usize % GRID_SIZE.max(2)) as i32;
            entry.x = x.min(GRID_SIZE as i32 - 1);
            entry.x_n = (entry.x + 1).min(GRID_SIZE as i32 - 1);
            let bits = (k >> 16) & 0x7FFF_FFFF;
            entry.w = bits as f32 / 0x8000_0000u32 as f32; // [0, 1)
        }
        w
    }

    fn scalar_trilinear_ref<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<f32>; 256],
        cube: &[Aligned4F32],
    ) -> [f32; 4] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [f32; 4] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let lerp = |a: [f32; 4], b: [f32; 4], t: f32| -> [f32; 4] {
            // Match the probe: (b - a) is fused as b*t + a*(1-t) via mul_add
            let it = 1.0 - t;
            [
                b[0].mul_add(t, a[0] * it),
                b[1].mul_add(t, a[1] * it),
                b[2].mul_add(t, a[2] * it),
                b[3].mul_add(t, a[3] * it),
            ]
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

    /// Mirrors `Pyramidal::interpolate` in `interpolator.rs` —
    /// four-term decomposition with a bilinear correction.
    fn scalar_pyramid_ref<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<f32>; 256],
        cube: &[Aligned4F32],
    ) -> [f32; 4] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [f32; 4] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
        };
        let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
        };
        let mla = |acc: [f32; 4], a: [f32; 4], k: f32| -> [f32; 4] {
            [
                a[0].mul_add(k, acc[0]),
                a[1].mul_add(k, acc[1]),
                a[2].mul_add(k, acc[2]),
                a[3].mul_add(k, acc[3]),
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
            let s0 = mla(c0, c1, db);
            let s1 = mla(s0, c2, dr);
            let s2 = mla(s1, c3, dg);
            mla(s2, c4, dr * dg)
        } else if db > dr && dg > dr {
            let x0 = fetch(x, y, z_n);
            let x1 = fetch(x_n, y_n, z_n);
            let x2 = fetch(x, y_n, z_n);
            let x3 = fetch(x, y_n, z);
            let c1 = sub(x0, c0);
            let c2 = sub(x1, x2);
            let c3 = sub(x3, c0);
            let c4 = add(sub(sub(c0, x3), x0), x2);
            let s0 = mla(c0, c1, db);
            let s1 = mla(s0, c2, dr);
            let s2 = mla(s1, c3, dg);
            mla(s2, c4, dg * db)
        } else {
            let x0 = fetch(x, y, z_n);
            let x1 = fetch(x_n, y, z);
            let x2 = fetch(x_n, y, z_n);
            let x3 = fetch(x_n, y_n, z_n);
            let c1 = sub(x0, c0);
            let c2 = sub(x1, c0);
            let c3 = sub(x3, x2);
            let c4 = add(sub(sub(c0, x1), x0), x2);
            let s0 = mla(c0, c1, db);
            let s1 = mla(s0, c2, dr);
            let s2 = mla(s1, c3, dg);
            mla(s2, c4, db * dr)
        }
    }

    /// Mirrors `Prismatic::interpolate` in `interpolator.rs` —
    /// five-term decomposition with two bilinear corrections.
    fn scalar_prism_ref<const GRID_SIZE: usize>(
        in_r: u8,
        in_g: u8,
        in_b: u8,
        lut: &[BarycentricWeight<f32>; 256],
        cube: &[Aligned4F32],
    ) -> [f32; 4] {
        let (x, y, z, x_n, y_n, z_n, dr, dg, db) = load_bary_weights(lut, in_r, in_g, in_b);
        let fetch = |x: i32, y: i32, z: i32| -> [f32; 4] {
            let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
                + y as u32 * GRID_SIZE as u32
                + z as u32) as usize;
            cube[offset].0
        };
        let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
        };
        let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
        };
        let mla = |acc: [f32; 4], a: [f32; 4], k: f32| -> [f32; 4] {
            [
                a[0].mul_add(k, acc[0]),
                a[1].mul_add(k, acc[1]),
                a[2].mul_add(k, acc[2]),
                a[3].mul_add(k, acc[3]),
            ]
        };
        let c0 = fetch(x, y, z);
        if db >= dr {
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
            let s3 = mla(s2, c4, dg * db);
            mla(s3, c5, dr * dg)
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
            let s3 = mla(s2, c4, dg * db);
            mla(s3, c5, dr * dg)
        }
    }

    /// Shared input set — every tetra branch firing + boundary cases.
    const EQUIV_INPUTS: &[(u8, u8, u8)] = &[
        (0, 0, 0),
        (255, 255, 255),
        (128, 64, 200),
        (1, 254, 128),
        (80, 80, 80),
        (200, 100, 50),
        (10, 200, 100),
        (33, 77, 222),
    ];

    fn assert_f32x4_close(got: [f32; 4], expect: [f32; 4], ctx: &str) {
        for lane in 0..4 {
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

    #[test]
    fn tetra_matches_scalar_reference() {
        // Cover every possible (rx, ry, rz) ordering by spanning 64
        // input triples across the [0, 255] byte space. GRID_SIZE=9
        // forces multiple cube cells per axis so ordering branches fire.
        const GRID_SIZE: usize = 9;
        let weights = random_weights::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube::<GRID_SIZE>(0xDEAD_BEEF);

        for &(ir, ig, ib) in EQUIV_INPUTS {
            let expect = scalar_tetra_ref::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0f32; 4];
            incant!(
                interpolate_tetra::<u8, GRID_SIZE, 256>(ir, ig, ib, &*weights, &cube, &mut got,),
                [v3, neon, wasm128, scalar]
            );
            assert_f32x4_close(got, expect, &format!("tetra (r={ir},g={ig},b={ib})"));
        }
    }

    #[test]
    fn trilinear_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS {
            let expect = scalar_trilinear_ref::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0f32; 4];
            incant!(
                interpolate_trilinear::<u8, GRID_SIZE, 256>(ir, ig, ib, &*weights, &cube, &mut got,),
                [v3, neon, wasm128, scalar]
            );
            assert_f32x4_close(got, expect, &format!("trilinear (r={ir},g={ig},b={ib})"));
        }
    }

    #[cfg(feature = "options")]
    #[test]
    fn pyramid_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS {
            let expect = scalar_pyramid_ref::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0f32; 4];
            incant!(
                interpolate_pyramid::<u8, GRID_SIZE, 256>(ir, ig, ib, &*weights, &cube, &mut got,),
                [v3, neon, wasm128, scalar]
            );
            assert_f32x4_close(got, expect, &format!("pyramid (r={ir},g={ig},b={ib})"));
        }
    }

    #[cfg(feature = "options")]
    #[test]
    fn prism_matches_scalar_reference() {
        const GRID_SIZE: usize = 9;
        let weights = random_weights::<GRID_SIZE>(0xC01D_F00D);
        let cube = random_cube::<GRID_SIZE>(0xDEAD_BEEF);
        for &(ir, ig, ib) in EQUIV_INPUTS {
            let expect = scalar_prism_ref::<GRID_SIZE>(ir, ig, ib, &weights, &cube);
            let mut got = [0f32; 4];
            incant!(
                interpolate_prism::<u8, GRID_SIZE, 256>(ir, ig, ib, &*weights, &cube, &mut got,),
                [v3, neon, wasm128, scalar]
            );
            assert_f32x4_close(got, expect, &format!("prism (r={ir},g={ig},b={ib})"));
        }
    }

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

    fn make_cube<const GRID_SIZE: usize>() -> Vec<Aligned4F32> {
        (0..GRID_SIZE.pow(3))
            .map(|i| Aligned4F32([i as f32, i as f32, i as f32, 0.0]))
            .collect()
    }

    #[test]
    fn tetra_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 4];
        incant!(
            interpolate_tetra::<u8, GRID_SIZE, BINS>(0u8, 0u8, 0u8, &*weights, &cube, &mut out,),
            [v3, neon, wasm128, scalar]
        );
    }

    #[test]
    fn trilinear_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 4];
        incant!(
            interpolate_trilinear::<u8, GRID_SIZE, BINS>(
                128u8, 64u8, 200u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[test]
    fn pyramid_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 4];
        incant!(
            interpolate_pyramid::<u8, GRID_SIZE, BINS>(
                50u8, 100u8, 150u8, &*weights, &cube, &mut out,
            ),
            [v3, neon, wasm128, scalar]
        );
    }

    #[cfg(feature = "options")]
    #[test]
    fn prism_smoke() {
        const GRID_SIZE: usize = 2;
        const BINS: usize = 256;
        let weights = make_identity_ramp::<BINS, GRID_SIZE>();
        let cube = make_cube::<GRID_SIZE>();
        let mut out = [0f32; 4];
        incant!(
            interpolate_prism::<u8, GRID_SIZE, BINS>(25u8, 75u8, 175u8, &*weights, &cube, &mut out,),
            [v3, neon, wasm128, scalar]
        );
    }
}

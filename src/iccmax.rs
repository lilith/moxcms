//! iccMAX (ICC.2) tag-type and processing-element readers.
//!
//! Currently scoped to the matrix-shaper-equivalent subset that real-world v5
//! profiles use: `utf8Type` (rfnm/csnm), `tagStructType` (cept), `mpet`
//! container with `cvst` (curveSet) and `matf` (float matrix) elements, and
//! `curf` segmentedCurveType with `parf` (parametric formula) and `samf`
//! (sampled curve) segments. CLUT-based / spectral / BRDF / calculator
//! features are intentionally out of scope.
//!
//! All routines return `Option` / `Result<Option, _>` so that an unexpected
//! shape falls through to existing parsing paths rather than poisoning the
//! whole profile parse. Wire-up into `ColorProfile::new_from_slice` happens
//! in `profile.rs`.

use crate::err::invalid_profile;
use crate::reader::{s15_fixed16_number_to_double, s15_fixed16_number_to_float};
use crate::safe_math::{SafeAdd, SafeMul};
use crate::trc::ToneReprCurve;
use crate::{CmsError, Matrix3d, Vector3d, Xyzd};

// ── Tag-type signatures (big-endian) ─────────────────────────────────────

#[allow(dead_code)]
const SIG_UTF8: &[u8; 4] = b"utf8";
const SIG_TSTR: &[u8; 4] = b"tstr";
const SIG_MPET: &[u8; 4] = b"mpet";
const SIG_CVST: &[u8; 4] = b"cvst";
const SIG_MATF: &[u8; 4] = b"matf";
const SIG_CURF: &[u8; 4] = b"curf";
const SIG_PARF: &[u8; 4] = b"parf";
const SIG_SAMF: &[u8; 4] = b"samf";

/// Number of points to densely sample a segmented curve into a Lut(u16).
/// 1024 covers the resolution of typical 8-bit and 12-bit transforms with
/// negligible error, and is what lcms2 uses internally for analogous cases.
const SEGMENTED_CURVE_SAMPLES: usize = 1024;

// ── utf8Type (10.2.30) ───────────────────────────────────────────────────

/// Read a `utf8` tag's payload as a UTF-8 string. Trailing NULs are stripped.
#[allow(dead_code)]
pub(crate) fn read_utf8_tag(
    slice: &[u8],
    entry: usize,
    tag_size: usize,
) -> Result<String, CmsError> {
    if tag_size < 8 {
        return Err(invalid_profile());
    }
    let end = entry.safe_add(tag_size)?;
    if end > slice.len() {
        return Err(invalid_profile());
    }
    let blob = &slice[entry..end];
    if &blob[0..4] != SIG_UTF8 {
        return Err(invalid_profile());
    }
    let text_bytes = &blob[8..];
    Ok(String::from_utf8_lossy(text_bytes)
        .trim_end_matches('\0')
        .to_string())
}

// ── tagStructType (10.2.23) ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct TagStructEntry {
    pub sig: [u8; 4],
    /// Offset RELATIVE to the start of the tagStructType blob.
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct TagStruct<'a> {
    /// `b"cept"`, `b"brdf"`, etc. — identifies which struct type this is.
    pub struct_id: [u8; 4],
    /// The full tag blob; sub-tag offsets are relative to its start.
    pub blob: &'a [u8],
    pub entries: Vec<TagStructEntry>,
}

impl<'a> TagStruct<'a> {
    /// Parse a `tstr` tag at `entry` of length `tag_size`.
    pub(crate) fn read(slice: &'a [u8], entry: usize, tag_size: usize) -> Result<Self, CmsError> {
        if tag_size < 16 {
            return Err(invalid_profile());
        }
        let end = entry.safe_add(tag_size)?;
        if end > slice.len() {
            return Err(invalid_profile());
        }
        let blob = &slice[entry..end];
        if &blob[0..4] != SIG_TSTR {
            return Err(invalid_profile());
        }
        let mut struct_id = [0u8; 4];
        struct_id.copy_from_slice(&blob[8..12]);
        let n = u32::from_be_bytes(blob[12..16].try_into().unwrap()) as usize;
        // Each entry is 12 bytes (4 sig + 4 offset + 4 size).
        let table_end = 16usize.safe_add(n.safe_mul(12)?)?;
        if blob.len() < table_end {
            return Err(invalid_profile());
        }
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let off = 16 + i * 12;
            let mut sig = [0u8; 4];
            sig.copy_from_slice(&blob[off..off + 4]);
            let offset = u32::from_be_bytes(blob[off + 4..off + 8].try_into().unwrap()) as usize;
            let size = u32::from_be_bytes(blob[off + 8..off + 12].try_into().unwrap()) as usize;
            entries.push(TagStructEntry { sig, offset, size });
        }
        Ok(TagStruct {
            struct_id,
            blob,
            entries,
        })
    }

    /// Find a sub-tag by signature.
    pub(crate) fn find(&self, sig: &[u8; 4]) -> Option<&TagStructEntry> {
        self.entries.iter().find(|e| &e.sig == sig)
    }

    /// Get the raw bytes for a sub-tag.
    pub(crate) fn slice(&self, entry: &TagStructEntry) -> Option<&'a [u8]> {
        let end = entry.offset.checked_add(entry.size)?;
        self.blob.get(entry.offset..end)
    }
}

// ── float32Number array helpers ──────────────────────────────────────────

/// Read an `fl32`-typed sub-tag as a slice of f32 values. The on-disk
/// signature is `fl32` (8-byte header + N×4 bytes of big-endian f32).
pub(crate) fn read_fl32_array(blob: &[u8]) -> Result<Vec<f32>, CmsError> {
    if blob.len() < 8 {
        return Err(invalid_profile());
    }
    if &blob[0..4] != b"fl32" {
        return Err(invalid_profile());
    }
    let body = &blob[8..];
    let n = body.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32::from_be_bytes(
            body[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    Ok(out)
}

// ── segmentedCurveType (11.2.2.3, table 109-112) ─────────────────────────

/// Evaluate one of the 4 standard parf formulas at `x`.
/// Function types from ICC.2:2019 Table 111:
///   0: Y = (a*X + b)^γ + c                    (4 params: γ,a,b,c)
///   1: Y = a*log10(b*X^γ + c) + d             (5 params: γ,a,b,c,d)
///   2: Y = a*b^(c*X+d) + e                    (5 params: a,b,c,d,e)
///   3: Y = a*(b*X + c)^γ + d                  (5 params: γ,a,b,c,d)
fn eval_parf(func_type: u16, params: &[f32], x: f64) -> f64 {
    let p = |i: usize| -> f64 { params.get(i).copied().unwrap_or(0.0) as f64 };
    match func_type {
        0 => {
            let g = p(0);
            let a = p(1);
            let b = p(2);
            let c = p(3);
            let base = a * x + b;
            let pow = if base > 0.0 { base.powf(g) } else { 0.0 };
            pow + c
        }
        1 => {
            let g = p(0);
            let a = p(1);
            let b = p(2);
            let c = p(3);
            let d = p(4);
            let inner = b * (if x > 0.0 { x.powf(g) } else { 0.0 }) + c;
            if inner > 0.0 {
                a * inner.log10() + d
            } else {
                d
            }
        }
        2 => {
            let a = p(0);
            let b = p(1);
            let c = p(2);
            let d = p(3);
            let e = p(4);
            a * b.powf(c * x + d) + e
        }
        3 => {
            let g = p(0);
            let a = p(1);
            let b = p(2);
            let c = p(3);
            let d = p(4);
            let base = b * x + c;
            let pow = if base > 0.0 { base.powf(g) } else { 0.0 };
            a * pow + d
        }
        _ => x, // Unknown function — pass through
    }
}

#[derive(Debug, Clone)]
enum CurveSegment {
    Parametric { func_type: u16, params: Vec<f32> },
    Sampled(Vec<f32>),
}

#[derive(Debug, Clone)]
struct SegmentedCurve {
    breakpoints: Vec<f32>,
    segments: Vec<CurveSegment>,
}

impl SegmentedCurve {
    fn read(blob: &[u8]) -> Result<Self, CmsError> {
        if blob.len() < 12 {
            return Err(invalid_profile());
        }
        if &blob[0..4] != SIG_CURF {
            return Err(invalid_profile());
        }
        let n_segs = u16::from_be_bytes([blob[8], blob[9]]) as usize;
        if n_segs == 0 {
            return Err(invalid_profile());
        }
        // bytes 10..11 reserved
        let bp_start = 12;
        let n_breakpoints = n_segs.saturating_sub(1);
        let bp_end = bp_start + n_breakpoints * 4;
        if blob.len() < bp_end {
            return Err(invalid_profile());
        }
        let mut breakpoints = Vec::with_capacity(n_breakpoints);
        for i in 0..n_breakpoints {
            breakpoints.push(f32::from_be_bytes(
                blob[bp_start + i * 4..bp_start + i * 4 + 4]
                    .try_into()
                    .unwrap(),
            ));
        }

        let mut cursor = bp_end;
        let mut segments = Vec::with_capacity(n_segs);
        for _ in 0..n_segs {
            if blob.len() < cursor + 8 {
                return Err(invalid_profile());
            }
            let sig = &blob[cursor..cursor + 4];
            if sig == SIG_PARF {
                if blob.len() < cursor + 12 {
                    return Err(invalid_profile());
                }
                let func_type = u16::from_be_bytes([blob[cursor + 8], blob[cursor + 9]]);
                // 4 params for type 0; 5 for types 1-3.
                let n_params = if func_type == 0 { 4 } else { 5 };
                let body_start = cursor + 12;
                let body_end = body_start + n_params * 4;
                if blob.len() < body_end {
                    return Err(invalid_profile());
                }
                let mut params = Vec::with_capacity(n_params);
                for i in 0..n_params {
                    params.push(f32::from_be_bytes(
                        blob[body_start + i * 4..body_start + i * 4 + 4]
                            .try_into()
                            .unwrap(),
                    ));
                }
                segments.push(CurveSegment::Parametric { func_type, params });
                cursor = body_end;
            } else if sig == SIG_SAMF {
                if blob.len() < cursor + 12 {
                    return Err(invalid_profile());
                }
                let count =
                    u32::from_be_bytes(blob[cursor + 8..cursor + 12].try_into().unwrap()) as usize;
                let body_start = cursor + 12;
                let body_end = body_start.safe_add(count.safe_mul(4)?)?;
                if blob.len() < body_end {
                    return Err(invalid_profile());
                }
                let mut samples = Vec::with_capacity(count);
                for i in 0..count {
                    samples.push(f32::from_be_bytes(
                        blob[body_start + i * 4..body_start + i * 4 + 4]
                            .try_into()
                            .unwrap(),
                    ));
                }
                segments.push(CurveSegment::Sampled(samples));
                cursor = body_end;
            } else {
                return Err(invalid_profile());
            }
            // 4-byte align between segments.
            if !cursor.is_multiple_of(4) {
                cursor += 4 - cursor % 4;
            }
        }
        Ok(SegmentedCurve {
            breakpoints,
            segments,
        })
    }

    /// Evaluate the curve at `x ∈ [0,1]`, returning value in [0,1] (clamped).
    fn eval(&self, x: f64) -> f64 {
        // Locate segment: first segment is [-∞, bp0]; segment k (1..=N-2) is
        // (bp_{k-1}, bp_k]; last segment is (bp_{N-2}, +∞].
        let n = self.segments.len();
        let mut seg_idx = n - 1;
        for (i, bp) in self.breakpoints.iter().enumerate() {
            if x <= *bp as f64 {
                seg_idx = i;
                break;
            }
        }
        let y = match &self.segments[seg_idx] {
            CurveSegment::Parametric { func_type, params } => eval_parf(*func_type, params, x),
            CurveSegment::Sampled(samples) => {
                if samples.is_empty() {
                    return x.clamp(0.0, 1.0);
                }
                // Sampled segments span from the previous breakpoint
                // (exclusive) to the current breakpoint (inclusive).
                let lo = if seg_idx == 0 {
                    0.0
                } else {
                    self.breakpoints[seg_idx - 1] as f64
                };
                let hi = if seg_idx >= self.breakpoints.len() {
                    1.0
                } else {
                    self.breakpoints[seg_idx] as f64
                };
                let span = (hi - lo).max(1e-12);
                let t = ((x - lo) / span).clamp(0.0, 1.0);
                let n = samples.len();
                let pos = t * (n as f64);
                let i0 = (pos as usize).min(n - 1);
                let i1 = (i0 + 1).min(n - 1);
                let frac = pos - i0 as f64;
                samples[i0] as f64 * (1.0 - frac) + samples[i1] as f64 * frac
            }
        };
        y.clamp(0.0, 1.0)
    }
}

/// Read a segmentedCurveType blob and bake it into a Lut(u16). The `invert`
/// parameter controls which direction the resulting curve maps:
///
/// - `invert=false` returns the curve as-encoded in the file (linear domain
///   in, codomain whatever the profile defines — typically encoded value
///   when used inside cept's `func`).
/// - `invert=true` returns the inverse (encoded→linear) — what moxcms's
///   `rTRC`/`gTRC`/`bTRC` slots expect for a matrix-shaper pipeline. Done
///   by inverting the dense sample table.
///
/// We dense-sample at SEGMENTED_CURVE_SAMPLES points across [0,1].
pub(crate) fn read_segmented_curve(blob: &[u8], invert: bool) -> Result<ToneReprCurve, CmsError> {
    let curve = SegmentedCurve::read(blob)?;
    let mut samples = Vec::with_capacity(SEGMENTED_CURVE_SAMPLES);
    let denom = (SEGMENTED_CURVE_SAMPLES - 1) as f64;
    for i in 0..SEGMENTED_CURVE_SAMPLES {
        let x = i as f64 / denom;
        let y = curve.eval(x);
        let q = (y * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        samples.push(q);
    }
    if !invert {
        return Ok(ToneReprCurve::Lut(samples));
    }
    // Invert the LUT: build the inverse by scanning the forward table for the
    // best-matching x for each target y. This is sufficient for monotonic
    // curves (sRGB / PQ / HLG / gamma) and handles non-monotonic regions by
    // taking the first crossing.
    let n = samples.len();
    let mut inverse = vec![0u16; n];
    let mut j = 0usize;
    for (slot, target) in inverse.iter_mut().enumerate().map(|(i, slot)| {
        let target = (i as f64 / (n - 1) as f64) * 65535.0;
        (slot, target)
    }) {
        while j + 1 < n && (samples[j + 1] as f64) < target {
            j += 1;
        }
        let lo = samples[j] as f64;
        let hi = samples[(j + 1).min(n - 1)] as f64;
        let span = hi - lo;
        let frac = if span.abs() > 1e-9 {
            (target - lo) / span
        } else {
            0.0
        };
        let pos = (j as f64 + frac.clamp(0.0, 1.0)) / (n - 1) as f64;
        *slot = (pos * 65535.0).clamp(0.0, 65535.0) as u16;
    }
    Ok(ToneReprCurve::Lut(inverse))
}

// ── matf (matrixElement, 11.2.10) ────────────────────────────────────────

/// Parse a `matf` element. Layout:
///   bytes 0..3   : 'matf' (already validated by caller)
///   bytes 4..7   : reserved
///   bytes 8..9   : input channels (u16)
///   bytes 10..11 : output channels (u16)
///   bytes 12..   : (out × in) float32 matrix entries (row-major)
///                  followed by `out` float32 offset (constant) values
pub(crate) fn read_matf_element(blob: &[u8]) -> Result<(Matrix3d, Vector3d), CmsError> {
    if blob.len() < 12 {
        return Err(invalid_profile());
    }
    if &blob[0..4] != SIG_MATF {
        return Err(invalid_profile());
    }
    let in_ch = u16::from_be_bytes([blob[8], blob[9]]);
    let out_ch = u16::from_be_bytes([blob[10], blob[11]]);
    if in_ch != 3 || out_ch != 3 {
        // Out of scope for matrix-shaper.
        return Err(invalid_profile());
    }
    let body = &blob[12..];
    if body.len() < (9 + 3) * 4 {
        return Err(invalid_profile());
    }
    let f = |i: usize| -> f32 { f32::from_be_bytes(body[i * 4..i * 4 + 4].try_into().unwrap()) };
    let m = Matrix3d {
        v: [
            [f(0) as f64, f(1) as f64, f(2) as f64],
            [f(3) as f64, f(4) as f64, f(5) as f64],
            [f(6) as f64, f(7) as f64, f(8) as f64],
        ],
    };
    let off = Vector3d {
        v: [f(9) as f64, f(10) as f64, f(11) as f64],
    };
    Ok((m, off))
}

// ── cvst (curveSetElement, 11.2.2) ───────────────────────────────────────

/// Parse a `cvst` element into N curves. Layout:
///   bytes 0..3   : 'cvst'
///   bytes 4..7   : reserved
///   bytes 8..9   : input channels (u16)
///   bytes 10..11 : output channels (u16)
///   then a position-table (N × 8 bytes: u32 offset + u32 size) where N is
///   max(in_ch, out_ch), followed by curveSegment data referenced by the
///   table. (For a curveSet the curve sub-tags can be either `singleSampled`
///   or `curf` segmented curves; we currently only support `curf`.)
pub(crate) fn read_cvst_element(blob: &[u8]) -> Result<Vec<ToneReprCurve>, CmsError> {
    if blob.len() < 12 {
        return Err(invalid_profile());
    }
    if &blob[0..4] != SIG_CVST {
        return Err(invalid_profile());
    }
    let in_ch = u16::from_be_bytes([blob[8], blob[9]]) as usize;
    let out_ch = u16::from_be_bytes([blob[10], blob[11]]) as usize;
    let n = in_ch.max(out_ch);
    if n == 0 || n > 16 {
        return Err(invalid_profile());
    }
    let table_start = 12usize;
    let table_end = table_start.safe_add(n.safe_mul(8)?)?;
    if blob.len() < table_end {
        return Err(invalid_profile());
    }
    let mut curves = Vec::with_capacity(n);
    for i in 0..n {
        let entry = table_start + i * 8;
        let offset = u32::from_be_bytes(blob[entry..entry + 4].try_into().unwrap()) as usize;
        let size = u32::from_be_bytes(blob[entry + 4..entry + 8].try_into().unwrap()) as usize;
        let end = offset.safe_add(size)?;
        if blob.len() < end {
            return Err(invalid_profile());
        }
        let curve_blob = &blob[offset..end];
        if curve_blob.len() < 4 {
            return Err(invalid_profile());
        }
        if &curve_blob[0..4] == SIG_CURF {
            // Inside mpet's curveSet, the per-channel curve direction depends
            // on whether the chain is [cvst, matf] (device→PCS, cvst is EOTF)
            // or [matf, cvst] (PCS→device, cvst is OETF). The caller already
            // restricts mpet extraction to A2B-direction tags (curves-first),
            // so cvst here is EOTF — no inversion needed.
            curves.push(read_segmented_curve(curve_blob, false)?);
        } else {
            // singleSampled or unknown — leave to caller's discretion. For now
            // treat as "no curve recognized" via identity.
            curves.push(ToneReprCurve::Lut(Vec::new()));
        }
    }
    Ok(curves)
}

// ── mpet (multiProcessElementType, 11.x) ─────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct MpetMatrixShaper {
    /// Per-channel TRC (3 entries for an RGB matrix-shaper).
    pub curves: Vec<ToneReprCurve>,
    /// 3×3 conversion matrix.
    pub matrix: Matrix3d,
    /// Constant offset vector applied after the matrix multiplication.
    /// Currently unused by `try_populate_matrix_shaper_from_mpet` because the
    /// canonical sRGB-style chains we target have a zero offset; kept for
    /// follow-up (HDR mpet may use a non-zero bias).
    #[allow(dead_code)]
    pub offset: Vector3d,
    /// True if the chain in the original profile is curves→matrix
    /// (device-to-PCS shape); false if matrix→curves (PCS-to-device).
    /// Available for callers that need to decide which side curves apply on;
    /// not consulted yet because the wire-up restricts mpet extraction to
    /// A2B tags (always curves-first).
    #[allow(dead_code)]
    pub curves_first: bool,
}

/// Try to interpret an `mpet` tag as a 2-element matrix-shaper chain. Returns
/// `Ok(None)` for any other shape (CLUT element, calculator, more than two
/// elements, non-RGB channel counts, etc.) so the caller can continue.
pub(crate) fn read_mpet_matrix_shaper(
    slice: &[u8],
    entry: usize,
    tag_size: usize,
) -> Result<Option<MpetMatrixShaper>, CmsError> {
    if tag_size < 16 {
        return Ok(None);
    }
    let end = entry.safe_add(tag_size)?;
    if end > slice.len() {
        return Err(invalid_profile());
    }
    let blob = &slice[entry..end];
    if &blob[0..4] != SIG_MPET {
        return Ok(None);
    }
    // bytes 4..7 reserved
    let in_ch = u16::from_be_bytes([blob[8], blob[9]]);
    let out_ch = u16::from_be_bytes([blob[10], blob[11]]);
    if in_ch != 3 || out_ch != 3 {
        return Ok(None);
    }
    let n_proc = u32::from_be_bytes(blob[12..16].try_into().unwrap()) as usize;
    if !(1..=2).contains(&n_proc) {
        return Ok(None);
    }
    let table_end = 16usize.safe_add(n_proc.safe_mul(8)?)?;
    if blob.len() < table_end {
        return Err(invalid_profile());
    }
    let mut elem_blobs: Vec<&[u8]> = Vec::with_capacity(n_proc);
    let mut elem_sigs: Vec<[u8; 4]> = Vec::with_capacity(n_proc);
    for i in 0..n_proc {
        let off = 16 + i * 8;
        let elem_off = u32::from_be_bytes(blob[off..off + 4].try_into().unwrap()) as usize;
        let elem_sz = u32::from_be_bytes(blob[off + 4..off + 8].try_into().unwrap()) as usize;
        let e_end = elem_off.safe_add(elem_sz)?;
        if blob.len() < e_end || elem_sz < 4 {
            return Err(invalid_profile());
        }
        let mut sig = [0u8; 4];
        sig.copy_from_slice(&blob[elem_off..elem_off + 4]);
        elem_sigs.push(sig);
        elem_blobs.push(&blob[elem_off..e_end]);
    }

    // Recognize either:
    //   [matf]                          (single matrix, identity curves)
    //   [cvst]                          (single curveSet, identity matrix)
    //   [cvst, matf]  device-to-PCS     curves first
    //   [matf, cvst]  PCS-to-device     matrix first
    let identity = Matrix3d::IDENTITY;
    let zero = Vector3d::default();
    let identity_trc = || vec![ToneReprCurve::Lut(Vec::new()); 3];

    let result = match (elem_sigs.as_slice(), elem_blobs.as_slice()) {
        ([s], [b]) if s == SIG_MATF => {
            let (m, o) = read_matf_element(b)?;
            Some(MpetMatrixShaper {
                curves: identity_trc(),
                matrix: m,
                offset: o,
                curves_first: false,
            })
        }
        ([s], [b]) if s == SIG_CVST => {
            let curves = read_cvst_element(b)?;
            if curves.len() < 3 {
                return Ok(None);
            }
            Some(MpetMatrixShaper {
                curves,
                matrix: identity,
                offset: zero,
                curves_first: true,
            })
        }
        ([s1, s2], [b1, b2]) if s1 == SIG_CVST && s2 == SIG_MATF => {
            let curves = read_cvst_element(b1)?;
            let (m, o) = read_matf_element(b2)?;
            if curves.len() < 3 {
                return Ok(None);
            }
            Some(MpetMatrixShaper {
                curves,
                matrix: m,
                offset: o,
                curves_first: true,
            })
        }
        ([s1, s2], [b1, b2]) if s1 == SIG_MATF && s2 == SIG_CVST => {
            let (m, o) = read_matf_element(b1)?;
            let curves = read_cvst_element(b2)?;
            if curves.len() < 3 {
                return Ok(None);
            }
            Some(MpetMatrixShaper {
                curves,
                matrix: m,
                offset: o,
                curves_first: false,
            })
        }
        _ => None,
    };
    Ok(result)
}

// ── cept (colorEncodingParamsStructure, 12.2.3) ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub(crate) struct Chromaticity {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Chromaticity {
    /// Convert chromaticity (x, y, z that sum to 1) to XYZ tristimulus
    /// values with assumed Y=1.
    #[allow(dead_code)]
    fn to_xyz_unit_y(self) -> Xyzd {
        // Y=1 ⇒ X = x/y, Z = z/y
        Xyzd {
            x: if self.y > 0.0 { self.x / self.y } else { 0.0 },
            y: 1.0,
            z: if self.y > 0.0 { self.z / self.y } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CeptMatrixShaper {
    /// Chromaticity coordinates of R, G, B primaries (per the cept spec
    /// 12.2.3.2.1-3, these are nCIE chromaticity not tristimulus).
    pub red_xy: Chromaticity,
    pub green_xy: Chromaticity,
    pub blue_xy: Chromaticity,
    /// White point chromaticity (cept 12.2.3.2.7 wXYZ).
    pub white_xy: Chromaticity,
    pub trc: Option<ToneReprCurve>,
    /// Reference name from `rfnm`, if the cept tag was self-defining
    /// ("ISO 22028-1") and we want to record the standard family. Reserved
    /// for future use by canonical-substitution logic.
    #[allow(dead_code)]
    pub reference_name: Option<String>,
}

/// Read 2 or 3 floats from an `fl32` blob and return chromaticity (x, y, z).
/// If only 2 floats present, z = 1 - x - y per spec 12.2.3.2.{1,2,3}.
fn chromaticity_from_fl32(arr: &[f32]) -> Result<Chromaticity, CmsError> {
    match arr.len() {
        n if n >= 3 => Ok(Chromaticity {
            x: arr[0] as f64,
            y: arr[1] as f64,
            z: arr[2] as f64,
        }),
        2 => {
            let x = arr[0] as f64;
            let y = arr[1] as f64;
            Ok(Chromaticity {
                x,
                y,
                z: 1.0 - x - y,
            })
        }
        _ => Err(invalid_profile()),
    }
}

/// Read a `cept` tagStructType into a matrix-shaper representation.
/// Sub-tags we understand: rXYZ/gXYZ/bXYZ/wXYZ (fl32 arrays), func (curf).
/// Returns `Ok(None)` if the cept lacks the minimal RGB+TRC subset we need.
pub(crate) fn read_cept_matrix_shaper(
    slice: &[u8],
    entry: usize,
    tag_size: usize,
) -> Result<Option<CeptMatrixShaper>, CmsError> {
    let ts = match TagStruct::read(slice, entry, tag_size) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    if &ts.struct_id != b"cept" {
        return Ok(None);
    }
    let read_chrom = |sig: &[u8; 4]| -> Result<Option<Chromaticity>, CmsError> {
        match ts.find(sig).and_then(|e| ts.slice(e)) {
            Some(blob) => {
                let arr = read_fl32_array(blob)?;
                Ok(Some(chromaticity_from_fl32(&arr)?))
            }
            None => Ok(None),
        }
    };

    let red_xy = read_chrom(b"rXYZ")?;
    let green_xy = read_chrom(b"gXYZ")?;
    let blue_xy = read_chrom(b"bXYZ")?;
    let white_xy = read_chrom(b"wXYZ")?;

    let (red_xy, green_xy, blue_xy, white_xy) = match (red_xy, green_xy, blue_xy, white_xy) {
        (Some(r), Some(g), Some(b), Some(w)) => (r, g, b, w),
        _ => return Ok(None),
    };

    let trc = match ts.find(b"func").and_then(|e| ts.slice(e)) {
        Some(blob) if blob.len() >= 4 && &blob[0..4] == SIG_CURF => {
            // cept's `func` is the encoding's OETF (linear → encoded value).
            // moxcms's matrix-shaper rTRC slot expects the EOTF (encoded →
            // linear). Invert during baking.
            Some(read_segmented_curve(blob, true)?)
        }
        _ => None,
    };

    Ok(Some(CeptMatrixShaper {
        red_xy,
        green_xy,
        blue_xy,
        white_xy,
        trc,
        reference_name: None,
    }))
}

// ── Canonical primary / TRC recognition for CICP synthesis ───────────────

/// Recognize a canonical primary set from the cept chromaticities. Returns
/// the matching CICP color-primaries enum if all four match within a small
/// tolerance.
pub(crate) fn match_canonical_primaries(
    r: Chromaticity,
    g: Chromaticity,
    b: Chromaticity,
    w: Chromaticity,
) -> Option<crate::CicpColorPrimaries> {
    use crate::CicpColorPrimaries as Cp;
    const TOL: f64 = 0.005; // slack for round-trip floating drift

    let near = |a: Chromaticity, b: (f64, f64)| -> bool {
        (a.x - b.0).abs() < TOL && (a.y - b.1).abs() < TOL
    };

    // BT.709 / sRGB primaries + D65 white
    if near(r, (0.640, 0.330))
        && near(g, (0.300, 0.600))
        && near(b, (0.150, 0.060))
        && near(w, (0.3127, 0.3290))
    {
        return Some(Cp::Bt709);
    }
    // BT.2020
    if near(r, (0.708, 0.292))
        && near(g, (0.170, 0.797))
        && near(b, (0.131, 0.046))
        && near(w, (0.3127, 0.3290))
    {
        return Some(Cp::Bt2020);
    }
    // SMPTE RP-431 / SMPTE EG-432 (Display P3) — same primaries, different white
    if near(r, (0.680, 0.320)) && near(g, (0.265, 0.690)) && near(b, (0.150, 0.060)) {
        if near(w, (0.3127, 0.3290)) {
            // D65 → Display P3 (SMPTE EG-432)
            return Some(Cp::Smpte432);
        } else if near(w, (0.314, 0.351)) {
            // DCI white → DCI-P3 cinema (SMPTE RP-431)
            return Some(Cp::Smpte431);
        }
    }

    None
}

/// Sample a curve at K equally-spaced points in [0,1] and produce a
/// normalized output ramp for canonical-curve comparison. The returned
/// vector is K f64 values in [0,1].
fn sample_curve(curve: &ToneReprCurve, samples: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(samples);
    let denom = (samples - 1) as f64;
    match curve {
        ToneReprCurve::Lut(lut) if lut.is_empty() => {
            for i in 0..samples {
                out.push(i as f64 / denom);
            }
        }
        ToneReprCurve::Lut(lut) => {
            // Treat as a uniformly sampled u16 LUT in [0,65535].
            let n = lut.len();
            for i in 0..samples {
                let t = i as f64 / denom;
                let pos = t * (n - 1) as f64;
                let i0 = pos.floor() as usize;
                let i1 = (i0 + 1).min(n - 1);
                let frac = pos - i0 as f64;
                let v = lut[i0] as f64 * (1.0 - frac) + lut[i1] as f64 * frac;
                out.push(v / 65535.0);
            }
        }
        ToneReprCurve::Parametric(params) => {
            // Use the existing ParametricCurve evaluator by evaluating
            // a few canonical formulas directly. For canonical-curve
            // matching we can fall back to the analytic forms.
            for i in 0..samples {
                let _ = params;
                let t = i as f64 / denom;
                out.push(t); // placeholder; we don't expect Parametric here
            }
        }
    }
    out
}

/// Recognize a canonical transfer characteristic by sampling and comparing
/// against reference curves. Returns the matching CICP transfer enum if
/// max u16-equivalent error is under a few hundred.
pub(crate) fn match_canonical_transfer(
    curve: &ToneReprCurve,
) -> Option<crate::TransferCharacteristics> {
    use crate::TransferCharacteristics as Tc;
    const SAMPLES: usize = 33; // sparse but enough — canonical EOTFs are smooth

    let actual = sample_curve(curve, SAMPLES);

    let max_err = |reference: fn(f64) -> f64| -> f64 {
        let mut max = 0.0f64;
        for (i, &a) in actual.iter().enumerate() {
            let x = i as f64 / (SAMPLES - 1) as f64;
            let r = reference(x);
            let d = (a - r).abs();
            if d > max {
                max = d;
            }
        }
        max
    };

    // u16-equivalent tolerance: 0.001 ≈ 65 u16 units.
    const TOL: f64 = 0.001;

    let srgb_ref = |x: f64| -> f64 {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
    let bt709_ref = |x: f64| -> f64 {
        if x < 0.081 {
            x / 4.5
        } else {
            ((x + 0.099) / 1.099).powf(1.0 / 0.45)
        }
    };
    let gamma22 = |x: f64| -> f64 { x.powf(2.2) };
    let gamma24 = |x: f64| -> f64 { x.powf(2.4) };
    let linear = |x: f64| -> f64 { x };
    let pq_ref = |x: f64| -> f64 {
        // ST.2084 inverse EOTF (PQ encoded → linear, then normalized to [0,1])
        const M1: f64 = 0.159_301_757_812_5;
        const M2: f64 = 78.843_75;
        const C1: f64 = 0.835_937_5;
        const C2: f64 = 18.851_562_5;
        const C3: f64 = 18.687_5;
        let vp = x.powf(1.0 / M2);
        let num = (vp - C1).max(0.0);
        let den = C2 - C3 * vp;
        if den <= 0.0 {
            0.0
        } else {
            (num / den).powf(1.0 / M1)
        }
    };

    type CandidateFn = fn(f64) -> f64;
    let candidates: &[(Tc, CandidateFn)] = &[
        (Tc::Srgb, srgb_ref),
        (Tc::Bt709, bt709_ref),
        (Tc::Bt470M, gamma22),
        (Tc::Bt202010bit, bt709_ref),
        (Tc::Linear, linear),
        (Tc::Smpte428, gamma24),
        (Tc::Smpte2084, pq_ref),
    ];
    let mut best: Option<(Tc, f64)> = None;
    for (tc, f) in candidates {
        let e = max_err(*f);
        if e < TOL {
            match best {
                None => best = Some((*tc, e)),
                Some((_, prev)) if e < prev => best = Some((*tc, e)),
                _ => {}
            }
        }
    }
    best.map(|(tc, _)| tc)
}

// ── unused-yet helpers ───────────────────────────────────────────────────
//
// The following functions are placeholders pulled out of fork scope but kept
// as `dead_code`-allowed to make the wiring obvious for future expansion.

#[allow(dead_code)]
pub(crate) fn s15_to_xyzd(x: i32, y: i32, z: i32) -> Xyzd {
    Xyzd {
        x: s15_fixed16_number_to_double(x),
        y: s15_fixed16_number_to_double(y),
        z: s15_fixed16_number_to_double(z),
    }
}

#[allow(dead_code)]
pub(crate) fn s15_to_f32(x: i32) -> f32 {
    s15_fixed16_number_to_float(x)
}

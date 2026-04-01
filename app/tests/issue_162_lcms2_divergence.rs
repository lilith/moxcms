//! Minimal reproduction for moxcms#162 / zenpipe#15.
//!
//! The embedded ICC profile is a v4 Apple Wide Color scanner profile
//! (ICC-RGB v4.0, scnr, PCS=XYZ, 30252 bytes) that uses only A2B LUTs
//! — no matrix-shaper fallback. Both moxcms and lcms2 are given the
//! same profile and the same exhaustive 256³ RGB input, then compared.
//!
//! Run: cargo test --release -p app issue_162 -- --nocapture

use lcms2::{Intent, PixelFormat, Profile, Transform as LcmsTransform};
use moxcms::{ColorProfile, InterpolationMethod, Layout, TransformOptions};

/// The ICC profile embedded in wmc_d4e6bfcba7ee8f83.jpg (Apple Wide Color, v4 A2B-only).
const ICC_BYTES: &[u8] = include_bytes!("../../assets/Apple_Wide_Color.icc");

struct Stats {
    max: [i32; 3],
    p99: [i32; 3],
    avg: [f64; 3],
    above_2: usize,
    total: usize,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max={}/{}/{} p99={}/{}/{} avg={:.2}/{:.2}/{:.2} above_2={}/{}",
            self.max[0],
            self.max[1],
            self.max[2],
            self.p99[0],
            self.p99[1],
            self.p99[2],
            self.avg[0],
            self.avg[1],
            self.avg[2],
            self.above_2,
            self.total,
        )
    }
}

fn compare(a: &[u8], b: &[u8], ch: usize) -> Stats {
    let n = a.len() / ch;
    let mut max = [0i32; 3];
    let mut sum = [0u64; 3];
    let mut above_2 = 0usize;
    let mut hist = [[0u64; 256]; 3];
    for (pa, pb) in a.chunks_exact(ch).zip(b.chunks_exact(ch)) {
        let mut px_max = 0i32;
        for c in 0..3 {
            let d = (pa[c] as i32 - pb[c] as i32).abs();
            max[c] = max[c].max(d);
            sum[c] += d as u64;
            hist[c][d as usize] += 1;
            px_max = px_max.max(d);
        }
        if px_max > 2 {
            above_2 += 1;
        }
    }
    let avg = std::array::from_fn(|c| sum[c] as f64 / n as f64);
    let p99 = std::array::from_fn(|c| {
        let thr = (n as f64 * 0.99).ceil() as u64;
        let mut cum = 0u64;
        for (v, &cnt) in hist[c].iter().enumerate() {
            cum += cnt;
            if cum >= thr {
                return v as i32;
            }
        }
        255
    });
    Stats {
        max,
        p99,
        avg,
        above_2,
        total: n,
    }
}

/// Build exhaustive 256³ RGB source (48 MiB).
fn all_rgb() -> Vec<u8> {
    let n: usize = 256 * 256 * 256;
    let mut v = Vec::with_capacity(n * 3);
    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for b in 0..=255u8 {
                v.push(r);
                v.push(g);
                v.push(b);
            }
        }
    }
    v
}

fn moxcms_transform(src: &[u8], fixed: bool, interp: InterpolationMethod) -> Vec<u8> {
    let prof = ColorProfile::new_from_slice(ICC_BYTES).unwrap();
    let srgb = ColorProfile::new_srgb();
    let t = prof
        .create_transform_8bit(
            Layout::Rgb,
            &srgb,
            Layout::Rgba,
            TransformOptions {
                prefer_fixed_point: fixed,
                allow_use_cicp_transfer: false,
                interpolation_method: interp,
                ..Default::default()
            },
        )
        .unwrap();
    let mut dst = vec![0u8; (src.len() / 3) * 4];
    t.transform(src, &mut dst).unwrap();
    dst
}

fn lcms2_transform(src: &[u8]) -> Vec<u8> {
    let sp = Profile::new_icc(ICC_BYTES).unwrap();
    let dp = Profile::new_srgb();
    let t = LcmsTransform::new(
        &sp,
        PixelFormat::RGB_8,
        &dp,
        PixelFormat::RGBA_8,
        Intent::Perceptual,
    )
    .unwrap();
    let mut dst = vec![0u8; (src.len() / 3) * 4];
    t.transform_pixels(src, &mut dst);
    dst
}

#[test]
fn issue_162_moxcms_default_vs_lcms2() {
    let src = all_rgb();
    let mox = moxcms_transform(&src, true, InterpolationMethod::Linear);
    let lcm = lcms2_transform(&src);
    let s = compare(&mox, &lcm, 4);
    eprintln!("moxcms default (fixed trilinear) vs lcms2: {s}");
    // This demonstrates the divergence — max ~14-18 on some channels.
}

#[test]
fn issue_162_moxcms_float_tetra_vs_lcms2() {
    let src = all_rgb();
    let mox = moxcms_transform(&src, false, InterpolationMethod::Tetrahedral);
    let lcm = lcms2_transform(&src);
    let s = compare(&mox, &lcm, 4);
    eprintln!("moxcms float tetrahedral vs lcms2: {s}");
    // Still shows similar max — the divergence is in LUT interpretation, not precision.
}

#[test]
fn issue_162_moxcms_internal_fixed_vs_float() {
    let src = all_rgb();
    let fixed = moxcms_transform(&src, true, InterpolationMethod::Linear);
    let float = moxcms_transform(&src, false, InterpolationMethod::Linear);
    let s = compare(&fixed, &float, 4);
    eprintln!("moxcms internal (fixed trilinear vs float trilinear): {s}");
    // Max ~2 — moxcms is internally consistent.
}

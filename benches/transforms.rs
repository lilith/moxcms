// zenbench baseline for the transform paths the asm/unsafe audits pin.
// Produces paired, interleaved measurements across the four interpolator
// variants × two barycentric scales × three profile topologies
// (CMYK→sRGB, sRGB→CMYK, synthetic RGB-LUT→sRGB). Numbers land in
// `benchmarks/` so later safety refactors can be compared head-to-head
// on the same hardware.

use moxcms::{
    BarycentricWeightScale, ColorProfile, DataColorSpace, InterpolationMethod, Layout, LutDataType,
    LutStore, LutType, LutWarehouse, Matrix3d, TransformOptions,
};
use zenbench::Throughput;
use zenbench::black_box;

// 1024 pixels per transform call: large enough to amortize per-call
// setup, small enough to fit in L1 on most runners.
const N_PIXELS: usize = 1024;

fn sample_bytes(n: usize) -> Vec<u8> {
    // Deterministic Knuth-multiplicative hash — same input for every
    // comparison group, no RNG overhead.
    (0..n)
        .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_shr(24) as u8)
        .collect()
}

fn synthetic_rgb_lut_profile() -> ColorProfile {
    const GRID: u8 = 9;
    const TABLE_ENTRIES: u16 = 256;

    let ramp: Vec<u16> = (0..3)
        .flat_map(|_| (0..TABLE_ENTRIES).map(|i| (i as u32 * 257) as u16))
        .collect();

    let scale_factor = 65535u32 / (GRID - 1) as u32;
    let mut clut: Vec<u16> = Vec::with_capacity((GRID as usize).pow(3) * 3);
    for r in 0..GRID {
        for g in 0..GRID {
            for b in 0..GRID {
                clut.push((r as u32 * scale_factor) as u16);
                clut.push((g as u32 * scale_factor) as u16);
                clut.push((b as u32 * scale_factor) as u16);
            }
        }
    }

    let lut = LutDataType {
        num_input_channels: 3,
        num_output_channels: 3,
        num_clut_grid_points: GRID,
        matrix: Matrix3d::IDENTITY,
        num_input_table_entries: TABLE_ENTRIES,
        num_output_table_entries: TABLE_ENTRIES,
        input_table: LutStore::Store16(ramp.clone()),
        clut_table: LutStore::Store16(clut),
        output_table: LutStore::Store16(ramp),
        lut_type: LutType::Lut16,
    };

    let mut profile = ColorProfile::new_srgb();
    profile.pcs = DataColorSpace::Lab;
    profile.lut_a_to_b_perceptual = Some(LutWarehouse::Lut(lut.clone()));
    profile.lut_a_to_b_colorimetric = Some(LutWarehouse::Lut(lut));
    profile
}

const METHODS: &[(&str, InterpolationMethod)] = &[
    ("tetra", InterpolationMethod::Tetrahedral),
    ("pyramid", InterpolationMethod::Pyramid),
    ("prism", InterpolationMethod::Prism),
    ("linear", InterpolationMethod::Linear),
];

const SCALES: &[(&str, BarycentricWeightScale)] = &[
    ("lo", BarycentricWeightScale::Low),
    ("hi", BarycentricWeightScale::High),
];

zenbench::main!(|suite| {
    let cmyk_profile = std::fs::read("./assets/us_swop_coated.icc")
        .ok()
        .map(|b| ColorProfile::new_from_slice(&b).unwrap());

    // --- Group 1: CMYK → sRGB via Lut4x3 ------------------------------
    if let Some(cmyk) = cmyk_profile.clone() {
        let srgb = ColorProfile::new_srgb();
        suite.compare("cmyk_to_srgb", |group| {
            group.throughput(Throughput::Elements(N_PIXELS as u64));
            group.throughput_unit("pixels");

            for &(method_name, method) in METHODS {
                for &(scale_name, scale) in SCALES {
                    let name = format!("{method_name}_{scale_name}");
                    let cmyk = cmyk.clone();
                    let srgb = srgb.clone();
                    group.bench(&name, move |b| {
                        let opts = TransformOptions {
                            interpolation_method: method,
                            barycentric_weight_scale: scale,
                            ..Default::default()
                        };
                        let transform = cmyk
                            .create_transform_8bit(Layout::Rgba, &srgb, Layout::Rgb, opts)
                            .unwrap();
                        let src = sample_bytes(N_PIXELS * 4);
                        let mut dst = vec![0u8; N_PIXELS * 3];
                        b.iter(|| {
                            transform
                                .transform(black_box(&src), black_box(&mut dst))
                                .unwrap();
                            black_box(&dst);
                        })
                    });
                }
            }
        });
    }

    // --- Group 2: sRGB → CMYK via Lut3x4 ------------------------------
    if let Some(cmyk) = cmyk_profile {
        let srgb = ColorProfile::new_srgb();
        suite.compare("srgb_to_cmyk", |group| {
            group.throughput(Throughput::Elements(N_PIXELS as u64));
            group.throughput_unit("pixels");

            for &(method_name, method) in METHODS {
                for &(scale_name, scale) in SCALES {
                    let name = format!("{method_name}_{scale_name}");
                    let cmyk = cmyk.clone();
                    let srgb = srgb.clone();
                    group.bench(&name, move |b| {
                        let opts = TransformOptions {
                            interpolation_method: method,
                            barycentric_weight_scale: scale,
                            ..Default::default()
                        };
                        let transform = srgb
                            .create_transform_8bit(Layout::Rgb, &cmyk, Layout::Rgba, opts)
                            .unwrap();
                        let src = sample_bytes(N_PIXELS * 3);
                        let mut dst = vec![0u8; N_PIXELS * 4];
                        b.iter(|| {
                            transform
                                .transform(black_box(&src), black_box(&mut dst))
                                .unwrap();
                            black_box(&dst);
                        })
                    });
                }
            }
        });
    }

    // --- Group 3: synthetic RGB-LUT → sRGB via Lut3x3 -----------------
    {
        let src_lut = synthetic_rgb_lut_profile();
        let dst_srgb = ColorProfile::new_srgb();
        suite.compare("rgb_lut_to_srgb", |group| {
            group.throughput(Throughput::Elements(N_PIXELS as u64));
            group.throughput_unit("pixels");

            for &(method_name, method) in METHODS {
                for &(scale_name, scale) in SCALES {
                    let name = format!("{method_name}_{scale_name}");
                    let src_lut = src_lut.clone();
                    let dst_srgb = dst_srgb.clone();
                    group.bench(&name, move |b| {
                        let opts = TransformOptions {
                            interpolation_method: method,
                            barycentric_weight_scale: scale,
                            ..Default::default()
                        };
                        let transform = src_lut
                            .create_transform_8bit(Layout::Rgb, &dst_srgb, Layout::Rgb, opts)
                            .unwrap();
                        let src = sample_bytes(N_PIXELS * 3);
                        let mut dst = vec![0u8; N_PIXELS * 3];
                        b.iter(|| {
                            transform
                                .transform(black_box(&src), black_box(&mut dst))
                                .unwrap();
                            black_box(&dst);
                        })
                    });
                }
            }
        });
    }
});

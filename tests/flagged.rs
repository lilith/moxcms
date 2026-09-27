//! Regression tests for 9 real-world ICC profiles that lcms2 accepts but
//! moxcms currently rejects, identified by the corpus-builder cross-CMS
//! report on 2026-04-16. See assets/flagged/README.md for details.

use moxcms::ColorProfile;

const FLAGGED: &[&str] = &[
    // skcms profiles/color.org/ — ICC.org canonical sRGB reference profiles
    "assets/flagged/sRGB_D65_MAT.icc",
    "assets/flagged/sRGB_D65_colorimetric.icc",
    "assets/flagged/sRGB_ISO22028.icc",
    // skcms profiles/misc/ — real-world profiles Skia uses for test coverage
    "assets/flagged/AdobeColorSpin.icc",
    "assets/flagged/ColorGATE_Sihl_PhotoPaper.icc",
    // OS-bundled Lab profiles (named-color palettes)
    "assets/flagged/installed-linux-arm64-Crayons.icc",
    "assets/flagged/installed-linux-arm64-x11-colors.icc",
    "assets/flagged/installed-linux-x64-Crayons.icc",
    "assets/flagged/installed-linux-x64-x11-colors.icc",
];

#[test]
fn report_flagged_parse_results() {
    let mut fails = Vec::new();
    for path in FLAGGED {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        match ColorProfile::new_from_slice(&data) {
            Ok(_) => println!("OK    {path}"),
            Err(e) => {
                println!("FAIL  {path}  {e:?}");
                fails.push((*path, format!("{e:?}")));
            }
        }
    }
    if !fails.is_empty() {
        for (p, e) in &fails {
            eprintln!("{p}: {e}");
        }
    }
}

// Individual parse assertions — only meaningful under the `permissive`
// feature (the default in this fork). In strict mode these profiles are
// expected to fail, so gate them.
#[cfg(feature = "permissive")]
#[test]
fn srgb_d65_mat_parses() {
    let data = std::fs::read("assets/flagged/sRGB_D65_MAT.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("sRGB_D65_MAT should parse");
}

#[cfg(feature = "permissive")]
#[test]
fn srgb_d65_colorimetric_parses() {
    let data = std::fs::read("assets/flagged/sRGB_D65_colorimetric.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("sRGB_D65_colorimetric should parse");
}

#[cfg(feature = "permissive")]
#[test]
fn srgb_iso22028_parses() {
    let data = std::fs::read("assets/flagged/sRGB_ISO22028.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("sRGB_ISO22028 should parse");
}

#[cfg(feature = "permissive")]
#[test]
fn adobe_color_spin_parses() {
    let data = std::fs::read("assets/flagged/AdobeColorSpin.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("AdobeColorSpin should parse");
}

#[cfg(feature = "permissive")]
#[test]
fn colorgate_sihl_photopaper_parses() {
    let data = std::fs::read("assets/flagged/ColorGATE_Sihl_PhotoPaper.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("ColorGATE_Sihl_PhotoPaper should parse");
}

#[cfg(feature = "permissive")]
#[test]
fn crayons_arm64_parses() {
    let data = std::fs::read("assets/flagged/installed-linux-arm64-Crayons.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("Crayons (arm64) should parse");
}

#[cfg(feature = "permissive")]
#[test]
fn crayons_x64_parses() {
    let data = std::fs::read("assets/flagged/installed-linux-x64-Crayons.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("Crayons (x64) should parse");
}

#[cfg(feature = "permissive")]
#[test]
fn x11_colors_arm64_parses() {
    let data = std::fs::read("assets/flagged/installed-linux-arm64-x11-colors.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("x11-colors (arm64) should parse");
}

#[cfg(feature = "permissive")]
#[test]
fn x11_colors_x64_parses() {
    let data = std::fs::read("assets/flagged/installed-linux-x64-x11-colors.icc").unwrap();
    ColorProfile::new_from_slice(&data).expect("x11-colors (x64) should parse");
}

/// Strict mode: every flagged profile is expected to fail to parse.
/// Guards against accidentally relaxing strict mode on upstream sync.
#[cfg(not(feature = "permissive"))]
#[test]
fn strict_mode_rejects_all_flagged() {
    let mut unexpected_ok = Vec::new();
    for path in FLAGGED {
        let data = std::fs::read(path).unwrap();
        if ColorProfile::new_from_slice(&data).is_ok() {
            unexpected_ok.push(*path);
        }
    }
    assert!(
        unexpected_ok.is_empty(),
        "strict mode accepted profiles it shouldn't: {unexpected_ok:?}"
    );
}

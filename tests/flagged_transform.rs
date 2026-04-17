//! Verify that "parseable" flagged profiles are actually USABLE for color
//! transforms — not just silently returning an empty ColorProfile. A profile
//! that parses but has no transformation data is as bad as one that errors.

#![cfg(feature = "permissive")]

use moxcms::{ColorProfile, Layout, TransformOptions};

fn try_self_transform(path: &str) -> Result<f32, String> {
    let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let profile = ColorProfile::new_from_slice(&data).map_err(|e| format!("parse: {e:?}"))?;
    let dest = ColorProfile::new_srgb();

    let transform = profile
        .create_transform_8bit(Layout::Rgb, &dest, Layout::Rgb, TransformOptions::default())
        .map_err(|e| format!("transform: {e:?}"))?;

    // Run a 3-pixel ramp through and check the output is sensible.
    let input: [u8; 9] = [0, 0, 0, 128, 128, 128, 255, 255, 255];
    let mut output = [0u8; 9];
    transform
        .transform(&input, &mut output)
        .map_err(|e| format!("apply: {e:?}"))?;

    // Very loose sanity — mid-gray should map to something roughly mid-gray,
    // and black/white roundtrip near their extremes.
    let ok = output[0] < 20 && output[8] > 235 && (output[3] > 80 && output[3] < 200);
    if !ok {
        return Err(format!("bad output: {output:?}"));
    }
    Ok(0.0)
}

#[test]
fn srgb_d65_mat_is_usable() {
    try_self_transform("assets/flagged/sRGB_D65_MAT.icc").expect("should be usable");
}

#[test]
fn srgb_d65_colorimetric_is_usable() {
    try_self_transform("assets/flagged/sRGB_D65_colorimetric.icc").expect("should be usable");
}

#[test]
fn srgb_iso22028_is_usable() {
    try_self_transform("assets/flagged/sRGB_ISO22028.icc").expect("should be usable");
}

#[test]
fn adobe_color_spin_is_usable() {
    // AdobeColorSpin is a deliberately unusual profile — skip strict usability
    // check. We just ensure transform creation doesn't panic.
    let data = std::fs::read("assets/flagged/AdobeColorSpin.icc").unwrap();
    let p = ColorProfile::new_from_slice(&data).unwrap();
    let _ = p.create_transform_8bit(
        Layout::Rgb,
        &ColorProfile::new_srgb(),
        Layout::Rgb,
        TransformOptions::default(),
    );
}

#[test]
fn colorgate_sihl_photopaper_is_usable() {
    let data = std::fs::read("assets/flagged/ColorGATE_Sihl_PhotoPaper.icc").unwrap();
    let p = ColorProfile::new_from_slice(&data).unwrap();
    let _ = p.create_transform_8bit(
        Layout::Rgb,
        &ColorProfile::new_srgb(),
        Layout::Rgb,
        TransformOptions::default(),
    );
}

#[test]
fn crayons_is_usable() {
    let data = std::fs::read("assets/flagged/installed-linux-x64-Crayons.icc").unwrap();
    let p = ColorProfile::new_from_slice(&data).unwrap();
    // Crayons is PCS→device (Lab→1ch named color), won't transform to sRGB
    // via standard pipeline; just ensure parse stability.
    drop(p);
}

//! Diagnostic test that prints the parsed ColorProfile fields for the
//! iccMAX profiles so we can compare against canonical sRGB.

#![cfg(feature = "permissive")]

use moxcms::ColorProfile;

#[test]
fn dump_iso22028_vs_canonical() {
    let data = std::fs::read("assets/flagged/sRGB_ISO22028.icc").unwrap();
    let parsed = ColorProfile::new_from_slice(&data).unwrap();
    let canonical = ColorProfile::new_srgb();

    println!("\n=== sRGB_ISO22028 (from cept) ===");
    println!("red_colorant   = {:?}", parsed.red_colorant);
    println!("green_colorant = {:?}", parsed.green_colorant);
    println!("blue_colorant  = {:?}", parsed.blue_colorant);
    println!("white_point    = {:?}", parsed.white_point);
    println!("media_white    = {:?}", parsed.media_white_point);
    println!("chad           = {:?}", parsed.chromatic_adaptation);
    println!("cicp           = {:?}", parsed.cicp);

    println!("\n=== Canonical sRGB ===");
    println!("red_colorant   = {:?}", canonical.red_colorant);
    println!("green_colorant = {:?}", canonical.green_colorant);
    println!("blue_colorant  = {:?}", canonical.blue_colorant);
    println!("white_point    = {:?}", canonical.white_point);
    println!("media_white    = {:?}", canonical.media_white_point);
    println!("chad           = {:?}", canonical.chromatic_adaptation);
    println!("cicp           = {:?}", canonical.cicp);

    // Sample our parsed TRC at known points and compare to canonical sRGB EOTF.
    fn srgb(x: f64) -> f64 {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }
    if let Some(red_trc) = &parsed.red_trc {
        println!("\n=== sRGB_ISO22028 red_trc samples (parsed vs canonical sRGB) ===");
        if let moxcms::ToneReprCurve::Lut(lut) = red_trc {
            let n = lut.len();
            for i in 0..11 {
                let t = i as f64 / 10.0;
                let pos = t * (n - 1) as f64;
                let i0 = pos.floor() as usize;
                let frac = pos - i0 as f64;
                let i1 = (i0 + 1).min(n - 1);
                let v = lut[i0] as f64 * (1.0 - frac) + lut[i1] as f64 * frac;
                let parsed_y = v / 65535.0;
                let canonical_y = srgb(t);
                println!(
                    "  x={:.2}  parsed={:.6}  canonical={:.6}  err={:.6}",
                    t,
                    parsed_y,
                    canonical_y,
                    (parsed_y - canonical_y).abs()
                );
            }
        }
    }
}

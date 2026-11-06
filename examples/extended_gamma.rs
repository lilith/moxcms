#![forbid(unsafe_code)]

use moxcms::{ColorProfile, Layout, TransformOptions, curve_from_gamma};

fn show(name: &str, input: &[f32], output: &[f32]) {
    let clipped: Vec<_> = output.iter().map(|x| x.clamp(0., 1.)).collect();
    println!("{name}\t{input:?}\t{output:?}\t{clipped:?}");
}

fn main() {
    println!("case\tinput\toutput\tclipped");
    let options = TransformOptions {
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };
    let source = ColorProfile::new_pro_photo_rgb();
    let target = ColorProfile::new_srgb();
    let transform = source
        .create_transform_f32(Layout::Rgb, &target, Layout::Rgb, options)
        .unwrap();
    for (name, input) in [
        ("prophoto_black", [0f32; 3]),
        ("prophoto_zero_channel", [0.5, 0.5, 0.]),
        ("prophoto_gray_control", [0.5; 3]),
    ] {
        let mut output = [0.; 3];
        transform.transform(&input, &mut output).unwrap();
        show(name, &input, &output);
    }
    let curve = curve_from_gamma(563. / 256.);
    for (name, evaluator) in [
        ("pure_gamma_encode", curve.make_gamma_evaluator().unwrap()),
        ("pure_gamma_decode", curve.make_linear_evaluator().unwrap()),
    ] {
        let input = [-0.5, 0., f32::from_bits(1)];
        let output = input.map(|x| evaluator.evaluate_value(x));
        show(name, &input, &output);
    }
}

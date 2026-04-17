# Rust ICC Management

Fast and safe conversion between ICC profiles; in pure Rust.

Supports CMYK⬌RGBX, RGBX⬌RGBX, RGBX⬌GRAY, LAB⬌RGBX and CMYK⬌LAB, GRAY⬌RGB, any 3/4 color profiles to RGB and vice versa. Also supports almost any to any Display Class ICC profiles up to 16 inks.

> **Imazen fork note**
>
> This fork extends `awxkee/moxcms` with broader compatibility for real-world
> ICC profiles. It adds a `permissive` feature (default on) that accepts
> profiles which lcms2 and skcms parse successfully but which upstream moxcms
> rejects on strict-spec grounds:
>
> - ICC v0 (zero version field) → treated as v2.4
> - ICC v5 / iccMAX → treated as v4.4 (functional subset)
> - Lut8/Lut16 with 0- or 1-entry input/output tables → synthesized identity
> - mAB/mBA nested curve arrays shorter than channel count → padded with identity
> - Unknown ProfileClass values (e.g. iccMAX `cenc`) → treated as `ColorSpace`
> - Unknown PCS color-space signature (all-zero) → treated as XYZ
> - Unknown LUT tag type (e.g. iccMAX `mpet`, corrupt vendor output) → skipped,
>   parser falls back to matrix/TRC tags from the same profile
>
> Test fixtures for the 9 real-world profiles that motivated these fixes
> (Skia-bundled sRGB variants from color.org, Adobe/ColorGATE print profiles,
> Linux-bundled Crayons / x11-colors named-color Lab profiles) live at
> `assets/flagged/` with regression coverage in `tests/flagged.rs`.
>
> To match upstream strict behavior, build with `--no-default-features`.
> A `trace-invalid-profile` feature is also available for diagnostic work;
> it prints file:line for every `InvalidProfile` construction via a new
> `err::invalid_profile()` helper.

## Example

```rust
let f_str = "./assets/dci_p3_profile.jpeg";
let file = File::open(f_str).expect("Failed to open file");

let img = image::ImageReader::open(f_str).unwrap().decode().unwrap();
let rgb = img.to_rgb8();

let mut decoder = JpegDecoder::new(BufReader::new(file)).unwrap();
let icc = decoder.icc_profile().unwrap().unwrap();
let color_profile = ColorProfile::new_from_slice(&icc).unwrap();
let dest_profile = ColorProfile::new_srgb();
let transform = color_profile
    .create_transform_8bit(&dest_profile, Layout::Rgb8, TransformOptions::default())
    .unwrap();
let mut dst = vec![0u8; rgb.len()];

for (src, dst) in rgb
    .chunks_exact(img.width() as usize * 3)
    .zip(dst.chunks_exact_mut(img.dimensions().0 as usize * 3))
{
    transform
        .transform(
            &src[..img.dimensions().0 as usize * 3],
            &mut dst[..img.dimensions().0 as usize * 3],
        )
        .unwrap();
}
image::save_buffer(
    "v1.jpg",
    &dst,
    img.dimensions().0,
    img.dimensions().1,
    image::ExtendedColorType::Rgb8,
)
    .unwrap();
```

## Benchmarks

### ICC Transform 8-Bit 

Tests were ran with a 1997×1331 resolution image.

| Conversion         | time(NEON) | Time(AVX2) |
|--------------------|:----------:|:----------:|
| moxcms RGB⮕RGB     |   2.68ms   |   4.52ms   |
| moxcms LUT RGB⮕RGB |   7.18ms   |  17.50ms   |
| moxcms RGBA⮕RGBA   |   2.96ms   |   4.83ms   |
| moxcms CMYK⮕RGBA   |  11.86ms   |  27.98ms   |
| lcms2 RGB⮕RGB      |   13.1ms   |  27.73ms   |
| lcms2 LUT RGB⮕RGB  |  27.60ms   |  58.26ms   |
| lcms2 RGBA⮕RGBA    |  21.97ms   |  35.70ms   |
| lcms2 CMYK⮕RGBA    |  39.71ms   |  79.40ms   |
| qcms RGB⮕RGB       |   6.47ms   |   4.59ms   |
| qcms LUT RGB⮕RGB   |  26.72ms   |  60.80ms   |
| qcms RGBA⮕RGBA     |   6.83ms   |   4.99ms   |
| qcms CMYK⮕RGBA     |  25.97ms   |  61.54ms   |

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.

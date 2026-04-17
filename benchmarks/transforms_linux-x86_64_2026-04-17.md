# Benchmark Results

**git:** `97d44665ef75582f83604a3115b8b30a70aaf23f`  
**total:** 66.9s  **waits:** 50 (0.0s)

## cmyk_to_srgb

*141 rounds × 52 calls*

| Benchmark | Min | Mean | vs Base | Throughput |
|-----------|-----|------|---------|------------|
| tetra_lo | 5.74µs | 6.32µs | 6.32µs | 162 Mpixels/s |
| tetra_hi | 7.06µs | 7.86µs | [+23.4%  +24.2%  +24.9%] | 130 Mpixels/s |
| pyramid_lo | 6.25µs | 7.40µs | [+8.6%  +9.4%  +10.1%] | 138 Mpixels/s |
| pyramid_hi | 7.79µs | 8.91µs | [+32.0%  +32.9%  +33.7%] | 115 Mpixels/s |
| prism_lo | 6.31µs | 7.26µs | [+8.6%  +9.2%  +9.8%] | 141 Mpixels/s |
| prism_hi | 7.61µs | 8.47µs | [+31.6%  +32.5%  +33.4%] | 121 Mpixels/s |
| linear_lo | 6.41µs | 7.36µs | [+12.2%  +13.0%  +13.8%] | 139 Mpixels/s |
| linear_hi | 8.53µs | 9.46µs | [+46.3%  +47.0%  +47.8%] | 108 Mpixels/s |

```
  tetra_lo    |██████████████████████████████| 162.0 Mpixels/s
  tetra_hi    |████████████████████████      | 130.2 Mpixels/s
  pyramid_lo  |█████████████████████████▋    | 138.4 Mpixels/s
  pyramid_hi  |█████████████████████▎        | 115.0 Mpixels/s
  prism_lo    |██████████████████████████▏   | 141.1 Mpixels/s
  prism_hi    |██████████████████████▍       | 120.9 Mpixels/s
  linear_lo   |█████████████████████████▊    | 139.1 Mpixels/s
  linear_hi   |████████████████████          | 108.2 Mpixels/s
```

## srgb_to_cmyk

*200 rounds × 123 calls*

| Benchmark | Min | Mean | vs Base | Throughput |
|-----------|-----|------|---------|------------|
| tetra_lo | 6.23µs | 6.66µs | 6.66µs | 154 Mpixels/s |
| tetra_hi | 7.13µs | 7.60µs | [+13.5%  +14.2%  +14.9%] | 135 Mpixels/s |
| pyramid_lo | 6.67µs | 7.04µs | [+5.4%  +6.0%  +6.6%] | 145 Mpixels/s |
| pyramid_hi | 7.86µs | 8.51µs | [+27.1%  +27.9%  +28.6%] | 120 Mpixels/s |
| prism_lo | 7.09µs | 7.47µs | [+11.7%  +12.3%  +12.8%] | 137 Mpixels/s |
| prism_hi | 8.54µs | 9.03µs | [+34.5%  +35.2%  +36.0%] | 113 Mpixels/s |
| linear_lo | 7.83µs | 8.31µs | [+23.2%  +23.9%  +24.6%] | 123 Mpixels/s |
| linear_hi | 9.23µs | 9.75µs | [+46.4%  +47.2%  +47.9%] | 105 Mpixels/s |

```
  tetra_lo    |██████████████████████████████| 153.7 Mpixels/s
  tetra_hi    |██████████████████████████▎   | 134.8 Mpixels/s
  pyramid_lo  |████████████████████████████▍ | 145.4 Mpixels/s
  pyramid_hi  |███████████████████████▌      | 120.4 Mpixels/s
  prism_lo    |██████████████████████████▊   | 137.1 Mpixels/s
  prism_hi    |██████████████████████▏       | 113.3 Mpixels/s
  linear_lo   |████████████████████████      | 123.2 Mpixels/s
  linear_hi   |████████████████████▌         | 105.0 Mpixels/s
```

## rgb_lut_to_srgb

*200 rounds × 175 calls*

| Benchmark | Min | Mean | vs Base | Throughput |
|-----------|-----|------|---------|------------|
| tetra_lo | 3.36µs | 3.82µs | 3.82µs | 268 Mpixels/s |
| tetra_hi | 4.64µs | 4.99µs | [+35.0%  +35.6%  +36.3%] | 205 Mpixels/s |
| pyramid_lo | 4.01µs | 4.90µs | [+16.0%  +16.6%  +17.3%] | 209 Mpixels/s |
| pyramid_hi | 5.28µs | 6.40µs | [+55.9%  +56.7%  +57.5%] | 160 Mpixels/s |
| prism_lo | 3.89µs | 4.83µs | [+15.4%  +16.1%  +16.8%] | 212 Mpixels/s |
| prism_hi | 5.32µs | 6.17µs | [+55.8%  +56.6%  +57.5%] | 166 Mpixels/s |
| linear_lo | 3.85µs | 4.48µs | [+14.0%  +14.6%  +15.2%] | 228 Mpixels/s |
| linear_hi | 5.45µs | 6.17µs | [+57.4%  +58.1%  +58.9%] | 166 Mpixels/s |

```
  tetra_lo    |██████████████████████████████| 268.2 Mpixels/s
  tetra_hi    |██████████████████████▉       | 205.1 Mpixels/s
  pyramid_lo  |███████████████████████▎      | 208.8 Mpixels/s
  pyramid_hi  |█████████████████▉            | 160.0 Mpixels/s
  prism_lo    |███████████████████████▋      | 212.0 Mpixels/s
  prism_hi    |██████████████████▌           | 165.8 Mpixels/s
  linear_lo   |█████████████████████████▌    | 228.4 Mpixels/s
  linear_hi   |██████████████████▌           | 166.0 Mpixels/s
```


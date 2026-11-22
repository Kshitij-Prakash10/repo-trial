# Alpha-conversion benchmark: ce56e17561c86e659cca61663d65a451bf753816

torch imported: False. IoU convention: scan() quad (whole image when no page is found) vs the true page mask, clipped to the frame.

## Paired images (B = A + one opaque far pixel unless the case says otherwise)

| case | photo | text |A-B| (grey) | contrast A / B | found A / B | quad IoU A-B | IoU vs truth A / B | upload auto IoU A-B | upload manual output |A-B| |
|---|---|---|---|---|---|---|---|---|
| antialiased_text_on_transparent (text alpha <= 200, RGB 0 under alpha 0) | - | 0.0 | 255.0 / 255.0 | False / False | 1.0 | - / - | 1.0 | 0.0 |
| semi_transparent_overlay_over_photo (alpha 128 everywhere) | 200000.jpg | 0.0 | 138.9 / 138.9 | True / True | 1.0 | 0.9134 / 0.9134 | 1.0 | 0.0 |
| partial_alpha_surroundings (page alpha 254, outside 128) | 200000.jpg | 0.0 | 138.9 / 138.9 | True / True | 1.0 | 0.9134 / 0.9134 | 1.0 | 0.0 |
| partial_alpha_surroundings: page alpha 254 (A) vs 255 (B), outside 128 | 200000.jpg | 0.0 | 138.9 / 138.9 | True / True | 1.0 | 0.9134 / 0.9134 | 1.0 | 0.0 |
| fully_transparent_cutout (page alpha 254, outside alpha 0 over black) | 200000.jpg | 0.0 | 207.9 / 207.9 | True / True | 1.0 | 0.9964 / 0.9964 | 1.0 | 0.0 |
| 16bit_grey: I;16 (x257) (A) vs 8-bit L (B) | 200000.jpg | 0.0 | 138.9 / 138.9 | False / False | 1.0 | 0.3703 / 0.3703 | 1.0 | 0.0 |
| 16bit_grey: I;16 holding 0..255 values (A) vs 8-bit L (B) | 200000.jpg | 47.076 | 1.0 / 138.9 | False / False | 1.0 | 0.3703 / 0.3703 | 1.0 | 27.236 |
| semi_transparent_overlay_over_photo (alpha 128 everywhere) | 200001.jpg | 0.0 | 90.5 / 90.5 | True / True | 1.0 | 0.9725 / 0.9725 | 1.0 | 0.0 |
| partial_alpha_surroundings (page alpha 254, outside 128) | 200001.jpg | 0.0 | 90.5 / 90.5 | True / True | 1.0 | 0.9725 / 0.9725 | 1.0 | 0.0 |
| partial_alpha_surroundings: page alpha 254 (A) vs 255 (B), outside 128 | 200001.jpg | 0.0 | 90.5 / 90.5 | True / True | 1.0 | 0.9725 / 0.9725 | 1.0 | 0.0 |
| fully_transparent_cutout (page alpha 254, outside alpha 0 over black) | 200001.jpg | 0.0 | 181.5 / 181.5 | True / True | 1.0 | 0.9943 / 0.9943 | 1.0 | 0.0 |
| 16bit_grey: I;16 (x257) (A) vs 8-bit L (B) | 200001.jpg | 0.0 | 90.5 / 90.5 | False / False | 1.0 | 0.4039 / 0.4039 | 1.0 | 0.0 |
| 16bit_grey: I;16 holding 0..255 values (A) vs 8-bit L (B) | 200001.jpg | 73.544 | 1.0 / 90.5 | False / False | 1.0 | 0.4039 / 0.4039 | 1.0 | 37.055 |
| semi_transparent_overlay_over_photo (alpha 128 everywhere) | 200002.jpg | 0.0 | 107.4 / 107.4 | True / True | 1.0 | 0.9968 / 0.9968 | 1.0 | 0.0 |
| partial_alpha_surroundings (page alpha 254, outside 128) | 200002.jpg | 0.0 | 107.4 / 107.4 | True / True | 1.0 | 0.9968 / 0.9968 | 1.0 | 0.0 |
| partial_alpha_surroundings: page alpha 254 (A) vs 255 (B), outside 128 | 200002.jpg | 0.0 | 107.4 / 107.4 | True / True | 1.0 | 0.9968 / 0.9968 | 1.0 | 0.0 |
| fully_transparent_cutout (page alpha 254, outside alpha 0 over black) | 200002.jpg | 0.0 | 193.4 / 193.4 | True / True | 1.0 | 0.993 / 0.993 | 1.0 | 0.0 |
| 16bit_grey: I;16 (x257) (A) vs 8-bit L (B) | 200002.jpg | 0.0 | 107.4 / 107.4 | True / True | 1.0 | 0.9489 / 0.9489 | 1.0 | 0.0 |
| 16bit_grey: I;16 holding 0..255 values (A) vs 8-bit L (B) | 200002.jpg | 61.626 | 1.0 / 107.4 | False / True | 0.3682 | 0.388 / 0.9489 | 0.3682 | 26.556 |

## 20 partial-alpha photos

| variant | found | mean IoU | IoU >= 0.9 |
|---|---|---|---|
| alpha128_outside | 19/20 (0.9500) | 0.9069 | 15/20 (0.7500) |
| alpha128_outside_page254 | 19/20 (0.9500) | 0.9069 | 15/20 (0.7500) |
| alpha128_outside_rgb0 | 20/20 (1.0000) | 0.9362 | 18/20 (0.9000) |
| alpha192_outside | 19/20 (0.9500) | 0.9069 | 15/20 (0.7500) |
| alpha64_outside | 19/20 (0.9500) | 0.9069 | 15/20 (0.7500) |
| cutout_alpha0_rgb0 | 19/20 (0.9500) | 0.9532 | 19/20 (0.9500) |
| rgb_reference | 19/20 (0.9500) | 0.9069 | 15/20 (0.7500) |

## Full synthetic sets (RGB)

| split | found | mean IoU | IoU >= 0.9 |
|---|---|---|---|
| ood | 46/80 (0.5750) | 0.6727 | 30/80 (0.3750) |
| test | 189/200 (0.9450) | 0.8867 | 151/200 (0.7550) |

Photo-level changes vs baseline (/tmp/claude-0/-home-user-repo-trial/aab1d323-54ba-58c1-877b-55ee24394209/scratchpad/review-wt/reports/review_2026-11/baseline_0c71106/alpha.json): partial-alpha 13, full sets 0.

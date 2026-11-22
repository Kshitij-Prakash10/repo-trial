# Page detection benchmark: ce56e17561c86e659cca61663d65a451bf753816

MIN_CONFIDENCE 0.6; 20 images per negative kind. scan() quad (whole image when no page) vs the true mask, both clipped to the frame.

## False alarms (photos without a page)

| kind | false alarms |
|---|---|
| blank_dark | 0/20 (0.0000) |
| blank_grey | 0/20 (0.0000) |
| blank_paper | 0/20 (0.0000) |
| blank_white | 0/20 (0.0000) |
| noise_uniform | 0/20 (0.0000) |
| ood_checker | 0/20 (0.0000) |
| ood_granite | 0/20 (0.0000) |
| ood_light | 1/20 (0.0500) |
| tiny_noise | 0/6 (0.0000) |
| tiny_pagelike | 2/6 (0.3333) |
| tiny_white | 0/6 (0.0000) |
| train_blotch | 2/20 (0.1000) |
| train_dots | 1/20 (0.0500) |
| train_fabric | 1/20 (0.0500) |
| train_gradient | 0/20 (0.0000) |
| train_light | 0/20 (0.0000) |
| train_solid | 0/20 (0.0000) |
| train_stripes | 0/20 (0.0000) |
| train_tiles | 6/20 (0.3000) |
| train_wood | 0/20 (0.0000) |

## Stored synthetic sets

| split | found | mean IoU | IoU >= 0.9 |
|---|---|---|---|
| ood | 46/80 (0.5750) | 0.6727 | 30/80 (0.3750) |
| test | 189/200 (0.9450) | 0.8867 | 151/200 (0.7550) |

## Generated pages (raw)

| case | found | mean IoU | IoU >= 0.9 | page lost (mean) | photos losing > 1 % of text |
|---|---|---|---|---|---|
| cut1 | 15/15 (1.0000) | 0.9717 | 13/15 (0.8667) | 0.0089 | 4/15 (0.2667) |
| cut2 | 15/15 (1.0000) | 0.9514 | 14/15 (0.9333) | 0.0125 | 5/15 (0.3333) |
| low_contrast | 19/20 (0.9500) | 0.6899 | 0/20 (0.0000) | 0.037 | 3/20 (0.1500) |
| regular | 20/20 (1.0000) | 0.9431 | 17/20 (0.8500) | 0.0328 | 4/20 (0.2000) |

## Generated pages by transform (all cases)

| transform | found | mean IoU | IoU >= 0.9 | page lost (mean) | photos losing > 1 % of text |
|---|---|---|---|---|---|
| cropped | 67/70 (0.9571) | 0.9405 | 55/70 (0.7857) | 0.0223 | 18/70 (0.2571) |
| raw | 69/70 (0.9857) | 0.8787 | 44/70 (0.6286) | 0.0245 | 16/70 (0.2286) |
| rot180 | 68/70 (0.9714) | 0.8617 | 42/70 (0.6000) | 0.0408 | 14/70 (0.2000) |
| rot90 | 69/70 (0.9857) | 0.8912 | 46/70 (0.6571) | 0.0211 | 13/70 (0.1857) |
| tilt4 | 67/70 (0.9571) | 0.828 | 40/70 (0.5714) | 0.0186 | 13/70 (0.1857) |

## Stored real-photo rows (threshold only)

highest threshold on doccorner_val keeping >= 0.995 of the IoU >= 0.9 pages: **0.6**. stored 'found' was written at threshold 0.35; the blank and noise guards cannot be measured from stored rows (they need the photos).

| set | found on pages | IoU>=0.9 kept | false alarms |
|---|---|---|---|
| val @ 0.6 | 831/880 (0.9443) | 600/603 (0.9950) | 5/120 (0.0417) |
| doccorner_test | 1385/1761 (0.7865) | 672/678 (0.9912) | 16/239 (0.0669) |
| smartdoc | 2313/2555 (0.9053) | 1263/1271 (0.9937) |  |

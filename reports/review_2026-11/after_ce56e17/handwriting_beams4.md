# Handwriting benchmark: ce56e17561c86e659cca61663d65a451bf753816

Data: synthetic, invented text, rendered with data/raw/fonts (seen in training: optimistic). Real held-out pages: BLOCKED: GNHK / IAM need huggingface.co (blocked here).

Automatic line detection: BLOCKED: docTR fast_base text-detector weights are missing: neither models/ocr/det_fast_base_ft.pt nor /root/.cache/doctr/models/fast_base-*.pt exists (download host doctr-static.mindee.com is blocked here).

Handwriting decoding options: {'requested': {'num_beams': 4}, 'constructor': {'num_beams': 4}, 'read': {}, 'ignored': {}}.

## validation: 20 pages, 113 lines

| reading | line CER / WER | page CER / WER | line exact | exact ignoring case | numeric lines exact | field value | field unit | field exact |
|---|---|---|---|---|---|---|---|---|
| printed | 0.0872 / 0.2385 | 0.0839 / 0.2385 | 49/113 (0.4336) | 53/113 (0.4690) | 41/96 (0.4271) | 42/76 (0.5526) | 37/76 (0.4868) | 37/76 (0.4868) |
| handwriting | 0.0229 / 0.0766 | 0.0221 / 0.0766 | 82/113 (0.7257) | 87/113 (0.7699) | 67/96 (0.6979) | 64/76 (0.8421) | 65/76 (0.8553) | 63/76 (0.8289) |
| shipped | 0.0217 / 0.07 | 0.0209 / 0.07 | 87/113 (0.7699) | 91/113 (0.8053) | 72/96 (0.7500) | 64/76 (0.8421) | 64/76 (0.8421) | 63/76 (0.8289) |
| shipped_if_routed | 0.0217 / 0.07 | 0.0209 / 0.07 | 87/113 (0.7699) | 91/113 (0.8053) | 72/96 (0.7500) | 64/76 (0.8421) | 64/76 (0.8421) | 63/76 (0.8289) |
| candidate | 0.02 / 0.0678 | 0.0193 / 0.0678 | 86/113 (0.7611) | 90/113 (0.7965) | 71/96 (0.7396) | 65/76 (0.8553) | 66/76 (0.8684) | 64/76 (0.8421) |

| diagnosis | n |
|---|---|
| reader_disagreement (text) | 73/113 (0.6460) |
| reader_disagreement (numbers) | 31/113 (0.2743) |
| routed_to_handwriting | 96/113 (0.8496) |
| lines_exact_by_either_reader (oracle selection) | 91/113 (0.8053) |
| lines_exact_shipped | 87/113 (0.7699) |
| selection_loss (one reader right, shipped wrong) | 4/113 (0.0354) |
| recognition_loss (both readers wrong) | 22/113 (0.1947) |
| routing_loss (handwriting right, line not routed) | 0/113 (0.0000) |

| handwriting conf >= | accepted | errors among accepted | number errors | wrong decimals |
|---|---|---|---|---|
| 0.50 | 113/113 (1.0000) | 31/113 (0.2743) | 14/96 (0.1458) | 2/21 (0.0952) |
| 0.55 | 113/113 (1.0000) | 31/113 (0.2743) | 14/96 (0.1458) | 2/21 (0.0952) |
| 0.60 | 112/113 (0.9912) | 30/112 (0.2679) | 14/95 (0.1474) | 2/21 (0.0952) |
| 0.65 | 112/113 (0.9912) | 30/112 (0.2679) | 14/95 (0.1474) | 2/21 (0.0952) |
| 0.70 | 111/113 (0.9823) | 29/111 (0.2613) | 14/94 (0.1489) | 2/21 (0.0952) |
| 0.75 | 108/113 (0.9558) | 27/108 (0.2500) | 12/91 (0.1319) | 2/21 (0.0952) |
| 0.80 | 100/113 (0.8850) | 24/100 (0.2400) | 11/83 (0.1325) | 2/19 (0.1053) |
| 0.85 | 78/113 (0.6903) | 14/78 (0.1795) | 6/62 (0.0968) | 2/11 (0.1818) |
| 0.90 | 52/113 (0.4602) | 8/52 (0.1538) | 4/39 (0.1026) | 2/6 (0.3333) |
| 0.95 | 27/113 (0.2389) | 0/27 (0.0000) | 0/17 (0.0000) | 0/2 (0.0000) |

## test: 30 pages, 167 lines

| reading | line CER / WER | page CER / WER | line exact | exact ignoring case | numeric lines exact | field value | field unit | field exact |
|---|---|---|---|---|---|---|---|---|
| printed | 0.0702 / 0.2372 | 0.0675 / 0.2372 | 78/167 (0.4671) | 82/167 (0.4910) | 58/133 (0.4361) | 67/104 (0.6442) | 57/104 (0.5481) | 56/104 (0.5385) |
| handwriting | 0.0212 / 0.0891 | 0.0204 / 0.0891 | 114/167 (0.6826) | 122/167 (0.7305) | 86/133 (0.6466) | 85/104 (0.8173) | 86/104 (0.8269) | 83/104 (0.7981) |
| shipped | 0.0168 / 0.074 | 0.0162 / 0.074 | 123/167 (0.7365) | 130/167 (0.7784) | 95/133 (0.7143) | 90/104 (0.8654) | 90/104 (0.8654) | 88/104 (0.8462) |
| shipped_if_routed | 0.0177 / 0.077 | 0.017 / 0.077 | 121/167 (0.7246) | 129/167 (0.7725) | 93/133 (0.6992) | 88/104 (0.8462) | 89/104 (0.8558) | 86/104 (0.8269) |
| candidate | 0.0186 / 0.0816 | 0.0179 / 0.0816 | 119/167 (0.7126) | 127/167 (0.7605) | 91/133 (0.6842) | 86/104 (0.8269) | 87/104 (0.8365) | 84/104 (0.8077) |

| diagnosis | n |
|---|---|
| reader_disagreement (text) | 105/167 (0.6287) |
| reader_disagreement (numbers) | 39/167 (0.2335) |
| routed_to_handwriting | 142/167 (0.8503) |
| lines_exact_by_either_reader (oracle selection) | 133/167 (0.7964) |
| lines_exact_shipped | 123/167 (0.7365) |
| selection_loss (one reader right, shipped wrong) | 10/167 (0.0599) |
| recognition_loss (both readers wrong) | 34/167 (0.2036) |
| routing_loss (handwriting right, line not routed) | 1/167 (0.0060) |

| handwriting conf >= | accepted | errors among accepted | number errors | wrong decimals |
|---|---|---|---|---|
| 0.50 | 167/167 (1.0000) | 53/167 (0.3174) | 22/133 (0.1654) | 5/27 (0.1852) |
| 0.55 | 167/167 (1.0000) | 53/167 (0.3174) | 22/133 (0.1654) | 5/27 (0.1852) |
| 0.60 | 167/167 (1.0000) | 53/167 (0.3174) | 22/133 (0.1654) | 5/27 (0.1852) |
| 0.65 | 167/167 (1.0000) | 53/167 (0.3174) | 22/133 (0.1654) | 5/27 (0.1852) |
| 0.70 | 166/167 (0.9940) | 52/166 (0.3133) | 21/132 (0.1591) | 5/27 (0.1852) |
| 0.75 | 163/167 (0.9760) | 49/163 (0.3006) | 20/129 (0.1550) | 5/26 (0.1923) |
| 0.80 | 151/167 (0.9042) | 42/151 (0.2781) | 20/117 (0.1709) | 5/21 (0.2381) |
| 0.85 | 126/167 (0.7545) | 30/126 (0.2381) | 14/93 (0.1505) | 5/16 (0.3125) |
| 0.90 | 92/167 (0.5509) | 13/92 (0.1413) | 4/61 (0.0656) | 2/11 (0.1818) |
| 0.95 | 45/167 (0.2695) | 6/45 (0.1333) | 1/18 (0.0556) | 0/3 (0.0000) |

Candidate selector: handwriting reading when its confidence exceeds the printed one by more than the margin (keep_printed still applies, routing assumed); margin chosen on validation by line CER; chosen margin -0.05.

Latency per line (ms, mean): printed_ms 33.6, route_ms 5.1, handwriting_ms 935.1

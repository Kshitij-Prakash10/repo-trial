# Handwriting benchmark: ce56e17561c86e659cca61663d65a451bf753816

Data: synthetic, invented text, rendered with data/raw/fonts (seen in training: optimistic). Real held-out pages: BLOCKED: GNHK / IAM need huggingface.co (blocked here).

Automatic line detection: BLOCKED: docTR fast_base text-detector weights are missing: neither models/ocr/det_fast_base_ft.pt nor /root/.cache/doctr/models/fast_base-*.pt exists (download host doctr-static.mindee.com is blocked here).

Handwriting decoding options: {'requested': {}, 'constructor': {}, 'read': {}, 'ignored': {}}.

## validation: 20 pages, 113 lines

| reading | line CER / WER | page CER / WER | line exact | exact ignoring case | numeric lines exact | field value | field unit | field exact |
|---|---|---|---|---|---|---|---|---|
| printed | 0.0872 / 0.2385 | 0.0839 / 0.2385 | 49/113 (0.4336) | 53/113 (0.4690) | 41/96 (0.4271) | 42/76 (0.5526) | 37/76 (0.4868) | 37/76 (0.4868) |
| handwriting | 0.0259 / 0.0853 | 0.0249 / 0.0853 | 80/113 (0.7080) | 85/113 (0.7522) | 65/96 (0.6771) | 63/76 (0.8289) | 64/76 (0.8421) | 62/76 (0.8158) |
| shipped | 0.0225 / 0.0744 | 0.0217 / 0.0744 | 86/113 (0.7611) | 91/113 (0.8053) | 71/96 (0.7396) | 65/76 (0.8553) | 64/76 (0.8421) | 63/76 (0.8289) |
| shipped_if_routed | 0.0225 / 0.0744 | 0.0217 / 0.0744 | 86/113 (0.7611) | 91/113 (0.8053) | 71/96 (0.7396) | 65/76 (0.8553) | 64/76 (0.8421) | 63/76 (0.8289) |
| candidate | 0.0213 / 0.0744 | 0.0205 / 0.0744 | 85/113 (0.7522) | 90/113 (0.7965) | 70/96 (0.7292) | 65/76 (0.8553) | 65/76 (0.8553) | 63/76 (0.8289) |

| diagnosis | n |
|---|---|
| reader_disagreement (text) | 74/113 (0.6549) |
| reader_disagreement (numbers) | 32/113 (0.2832) |
| routed_to_handwriting | 96/113 (0.8496) |
| lines_exact_by_either_reader (oracle selection) | 90/113 (0.7965) |
| lines_exact_shipped | 86/113 (0.7611) |
| selection_loss (one reader right, shipped wrong) | 4/113 (0.0354) |
| recognition_loss (both readers wrong) | 23/113 (0.2035) |
| routing_loss (handwriting right, line not routed) | 0/113 (0.0000) |

| handwriting conf >= | accepted | errors among accepted | number errors | wrong decimals |
|---|---|---|---|---|
| 0.50 | 113/113 (1.0000) | 33/113 (0.2920) | 15/96 (0.1562) | 2/21 (0.0952) |
| 0.55 | 112/113 (0.9912) | 32/112 (0.2857) | 14/95 (0.1474) | 2/21 (0.0952) |
| 0.60 | 111/113 (0.9823) | 31/111 (0.2793) | 14/94 (0.1489) | 2/21 (0.0952) |
| 0.65 | 111/113 (0.9823) | 31/111 (0.2793) | 14/94 (0.1489) | 2/21 (0.0952) |
| 0.70 | 109/113 (0.9646) | 29/109 (0.2661) | 13/92 (0.1413) | 2/21 (0.0952) |
| 0.75 | 104/113 (0.9204) | 25/104 (0.2404) | 11/87 (0.1264) | 2/21 (0.0952) |
| 0.80 | 96/113 (0.8496) | 20/96 (0.2083) | 10/79 (0.1266) | 2/17 (0.1176) |
| 0.85 | 78/113 (0.6903) | 14/78 (0.1795) | 6/62 (0.0968) | 2/11 (0.1818) |
| 0.90 | 52/113 (0.4602) | 8/52 (0.1538) | 4/39 (0.1026) | 2/6 (0.3333) |
| 0.95 | 27/113 (0.2389) | 0/27 (0.0000) | 0/17 (0.0000) | 0/2 (0.0000) |

## test: 30 pages, 167 lines

| reading | line CER / WER | page CER / WER | line exact | exact ignoring case | numeric lines exact | field value | field unit | field exact |
|---|---|---|---|---|---|---|---|---|
| printed | 0.0702 / 0.2372 | 0.0675 / 0.2372 | 78/167 (0.4671) | 82/167 (0.4910) | 58/133 (0.4361) | 67/104 (0.6442) | 57/104 (0.5481) | 56/104 (0.5385) |
| handwriting | 0.0244 / 0.1042 | 0.0234 / 0.1042 | 106/167 (0.6347) | 114/167 (0.6826) | 78/133 (0.5865) | 83/104 (0.7981) | 83/104 (0.7981) | 80/104 (0.7692) |
| shipped | 0.0177 / 0.077 | 0.017 / 0.077 | 121/167 (0.7246) | 128/167 (0.7665) | 93/133 (0.6992) | 91/104 (0.8750) | 90/104 (0.8654) | 88/104 (0.8462) |
| shipped_if_routed | 0.0186 / 0.0801 | 0.0179 / 0.0801 | 119/167 (0.7126) | 127/167 (0.7605) | 91/133 (0.6842) | 89/104 (0.8558) | 89/104 (0.8558) | 86/104 (0.8269) |
| candidate | 0.0194 / 0.0846 | 0.0187 / 0.0846 | 117/167 (0.7006) | 125/167 (0.7485) | 89/133 (0.6692) | 87/104 (0.8365) | 87/104 (0.8365) | 84/104 (0.8077) |

| diagnosis | n |
|---|---|
| reader_disagreement (text) | 110/167 (0.6587) |
| reader_disagreement (numbers) | 44/167 (0.2635) |
| routed_to_handwriting | 142/167 (0.8503) |
| lines_exact_by_either_reader (oracle selection) | 131/167 (0.7844) |
| lines_exact_shipped | 121/167 (0.7246) |
| selection_loss (one reader right, shipped wrong) | 10/167 (0.0599) |
| recognition_loss (both readers wrong) | 36/167 (0.2156) |
| routing_loss (handwriting right, line not routed) | 1/167 (0.0060) |

| handwriting conf >= | accepted | errors among accepted | number errors | wrong decimals |
|---|---|---|---|---|
| 0.50 | 167/167 (1.0000) | 61/167 (0.3653) | 28/133 (0.2105) | 5/27 (0.1852) |
| 0.55 | 167/167 (1.0000) | 61/167 (0.3653) | 28/133 (0.2105) | 5/27 (0.1852) |
| 0.60 | 167/167 (1.0000) | 61/167 (0.3653) | 28/133 (0.2105) | 5/27 (0.1852) |
| 0.65 | 166/167 (0.9940) | 60/166 (0.3614) | 27/132 (0.2045) | 5/27 (0.1852) |
| 0.70 | 163/167 (0.9760) | 57/163 (0.3497) | 25/129 (0.1938) | 5/26 (0.1923) |
| 0.75 | 156/167 (0.9341) | 51/156 (0.3269) | 24/122 (0.1967) | 5/25 (0.2000) |
| 0.80 | 146/167 (0.8743) | 44/146 (0.3014) | 23/112 (0.2054) | 5/21 (0.2381) |
| 0.85 | 124/167 (0.7425) | 31/124 (0.2500) | 16/91 (0.1758) | 5/16 (0.3125) |
| 0.90 | 92/167 (0.5509) | 13/92 (0.1413) | 4/61 (0.0656) | 2/11 (0.1818) |
| 0.95 | 45/167 (0.2695) | 6/45 (0.1333) | 1/18 (0.0556) | 0/3 (0.0000) |

Candidate selector: handwriting reading when its confidence exceeds the printed one by more than the margin (keep_printed still applies, routing assumed); margin chosen on validation by line CER; chosen margin -0.05.

Latency per line (ms, mean): printed_ms 33.2, route_ms 5.1, handwriting_ms 522.9

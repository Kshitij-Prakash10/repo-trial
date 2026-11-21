# Medication-warning benchmark: 0c71106e54f4d2d3010c9e89e24b70d011f6fa9a

295 invented lines (`warnings/fixtures.jsonl`, sha256 `021b6698b3ac`). Prominent warning = API `confirm_numbers`; any flag = `confirm_numbers` or `has_numbers`.

## misread_option_off (default)

**Original target** (>= 0.95): 37/39 = 0.9487 -> **FAIL**; missed: ['o39-16: Metformin SOO my twice daily', 'o39-39: Metformin SOO my']

| split | (a) any-flag recall | (a) any-flag on no-number lines | (b) P vs dose wrong | (b) R vs dose wrong | (b) P vs warranted | (b) R vs warranted | (c) prominent on correct print | (d) exact fields | silent dose errors |
|---|---|---|---|---|---|---|---|---|---|
| set=original_39 path=v1 | 2/3 (0.6667) | 0/0 | 0/2 (0.0000) | 0/1 (0.0000) | 0/2 (0.0000) | 0/1 (0.0000) | 2/2 (1.0000) | 1/2 (0.5000) | 1/1 (1.0000) |
| set=original_39 path=v2 | 35/36 (0.9722) | 0/0 | 13/20 (0.6500) | 13/14 (0.9286) | 20/20 (1.0000) | 20/21 (0.9524) | 0/15 (0.0000) | 12/26 (0.4615) | 1/14 (0.0714) |
| set=original_39 path=all | 37/39 (0.9487) | 0/0 | 13/22 (0.5909) | 13/15 (0.8667) | 20/22 (0.9091) | 20/22 (0.9091) | 2/17 (0.1176) | 13/28 (0.4643) | 2/15 (0.1333) |
| set=extended path=v1 | 45/47 (0.9574) | 1/25 (0.0400) | 7/46 (0.1522) | 7/9 (0.7778) | 9/46 (0.1957) | 9/11 (0.8182) | 37/60 (0.6167) | 23/32 (0.7188) | 2/9 (0.2222) |
| set=extended path=v2 | 104/107 (0.9720) | 7/77 (0.0909) | 22/37 (0.5946) | 22/41 (0.5366) | 28/37 (0.7568) | 28/47 (0.5957) | 7/125 (0.0560) | 47/87 (0.5402) | 19/41 (0.4634) |
| set=extended path=all | 149/154 (0.9675) | 8/102 (0.0784) | 29/83 (0.3494) | 29/50 (0.5800) | 37/83 (0.4458) | 37/58 (0.6379) | 44/185 (0.2378) | 70/119 (0.5882) | 21/50 (0.4200) |
| set=all path=v1 | 47/50 (0.9400) | 1/25 (0.0400) | 7/48 (0.1458) | 7/10 (0.7000) | 9/48 (0.1875) | 9/12 (0.7500) | 39/62 (0.6290) | 24/34 (0.7059) | 3/10 (0.3000) |
| set=all path=v2 | 139/143 (0.9720) | 7/77 (0.0909) | 35/57 (0.6140) | 35/55 (0.6364) | 48/57 (0.8421) | 48/68 (0.7059) | 7/140 (0.0500) | 59/113 (0.5221) | 20/55 (0.3636) |
| set=all path=all | 186/193 (0.9637) | 8/102 (0.0784) | 42/105 (0.4000) | 42/65 (0.6462) | 57/105 (0.5429) | 57/80 (0.7125) | 46/202 (0.2277) | 83/147 (0.5646) | 23/65 (0.3538) |

Silent dose errors (all lines): o39-16: Metformin SOO my twice daily; o39-39: Metformin SOO my; ext-137: 1. Omeprazole mg; ext-138: Omeprazole mg for 7 days; ext-140: Amlodipine 5 mg and Omeprazole mg; ext-141: 2. Tab Omeprazole mg OD; ext-142: Omeprazole mg x 14 days; ext-143: 3) Ramipril mg OD; ext-144: Metformin 500 mg BD, Gliclazide mg OD; ext-145: Review in 2 weeks: Omeprazole mg; ext-151: Metformin SOO my; ext-156: Metformin SOO my; ext-157: Metformin SOO my; ext-158: Metformin SOO my; ext-167: Ramipril 25 mg OD; ext-171: Lorazepam .5 mg; ext-172: Lorazepam 5 mg; ext-183: Levothyroxine 50 mcq OD; ext-185: Lactulose 10 mI bd; ext-190: Haemoglobin 11.8 q/dL; ext-217: 4. Cap Omeprazole 2O mg; ext-229: Collected: 02/06/2026 O9:15; ext-255: Warfarin B my

## misread_option_on

**Original target** (>= 0.95): 38/39 = 0.9744 -> **PASS**; missed: ['o39-39: Metformin SOO my']

| split | (a) any-flag recall | (a) any-flag on no-number lines | (b) P vs dose wrong | (b) R vs dose wrong | (b) P vs warranted | (b) R vs warranted | (c) prominent on correct print | (d) exact fields | silent dose errors |
|---|---|---|---|---|---|---|---|---|---|
| set=original_39 path=v1 | 2/3 (0.6667) | 0/0 | 0/2 (0.0000) | 0/1 (0.0000) | 0/2 (0.0000) | 0/1 (0.0000) | 2/2 (1.0000) | 1/2 (0.5000) | 1/1 (1.0000) |
| set=original_39 path=v2 | 36/36 (1.0000) | 0/0 | 14/21 (0.6667) | 14/14 (1.0000) | 21/21 (1.0000) | 21/21 (1.0000) | 0/15 (0.0000) | 12/26 (0.4615) | 0/14 (0.0000) |
| set=original_39 path=all | 38/39 (0.9744) | 0/0 | 14/23 (0.6087) | 14/15 (0.9333) | 21/23 (0.9130) | 21/22 (0.9545) | 2/17 (0.1176) | 13/28 (0.4643) | 1/15 (0.0667) |
| set=extended path=v1 | 45/47 (0.9574) | 1/25 (0.0400) | 7/46 (0.1522) | 7/9 (0.7778) | 9/46 (0.1957) | 9/11 (0.8182) | 37/60 (0.6167) | 23/32 (0.7188) | 2/9 (0.2222) |
| set=extended path=v2 | 105/107 (0.9813) | 8/77 (0.1039) | 24/40 (0.6000) | 24/41 (0.5854) | 30/40 (0.7500) | 30/47 (0.6383) | 8/125 (0.0640) | 47/87 (0.5402) | 17/41 (0.4146) |
| set=extended path=all | 150/154 (0.9740) | 9/102 (0.0882) | 31/86 (0.3605) | 31/50 (0.6200) | 39/86 (0.4535) | 39/58 (0.6724) | 45/185 (0.2432) | 70/119 (0.5882) | 19/50 (0.3800) |
| set=all path=v1 | 47/50 (0.9400) | 1/25 (0.0400) | 7/48 (0.1458) | 7/10 (0.7000) | 9/48 (0.1875) | 9/12 (0.7500) | 39/62 (0.6290) | 24/34 (0.7059) | 3/10 (0.3000) |
| set=all path=v2 | 141/143 (0.9860) | 8/77 (0.1039) | 38/61 (0.6230) | 38/55 (0.6909) | 51/61 (0.8361) | 51/68 (0.7500) | 8/140 (0.0571) | 59/113 (0.5221) | 17/55 (0.3091) |
| set=all path=all | 188/193 (0.9741) | 9/102 (0.0882) | 45/109 (0.4128) | 45/65 (0.6923) | 60/109 (0.5505) | 60/80 (0.7500) | 47/202 (0.2327) | 83/147 (0.5646) | 20/65 (0.3077) |

Silent dose errors (all lines): o39-39: Metformin SOO my; ext-137: 1. Omeprazole mg; ext-138: Omeprazole mg for 7 days; ext-140: Amlodipine 5 mg and Omeprazole mg; ext-141: 2. Tab Omeprazole mg OD; ext-142: Omeprazole mg x 14 days; ext-143: 3) Ramipril mg OD; ext-144: Metformin 500 mg BD, Gliclazide mg OD; ext-145: Review in 2 weeks: Omeprazole mg; ext-156: Metformin SOO my; ext-157: Metformin SOO my; ext-158: Metformin SOO my; ext-167: Ramipril 25 mg OD; ext-171: Lorazepam .5 mg; ext-172: Lorazepam 5 mg; ext-183: Levothyroxine 50 mcq OD; ext-185: Lactulose 10 mI bd; ext-190: Haemoglobin 11.8 q/dL; ext-229: Collected: 02/06/2026 O9:15; ext-255: Warfarin B my

Field parser limits: see `benchmarks/fields.py` (lexicon names only, first value per name, units compared without case, frequencies and durations not parsed).

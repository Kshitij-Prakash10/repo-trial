"""Build benchmarks/warnings/fixtures.jsonl: labelled, INVENTED OCR line readings (no real patient, no
dataset text). Deterministic; run `python -m benchmarks.warnings.build_fixtures` from the repository root
and commit the output. tests/test_benchmarks.py checks that the committed file matches this script.

Record fields
  id, set ("original_39" | "extended" | "extended_r2", the last appended in round 2), category, path ("v1" | "v2"),
  routing: printed_only (v2, never sent to the handwriting reader) | handwriting_selected (v2, the
           handwriting reading is shown, the printed one kept as printed_text) |
           printed_selected_after_handwriting (v2, the line went to the handwriting reader, whose reading
           was not chosen; kept as handwriting_text) | none_v1 (v1 pipeline: no routing, no script),
  source_script: what the page really has on that line (printed | handwritten),
  ground_truth, displayed_text, handwriting_text / printed_text (as the v2 Scanner stores them: when the
           handwriting reading is shown, handwriting_text equals displayed_text),
  has_numeric_content: the ground truth holds a digit, or a spelt-out quantity next to a unit,
  dose_incorrect_or_lost: the displayed text gets a number or unit wrong or drops it (numbers and units
           compared as exact strings, units without case), or its drug / value / unit fields
           (benchmarks.fields.parse_fields) differ from the ground truth's,
  warning_warranted: dose_incorrect_or_lost, or the line is handwritten and holds numeric content
           (handwritten doses are misread with high confidence, so a person must check them),
  expected_fields: parse_fields(ground_truth),
  origin: where the case comes from (the earlier confirm_cases.py lists, or "new").

The "original_39" set is exactly the 39 lines with a digit in the reference of CASES in the earlier
evaluation script confirm_cases.py (used for "flagged by either flag: 37/39"), in their original order,
mapped to routings as that script built them: P() -> printed_only, HW(chosen="hw") ->
handwriting_selected, HW(chosen="printed") -> printed_selected_after_handwriting, V1() -> none_v1.
"""
from __future__ import annotations

import json
import os
import re

from benchmarks.fields import numbers, parse_fields, units

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures.jsonl")
_QTY_WORD = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|half)\s+(?:a\s+)?(tab|tabs|tablet|tablets|cap|caps|"
                       r"capsule|capsules|puff|puffs|drop|drops)\b", re.I)


def P(ref, text=None, cat="printed_num"):
    return {"routing": "printed_only", "ground_truth": ref, "displayed_text": ref if text is None else text, "category": cat}


def HW(ref, hw, printed, chosen="hw", cat="hw_num"):
    if chosen == "hw":
        return {"routing": "handwriting_selected", "ground_truth": ref, "displayed_text": hw, "handwriting_text": hw,
                "printed_text": printed, "category": cat}
    return {"routing": "printed_selected_after_handwriting", "ground_truth": ref, "displayed_text": printed,
            "handwriting_text": hw, "category": cat}


def V1(ref, text=None, cat="v1", source="printed"):
    return {"routing": "none_v1", "ground_truth": ref, "displayed_text": ref if text is None else text, "category": cat,
            "source_script": source}


# ---------------------------------------------------------------- the earlier confirm_cases.py CASES, verbatim
OLD_CASES = [
    P("Date: 14/03/2026"), P("DOB: 02-11-1984"), P("Phone: 020 7946 0000"), P("Haemoglobin 13.2 g/dL 12.0-15.5"),
    P("Fasting glucose 5.4 mmol/L 3.9-5.5"), P("HbA1c 6.1 %"), P("Paracetamol 500 mg two tablets"),
    P("Amlodipine 5 mg once daily"), P("Sample ID: LAB-20431"), P("Platelets 250 x10^9/L"),
    P("Blood pressure 128/82 mmHg"), P("Next visit in 2 weeks"), P("Age: 41 years"), P("Ward 7, Bed 12"),
    P("Cholesterol 4.8 mmol/L"),
    P("Metformin 500 mg twice daily", "Metformin SOO my twice daily", cat="printed_num_misread"),
    P("Amlodipine 5 mg daily", "Amlodipine S mg daily", cat="printed_num_misread"),
    P("Atorvastatin 20 mg at night", "Atorvastatin ZO mg at night", cat="printed_num_misread"),
    P("Levothyroxine 50 mcg", "Levothyroxine SO mcg", cat="printed_num_misread"),
    P("PRESCRIPTION", cat="printed_text"), P("Patient: Jane Sample", cat="printed_text"),
    P("Referred by Dr A. Example", cat="printed_text"), P("Take with food", cat="printed_text"),
    P("Avoid alcohol while taking this medicine", cat="printed_text"), P("Signature of the doctor", cat="printed_text"),
    P("COMPLETE BLOOD COUNT", cat="printed_text"), P("Test Result Units Reference range", cat="printed_text"),
    P("Interpretation: within normal limits", cat="printed_text"), P("Specimen: venous blood", cat="printed_text"),
    P("Report verified by the laboratory", cat="printed_text"), P("Address: Green Lane, Sampletown", cat="printed_text"),
    HW("Ramipril 2.5 mg OD", "Ramipril 2.5 mg OD", "Ramipril 2.5 mg OD"),
    HW("Insulin 10 units at night", "Insulin 10 units at night", "lnsulin 1O units at night"),
    HW("Review on 21/04", "Review on 21/04", "Review on 2l/04"),
    HW("Amoxicillin 500 mg tds 5 days", "Amoxicillin 500 mg tds 5 days", "Amoxicilin 5OO mg tds S days"),
    HW("BP 140/90", "BP 140/90", "BP 14O/9O"),
    HW("Ramipril 2.5 mg", "Ramipril Z.S my", "Ramipril 2.5 mg", cat="hw_num_misread_one"),
    HW("Warfarin 3 mg", "Warfarin 3 mg", "Warfarin B my", chosen="hw", cat="hw_num_misread_one"),
    HW("Prednisolone 40 mg", "Prednisolone 4O my", "Prednisolone 40 mg", cat="hw_num_misread_one"),
    HW("Salbutamol 2 puffs", "Salbutamol Z puffs", "Salbutamol 2 puffs", chosen="printed", cat="hw_num_misread_one"),
    HW("Folic acid 5 mg", "Folic acid S my", "Folic acid 5 mg", cat="hw_num_misread_one"),
    HW("Ramipril 2.5 mg", "Ramipril Z.S my", "Ramipril Z5 my", cat="hw_num_misread_both"),
    HW("Insulin 10 units", "InSulin O Units", "lnsulin lO Units", cat="hw_num_misread_both"),
    HW("Metformin 500 mg", "Metformin SOO my", "Metformin S0O rng", cat="hw_num_misread_both"),
    HW("Insulin 10 Units", "InSulin O Units", "lnsulin O Units", cat="unit_only"),
    P("Omeprazole 20 mg", "Omeprazole mg", cat="unit_only"),
    P("Insulin glargine 12 units", "Insulin glargine IZ Units", cat="unit_only"),
    HW("Morphine 5 mL", "Morphine S mL", "Morphine S ML", cat="unit_only"),
    HW("Dr A. Example", "Dr A. Example", "Dr A Exarnple", cat="hw_text"),
    HW("Continue same medicines", "Continue same medicines", "Contlnue same rnedicines", cat="hw_text"),
    HW("Signed", "Signed", "Slgned", chosen="printed", cat="hw_text"),
    HW("See me if worse", "See me if worse", "See rne if worse", cat="hw_text"),
    HW("Good progress", "Good progress", "Gocd progress", cat="hw_text"),
    P("Mehta OD Clinic, Station Road", cat="collision"), P("Dr L Example", cat="collision"),
    P("G Sample, Pharmacist", cat="collision"), P("Department of Mol Biology", cat="collision"),
    P("Tab Pharmacy, High Street", cat="collision"), P("Caps and gowns must be worn", cat="collision"),
    P("IU Health partner clinic", cat="collision"), P("Prn as directed by the doctor", cat="collision"),
    P("BD Diagnostics Ltd", cat="collision"), P("Mane Street Surgery", cat="collision"),
    P("Name: L Sample", cat="collision"), P("ML Sample", cat="collision"), P("Dl Sample", cat="collision"),
    P("Initials: A G", cat="collision"), P("TDS Medical Group", cat="collision"),
    V1("Paracetamol 500 mg"), V1("Date: 03/02/2026"), V1("Metformin 500 mg", "Metformin SOO my", cat="v1_misread"),
    V1("Patient: Jane Sample", cat="v1_text"), V1("Take after meals", cat="v1_text"), V1("Dr L Example", cat="v1_text"),
]

# The two one-page lists of the earlier script (all printed and read correctly, except the two HW lines).
LAB_REPORT = ["SAMPLE DIAGNOSTICS LABORATORY", "12 Example Road, Sampletown", "Phone: 020 7946 0000",
              "Patient: Jane Sample", "Age/Sex: 41 / F", "Patient ID: HD-000123", "Referred by: Dr A. Example",
              "Collected: 14/03/2026 08:10", "Reported: 14/03/2026 13:45", "COMPLETE BLOOD COUNT",
              "Test Result Units Reference range", "Haemoglobin 13.2 g/dL 12.0-15.5", "RBC count 4.6 x10^12/L 3.8-5.2",
              "WBC count 7.1 x10^9/L 4.0-11.0", "Platelets 250 x10^9/L 150-400", "Haematocrit 40 % 36-46",
              "MCV 88 fL 80-100", "BIOCHEMISTRY", "Fasting glucose 5.4 mmol/L 3.9-5.5", "HbA1c 6.1 % 4.0-6.0",
              "Total cholesterol 4.8 mmol/L <5.0", "Creatinine 72 umol/L 45-90", "Sodium 139 mmol/L 135-145",
              "Potassium 4.2 mmol/L 3.5-5.0", "TSH 2.1 mIU/L 0.4-4.0", "Interpretation: HbA1c slightly above range",
              "Please correlate clinically", "Report verified by the laboratory", "Page 1 of 1", "End of report"]
PRESCRIPTION = ["SAMPLE FAMILY CLINIC", "Dr A. Example, MBBS, MD", "Reg. No. 00000", "3 Example Street, Sampletown",
                "Date: 14/03/2026", "Patient: Jane Sample  Age: 41  Sex: F", "Diagnosis: Hypertension, type 2 diabetes",
                "Rx", "1. Tab Amlodipine 5 mg OD x 30 days", "2. Tab Metformin 500 mg BD after meals",
                "3. Tab Atorvastatin 10 mg at night", "4. Cap Omeprazole 20 mg before breakfast",
                "Advice:", "Low salt diet", "Walk for 30 minutes daily", "Check blood sugar weekly",
                "Review after 4 weeks", "Signature"]
PRESCRIPTION_HW = [HW("Increase Amlodipine to 10 mg", "Increase Amlodipine to 10 mg", "lncrease Arnlodipine to 1O rng",
                      cat="page_rx"),
                   HW("Dr A. Example", "Dr A. Example", "Dr A Exarnple", cat="page_rx")]

# ---------------------------------------------------------------- new cases (all invented)
NEW = [
    # unit without its number, with a list index / duration / another medicine's dose elsewhere on the line
    P("Omeprazole 20 mg", "Omeprazole mg", cat="unit_lost_context"),
    P("1. Omeprazole 20 mg", "1. Omeprazole mg", cat="unit_lost_context"),
    P("Omeprazole 20 mg for 7 days", "Omeprazole mg for 7 days", cat="unit_lost_context"),
    P("1. Omeprazole 20 mg", cat="numbered_list"),
    P("Amlodipine 5 mg and Omeprazole 20 mg", "Amlodipine 5 mg and Omeprazole mg", cat="unit_lost_context"),
    P("2. Tab Omeprazole 20 mg OD", "2. Tab Omeprazole mg OD", cat="unit_lost_context"),
    P("Omeprazole 20 mg x 14 days", "Omeprazole mg x 14 days", cat="unit_lost_context"),
    P("3) Ramipril 2.5 mg OD", "3) Ramipril mg OD", cat="unit_lost_context"),
    P("Metformin 500 mg BD, Gliclazide 80 mg OD", "Metformin 500 mg BD, Gliclazide mg OD", cat="unit_lost_context"),
    P("Review in 2 weeks: Omeprazole 20 mg", "Review in 2 weeks: Omeprazole mg", cat="unit_lost_context"),
    HW("1. Omeprazole 20 mg", "1. Omeprazole mg", "1. Omeprazole mg", cat="unit_lost_context"),
    HW("Omeprazole 20 mg for 7 days", "Omeprazole mg for 7 days", "Omeprazole mg for 7 days", cat="unit_lost_context"),
    V1("1. Omeprazole 20 mg", "1. Omeprazole mg", cat="unit_lost_context"),
    V1("Omeprazole 20 mg for 7 days", "Omeprazole mg for 7 days", cat="unit_lost_context"),
    V1("Amlodipine 5 mg and Omeprazole 20 mg", "Amlodipine 5 mg and Omeprazole mg", cat="unit_lost_context"),
    # "Metformin SOO my" under every routing, the misread reading in either position
    P("Metformin 500 mg", "Metformin SOO my", cat="metformin_soo"),
    HW("Metformin 500 mg", "Metformin SOO my", "Metformin 500 mg", cat="metformin_soo"),
    HW("Metformin 500 mg", "Metformin 500 mg", "Metformin SOO my", cat="metformin_soo"),
    HW("Metformin 500 mg", "Metformin 500 mg", "Metformin SOO my", chosen="printed", cat="metformin_soo"),
    HW("Metformin 500 mg", "Metformin SOO my", "Metformin 500 mg", chosen="printed", cat="metformin_soo"),
    HW("Metformin 500 mg", "Metformin SOO my", "Metformin SOO my", cat="metformin_soo"),
    HW("Metformin 500 mg", "Metformin SOO my", "Metformin SOO my", chosen="printed", cat="metformin_soo"),
    V1("Metformin 500 mg", "Metformin SOO my", cat="metformin_soo", source="handwritten"),
    # "InSulin O Units"
    P("Insulin 10 units", "InSulin O Units", cat="insulin_o_units"),
    HW("Insulin 10 units", "InSulin O Units", "Insulin 10 units", cat="insulin_o_units"),
    HW("Insulin 10 units", "InSulin O Units", "InSulin O Units", chosen="printed", cat="insulin_o_units"),
    V1("Insulin 10 units", "InSulin O Units", cat="insulin_o_units", source="handwritten"),
    # the dose survives only in the reading that is not shown
    HW("Ramipril 2.5 mg", "Ramipril mg", "Ramipril 2.5 mg", cat="dose_only_in_alternate"),
    HW("Warfarin 3 mg", "Warfarin 3 mg", "Warfarin my", chosen="printed", cat="dose_only_in_alternate"),
    HW("Bisoprolol 2.5 mg OD", "Bisoprolol OD", "Bisoprolol 2.5 mg OD", cat="dose_only_in_alternate"),
    HW("Levothyroxine 75 mcg", "Levothyroxine 75 mcg", "Levothyroxine mcg", chosen="printed",
       cat="dose_only_in_alternate"),
    # decimals
    P("Ramipril 2.5 mg OD", "Ramipril 25 mg OD", cat="decimal"),
    HW("Ramipril 2.5 mg OD", "Ramipril 25 mg OD", "Ramipril 2.5 mg OD", cat="decimal"),
    HW("Ramipril 2.5 mg OD", "Ramipril 25 mg OD", "Ramipril 25 mg OD", cat="decimal"),
    V1("Ramipril 2.5 mg OD", "Ramipril 25 mg OD", cat="decimal"),
    P("Lorazepam 0.5 mg", "Lorazepam .5 mg", cat="decimal"),
    P("Lorazepam 0.5 mg", "Lorazepam 5 mg", cat="decimal"),
    P("Lorazepam 0.5 mg", cat="decimal"),
    HW("Lorazepam 0.5 mg prn", "Lorazepam 05 mg prn", "Lorazepam 0.5 mg prn", cat="decimal"),
    HW("Digoxin 62.5 mcg OD", "Digoxin 625 mcg OD", "Digoxin 62.5 mcq OD", cat="decimal"),
    P("Warfarin 2.5 mg", cat="decimal"),
    V1("Lorazepam 0.5 mg", "Lorazepam .5 mg", cat="decimal"),
    # missing quantities on the page itself (nothing lost)
    P("Paracetamol as needed", cat="no_quantity"),
    P("Omeprazole before breakfast", cat="no_quantity"),
    HW("Continue Metformin", "Continue Metformin", "Contlnue Metformin", cat="no_quantity"),
    V1("Stop Aspirin", cat="no_quantity"),
    # units
    P("Levothyroxine 50 mcg OD", cat="units"), P("Levothyroxine 50 mcg OD", "Levothyroxine 50 mcq OD", cat="units"),
    P("Lactulose 10 mL bd", cat="units"), P("Lactulose 10 mL bd", "Lactulose 10 mI bd", cat="units"),
    P("Vitamin D 1000 IU daily", cat="units"), P("Vitamin D 1000 IU daily", "Vitamin D IOOO IU daily", cat="units"),
    P("Potassium 4.2 mmol/L 3.5-5.0", cat="units"), P("Haemoglobin 11.8 g/dL", cat="units"),
    P("Haemoglobin 11.8 g/dL", "Haemoglobin 11.8 q/dL", cat="units"),
    HW("Enoxaparin 40 mg SC OD", "Enoxaparin 40 mg SC OD", "Enoxaparin 4O rng SC OD", cat="units"),
    HW("Lactulose 15 mL bd", "Lactulose 1S mL bd", "Lactulose 15 ML bd", cat="units"),
    V1("Vitamin D 1000 IU daily", cat="units"), V1("Potassium 4.2 mmol/L", cat="units"),
    # frequencies
    P("Paracetamol 1 g prn", cat="frequency"), P("Amitriptyline 10 mg nocte", cat="frequency"),
    P("Metformin 500 mg bd", cat="frequency"), P("Amoxicillin 500 mg tds", cat="frequency"),
    P("Aspirin 75 mg od", cat="frequency"), P("Salbutamol 2 puffs twice daily", cat="frequency"),
    P("Amitriptyline 10 mg nocte", "Amitriptyline mg nocte", cat="frequency"),
    P("Amitriptyline 10 mg nocte", "Amitriptyline nocte", cat="frequency"),
    P("Paracetamol 1 g prn", "Paracetamol g prn", cat="frequency"),
    HW("Aspirin 75 mg od", "Aspirin 75 mg od", "Asplrin 7S mg od", cat="frequency"),
    HW("Amoxicillin 500 mg tds", "Amoxicillin 500 mg tds", "Amoxicillin 500 mg tds", chosen="printed", cat="frequency"),
    V1("Amitriptyline 10 mg nocte", cat="frequency"),
    # frequency or count words without a digit, read correctly (no number was lost)
    P("Apply the cream bd", cat="frequency_no_number"), P("Salbutamol inhaler prn", cat="frequency_no_number"),
    P("Take one tablet at night", cat="frequency_no_number"), P("Take two tablets twice daily", cat="frequency_no_number"),
    P("Eye drops nocte", cat="frequency_no_number"), P("Paracetamol tablets as needed", cat="frequency_no_number"),
    V1("Apply the cream bd", cat="frequency_no_number"),
    # numbered lists
    P("1. Tab Amlodipine 5 mg OD", cat="numbered_list"), P("2) Metformin 500 mg BD", cat="numbered_list"),
    P("3. Paracetamol", cat="numbered_list"), P("4. Cap Omeprazole 20 mg", "4. Cap Omeprazole 2O mg", cat="numbered_list"),
    HW("2. Ramipril 5 mg", "2. Ramipril S mg", "2. Ramipril 5 mg", cat="numbered_list"),
    V1("1. Tab Amlodipine 5 mg OD", cat="numbered_list"),
    # ordinary printed values and dates, read correctly
    P("Sodium 139 mmol/L 135-145", cat="printed_value"), P("Creatinine 72 umol/L", cat="printed_value"),
    P("TSH 2.1 mIU/L", cat="printed_value"), P("Collected: 02/06/2026 09:15", cat="printed_value"),
    P("Invoice no. 4471", cat="printed_value"), P("Room 3B, Floor 2", cat="printed_value"),
    P("Pulse 72 bpm", cat="printed_value"), P("Weight 68.5 kg", cat="printed_value"),
    P("Date of birth: 1984", cat="printed_value"),
    P("Collected: 02/06/2026 09:15", "Collected: 02/06/2026 O9:15", cat="printed_value_misread"),
    # unit-word collisions (no number on the line)
    P("Dr L Example", cat="collision"), P("Mehta OD Clinic", cat="collision"),
    P("Test Result Units Reference range", cat="collision"), P("Mg Road Pharmacy", cat="collision"),
    P("Units: see reference column", cat="collision"), P("G Example, MD", cat="collision"),
    P("Caps Lock Tutoring Centre", cat="collision"), P("Bo Tab and Sons Chemists", cat="collision"),
    P("Ask Bo my pharmacist", cat="collision"), P("So my doctor said to continue", cat="collision"),
    P("OBS stable overnight", cat="collision"), P("ZOO Road Pharmacy", cat="collision"),
    HW("Dr L Example", "Dr L Example", "Dr L Exarnple", cat="collision"),
    HW("Mehta OD Clinic", "Mehta OD Clinic", "Mehta 0D Clinic", cat="collision"),
    HW("Test Result Units Reference range", "Test Resu1t Units Reference range", "Test Result Units Reference range",
       chosen="printed", cat="collision"),
    V1("Mehta OD Clinic", cat="collision"), V1("Test Result Units Reference range", cat="collision"),
    # plain text negative controls
    P("Please bring this letter to your next appointment", cat="plain_text"), P("Allergies: none known", cat="plain_text"),
    P("Keep out of the reach of children", cat="plain_text"), P("Ward round notes", cat="plain_text"),
    HW("Feeling better today", "Feeling better today", "Feeling betler today", cat="plain_text"),
    HW("Seen by the nurse", "Seen by the nurse", "Seen by the nurse", chosen="printed", cat="plain_text"),
    V1("Keep out of the reach of children", cat="plain_text"),
    # handwritten lines read correctly on v1 (v1 cannot tell print from handwriting)
    V1("Ramipril 2.5 mg OD", cat="v1_handwritten", source="handwritten"),
    V1("Warfarin 3 mg", "Warfarin B my", cat="v1_handwritten", source="handwritten"),
    V1("Prednisolone 40 mg", "Prednisolone 4O my", cat="v1_handwritten", source="handwritten"),
]

# ---------------------------------------------------------------- round 2 additions (set "extended_r2", appended only)
NEW_R2 = [
    # count or frequency words without a digit: correct print (no warning warranted) and doses that were lost
    P("Take one tablet at night", cat="r2_count_freq_no_digit"), P("Take two tablets twice daily", cat="r2_count_freq_no_digit"),
    P("Apply the cream bd", cat="r2_count_freq_no_digit"), P("Eye drops nocte", cat="r2_count_freq_no_digit"),
    P("Salbutamol inhaler prn", cat="r2_count_freq_no_digit"), P("Paracetamol tablets as needed", cat="r2_count_freq_no_digit"),
    P("One puff when needed", cat="r2_count_freq_no_digit"), P("Two drops in each eye tds", cat="r2_count_freq_no_digit"),
    P("Half a tablet in the morning", cat="r2_count_freq_no_digit"),
    V1("Take one tablet at night", cat="r2_count_freq_no_digit"), V1("Eye drops nocte", cat="r2_count_freq_no_digit"),
    V1("Use two puffs prn", cat="r2_count_freq_no_digit"),
    HW("Take one tablet daily", "Take one tablet daily", "Take one tablet daily", cat="r2_count_freq_no_digit"),
    P("Amitriptyline 10 mg nocte", "Amitriptyline nocte", cat="r2_count_freq_no_digit"),
    P("Paracetamol 1 g prn", "Paracetamol prn", cat="r2_count_freq_no_digit"),
    P("Salbutamol 2 puffs prn", "Salbutamol puffs prn", cat="r2_count_freq_no_digit"),
    P("Bisoprolol 2.5 mg od", "Bisoprolol od", cat="r2_count_freq_no_digit"),
    V1("Amitriptyline 10 mg nocte", "Amitriptyline mg nocte", cat="r2_count_freq_no_digit"),
    # lab layouts with the unit before the value, and unit column headers
    P("Glucose (mmol/L) 5.4", cat="r2_unit_before_value"), P("Glucose mmol/L 5.4", cat="r2_unit_before_value"),
    P("Test (units) Result Reference range", cat="r2_unit_before_value"), P("Test Units Result", cat="r2_unit_before_value"),
    P("Haemoglobin (g/dL) 13.2", cat="r2_unit_before_value"), P("Potassium mmol/L 4.2 3.5-5.0", cat="r2_unit_before_value"),
    P("Ferritin (ug/L) 45", cat="r2_unit_before_value"), P("Creatinine (umol/L) 72", cat="r2_unit_before_value"),
    P("Analyte (mmol/L)", cat="r2_unit_before_value"),
    V1("Glucose (mmol/L) 5.4", cat="r2_unit_before_value"), V1("Test (units) Result Reference range", cat="r2_unit_before_value"),
    P("Glucose (mmol/L) 5.4", "Glucose (mmol/L)", cat="r2_unit_before_value"),
    P("Glucose mmol/L 5.4", "Glucose mmol/L S.4", cat="r2_unit_before_value"),
    P("Haemoglobin (g/dL) 13.2", "Haemoglobin (g/dL) l3.2", cat="r2_unit_before_value"),
    P("Ferritin (ug/L) 45", "Ferritin (ug/L) 4S", cat="r2_unit_before_value"),
    V1("Potassium mmol/L 4.2", "Potassium mmol/L", cat="r2_unit_before_value"),
    # a lower-case form word before the medicine
    P("tab Metformin 500 mg", cat="r2_form_word"), P("cap Omeprazole 20 mg", cat="r2_form_word"),
    P("tab Amlodipine 5 mg od", cat="r2_form_word"), P("1. tab Ramipril 2.5 mg", cat="r2_form_word"),
    P("cap Amoxicillin 500 mg tds", cat="r2_form_word"), P("tabs Paracetamol 1 g prn", cat="r2_form_word"),
    P("inj Insulin 10 units", cat="r2_form_word"),
    V1("tab Metformin 500 mg", cat="r2_form_word"), V1("cap Omeprazole 20 mg", cat="r2_form_word"),
    P("tab Metformin 500 mg", "tab Metformin mg", cat="r2_form_word"),
    P("cap Omeprazole 20 mg", "cap Omeprazole 2O mg", cat="r2_form_word"),
    V1("tab Amlodipine 5 mg", "tab Amlodipine S mg", cat="r2_form_word"),
    HW("tab Ramipril 2.5 mg", "tab Ramipril 25 mg", "tab Ramipril 2.5 mg", cat="r2_form_word"),
    # v1 readings with digits misread as letters, and v1 sentences made of look-alike letters
    V1("Metformin 500 mg", "Metformin SOO my", cat="r2_v1_misread"), V1("Warfarin 3 mg", "Warfarin B my", cat="r2_v1_misread"),
    V1("Amlodipine 5 mg", "Amlodipine S my", cat="r2_v1_misread"),
    V1("Atorvastatin 20 mg at night", "Atorvastatin ZO my at night", cat="r2_v1_misread"),
    V1("Prednisolone 40 mg", "Prednisolone 4O my", cat="r2_v1_misread"),
    V1("Levothyroxine 50 mcg", "Levothyroxine SO mcq", cat="r2_v1_misread"),
    V1("Insulin 10 units", "lnsulin lO Units", cat="r2_v1_misread"), V1("Ramipril 2.5 mg", "Ramipril Z.S my", cat="r2_v1_misread"),
    V1("So my doctor said to continue", cat="r2_v1_misread"), V1("Ask Bo my pharmacist", cat="r2_v1_misread"),
    V1("OBS stable overnight", cat="r2_v1_misread"), V1("ZOO Road Pharmacy", cat="r2_v1_misread"),
    V1("Is my dose right", cat="r2_v1_misread"), V1("Boss my manager", cat="r2_v1_misread"),
    # misread units next to digits that were kept
    P("Levothyroxine 50 mcg OD", "Levothyroxine 50 mcq OD", cat="r2_misread_unit"),
    P("Lactulose 10 mL bd", "Lactulose 10 mI bd", cat="r2_misread_unit"),
    P("Haemoglobin 11.8 g/dL", "Haemoglobin 11.8 q/dL", cat="r2_misread_unit"),
    P("Metformin 500 mg bd", "Metformin 500 my bd", cat="r2_misread_unit"),
    P("Atorvastatin 20 mg", "Atorvastatin 20 rng", cat="r2_misread_unit"),
    V1("Levothyroxine 50 mcg", "Levothyroxine 50 mcq", cat="r2_misread_unit"),
    V1("Lactulose 10 mL", "Lactulose 10 mI", cat="r2_misread_unit"),
    HW("Digoxin 125 mcg OD", "Digoxin 125 mcq OD", "Digoxin 125 mcg OD", cat="r2_misread_unit"),
    P("Levothyroxine 50 mcg OD", cat="r2_misread_unit"), P("Lactulose 10 mL bd", cat="r2_misread_unit"),
    P("Haemoglobin 11.8 g/dL", cat="r2_misread_unit"), V1("Metformin 500 mg bd", cat="r2_misread_unit"),
    P("Since 2019 my blood pressure is fine", cat="r2_misread_unit"), P("In 2024 my dose was reduced", cat="r2_misread_unit"),
    V1("Since 2019 my blood pressure is fine", cat="r2_misread_unit"),
]


def _is_numeric(text: str) -> bool:
    return bool(re.search(r"\d", text)) or bool(_QTY_WORD.search(text))


def _dose_wrong(gt: str, shown: str) -> bool:
    return numbers(gt) != numbers(shown) or units(gt) != units(shown) or parse_fields(gt) != parse_fields(shown)


def finish(r: dict, rid: str, set_name: str, origin: str) -> dict:
    r = dict(r)
    r["id"], r["set"], r["origin"] = rid, set_name, origin
    r["path"] = "v1" if r["routing"] == "none_v1" else "v2"
    if "source_script" not in r:
        r["source_script"] = "printed" if r["routing"] in ("printed_only", "none_v1") else "handwritten"
    gt, shown = r["ground_truth"], r["displayed_text"]
    r["has_numeric_content"] = _is_numeric(gt)
    r["dose_incorrect_or_lost"] = _dose_wrong(gt, shown)
    r["warning_warranted"] = bool(r["dose_incorrect_or_lost"]
                                  or (r["source_script"] == "handwritten" and r["has_numeric_content"]))
    r["expected_fields"] = parse_fields(gt)
    keys = ("id", "set", "category", "path", "routing", "source_script", "ground_truth", "displayed_text",
            "handwriting_text", "printed_text", "has_numeric_content", "dose_incorrect_or_lost", "warning_warranted",
            "expected_fields", "origin")
    return {k: r[k] for k in keys if k in r}


def build() -> list[dict]:
    recs = []
    o = e = 0
    for r in OLD_CASES:
        if re.search(r"\d", r["ground_truth"]):
            o += 1
            recs.append(finish(r, f"o39-{o:02d}", "original_39", "confirm_cases.CASES"))
    for r in OLD_CASES:
        if not re.search(r"\d", r["ground_truth"]):
            e += 1
            recs.append(finish(r, f"ext-{e:03d}", "extended", "confirm_cases.CASES"))
    pages = ([P(t, cat="page_lab") for t in LAB_REPORT] + [P(t, cat="page_rx") for t in PRESCRIPTION] + PRESCRIPTION_HW
             + [V1(t, cat="page_lab_v1") for t in LAB_REPORT] + [V1(t, cat="page_rx_v1") for t in PRESCRIPTION]
             + [V1(r["ground_truth"], cat="page_rx_v1", source="handwritten") for r in PRESCRIPTION_HW])
    for r in pages:
        e += 1
        recs.append(finish(r, f"ext-{e:03d}", "extended", "confirm_cases.LAB_REPORT/PRESCRIPTION"))
    for r in NEW:
        e += 1
        recs.append(finish(r, f"ext-{e:03d}", "extended", "new"))
    for k, r in enumerate(NEW_R2, 1):  # appended after every round-1 record, which stay as they were
        recs.append(finish(r, f"r2-{k:03d}", "extended_r2", "new_r2"))
    return recs


def dumps(recs: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in recs)


if __name__ == "__main__":
    recs = build()
    with open(OUT, "w") as f:
        f.write(dumps(recs))
    n39 = sum(r["set"] == "original_39" for r in recs)
    print(f"wrote {len(recs)} records ({n39} original_39) to {OUT}")

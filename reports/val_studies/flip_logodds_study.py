"""Validation study: should the 180-degree check also use the orientation classifier's log-odds?

    python reports/val_studies/flip_logodds_study.py > reports/val_studies/flip_logodds_val.txt

No model is loaded: it reads orientation_flip_val.json (232 validation pages; for each, the classifier's
probabilities for all 4 quarter-turns of the page, and the printed recogniser's mean confidence on the
page's largest word crops as they are (c0) and turned 180 degrees (c2)). As in analyse.py's flip sweep,
each page is shown in all 4 turns; a turned page's crops are approximated by turning the upright
crops (so the flip margin of an upside-down page is c2 - c0).

Pipeline simulated (ocr.scan.Scanner.run): the classifier's turn is applied when its probability is at
least ORIENT_MIN_CONF; then, if the page is now upright or upside down, it is turned 180 degrees when

    margin + a * L < -m

where margin is the recogniser flip margin and L = log(P(as is) / P(turned 180)) from the classifier
after its turn (a = 0 is today's rule, m = FLIP_MARGIN = 0.10). Pages left at 90 degrees cannot be
fixed by a 180-degree turn and count as errors for every rule.
"""
import json
import os

import numpy as np

V = os.path.dirname(os.path.abspath(__file__))
ORIENT_MIN_CONF = 0.6  # ocr.scan.ORIENT_MIN_CONF
SETS = ("funsd", "sroie", "cord", "gnhk")


def simulate(rows, a: float, m: float) -> dict:
    hit = {s: [] for s in SETS}
    for x in rows:
        p, d = np.clip(np.array(x["p"], float), 1e-6, 1), x["c0"] - x["c2"]
        for k in range(4):
            pred = int(p[k].argmax())
            turn = pred if p[k, pred] >= ORIENT_MIN_CONF else 0
            res = (k - turn) % 4  # quarter-turns still left on the page
            if res in (0, 2):
                margin = d if res == 0 else -d
                L = float(np.log(p[k, turn] / p[k, (turn + 2) % 4]))  # classifier: as is vs turned 180
                if margin + a * L < -m:
                    res = (res + 2) % 4
            hit[x["src"]].append(res == 0)
    out = {s: float(np.mean(h)) for s, h in hit.items()}
    out["all"] = float(np.mean([v for h in hit.values() for v in h]))
    out["errors"] = int(sum(len(h) - sum(h) for h in hit.values()))
    return out


def main() -> None:
    rows = json.load(open(os.path.join(V, "orientation_flip_val.json")))
    print(f"{len(rows)} validation pages x 4 turns = {4 * len(rows)} cases")
    print(f"{'a':>6} {'m':>5}  {'all':>6} {'errors':>6}  " + "  ".join(f"{s:>6}" for s in SETS))
    for a in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
        for m in (0.06, 0.08, 0.10, 0.12, 0.15):
            r = simulate(rows, a, m)
            mark = "  <- today" if a == 0 and m == 0.10 else ""
            print(f"{a:>6} {m:>5}  {r['all']:.4f} {r['errors']:>6}  " + "  ".join(f"{r[s]:.4f}" for s in SETS) + mark)
    r = simulate(rows, 0.0, float("inf"))
    print(f"classifier alone (no flip check): all {r['all']:.4f}, errors {r['errors']}")


if __name__ == "__main__":
    main()

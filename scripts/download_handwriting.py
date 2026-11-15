#!/usr/bin/env python3
"""Download the handwriting datasets into data/raw/<name>/ (pinned revisions, no login needed).

    python scripts/download_handwriting.py            # IAM lines, GNHK, handwriting fonts
    python scripts/download_handwriting.py --only iam

Nothing inside the downloads is executed. Archives are only read by scripts/prepare_handwriting.py.
Sources, licences and sizes are recorded in data/DATASETS.md. Stops before a download that would
leave less than MIN_FREE_GB free or would exceed MAX_TOTAL_GB in total.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
MAX_TOTAL_GB = 20.0
MIN_FREE_GB = 5.0

HF = {
    "iam": {"repo": "Teklia/IAM-line", "revision": "fbdad97500ce54635c0d1ba306bf535cb40656cf", "gb": 0.27,
            "files": ["data/train.parquet", "data/validation.parquet", "data/test.parquet", "README.md"]},
    "gnhk": {"repo": "staghado/GNHK-Dataset", "revision": "eaa67d396fd29b8b04e38d630da79f67db11b212", "gb": 1.01,
             "files": ["GNHK-dataset.zip"]},
}
# Handwriting fonts from github.com/google/fonts (SIL Open Font License 1.1), pinned commit.
FONTS_COMMIT = "main"
FONTS = ["ofl/caveat/Caveat%5Bwght%5D.ttf",
         "ofl/kalam/Kalam-Regular.ttf",
         "ofl/kalam/Kalam-Light.ttf",
         "ofl/indieflower/IndieFlower-Regular.ttf",
         "ofl/shadowsintolight/ShadowsIntoLight.ttf",
         "ofl/patrickhand/PatrickHand-Regular.ttf",
         "ofl/gloriahallelujah/GloriaHallelujah.ttf",
         "ofl/reeniebeanie/ReenieBeanie.ttf",
         "ofl/nothingyoucoulddo/NothingYouCouldDo.ttf",
         "ofl/coveredbyyourgrace/CoveredByYourGrace.ttf",
         "ofl/gochihand/GochiHand-Regular.ttf",
         "ofl/handlee/Handlee-Regular.ttf",
         "ofl/dekko/Dekko-Regular.ttf",
         "ofl/zeyada/Zeyada.ttf",
         "ofl/waitingforthesunrise/WaitingfortheSunrise.ttf",
         "ofl/swankyandmoomoo/SwankyandMooMoo.ttf",
         "ofl/sueellenfrancisco/SueEllenFrancisco-Regular.ttf",
         "ofl/overtherainbow/OvertheRainbow.ttf",
         "ofl/lovedbytheking/LovedbytheKing.ttf",
         "ofl/cedarvillecursive/Cedarville-Cursive.ttf",
         "ofl/dawningofanewday/DawningofaNewDay.ttf",
         "ofl/shortstack/ShortStack-Regular.ttf",
         "ofl/neucha/Neucha.ttf",
         "ofl/annieuseyourtelescope/AnnieUseYourTelescope-Regular.ttf"]


def check_space(need_gb: float) -> None:
    used = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(RAW) for f in fs) / 1e9 if os.path.isdir(RAW) else 0
    free = shutil.disk_usage(ROOT).free / 1e9
    if used + need_gb > MAX_TOTAL_GB or free - need_gb < MIN_FREE_GB:
        sys.exit(f"stop: {need_gb:.1f} GB more would pass the limits (raw data {used:.1f} GB, free {free:.1f} GB)")


def get_hf(name: str) -> None:
    from huggingface_hub import hf_hub_download
    spec = HF[name]
    check_space(spec["gb"])
    for f in spec["files"]:
        p = hf_hub_download(spec["repo"], f, repo_type="dataset", revision=spec["revision"],
                            local_dir=os.path.join(RAW, name))
        print(name, f, round(os.path.getsize(p) / 1e6, 1), "MB", flush=True)


def get_fonts() -> None:
    out = os.path.join(RAW, "fonts")
    os.makedirs(out, exist_ok=True)
    for f in FONTS:
        dst = os.path.join(out, os.path.basename(f).replace("%5B", "[").replace("%5D", "]"))
        if os.path.exists(dst):
            continue
        url = f"https://raw.githubusercontent.com/google/fonts/{FONTS_COMMIT}/{f.replace(' ', '')}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dst + ".part", "wb") as o:
                shutil.copyfileobj(r, o)
            os.replace(dst + ".part", dst)
            print("font", os.path.basename(dst), flush=True)
        except Exception as e:  # a missing font is reported, not fatal
            print("font FAILED", f, e, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["iam", "gnhk", "fonts"])
    a = ap.parse_args()
    for n in ([a.only] if a.only else ["iam", "gnhk", "fonts"]):
        get_fonts() if n == "fonts" else get_hf(n)

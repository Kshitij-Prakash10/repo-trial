"""Download the real-photo datasets used to fine-tune and test the scanner, and verify their checksums.

  DocCornerDataset   Hugging Face mapo80/DocCornerDataset at a pinned revision (Parquet, about 11 GB)
  SmartDoc 2015 Ch1  frames.tar.gz from the v2.0.0 GitHub release (about 1 GB), extracted

Nothing is downloaded at run time; this is a one-off setup step. The images are not committed to this
repository (size, and DocCornerDataset mixes sources with research-only licences); the checksums and the
exact lists of images used are in data/real/manifests/.
CLI: python -m scanner.finetune.download --data-home DIR   (then leakcheck, prep_real, finetune, report)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request

from scanner.infer import ROOT

DOCCORNER_REPO = "mapo80/DocCornerDataset"
DOCCORNER_REVISION = "2f6d2d16e2a2a2afe038c43afbc52fa59ab17d22"
SMARTDOC_URL = "https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/frames.tar.gz"
MANIFESTS = os.path.join(ROOT, "data", "real", "manifests")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected(name: str) -> dict[str, str]:
    """{relative path: sha256} from a manifest written by `sha256sum`."""
    with open(os.path.join(MANIFESTS, name)) as f:
        return {p: h for h, p in (line.split() for line in f if line.strip())}


def doccorner(data_home: str) -> None:
    from huggingface_hub import snapshot_download
    dst = os.path.join(data_home, "doccorner")
    snapshot_download(DOCCORNER_REPO, repo_type="dataset", revision=DOCCORNER_REVISION, local_dir=dst,
                      allow_patterns=["train/*", "val/*", "test/*", "README.md"])
    bad = [p for p, h in expected("doccorner_parquet_sha256.txt").items() if sha256(os.path.join(dst, p)) != h]
    if bad:
        raise SystemExit(f"DocCornerDataset checksum mismatch: {bad}")
    print("DocCornerDataset OK")


def smartdoc(data_home: str) -> None:
    home = os.path.join(data_home, "smartdoc_ch1_home")
    os.makedirs(home, exist_ok=True)
    tgz = os.path.join(home, "frames.tar.gz")
    if not os.path.exists(tgz):
        urllib.request.urlretrieve(SMARTDOC_URL, tgz)
    if sha256(tgz) != expected("smartdoc_frames_sha256.txt")["frames.tar.gz"]:
        raise SystemExit("SmartDoc frames.tar.gz checksum mismatch")
    frames = os.path.join(home, "frames")
    if not os.path.exists(os.path.join(frames, "metadata.csv.gz")):
        os.makedirs(frames, exist_ok=True)
        with tarfile.open(tgz) as t:
            t.extractall(frames, filter="data")
    print("SmartDoc Ch1 OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-home", default=os.path.join(ROOT, "data", "raw"))
    a = ap.parse_args()
    doccorner(a.data_home)
    smartdoc(a.data_home)

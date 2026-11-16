#!/usr/bin/env bash
# Fetch the datasets whose licences allow re-sharing (GNHK, SROIE, CORD: CC BY 4.0),
# verify them against data/shareable.sha256.json and commit them with Git LFS.
#
#   bash scripts/fetch_shareable_data.sh          # then: git push
#
# Needs: git-lfs, Python with huggingface_hub, pyarrow and pillow (pip install -r requirements-ocr.txt),
# about 8 GB of free disk, and the internet (about 3.3 GB of downloads). Safe to run again: steps
# already done are skipped. IAM and FUNSD are not touched (their licences do not allow re-sharing).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

command -v git-lfs >/dev/null || { echo "git-lfs is not installed: https://git-lfs.com"; exit 1; }
git lfs install --local

"$PY" scripts/download_handwriting.py --only gnhk
"$PY" scripts/download_data.py --only sroie --only cord

"$PY" scripts/shareable_manifest.py || {
  echo "The files differ from data/shareable.sha256.json. Nothing was committed."; exit 1; }

git add data/raw/gnhk/GNHK-dataset.zip data/sroie data/cord
if git diff --cached --quiet; then
  echo "Nothing new to commit: the data is already in the repository."
else
  git commit -m "Add the shareable datasets with Git LFS: GNHK, SROIE and CORD (CC BY 4.0)"
  echo "Committed. Now run: git push"
fi

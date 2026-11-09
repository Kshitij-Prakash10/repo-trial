"""Near-duplicate check: every DocCornerDataset image vs every SmartDoc 2015 Ch1 frame (dHash, 256 bits).

Writes <data_home>/manifests/doccorner_vs_smartdoc_dhash.csv.gz: per DocCorner image, the nearest SmartDoc
frame and the Hamming distance. CLI: python -m scanner.finetune.leakcheck --data-home /path/to/data
"""
import argparse, glob, io, os
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image

def dhash(im):
    g = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())

def sd(path):
    im = Image.open(os.environ["LC_DATA_HOME"] + "/smartdoc_ch1_home/frames/" + path); im.draft("L", (64, 64))
    return dhash(im)

def dc(f):
    out = []
    for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["image", "filename", "is_negative"]):
        d = b.to_pydict()
        for img, fn, neg in zip(d["image"], d["filename"], d["is_negative"]):
            im = Image.open(io.BytesIO(img["bytes"])); im.draft("L", (64, 64))
            out.append((fn, bool(neg), dhash(im)))
    return f.split("/")[-2], out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-home", required=True)
    os.environ["LC_DATA_HOME"] = os.path.abspath(ap.parse_args().data_home)
    m = pd.read_csv(os.environ["LC_DATA_HOME"] + "/smartdoc_ch1_home/frames/metadata.csv.gz")
    with Pool(3, maxtasksperchild=4) as p:
        sh = np.stack(p.map(sd, m.image_path, chunksize=200))
        parts = p.map(dc, sorted(glob.glob(os.environ["LC_DATA_HOME"] + "/doccorner/*/*.parquet")), chunksize=1)
    sbits = np.unpackbits(sh, axis=1).astype(np.uint8)
    rows = []
    for split, items in parts:
        for fn, neg, h in items:
            rows.append((split, fn, neg, h))
    H = np.unpackbits(np.stack([r[3] for r in rows]), axis=1).astype(np.uint8)
    S = sbits.astype(np.int32) * 2 - 1
    res = []
    for i in range(0, len(H), 2000):
        A = H[i:i+2000].astype(np.int32) * 2 - 1
        ham = (256 - A @ S.T) // 2
        j = ham.argmin(1)
        for k in range(len(A)):
            r = rows[i+k]
            res.append({"split": r[0], "filename": r[1], "is_negative": r[2],
                        "nearest_smartdoc": m.image_path.iloc[j[k]], "hamming": int(ham[k, j[k]])})
    df = pd.DataFrame(res)
    df.to_csv(os.environ["LC_DATA_HOME"] + "/manifests/doccorner_vs_smartdoc_dhash.csv.gz", index=False)
    named = df.filename.str.startswith("smartdoc")
    for t in (0, 10, 20, 30, 40):
        print(f"hamming<={t}: {int((df.hamming<=t).sum())} total, named smartdoc {int(((df.hamming<=t)&named).sum())}, other {int(((df.hamming<=t)&~named).sum())}")
    print("named smartdoc hamming quantiles", df[named].hamming.quantile([.1,.5,.9]).tolist())
    print("others hamming quantiles", df[~named].hamming.quantile([.001,.01,.1,.5]).tolist())

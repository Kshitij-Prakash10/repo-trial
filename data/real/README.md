# Real-photo data

The real photos are not in this repository. DocCornerDataset is about 11 GB and includes research-only
sources; the SmartDoc 2015 Ch1 frames are about 1 GB. `python -m scanner.finetune.download --data-home DIR`
fetches both once, during setup, and checks them against the checksums in `manifests/`.

`manifests/` holds what is needed to rebuild the exact subsets:
- the checksum of every downloaded file;
- the DocCorner images used per split, and the ones excluded and why;
- the SmartDoc frames used for testing;
- the near-duplicate check between the two datasets.

See `docs/finetune/results.md`.

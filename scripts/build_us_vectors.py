"""Rebuild data/job_vecs.npy after changing data/jobs.csv (row i of the vectors = row i of the CSV).

Usage: python scripts/build_us_vectors.py  (about 3 minutes on a laptop CPU)"""

import numpy as np
import pandas as pd

from jobagent.config import PROJECT_ROOT
from jobagent.embed import embed_texts, job_text

JOBS = PROJECT_ROOT / "data" / "jobs.csv"
VECS = PROJECT_ROOT / "data" / "job_vecs.npy"


def main():
    jobs = pd.read_csv(JOBS, dtype={"job_id": str})
    texts = [job_text(str(t), str(d)) for t, d in zip(jobs["title"], jobs["description"])]
    vecs = embed_texts(texts)
    np.save(VECS, vecs)
    print(f"Saved {vecs.shape[0]} vectors of size {vecs.shape[1]} to {VECS.name}")


if __name__ == "__main__":
    main()

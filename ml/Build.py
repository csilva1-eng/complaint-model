"""
Stage 2-3 driver - parse once, build once, cache both
-----------------------------------------------------
  1. Parse the CSV into compact arrays (ingest.py). The slow step, so it is cached and keyed to
     the file's size and modification time.
  2. Collapse complaints into distinct profiles and embed them (embed.py).
  3. Build the FAISS index over the profile vectors (vector_index.py).
  4. Fit k-means on the training period and assign every profile to a theme (cluster.py).

Everything in step 2-4 is cached too, so changing only trend or export settings reruns in seconds.
"""
import hashlib
import json
import shutil
from dataclasses import dataclass

import joblib
import numpy as np

import cluster
import config as cfg
import embed
import ingest
import vector_index


@dataclass
class Built:
    prof: embed.Profiles
    embedder: object
    kmeans: object
    index: object
    index_kind: str
    Z: np.ndarray            # (profiles x dims) float32, row i = profile i
    labels: np.ndarray       # profile -> theme (int16)
    train_counts: np.ndarray # complaints per profile in the training period


def _md5(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def load_complaints_cached(rebuild=False) -> ingest.Complaints:
    if not cfg.RAW_CSV.exists():
        raise ingest.DataError(f"{cfg.RAW_CSV} not found. Download the CFPB complaints CSV and save it there.")
    st = cfg.RAW_CSV.stat()
    sig = _md5([str(cfg.RAW_CSV), st.st_size, int(st.st_mtime), cfg.MIN_DATE, cfg.MAX_DATE])
    path = cfg.CACHE_DIR / f"complaints_{sig}.joblib"
    if path.exists() and not rebuild:
        print("ingest: cache hit (use --rebuild-data to re-read the CSV)")
        return joblib.load(path)
    print("ingest: reading the CSV")
    c = ingest.load_complaints()
    c.signature = sig
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(c, path)
    return c


def build_or_load(c, cutoff_day: int, rebuild=False) -> Built:
    params = [c.signature, cutoff_day, cfg.PROFILE_FIELDS, cfg.BLOCK_WEIGHTS, cfg.SVD_COMPONENTS, cfg.FIT_SAMPLE,
              cfg.N_CLUSTERS, cfg.INDEX_KIND, cfg.IVF_NPROBE, cfg.PQ_M, cfg.RANDOM_STATE]
    cache = cfg.CACHE_DIR / f"build_{_md5(params)}"
    if rebuild and cache.exists():
        shutil.rmtree(cache)
    if (cache / "DONE").exists():
        print("build: cache hit")
        state = joblib.load(cache / "state.joblib")
        return Built(index=vector_index.load(cache / "index.faiss"), Z=np.load(cache / "Z.npy"), **state)

    cache.mkdir(parents=True, exist_ok=True)
    prof = embed.build_profiles(c)
    print(f"build: {len(c):,} complaints collapse into {prof.n:,} distinct profiles")

    train_rows = np.flatnonzero(c.days < cutoff_day)
    train_counts = np.bincount(prof.inverse[train_rows], minlength=prof.n)

    print("build: embedding profiles")
    embedder = embed.ProfileEmbedder(prof.sizes).fit(prof, train_rows)
    Z = embedder.encode_profiles(prof)

    print("build: building the FAISS index")
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    sample = rng.choice(prof.n, size=min(prof.n, 200_000), replace=False)
    index, kind = vector_index.new_index(Z.shape[1], prof.n, Z[sample])
    for s in range(0, prof.n, 200_000):
        index.add(Z[s:s + 200_000])

    print("build: clustering")
    kmeans = cluster.fit(Z, train_counts)
    labels = cluster.assign(kmeans, Z)

    np.save(cache / "Z.npy", Z)
    vector_index.save(index, cache / "index.faiss")
    joblib.dump(dict(prof=prof, embedder=embedder, kmeans=kmeans, index_kind=kind,
                     labels=labels, train_counts=train_counts), cache / "state.joblib")
    (cache / "DONE").write_text("ok")
    return Built(prof=prof, embedder=embedder, kmeans=kmeans, index=index, index_kind=kind,
                 Z=Z, labels=labels, train_counts=train_counts)
"""
Stage 3a - FAISS index (cosine similarity at scale)
---------------------------------------------------
FAISS (Meta's similarity-search library) holds the vectors and answers "which complaints are
nearest to this one?" without ever comparing against every row in Python.

Vectors are L2-normalised, so inner product == cosine similarity; we use the inner-product
indexes. Three flavours, chosen with cfg.INDEX_KIND:

  flat    exact search. Memory = 4 bytes x dims per vector (64 dims -> 256 B each, ~1 GB per 4M profiles).
  ivf     approximate: vectors are bucketed into `nlist` cells and only IVF_NPROBE cells are searched.
          Same memory as flat, far faster on millions of rows, may miss a few true neighbours.
  ivfpq   ivf + product quantisation: each vector is compressed to PQ_M bytes. Smallest memory,
          slightly less accurate. Use when even the vectors will not fit in RAM.

evaluate.index_recall() reports how many true neighbours the approximate kinds actually find.
Row positions are the FAISS ids: vector i is profile i (see embed.Profiles).
"""
import math

import faiss
import numpy as np

import config as cfg

_KEEPALIVE = []   # FAISS objects do not own their quantizer; hold it here so it is not garbage-collected


def _set_nprobe(index):
    ivf = faiss.try_extract_index_ivf(index)
    if ivf is not None:
        ivf.nprobe = cfg.IVF_NPROBE


def new_index(d: int, n_total: int, train: np.ndarray):
    """Create (and, for IVF kinds, train) an empty index. Returns (index, kind_used)."""
    kind = cfg.INDEX_KIND
    if kind == "auto":
        kind = "flat" if n_total <= 500_000 else "ivf"
    if kind == "flat":
        return faiss.IndexFlatIP(d), kind
    if kind not in ("ivf", "ivfpq"):
        raise ValueError(f"INDEX_KIND must be auto, flat, ivf or ivfpq (got {kind!r})")

    # ~4*sqrt(N) cells is the usual rule of thumb; FAISS wants >= ~39 training points per cell.
    nlist = max(1, int(min(4 * math.sqrt(n_total), len(train) / 40)))
    quantizer = faiss.IndexFlatIP(d)
    if kind == "ivf":
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        if d % cfg.PQ_M:
            raise ValueError(f"SVD_COMPONENTS ({d}) must be divisible by PQ_M ({cfg.PQ_M})")
        index = faiss.IndexIVFPQ(quantizer, d, nlist, cfg.PQ_M, 8, faiss.METRIC_INNER_PRODUCT)
    _KEEPALIVE.append(quantizer)
    index.train(np.ascontiguousarray(train, dtype=np.float32))
    _set_nprobe(index)
    print(f"  FAISS index: {kind}, nlist={nlist}")
    return index, kind


def search(index, Zq: np.ndarray, k: int, batch: int = 10_000):
    """Top-k (scores, row positions) for each query vector. Missing results come back as -1."""
    D = np.empty((len(Zq), k), np.float32)
    I = np.empty((len(Zq), k), np.int64)
    for s in range(0, len(Zq), batch):
        D[s:s + batch], I[s:s + batch] = index.search(np.ascontiguousarray(Zq[s:s + batch], dtype=np.float32), k)
    return D, I


def save(index, path):
    faiss.write_index(index, str(path))


def load(path):
    index = faiss.read_index(str(path))
    _set_nprobe(index)                          # nprobe is not stored in the file
    return index
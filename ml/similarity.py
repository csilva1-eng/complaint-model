"""
Stage 3b - Related complaints (cosine similarity)
-------------------------------------------------
Exact brute-force nearest neighbours with the cosine metric. At prototype scale (tens of
thousands of rows) this is fast and avoids approximate-index tuning.

The top-k neighbours of every complaint are precomputed and shipped as JSON, so the Java
backend can answer "show me similar complaints" with a lookup and needs no ML runtime.
Trade-off: you can search *from an existing complaint*, not from arbitrary new text. For
free-text semantic search, expose `Embedder.transform` behind a small service.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

import config as cfg


def build_neighbors(Z: np.ndarray, ids: np.ndarray, k: int = cfg.NEIGHBORS_K) -> dict:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    out = {}
    for row, (d_row, i_row) in enumerate(zip(dist, idx)):
        items = [
            {"id": int(ids[j]), "score": round(float(1.0 - d), 4)}
            for d, j in zip(d_row, i_row) if j != row          # drop the complaint itself
        ]
        out[int(ids[row])] = items[:k]
    return out

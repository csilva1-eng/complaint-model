"""
Stage 2 - Profiles and embeddings
---------------------------------
With no complaint text, a complaint is described by categories. Two steps make that tractable:

  1. COLLAPSE: complaints with the same (product, sub-product, issue, sub-issue, company) are
     one "profile" with a count. Millions of complaints become far fewer profiles, and every
     later stage (embedding, FAISS, k-means) works on profiles, not rows.
  2. EMBED: each profile becomes a weighted one-hot vector (one block per field), compressed
     to SVD_COMPONENTS dimensions with TruncatedSVD and L2-normalised, so cosine similarity is a
     plain dot product. Profiles that share product/issue/company land close together.

BLOCK_WEIGHTS in config.py decides what "similar" means. This is a deliberate modelling choice,
not a learned one: it says how much an issue match matters compared with a company match.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import config as cfg


@dataclass
class Profiles:
    codes: dict                # field -> int32 category code, one entry per profile
    counts: np.ndarray         # complaints per profile
    inverse: np.ndarray        # complaint row -> profile index (int32)
    example_row: np.ndarray    # profile -> row of its most recent complaint
    sizes: dict                # field -> number of categories

    @property
    def n(self) -> int:
        return len(self.counts)


def build_profiles(c) -> Profiles:
    fields = cfg.PROFILE_FIELDS
    sizes = {f: len(c.cats[f].categories) for f in fields}
    if np.prod([float(s) for s in sizes.values()]) >= 2.0 ** 62:
        raise ValueError("Too many category combinations to pack into one integer key")

    key = np.zeros(len(c), np.int64)                       # mixed-radix key: one int per complaint
    for f in fields:
        key = key * sizes[f] + c.cats[f].codes
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)

    codes, rem = {}, uniq.copy()                           # decode the key back into per-field codes
    for f in reversed(fields):
        codes[f] = (rem % sizes[f]).astype(np.int32)
        rem //= sizes[f]

    order = np.lexsort((c.days, inverse))                  # sort by profile, then by date
    example_row = order[np.cumsum(counts) - 1]             # last (= newest) complaint of each profile
    return Profiles(codes, counts.astype(np.int64), inverse.astype(np.int32), example_row, sizes)


class ProfileEmbedder:
    def __init__(self, sizes: dict):
        self.fields = list(cfg.EMBEDDING_FIELDS)
        block = [sizes[f] for f in self.fields]
        self.offsets = np.concatenate([[0], np.cumsum(block)[:-1]]).astype(np.int64)
        self.n_features = int(sum(block))
        self.weights = np.array([cfg.BLOCK_WEIGHTS[f] for f in self.fields], dtype=np.float32)
        self.svd = TruncatedSVD(n_components=min(cfg.SVD_COMPONENTS, self.n_features - 1),
                                random_state=cfg.RANDOM_STATE)

    def onehot(self, codes: dict) -> sparse.csr_matrix:
        """Weighted one-hot, exactly one entry per field per row."""
        n, f = len(codes[self.fields[0]]), len(self.fields)
        cols = np.stack([self.offsets[j] + codes[name].astype(np.int64)
                         for j, name in enumerate(self.fields)], axis=1)
        return sparse.csr_matrix((np.tile(self.weights, n), cols.ravel(), np.arange(0, n * f + 1, f)),
                                 shape=(n, self.n_features))

    def fit(self, prof: Profiles, train_rows: np.ndarray):
        """Fit the SVD on complaints (not profiles), so common profiles count in proportion."""
        rng = np.random.default_rng(cfg.RANDOM_STATE)
        rows = rng.choice(train_rows, size=min(cfg.FIT_SAMPLE, len(train_rows)), replace=False)
        p = prof.inverse[rows]
        self.svd.fit(self.onehot({f: prof.codes[f][p] for f in self.fields}))
        return self

    def encode(self, codes: dict) -> np.ndarray:
        return normalize(self.svd.transform(self.onehot(codes))).astype(np.float32)

    def encode_profiles(self, prof: Profiles, batch: int = 200_000) -> np.ndarray:
        Z = np.empty((prof.n, self.svd.n_components), np.float32)
        for s in range(0, prof.n, batch):
            sl = slice(s, s + batch)
            Z[sl] = self.encode({f: prof.codes[f][sl] for f in self.fields})
        return Z
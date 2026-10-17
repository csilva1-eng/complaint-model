"""
Stage 5 - Evaluate (prove it finds useful patterns, not just that it runs)
---------------------------------------------------------------------------
Because the themes are built FROM the same categorical fields CFPB uses for its own labels,
comparing them with CFPB's Issue/Product labels would be circular (the answer is an input), so
that comparison is deliberately not reported. What is reported instead:

  1. Theme stability: refit k-means with a different random seed. Do complaints keep landing in
     the same themes? (adjusted Rand index; 1.0 = identical grouping)
  2. Theme compactness: silhouette on a random sample of complaints.
  3. Novelty: what share of held-out-period complaints have a product/issue/company combination
     that never appeared in the training period? High means the models see a lot of "new" territory.
  4. Index recall: if you chose an approximate FAISS index, how many of the true nearest
     neighbours does it actually return? (An exact index is 1.0 by definition.)
  5. Human review. This module writes sheets for you to label, then scores them on the next run:
        data/artifacts/pairs_to_label.csv   -> fill `similar_1_or_0`,     save as data/labels/pairs_labeled.csv
        data/artifacts/alerts_to_review.csv -> fill `reviewer_verdict`
                                               (meaningful / noise),      save as data/labels/alerts_labeled.csv
     These give you precision@rank for similar-complaint search and the false-alert rate for trends.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

import cluster
import config as cfg
import vector_index


def cluster_quality(built, c) -> dict:
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    rows = rng.choice(len(c), size=min(50_000, len(c)), replace=False)
    p = built.prof.inverse[rows]                                  # complaints -> profiles
    km2 = cluster.fit(built.Z, built.train_counts, seed=cfg.RANDOM_STATE + 1)
    stability = adjusted_rand_score(built.labels[p], km2.predict(built.Z[p]))
    sil_rows = p[:5000]
    return {
        "stability_ari_two_seeds": round(float(stability), 3),
        "silhouette_sampled": round(float(silhouette_score(built.Z[sil_rows], built.labels[sil_rows])), 3),
    }


def novelty(built, c, cutoff_day) -> dict:
    held_out = c.days >= cutoff_day
    unseen = built.train_counts[built.prof.inverse[held_out]] == 0
    return {"held_out_complaints_with_unseen_profile": round(float(unseen.mean()), 3)}


def index_recall(index, kind, Z, k=cfg.NEIGHBORS_K, n_queries=500, block=50_000):
    """
    How many of the true top-k neighbours does the (approximate) index return?
    A neighbour counts as found if its similarity is at least the exact k-th best, so a tie for
    10th place is not a miss. That matters here: profiles that differ only in a field the SVD
    barely sees have identical vectors, which creates many exact ties.
    """
    if kind == "flat":
        return 1.0
    n = len(Z)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    qpos = np.sort(rng.choice(n, size=min(n_queries, n), replace=False))
    Q = Z[qpos]
    _, approx = vector_index.search(index, Q, k + 1)

    best = np.full((len(Q), k + 1), -np.inf, np.float32)             # exact top scores, one block at a time
    for s in range(0, n, block):
        S = Q @ Z[s:s + block].T
        kk = min(k + 1, S.shape[1])
        top_in_block = -np.partition(-S, kk - 1, axis=1)[:, :kk]
        best = -np.sort(-np.concatenate([best, top_in_block], axis=1), axis=1)[:, :k + 1]
    kth = best[:, k]                                                   # k-th best excluding the query itself

    hits = []
    for q, a_row, threshold in zip(range(len(qpos)), approx, kth):
        found = [int(j) for j in a_row if j != qpos[q] and j >= 0][:k]
        scores = Z[found] @ Q[q] if found else np.array([])
        hits.append(float((scores >= threshold - 1e-5).sum()) / k)
    return round(float(np.mean(hits)), 3)


def alert_summary(signals) -> dict:
    return {"flagged_cluster_weeks": int(len(signals)),
            "distinct_clusters_flagged": int(signals["clusterId"].nunique()) if len(signals) else 0}


def monthly_counts(days, last=12) -> dict:
    """Complaints per month. Check the tail: the newest weeks are still being filled in."""
    months = days.astype("datetime64[D]").astype("datetime64[M]")
    u, n = np.unique(months, return_counts=True)
    return {str(m): int(k) for m, k in list(zip(u, n))[-last:]}


def write_label_sheets(c, neighbors, row_of_id, signals):
    """Create the human-review sheets described in the module docstring."""
    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    keys = [q for q in neighbors if len(neighbors[q]) >= 5]
    queries = rng.choice(keys, size=min(100, len(keys)), replace=False) if keys else []

    rows = []
    for q in queries:
        q = int(q)
        for rank in (1, 5):                                   # best match and a mid-ranked one
            nb = neighbors[q][rank - 1]
            rows.append({"query_id": q, "neighbor_id": nb["id"], "rank": rank, "score": nb["score"],
                         "query_profile": c.describe_row(row_of_id[q]),
                         "neighbor_profile": c.describe_row(row_of_id[nb["id"]]),
                         "similar_1_or_0": ""})
    pd.DataFrame(rows).to_csv(cfg.ARTIFACT_DIR / "pairs_to_label.csv", index=False)
    signals.assign(reviewer_verdict="").to_csv(cfg.ARTIFACT_DIR / "alerts_to_review.csv", index=False)


def score_human_labels() -> dict:
    out = {}
    pairs, alerts = cfg.LABEL_DIR / "pairs_labeled.csv", cfg.LABEL_DIR / "alerts_labeled.csv"
    if pairs.exists():
        d = pd.read_csv(pairs).dropna(subset=["similar_1_or_0"])
        out["labeled_pairs"] = int(len(d))
        out["labeled_precision_by_rank"] = {str(k): round(float(v), 3)
                                           for k, v in d.groupby("rank")["similar_1_or_0"].mean().items()}
    if alerts.exists():
        d = pd.read_csv(alerts).dropna(subset=["reviewer_verdict"])
        out["reviewed_alerts"] = int(len(d))
        out["alert_precision"] = round(float((d["reviewer_verdict"] == "meaningful").mean()), 3) if len(d) else None
    return out
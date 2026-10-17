"""
Stage 3b - Cluster into themes
------------------------------
MiniBatchKMeans over profile vectors. Each profile is weighted by log(1 + its complaints in the
TRAINING period), so common profiles pull harder than rare ones, but a handful of enormous ones
(for example credit-report disputes) cannot claim every cluster. Profiles with no training-period
complaints are not used to fit, but are still assigned to a theme with `predict`, so a brand-new
product/issue/company combination in the "future" lands somewhere sensible.

Because themes are built from the same categorical fields CFPB already uses, a theme is best read
as "a recurring mix of product, issue and company", not as a newly discovered topic. The value is
in watching each mix's volume over time (trends.py), not in the grouping itself.

Each theme is described by its biggest issues, sub-issues, products and companies, plus the
profiles closest to its centre as representatives.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

import config as cfg


def fit(Z: np.ndarray, train_counts: np.ndarray, seed: int = cfg.RANDOM_STATE) -> MiniBatchKMeans:
    used = train_counts > 0
    km = MiniBatchKMeans(n_clusters=min(cfg.N_CLUSTERS, int(used.sum())), n_init=5,
                         batch_size=4096, random_state=seed)
    return km.fit(Z[used], sample_weight=np.log1p(train_counts[used]).astype(np.float64))


def assign(km: MiniBatchKMeans, Z: np.ndarray, batch: int = 500_000) -> np.ndarray:
    return np.concatenate([km.predict(Z[s:s + batch]) for s in range(0, len(Z), batch)]).astype(np.int16)


def _mix(sub: pd.DataFrame, field: str, c, n: int = cfg.N_TOP):
    s = sub.groupby(field)["count"].sum().sort_values(ascending=False)
    total = float(s.sum())
    return [{"name": str(c.cats[field].categories[code]), "share": round(float(v) / total, 3)}
            for code, v in s.head(n).items()]


def _label(issues, sub_issues, companies) -> str:
    parts = [issues[0]["name"]]
    if sub_issues and sub_issues[0]["name"] != "Unknown" and sub_issues[0]["share"] >= 0.3:
        parts.append(sub_issues[0]["name"])
    label = " / ".join(parts)
    if companies[0]["share"] >= cfg.LABEL_COMPANY_SHARE and companies[0]["name"] != "Unknown":
        label += f" ({companies[0]['name']})"
    return label


def describe(prof, labels, Z, km, c):
    """One JSON-ready dict per theme. Field names match the backend's `Cluster` record."""
    tab = pd.DataFrame({"cluster": labels, "count": prof.counts})
    for f in cfg.PROFILE_FIELDS:
        tab[f] = prof.codes[f]
    span = pd.DataFrame({"d": c.days, "k": labels[prof.inverse]}).groupby("k")["d"].agg(["min", "max"])
    centres = normalize(km.cluster_centers_)

    out, used = [], set()
    for k in range(km.n_clusters):
        sub = tab[tab["cluster"] == k]
        if sub.empty:
            continue
        issues, sub_issues = _mix(sub, "issue", c), _mix(sub, "sub_issue", c)
        companies, products = _mix(sub, "company", c), _mix(sub, "product", c)
        label = _label(issues, sub_issues, companies)
        if label in used:
            label += f" [{k}]"                              # keep theme names unique
        used.add(label)

        pool = sub[sub["count"] >= cfg.MIN_REP_COUNT]
        pool = pool if len(pool) else sub
        idx = pool.index.to_numpy()
        reps = idx[np.argsort(-(Z[idx] @ centres[k]))[: cfg.N_REPRESENTATIVES]]

        out.append({
            "clusterId": int(k),
            "size": int(sub["count"].sum()),
            "label": label,
            "topProducts": products,
            "topIssues": issues,
            "topSubIssues": sub_issues,
            "topCompanies": companies,
            "representativeIds": [int(i) for i in c.ids[prof.example_row[reps]]],
            "firstDate": str(np.datetime64(int(span.loc[k, "min"]), "D")),
            "lastDate": str(np.datetime64(int(span.loc[k, "max"]), "D")),
            "_repProfiles": reps.tolist(),                  # stripped before export
        })
    return out
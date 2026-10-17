"""
Entry point: live CFPB complaints CSV -> JSON artifacts for the Spring Boot backend.

    cd ml
    python run_pipeline.py                  # reuses cached data and models from a previous run
    python run_pipeline.py --rebuild-data   # re-read the CSV (the slow step)
    python run_pipeline.py --rebuild-index  # refit embeddings, FAISS and clusters (keeps parsed data)

Pipeline (each stage lives in its own module)
    ingest        stream the CSV -> compact integer arrays (no text)             ingest.py
    profiles      collapse identical complaints into distinct profiles           embed.py
    embed/index   one-hot -> SVD vectors -> FAISS nearest-neighbour index        embed.py, vector_index.py
    cluster       weighted k-means -> themes                                     cluster.py
    trends        weekly counts per theme, baseline, flagged spikes              trends.py
    export        themes + trends for EVERY complaint; records and neighbours
                  for a bounded slice                                            (below)
    evaluate      stability, novelty, index recall, human-review sheets          evaluate.py

Output contract (data/artifacts/) -- field names are camelCase to match the Java records:
    complaints.json  clusters.json  trends.json  signals.json  neighbors.json  eval_report.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import build
import cluster
import config as cfg
import evaluate
import ingest
import trends
import vector_index


def _native(o):
    """json.dump hook: numpy scalars -> plain Python."""
    return o.item() if hasattr(o, "item") else str(o)


def dump_json(obj, name):
    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.ARTIFACT_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=_native, ensure_ascii=False)


def pick_export_slice(built, c, reps):
    """
    Choose which profiles leave the pipeline for the dashboard: those of a random sample of
    complaints (so frequent profiles are well represented), plus every theme's representatives.
    Their nearest neighbours are searched across the WHOLE FAISS index and shipped too, so
    "similar complaints" always has a record to show. A profile is shown as its newest complaint.
    Returns (neighbors {complaint id: [{id, score}]}, sorted profile indices to export).
    """
    prof, example = built.prof, built.prof.example_row
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    sample_rows = rng.choice(len(c), size=min(cfg.EXPORT_SAMPLE, len(c)), replace=False)
    query = np.union1d(np.unique(prof.inverse[sample_rows]), np.concatenate(reps))

    D, I = vector_index.search(built.index, built.Z[query], cfg.NEIGHBORS_K + 1)
    neighbors, keep = {}, set(query.tolist())
    for q, scores, found in zip(query, D, I):
        hits = [(int(j), float(s)) for s, j in zip(scores, found) if j >= 0 and j != q][:cfg.NEIGHBORS_K]
        neighbors[int(c.ids[example[q]])] = [{"id": int(c.ids[example[j]]), "score": round(s, 4)} for j, s in hits]
        keep.update(j for j, _ in hits)
    return neighbors, np.array(sorted(keep))


def main(rebuild_data=False, rebuild_index=False):
    t0 = time.time()

    # 1. Ingest & validate
    c = build.load_complaints_cached(rebuild=rebuild_data)
    cutoff_day = int(np.quantile(c.days, cfg.TRAIN_FRACTION))
    print(f"ingest: {len(c):,} complaints; train/eval split at {np.datetime64(cutoff_day, 'D')}")

    # 2-3. Profiles, embeddings, FAISS index, themes (cached)
    built = build.build_or_load(c, cutoff_day, rebuild=rebuild_index)
    prof = built.prof
    complaint_theme = built.labels[prof.inverse]                  # theme of every complaint

    clusters = cluster.describe(prof, built.labels, built.Z, built.kmeans, c)
    reps = [np.array(cl["_repProfiles"], dtype=np.int64) for cl in clusters]
    for cl in clusters:
        del cl["_repProfiles"]

    # 4. Trends and alerts (every complaint; needs only date + theme)
    trend_rows, signals = trends.detect(c.days, complaint_theme, pd.Timestamp(cutoff_day, unit="D"))
    signals = trends.explain(signals, c.days, complaint_theme,
                             {"company": c.cats["company"], "issue": c.cats["issue"], "sub_issue": c.cats["sub_issue"]})
    print(f"trends: {len(signals)} flagged theme-weeks in the held-out period")

    # 5. Bounded slice for the dashboard, with neighbours from the full index
    neighbors, keep = pick_export_slice(built, c, reps)
    rows = prof.example_row[keep]

    complaints = pd.DataFrame({
        "id": c.ids[rows], "date": c.date_strings(rows),
        "product": c.values("product", rows), "subProduct": c.values("sub_product", rows),
        "issue": c.values("issue", rows), "subIssue": c.values("sub_issue", rows),
        "company": c.values("company", rows), "state": c.values("state", rows),
        "submittedVia": c.values("submitted_via", rows), "companyResponse": c.values("company_response", rows),
        "timely": c.values("timely", rows), "clusterId": built.labels[keep].astype(int),
        "profileCount": prof.counts[keep],
    })
    print(f"export: {len(complaints):,} complaint records, {len(neighbors):,} with neighbours")

    # 6. Evaluate
    row_of_id = {int(c.ids[r]): int(r) for r in rows}
    evaluate.write_label_sheets(c, neighbors, row_of_id, signals)
    train = int((c.days < cutoff_day).sum())
    report = {
        "exclusions": c.log,
        "data": {"complaints": len(c), "distinct_profiles": prof.n, "train": train, "eval": len(c) - train,
                 "exported_records": len(complaints)},
        "monthly_counts": evaluate.monthly_counts(c.days),
        "index": {"kind": built.index_kind,
                  "recall@%d" % cfg.NEIGHBORS_K: evaluate.index_recall(built.index, built.index_kind, built.Z)},
        "cluster_quality": evaluate.cluster_quality(built, c),
        "novelty": evaluate.novelty(built, c, cutoff_day),
        "alerts": evaluate.alert_summary(signals),
        "human_review": evaluate.score_human_labels(),
        "notes": ["No complaint text is used. Themes are mixes of product, issue and company.",
                  "Agreement with CFPB Issue/Product labels is not reported: the themes are built from those fields."],
        "config": {k: getattr(cfg, k) for k in dir(cfg)
                   if k.isupper() and isinstance(getattr(cfg, k), (int, float, str, bool, type(None), dict, list))},
    }

    # 7. Export for the backend
    dump_json(complaints.to_dict(orient="records"), "complaints.json")
    dump_json(clusters, "clusters.json")
    dump_json(trend_rows.to_dict(orient="records"), "trends.json")
    dump_json(signals.to_dict(orient="records"), "signals.json")
    dump_json(neighbors, "neighbors.json")
    report["runtime_seconds"] = round(time.time() - t0, 1)
    dump_json(report, "eval_report.json")
    print(f"done in {report['runtime_seconds']}s -> {cfg.ARTIFACT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild-data", action="store_true", help="re-read the CSV")
    parser.add_argument("--rebuild-index", action="store_true", help="refit embeddings, FAISS and clusters")
    args = parser.parse_args()
    try:
        main(rebuild_data=args.rebuild_data, rebuild_index=args.rebuild_index)
    except ingest.DataError as e:
        raise SystemExit(f"\nInput problem: {e}")
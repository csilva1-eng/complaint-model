"""
Stage 4 - Detect changes
------------------------
Counts complaints per theme per week, then compares each week with that theme's own trailing
baseline (mean and spread of the previous BASELINE_WEEKS weeks).

  z = (count - baseline_mean) / max(baseline_std, sqrt(baseline_mean), 1)

The denominator floor is a Poisson-style guard: with small counts the sample std can be ~0,
which would turn any wiggle into an "anomaly". A theme-week is flagged only if z and the
absolute count clear the theme alert thresholds and the week falls in the held-out period.

Works on two plain arrays (day numbers and theme per complaint), so it handles every complaint
in the file without building a DataFrame of them. Partial first/last weeks are dropped so
incomplete data is never mistaken for a drop or spike. This is deliberately simple and
explainable; seasonality, STL, or Poisson/NB models are the natural next steps if alerts prove noisy.
"""
import numpy as np
import pandas as pd

import config as cfg


def weekly_counts(days: np.ndarray, clusters: np.ndarray) -> pd.DataFrame:
    """Rows = week start (Monday), columns = theme id, values = complaint count."""
    days = days.astype(np.int64)
    monday = days - ((days + 3) % 7)          # 1970-01-01 was a Thursday, so shift by 3 to land on Monday
    counts = (pd.DataFrame({"week": monday, "cluster": clusters})
              .groupby(["week", "cluster"]).size().unstack("cluster", fill_value=0))
    counts.index = pd.to_datetime(counts.index.to_numpy(), unit="D")
    counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="7D"), fill_value=0)

    if pd.Timestamp(int(days.min()), unit="D") > counts.index[0]:
        counts = counts.iloc[1:]
    if pd.Timestamp(int(days.max()), unit="D") < counts.index[-1] + pd.Timedelta(days=6):
        counts = counts.iloc[:-1]
    return counts


def score(counts: pd.DataFrame, eval_start):
    w = cfg.BASELINE_WEEKS
    prior = counts.shift(1)                                  # baseline never includes the week itself
    mean = prior.rolling(w, min_periods=w).mean()
    std = prior.rolling(w, min_periods=w).std()
    floor = np.sqrt(mean).clip(lower=1.0)
    z = (counts - mean) / std.where(std > floor, floor)
    flagged = ((z >= cfg.THEME_ALERT_Z_THRESHOLD)
               & (counts >= cfg.THEME_ALERT_MIN_WEEKLY_COUNT))
    flagged.loc[counts.index < eval_start] = False           # only alert on the held-out period
    return mean, z, flagged


def _long(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    f = frame.copy()
    f.index.name, f.columns.name = "weekStart", "clusterId"
    return f.reset_index().melt(id_vars="weekStart", var_name="clusterId", value_name=name)


def detect(days: np.ndarray, clusters: np.ndarray, eval_start: pd.Timestamp):
    """Returns (trend rows for every theme-week, ranked alert rows)."""
    counts = weekly_counts(days, clusters)
    mean, z, flagged = score(counts, eval_start)

    long = _long(counts, "count")
    for name, frame in (("baselineMean", mean), ("zScore", z), ("flagged", flagged)):
        long = long.merge(_long(frame, name), on=["weekStart", "clusterId"])
    long[["baselineMean", "zScore"]] = long[["baselineMean", "zScore"]].fillna(0.0).round(2)
    long["flagged"] = long["flagged"].astype(bool)
    long["weekStart"] = long["weekStart"].dt.strftime("%Y-%m-%d")
    long = long.sort_values(["clusterId", "weekStart"]).reset_index(drop=True)

    signals = long[long["flagged"]].copy()
    signals["reason"] = [
        f"{n} complaints in the week of {wk} against a trailing average of {m:.1f} (z = {zz:.1f})"
        for n, wk, m, zz in zip(signals["count"], signals["weekStart"], signals["baselineMean"], signals["zScore"])
    ]
    cols = ["clusterId", "weekStart", "count", "baselineMean", "zScore", "reason"]
    return long, signals.sort_values("zScore", ascending=False)[cols].reset_index(drop=True)


def detect_company_signals(days: np.ndarray, company_codes: np.ndarray, company_names,
                           eval_start: pd.Timestamp) -> list[dict]:
    """Classify company-week increases as alert or watch against each company's baseline."""
    counts = weekly_counts(days, company_codes)
    w = cfg.BASELINE_WEEKS
    prior = counts.shift(1)
    mean = prior.rolling(w, min_periods=w).mean()
    std = prior.rolling(w, min_periods=w).std()
    floor = np.sqrt(mean).clip(lower=1.0)
    z = (counts - mean) / std.where(std > floor, floor)

    watch = ((z >= cfg.COMPANY_WATCH_Z_THRESHOLD)
             & (counts >= cfg.COMPANY_WATCH_MIN_WEEKLY_COUNT))
    watch.loc[counts.index < eval_start] = False
    week_rows, company_columns = np.where(watch.to_numpy())
    signals = []
    for row, col in zip(week_rows, company_columns):
        count = int(counts.iat[row, col])
        baseline = float(mean.iat[row, col])
        z_score = float(z.iat[row, col])
        week = counts.index[row].strftime("%Y-%m-%d")
        company_code = int(counts.columns[col])
        is_alert = (z_score >= cfg.COMPANY_ALERT_Z_THRESHOLD
                    and count >= cfg.COMPANY_ALERT_MIN_WEEKLY_COUNT)
        signals.append({
            "company": str(company_names[company_code]),
            "tier": "alert" if is_alert else "watch",
            "weekStart": week,
            "count": count,
            "baselineMean": round(baseline, 2),
            "zScore": round(z_score, 2),
            "reason": f"{count} complaints in the week of {week} against a trailing average of {baseline:.1f} (z = {z_score:.1f})",
        })
    return sorted(signals, key=lambda item: item["zScore"], reverse=True)


def explain(signals: pd.DataFrame, days: np.ndarray, themes: np.ndarray, fields: dict,
            top: int = cfg.EXPLAIN_TOP_SIGNALS) -> pd.DataFrame:
    """
    Themes are broad, so an alert alone does not say who is behind it. For the strongest alerts,
    compare the flagged week with the theme's own trailing weeks and name the company, issue and
    sub-issue whose count rose the most. `fields` maps a label to a pd.Categorical aligned with `days`.
    """
    signals = signals.copy()
    days = days.astype(np.int64)
    for i, r in signals.head(top).iterrows():
        week = (pd.Timestamp(r["weekStart"]) - pd.Timestamp(0)).days
        in_theme = themes == int(r["clusterId"])
        now_rows = in_theme & (days >= week) & (days < week + 7)
        base_rows = in_theme & (days >= week - 7 * cfg.BASELINE_WEEKS) & (days < week)
        parts = []
        for name, cat in fields.items():
            k = len(cat.categories)
            now = np.bincount(cat.codes[now_rows], minlength=k)
            usual = np.bincount(cat.codes[base_rows], minlength=k) / cfg.BASELINE_WEEKS
            j = int(np.argmax(now - usual))
            if now[j] - usual[j] > 0:
                parts.append(f"{name.replace('_', '-')} '{cat.categories[j]}' ({int(now[j])} vs about {usual[j]:.0f} usual)")
        if parts:
            signals.at[i, "reason"] += ". Biggest increases: " + "; ".join(parts)
    return signals
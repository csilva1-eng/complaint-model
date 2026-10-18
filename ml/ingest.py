"""
Stage 1 - Ingest & validate (streaming, structured fields only)
---------------------------------------------------------------
Reads the CFPB CSV a chunk at a time and keeps each field as a small integer code, so the whole
file ends up as a few hundred MB of numpy arrays instead of 5 GB of text.

  per chunk: find columns by name -> parse id + date -> drop bad rows and rows outside the
             date window -> store ids, day numbers and categorical codes
  at the end: drop duplicate complaint ids

Fields used (names from the CFPB field reference; matched loosely, so a renamed header does not
crash the run): Complaint ID, Date received, Product, Sub-product, Issue, Sub-issue, Company,
State, Submitted via, Company response to consumer, Timely response?

Deliberately NOT read: ZIP code, Tags and the consumer-narrative-style fields. ZIP and Tags are
personal-information-adjacent and nothing here needs them. Every drop is counted so the eval
report can say exactly why rows were excluded.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

import config as cfg

# canonical name -> acceptable headers, compared after lower-casing and stripping non-alphanumerics
HEADER_ALIASES = {
    "id": ["complaintid", "id"],
    "date": ["datereceived", "date", "receiveddate"],
    "product": ["product"],
    "sub_product": ["subproduct"],
    "issue": ["issue"],
    "sub_issue": ["subissue"],
    "company": ["company"],
    "state": ["state"],
    "submitted_via": ["submittedvia"],
    "company_response": ["companyresponsetoconsumer", "companyresponse"],
    "timely": ["timelyresponse", "timely"],
    "Tags": ["tag", "tags", "tags?"],
    "Zip Code": ["zipcode", "zip", "zip_code", "zip code"],
}
REQUIRED = ["id", "date", "product", "issue", "company"]
CATEGORICAL = ["product", "sub_product", "issue", "sub_issue", "company",
               "state", "submitted_via", "company_response", "timely",
               "Tags", "Zip Code"]


class DataError(RuntimeError):
    """The input file is not what the pipeline expects. The message says what to do."""


@dataclass
class Complaints:
    """Every complaint as parallel arrays. Row i is one complaint in every array."""
    ids: np.ndarray            # int64 complaint id
    days: np.ndarray           # int32 days since 1970-01-01 (the date received)
    cats: dict                 # field -> pd.Categorical (integer codes + the category names)
    log: dict                  # exclusion counts
    signature: str = ""        # fingerprint of the source file, used for cache keys

    def __len__(self):
        return len(self.ids)

    def values(self, field, rows) -> np.ndarray:
        """Text values of `field` for the given rows."""
        cat = self.cats[field]
        return np.asarray(cat.categories)[cat.codes[rows]]

    def date_strings(self, rows) -> list:
        return [str(np.datetime64(int(d), "D")) for d in self.days[rows]]

    def describe_row(self, row) -> str:
        return " | ".join(str(self.values(f, row)) for f in cfg.PROFILE_FIELDS)


def _norm(header) -> str:
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


def resolve_columns(header) -> dict:
    """canonical name -> the actual header present in the file."""
    by_norm = {_norm(h): h for h in header}
    found = {}
    for canon, aliases in HEADER_ALIASES.items():
        for a in aliases:
            if a in by_norm:
                found[canon] = by_norm[a]
                break
    return found


def parse_dates(s: pd.Series) -> pd.Series:
    """ISO first (fast); anything left over gets the slower mixed-format parser."""
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    miss = d.isna() & s.notna()
    if miss.any():
        d[miss] = pd.to_datetime(s[miss], format="mixed", errors="coerce")
    return d


def _clean_text(s: pd.Series) -> pd.Series:
    return s.fillna("Unknown").str.strip().replace("", "Unknown")


def _normalize_tag(value):
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return "Unknown"
    parts = re.split(r"[;|,/]+", text)
    norm = []
    for p in parts:
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(p).strip().lower()).strip("_")
        if cleaned:
            norm.append(cleaned)
    return "|".join(norm) if norm else "Unknown"


def _zip3(value):
    if pd.isna(value):
        return "Unknown"
    digits = re.sub(r"\D", "", str(value))
    if len(digits) < 3:
        return "Unknown"
    return digits[:3]


def load_complaints(path=None) -> Complaints:
    path = Path(path or cfg.RAW_CSV)
    if not path.exists():
        raise DataError(f"{path} not found. Download the CFPB complaints CSV and save it there.")

    header = list(pd.read_csv(path, nrows=0, dtype=str).columns)
    cols = resolve_columns(header)
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise DataError(f"Could not find these columns: {missing}. Columns present: {header}. "
                        "Add the right header name to HEADER_ALIASES in ingest.py.")
    optional_missing = [c for c in CATEGORICAL if c not in cols]
    if optional_missing:
        print(f"  note: no column for {optional_missing}; those fields will read 'Unknown'")
    inverse = {v: k for k, v in cols.items()}

    log = {"rows_seen": 0, "dropped_bad_id_or_date": 0, "dropped_outside_date_window": 0,
           "dropped_duplicate_id": 0}
    ids_p, days_p, cat_p = [], [], {c: [] for c in CATEGORICAL}

    reader = pd.read_csv(path, usecols=list(cols.values()), dtype=str, chunksize=cfg.CHUNK_ROWS)
    for i, chunk in enumerate(reader):
        chunk = chunk.rename(columns=inverse)
        log["rows_seen"] += len(chunk)

        d = parse_dates(chunk["date"])
        idn = pd.to_numeric(chunk["id"], errors="coerce")
        ok = d.notna() & idn.notna()
        log["dropped_bad_id_or_date"] += int((~ok).sum())
        window = d >= cfg.MIN_DATE
        if cfg.MAX_DATE:
            window &= d <= cfg.MAX_DATE
        log["dropped_outside_date_window"] += int((ok & ~window).sum())
        keep = (ok & window).to_numpy()

        ids_p.append(idn[keep].to_numpy().astype("int64"))
        days_p.append(d[keep].to_numpy().astype("datetime64[D]").astype("int32"))
        zip_col = chunk["Zip Code"] if "Zip Code" in chunk else pd.Series("Unknown", index=chunk.index)
        tag_col = chunk["Tags"] if "Tags" in chunk else pd.Series("Unknown", index=chunk.index)
        for c in CATEGORICAL:
            if c == "Tags":
                values = tag_col[keep].map(_normalize_tag).fillna("Unknown")
                cat_p[c].append(pd.Categorical(values))
            elif c == "Zip Code":
                values = zip_col[keep].map(_zip3).fillna("Unknown")
                cat_p[c].append(pd.Categorical(values))
            else:
                col = chunk[c] if c in chunk else pd.Series("Unknown", index=chunk.index)
                cat_p[c].append(pd.Categorical(_clean_text(col[keep])))
        if i % 10 == 0:
            print(f"  read {log['rows_seen']:,} rows")

    ids, days = np.concatenate(ids_p), np.concatenate(days_p)
    if len(ids) == 0:
        raise DataError("No complaints survived cleaning. Check MIN_DATE / MAX_DATE in config.py.")
    cats = {c: union_categoricals(parts) for c, parts in cat_p.items()}

    _, first = np.unique(ids, return_index=True)          # keep the first row of any repeated id
    if len(first) < len(ids):
        first.sort()
        log["dropped_duplicate_id"] = int(len(ids) - len(first))
        ids, days = ids[first], days[first]
        cats = {c: cat[first] for c, cat in cats.items()}

    log["rows_final"] = int(len(ids))
    return Complaints(ids=ids, days=days, cats=cats, log=log)
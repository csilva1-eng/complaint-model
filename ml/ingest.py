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

REQUIRED = ["Complaint ID", "Date received", "Product", "Issue", "Company"]
CATEGORICAL = ["Product", "Sub-product", "Issue", "Sub-issue", "Company",
               "State", "Submitted via",
               "Tags", "ZIP code"]


class DataError(RuntimeError):
    """The input file is not what the pipeline expects. The message says what to do."""


@dataclass
class Complaints:
    """Every complaint as parallel arrays. Row i is one complaint in every array."""
    ids: np.ndarray            # int64 complaint id
    days: np.ndarray           # int32 days since 1970-01-01 (the date received)
    cats: dict                 # field -> pd.Categorical (integer codes + the category names)
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
    rows_seen = 0
    read = pd.read_csv('../data/raw/complaints.csv', chunksize=250000)
    chunks_to_drop = ['Company public response', 'Company response to consumer', 'Timely response?', 'Date sent to company']
    
    ids_p, dates_p = [], []
    cat_p = {c: [] for c in CATEGORICAL}
    
    for i, chunk in enumerate(read):
        chunk = chunk.drop(columns = chunks_to_drop)
        rows_seen += chunk.shape[0]
    
        ids_p.append(pd.to_numeric(chunk['Complaint ID']).to_numpy())
        dates_p.append(np.asarray(parse_dates(chunk['Date received'])))

        for c in CATEGORICAL:
            col = chunk[c] if c in chunk else pd.Series("Unknown", index=chunk.index)
            cat_p[c].append(pd.Categorical(col.fillna("Unknown")))
        if i % 10 == 0:
            print(f'read {rows_seen} rows')
    idNum = np.concatenate(ids_p)
    dates = np.concatenate(dates_p).astype("datetime64[D]").astype(np.int32)
    cats = {c: union_categoricals(parts) for c, parts in cat_p.items()}

    _, first = np.unique(idNum, return_index=True)
    if len(first) < len(idNum):
        first.sort()
        idNum, dates = idNum[first], dates[first]
        cats = {c: cat[first] for c, cat in cats.items()}
    return Complaints(ids=idNum, days=dates, cats=cats)
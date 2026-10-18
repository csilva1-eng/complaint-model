"""
Central configuration for the ML pipeline.

Everything tunable lives here so a run is reproducible and the README can point at one file.
Paths are relative to the repo root; the Spring Boot backend reads ARTIFACT_DIR.

Input: the live CFPB complaints CSV. It has NO narrative text, so this pipeline works only from
the structured fields (product, sub-product, issue, sub-issue, company, state, response, ...).

Memory model: the CSV is streamed in chunks and reduced to small integer codes, which is what
lets a 5 GB file fit in well under 2 GB of RAM. Identical complaint "profiles" are collapsed
before any vector work, so FAISS indexes distinct profiles, not 18 million near-duplicates.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "complaints.csv"   # the live CFPB file
CACHE_DIR = ROOT / "data" / "cache"                  # parsed data, fitted models, FAISS index
ARTIFACT_DIR = ROOT / "data" / "artifacts"           # JSON handed to the backend
LABEL_DIR = ROOT / "data" / "labels"                 # human-reviewed pairs / alerts (optional)

# --- ingest -----------------------------------------------------------------------------
CHUNK_ROWS = 250_000        # CSV rows read at a time
MIN_DATE = "2023-01-01"     # use "2011-12-01" for the full history (more memory, more rows)
MAX_DATE = None             # e.g. "2026-09-30". The newest weeks are partial; cap here if needed.
RANDOM_STATE = 42
TRAIN_FRACTION = 0.7        # earliest 70% (by date) fits the models; the rest is held-out "future"

# --- profiles and embeddings --------------------------------------------------------------
# A "profile" is a distinct combination of these fields. Complaints with the same profile are
# indistinguishable to the model, so they are collapsed into one row with a count.
PROFILE_FIELDS = ["product", "sub_product", "issue", "sub_issue", "company", "Tags", "Zip Code"]
# Company stays in profile identity/reporting and filters neighbor candidates, but is not a
# similarity feature in the embedding score.
EMBEDDING_FIELDS = ["product", "sub_product", "issue", "sub_issue", "Tags", "Zip Code"]
BLOCK_WEIGHTS = {"product": 1.0, "sub_product": 1.0, "issue": 1.5, "sub_issue": 1.0,
                 "Tags": 0.35, "Zip Code": 0.15}
SVD_COMPONENTS = 64         # divisible by PQ_M so the compressed index option works
FIT_SAMPLE = 300_000        # training-period complaints used to fit the SVD

# --- clustering -----------------------------------------------------------------------------
N_CLUSTERS = 40
N_REPRESENTATIVES = 5
MIN_REP_COUNT = 3           # representative profiles should be common enough to be typical
N_TOP = 3                   # top issues / sub-issues / products / companies listed per theme
LABEL_COMPANY_SHARE = 0.4   # append the top company to a theme's name if it holds this share

# --- FAISS nearest-neighbour index ------------------------------------------------------------
INDEX_KIND = "auto"         # auto | flat | ivf | ivfpq
                            #   flat  exact, 4 bytes x dims per vector (64 dims = 256 B/profile)
                            #   ivf   approximate, same RAM, much faster to search
                            #   ivfpq approximate + compressed (PQ_M bytes per vector)
IVF_NPROBE = 16             # inverted lists searched per query: higher = better recall, slower
PQ_M = 16                   # bytes per compressed vector for ivfpq (SVD_COMPONENTS % PQ_M == 0)
NEIGHBORS_K = 10

# --- what the dashboard receives ----------------------------------------------------------------
# Themes, counts and trends always cover EVERY complaint. Individual complaint records and
# "similar complaints" are shipped for this many random complaints' profiles, plus their
# neighbours and each theme's representatives. Keeps complaints.json small enough for the
# backend to hold in memory.
EXPORT_SAMPLE = 5_000

# --- change detection -----------------------------------------------------------------------
BASELINE_WEEKS = 8          # trailing window that defines "normal"
THEME_ALERT_Z_THRESHOLD = 3.0
THEME_ALERT_MIN_WEEKLY_COUNT = 5
COMPANY_ALERT_Z_THRESHOLD = 3.0
COMPANY_ALERT_MIN_WEEKLY_COUNT = 5
COMPANY_WATCH_Z_THRESHOLD = 1.0
COMPANY_WATCH_MIN_WEEKLY_COUNT = 3 
EXPLAIN_TOP_SIGNALS = 50    # for this many strongest alerts, name which company/issue drove the rise
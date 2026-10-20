# Complaint Intelligence Engine

Turns the CFPB Consumer Complaint Database into ranked, evidence-backed investigations: it groups
complaints into themes, flags unusual weekly increases, says which company and issue drove each
increase, and drafts Jira tickets that **a person approves**.

**No complaint text is used.** The live CFPB file has no narrative column (the CFPB stopped publishing
narratives on 2026-08-14), so everything works from structured fields: product, sub-product, issue,
sub-issue, company, state, submitted-via, company response, timely response, and dates.

```
data/raw/complaints.csv (5 GB, streamed in chunks)
   |
   v
ml/  (Python, pandas, scikit-learn, FAISS)             backend/  (Java 21, Spring Boot)       frontend/  (C#, Blazor WASM)
 ingest -> compact integer arrays                       ArtifactStore -> REST API              Signals -> Theme -> Ticket review
 collapse identical complaints into profiles            EvidenceSummarizer (template | LLM)
 one-hot -> SVD vectors -> FAISS index                  TicketService (DRAFT -> APPROVED -> CREATED)
 weighted k-means themes -> weekly trends -> alerts     JiraClient (mock | real)
   |
   +--- data/artifacts/*.json -----------------------> (the only coupling between tiers)
```

## What it can and cannot tell you

- **Can:** which mixes of product, issue and company are growing faster than their own baseline, and which
  company, issue and sub-issue account for the growth ("company 'X' (1518 vs about 0 usual)").
- **Cannot:** what customers actually said, or why. Without text there is no semantic search, and the
  hypothesis in a ticket can only say "company-specific" versus "industry-wide", always marked unverified.
- A **theme** is a recurring mix of product, issue and company built from the same fields CFPB uses for its
  own labels. Treat it as a way to watch volume, not as a newly discovered topic.
- **Similar complaints** are selected from the same company, then ranked by cosine similarity over
  product, sub-product, issue, sub-issue, Tags and ZIP3. Company filters eligible candidates but does
  not affect the similarity score.

The dashboard opens with a company selector. Signals, ticket drafts and complaint search remain scoped
to that selection. Company signals compare each company's weekly volume with its own trailing baseline;
they are calculated independently from the global theme signals.

## How a 5 GB file fits in memory

1. **Stream and encode.** The CSV is read 250k rows at a time and each field is stored as a small integer
   code. The whole file becomes a few hundred MB of numpy arrays.
2. **Collapse to profiles.** Complaints with the same product, sub-product, issue, sub-issue, company,
  Tags and ZIP code are one *profile* with a count. Company remains part of profile identity, but is
  excluded from the embedding used to rank similar complaints.
3. **FAISS over profiles.** Each profile is a weighted one-hot vector compressed to 64 dimensions (SVD) and
   L2-normalised, so inner product is cosine similarity. `INDEX_KIND` in `ml/config.py` picks `flat` (exact),
   `ivf` (approximate, faster) or `ivfpq` (approximate, compressed). The default switches to `ivf` above
   500k profiles.
4. **Cache.** Parsed data, models and the index are cached, so only the first run is slow.

Measured on a synthetic CSV with the real column names (3.0M rows, 549 MB, 199k distinct profiles): 27 s
and 0.68 GB peak memory. That is roughly 75 bytes per row, which extrapolates to about 2 GB for ~18M rows,
but real data will have more distinct profiles than my synthetic data, which costs more for the vectors
and the index. I have not run it on the real file.

## Run it

```
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py                  # first run reads the CSV; later runs reuse caches
python run_pipeline.py --rebuild-data   # re-read the CSV
python run_pipeline.py --rebuild-index  # refit embeddings, FAISS and themes (keeps parsed data)
```

Save the CFPB CSV as `data/raw/complaints.csv`. Column names are matched loosely; if a required one
(Complaint ID, Date received, Product, Issue, Company) is missing you get an error listing the columns found.
`MIN_DATE` defaults to 2023-01-01; set it to `2011-12-01` in `ml/config.py` for the full history.
Tags are normalized as categorical values; ZIP codes are reduced to their first three digits (ZIP3).
Both are low-weight similarity features, and missing values are treated as `Unknown`.

Then the backend (`cd backend && mvn spring-boot:run`, Java 21, port 8080, Jira in mock mode) and the
frontend (`cd frontend/CIE.Web && dotnet run --urls http://localhost:5173`, .NET 8 SDK).

## Artifact contract (`data/artifacts/`)

| File | Contents | Scope |
|---|---|---|
| `clusters.json` | size, label, top products/issues/sub-issues/companies, representative ids | every complaint |
| `trends.json` | weekly count, baseline mean, z-score, flagged, per theme | every complaint |
| `signals.json` | flagged theme-weeks, with the company/issue/sub-issue that drove each | every complaint |
| `companies.json` | company names available in the dashboard selector | every company |
| `company_signals.json` | flagged company-weeks compared with that company's trailing baseline | every company |
| `company_clusters.json` | complaint counts for each company and theme combination | all complaints |
| `complaints.json` | one record per exported profile (its newest complaint) plus `profileCount` | **export slice only** |
| `neighbors.json` | top-10 similar profiles, searched across the whole FAISS index | export slice only |
| `eval_report.json` | exclusions, monthly counts, index recall, stability, novelty, human-review scores | |
| `pairs_to_label.csv`, `alerts_to_review.csv` | sheets for manual labelling | |

**Export slice:** counts, themes and trends reflect every complaint. Individual records and "similar
complaints" are shipped for the profiles of `EXPORT_SAMPLE` random complaints (default 5,000), their
neighbours, and each theme's representatives, so the backend can hold `complaints.json` in memory.
Dashboard search therefore covers that slice, not the whole file.

## Evaluation

Agreement with CFPB's Issue/Product labels is **not** reported: the themes are built from those fields, so
the comparison would be circular. `eval_report.json` reports instead:

- **Stability:** refit k-means with a different seed; do complaints land in the same themes? (adjusted Rand index)
- **Silhouette:** how compact the themes are, on a sample.
- **Novelty:** share of held-out-period complaints whose product/issue/company combination never appeared in training.
- **Index recall:** for approximate FAISS indexes, how many of the true nearest neighbours are returned. A
  neighbour counts as found if it is at least as similar as the exact k-th best, so ties are not misses.
  On synthetic data `ivf` scored 1.0 and `ivfpq` 0.25, so check this number if you use compression.
- **Human review:** fill `pairs_to_label.csv` and `alerts_to_review.csv`, save them in `data/labels/` as
  `pairs_labeled.csv` and `alerts_labeled.csv`, and rerun for precision@rank and the false-alert rate.

## Design decisions and trade-offs

- **Models are fitted on the earliest 70% of complaints; alerts are only raised in the remaining 30%.**
- **Themes are weighted by log(complaints)** so a few enormous profiles (for example credit-report disputes)
  cannot claim every theme.
- **`BLOCK_WEIGHTS` decides what "similar" means.** It is a modelling choice, not a learned one.
- **Facts come from code, hypotheses from the template or the LLM.** An LLM reply must start with
  "Unverified", cite complaint ids, and cite only ids it was given, or the template summary is used instead.
  With no text, the LLM can rephrase a lead but cannot discover one.
- **Jira defaults to mock mode.** Only IDs and generated text go into a ticket.

## Known limitations

- A *sustained* spike stops alerting after a few weeks, because the trailing baseline absorbs it.
- The newest weeks of the CFPB file are still being filled in; check `monthly_counts` and set `MAX_DATE`.
- Complaint volume reflects reporting behaviour and company size, not quality. The UI says so.
- Tickets are held in memory and there is no login, so audit entries say "analyst".
- The Java and C# code has not been compiled; only the Python pipeline has been run.
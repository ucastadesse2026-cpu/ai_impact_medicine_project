# Which AI-augmented papers actually gain impact?
### Heterogeneous citation-impact prediction in Medicine (2015–2021)

UCAS Data Science team project. We predict, **at publication time**, whether a
Medicine journal article will end up in the **top-10% cited** of its
publication-year cohort, and we estimate how the association between AI
adoption and citation impact **varies across contexts** (journal prestige,
team size).

- **Stakeholders:** funding agencies, early-career researchers, journal editors.
- **Estimand / prediction target:** P(top-10% cited within publication-year
  cohort | features available at publication, AI-adoption status).
- **Data:** 25,000 journal articles sampled from Europe PMC (PubMed, SRC:MED),
  published 2015–2021, fetched 2026-11-12 (citations cumulative to that date).
- **Claim tested:** AI-augmented papers are more likely to be highly cited, but
  the association is confounded by venue prestige and heterogeneous across
  contexts.

## Repository layout

```
configs/config.yaml           # FROZEN configuration: seeds, split, lexicon, thresholds
src/fetch_data.py             # step 1 — seeded sampling from Europe PMC REST API
src/build_dataset.py          # step 2 — labels, outcome, leakage-safe features
src/make_eda_figures.py       # step 3 — EDA figures
src/train_evaluate.py         # step 4 — baseline, GBM, ablations, sensitivity
src/association_and_uplift.py # step 5 — adjusted ORs, AI x context interactions, uplift bootstrap
data/raw/                     # fetch via src/fetch_data.py (see data/raw/README.md)
data/processed/dataset.csv    # FROZEN analysis dataset, 25,000 rows x 67 cols (one row per paper)
results/                      # metrics*.json, test_predictions.csv, figures/
notebooks/01_reproduce_report.ipynb  # re-derives every report table & figure from the frozen outputs
notebooks/02_full_pipeline.ipynb     # reruns the whole pipeline (fetch -> build -> train -> uplift)
cards/data_card.md            # data documentation (incl. full variable list)
cards/model_card.md           # model documentation
report/report.md | report.pdf # 8–10 page report (Table 2 = dataset variables)
slides/                       # final presentation, 13 pages (.pptd) [git-ignored]
proposal/                     # proposal presentation (.pptd)   [git-ignored]
AI_USE_RECORD.md              # AI-assistance log with independent checks
```

> **GitHub note:** the `.gitignore` keeps graded material out of the public
> repo — `slides/`, `proposal/`, `report/`, the dataset CSVs/JSONL, and the
> large per-paper predictions file. Everything is reproducible: the pipeline
> rebuilds the data and outputs from `configs/config.yaml`, and the frozen
> raw copy is available via the course Drive (see `data/raw/README.md`).

## Reproduce (tested on Python 3.12, Linux)

```bash
pip install -r requirements.txt
python3 src/fetch_data.py            # ~4 min, needs internet (see data/raw/README.md for the frozen copy)
python3 src/build_dataset.py         # -> data/processed/dataset.csv, results/label_stats.json
python3 src/make_eda_figures.py      # -> results/figures/eda_*.png
python3 src/train_evaluate.py        # -> results/metrics.json + model figures
python3 src/association_and_uplift.py# -> results/metrics_supplement.json, metrics_uplift_boot.json, uplift.png
```

All randomness is seeded (`sampling=42, modeling=42, subsample=42` in
`configs/config.yaml`). The train/test split is **time-based and frozen**
(train 2015–2018, test 2019–2021) — no shuffling anywhere.

### Prefer Jupyter?

- `notebooks/01_reproduce_report.ipynb` — open and "Run All": recomputes every
  table and figure in `report/report.pdf` from the shipped frozen files and
  shows the figures inline (no download needed). Shipped **pre-executed**.
- `notebooks/02_full_pipeline.ipynb` — reruns stages 1–5 above cell by cell,
  with explanations.

### Dataset variables

The frozen dataset is `data/processed/dataset.csv` (25,000 rows × 67 columns,
one row per paper). Full variable list: report **Table 2** and
`cards/data_card.md`.

> Note: re-running `fetch_data.py` queries a live database; citation counts
> grow over time, so exact numbers drift. The frozen raw file (fetched
> 2026-11-12, shared via the course Drive — see `data/raw/README.md`)
> reproduces every number in the report exactly.

## Headline results (frozen run; test = 2019–2021, n = 10,426)

| finding | value |
|---|---|
| Logistic baseline ROC-AUC / PR-AUC | **0.787 / 0.322 — best model, kept as primary** |
| Gradient boosting ROC-AUC / PR-AUC | 0.754 / 0.319 (no gain over baseline) |
| XGBoost ROC-AUC / PR-AUC (`scale_pos_weight` = 8.65) | 0.724 / 0.303 (worse; calibration degrades, Brier 0.123) |
| LightGBM ROC-AUC / PR-AUC (`class_weight` balanced) | 0.720 / 0.294 (worse; Brier 0.120) |
| Unadjusted odds ratio, AI vs non-AI (top-10%) | 1.63 [1.20, 2.23] |
| Adjusted odds ratio (venue, team, topics, OA…) | 0.55 [0.28, 1.13] — vanishes |
| Estimated AI uplift by journal-prestige quartile | +3pp (Q1) → +41pp (Q4) |
| Estimated AI uplift by team-size quartile | +12pp (small) → +26pp (large) |

Interpretation: the raw citation advantage of AI-augmented papers is **fully
absorbed by confounders** (mainly journal prestige); the remaining association
is concentrated in high-prestige venues and larger teams. On the modeling side,
performance **decreases monotonically with model complexity** (logistic > GBM >
XGBoost > LightGBM): publication-time citation impact is dominated by additive
linear structure, and flexible trees overfit the sparse positive class. All
findings are **associational**, not causal — see report §5.

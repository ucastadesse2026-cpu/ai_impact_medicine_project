# Data Card — Europe PMC Medicine sample (2015–2021)

## Provenance
- **Source:** Europe PMC REST API (`/webservices/rest/search`, `resultType=core`),
  PubMed records only (`SRC:MED`), journal articles with abstracts
  (`PUB_TYPE:"journal article" AND HAS_ABSTRACT:y`).
- **License/permitted use:** Europe PMC metadata is openly available for
  research use under the Europe PMC terms of service; abstracts remain
  copyrighted by publishers and are used here only for text classification,
  not redistribution. No personal data beyond author names on public papers.
- **Fetch date:** 2026-11-12. Citation counts (`citedByCount`, Europe PMC's
  own citation network) are cumulative to this date.

## Sampling
- Unit of observation: one journal article (PubMed ID).
- Frame: all PubMed journal articles with abstracts, first-published 2015–2021.
- Design: per year, 24 random 2-day windows (seed 42); first 160 records per
  window sorted by publication date. Within-window order is arbitrary w.r.t.
  citations and AI use, approximating simple random sampling. De-duplicated by
  PubMed ID; subsampled to 25,000 (seed 42).
- Known sampling caveats: windows with >160 hits contribute early-window
  records only; Europe PMC coverage skews to English-language, indexed
  journals (98.7% English in sample); very new journals are under-represented.

## Sample composition (frozen run)
- 25,000 papers: 2015: 3,794 · 2016: 3,455 · 2017: 3,935 · 2018: 3,390 ·
  2019: 3,935 · 2020: 3,459 · 2021: 3,032.
- 2,450 distinct journals; 83.5% open access; 14.2% reviews;
  median 5 authors; median abstract 204 words; 71.6% carry author keywords.

## Variables (full list, by group)

One row per paper; 25,000 rows × 67 columns in `data/processed/dataset.csv`.

| group | variables | role |
|---|---|---|
| Identifiers & text | `id`, `title`, `abstract`, `journal`, `keywords`, `window_hits` | lexicon & topic input; joins |
| Bibliographic features | `year`, `n_authors` (team size), `n_keywords`, `n_grants`, `open_access`, `is_review`, `has_references`, `english`, `abstract_words`, `title_words` | publication-time predictors |
| Citation outcome | `cited_by` → `y_top10` (primary), `y_top5`, `y_top20` (sensitivity); within-year percentile | prediction target |
| AI-adoption label | `ai_score`, `ai_flag` (score ≥ 2, primary), `ai_flag_strict` (≥ 3), `ai_flag_loose` (≥ 1) | treatment proxy |
| Leakage-safe derived | `journal_train_mean_cites` (train years only), `journal_train_n_papers`, 40 × `kw::*` topic dummies (train-years vocabulary) | confounder controls |

### Leakage notes (selected)
| variable | definition | leakage-safe? |
|---|---|---|
| `cited_by` | Europe PMC citation count at fetch date | outcome only |
| `y_top10` | cited_by ≥ 90th percentile of publication-year cohort | outcome |
| `ai_score`, `ai_flag` | lexicon score on title+abstract; flag = score ≥ 2 | yes (text at publication) |
| `journal_train_mean_cites` | journal's mean citations computed from TRAIN years (2015–2018) only | yes |
| `kw::*` (40) | keyword dummies; vocabulary selected on TRAIN only | yes |

## Missingness & bias
- `abstract` required by query; residual short abstracts (<100 chars) dropped (26,400 → 25,434 before subsampling).
- Author keywords missing for 28.4% of records (topic dummies = 0) — likely
  missing-not-at-random (varies by journal), noted as limitation.
- Citation counts from Europe PMC's network undercount citations outside
  biomedical literature.

## Ethical considerations
Public bibliographic metadata only; no human subjects; no attempt to identify
or profile individual researchers; aggregate findings reported at group level.
AI-adoption labels are lexicon-based proxies and must not be used to judge
individual papers or authors.

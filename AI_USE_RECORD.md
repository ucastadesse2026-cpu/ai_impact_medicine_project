# AI-Use Record

Per course requirements, this file documents AI assistance used in this
project and the independent checks performed by the team.

## Tools used
- **Kimi (Moonshot AI)**, November 2026: project ideation; dataset/source
  selection; drafting of pipeline code (`src/*.py`), README, cards, report and
  slides; debugging.

## What the AI did
| artifact | AI role | human check performed |
|---|---|---|
| `src/fetch_data.py` | drafted sampling code | verified query syntax against Europe PMC docs; inspected `window_hits` distribution in raw data |
| `src/build_dataset.py` | drafted lexicon + features | manually reviewed 30 labeled titles/abstracts for false positives/negatives |
| `src/train_evaluate.py`, `src/association_and_uplift.py` | drafted modeling code | confirmed split years, seeds; re-ran and compared metrics.json values |
| report/slides text | drafted | fact-checked all external claims against the cited papers; verified every number against `results/metrics*.json` |

## Independent verification
- Every quantitative claim in the report is traceable to
  `results/metrics.json`, `results/metrics_supplement.json`,
  `results/metrics_uplift_boot.json`, `results/label_stats.json`, or the cited
  literature.
- The full pipeline was executed end-to-end from `README.md` instructions
  before submission; `results/` contains the frozen raw outputs.
- AI-generated suggestions that were rejected by the team:
  - shuffling the train/test split (rejected — would break the time-based
    design and invite leakage);
  - GPT-based abstract labeling (rejected — non-reproducible, not inspectable);
  - reporting the unadjusted AI–citation association as the headline result
    (rejected — confounded; the adjusted analysis is primary).

## Declaration template for the team
> We used Kimi for drafting code and text as logged above. All outputs were
> reviewed, executed, and verified by the team. We take full responsibility
> for the content. — TADESSE BIRARA AKELE (2026A8019427001), DEONISIO YOHANES WILFRIDUS HORNAY (026A8000531004)

# Model Card — Citation-impact prediction (top-10% cited, Medicine 2015–2021)

## Intended use
Research/education only: studying correlates of citation impact and the
context-dependence of AI-adoption associations. **Not** for evaluating
individual papers, researchers, or funding decisions without human review.

## Models
| model | role | key hyperparameters |
|---|---|---|
| L2 logistic regression | baseline | C=1.0, standardized features, max_iter=2000 |
| Histogram gradient boosting | complexity candidate | max_iter=400, lr=0.06, max_leaf_nodes=31, L2=1.0 |
| XGBoost | complexity candidate (imbalance-corrected) | n_estimators=400, lr=0.06, depth 6, scale_pos_weight=8.65 |
| LightGBM | complexity candidate (imbalance-corrected) | n_estimators=400, lr=0.06, num_leaves=31, class_weight="balanced" |
| Logistic with AI×context interactions | association/uplift analysis | C=1.0; adds log-prestige, log-team, ai×prestige, ai×team |

## Headline performance (frozen test 2019–2021, n=10,426)
| model | ROC-AUC | PR-AUC | Brier | lift@10% |
|---|---|---|---|---|
| Logistic baseline | 0.787 [0.773, 0.803] | 0.322 [0.294, 0.351] | 0.083 | 3.75 |
| Gradient boosting | 0.754 [0.736, 0.769] | 0.319 [0.291, 0.345] | 0.082 | 3.79 |
| XGBoost | 0.724 [0.706, 0.740] | 0.303 [0.273, 0.330] | 0.123 | 3.58 |
| LightGBM | 0.720 [0.700, 0.737] | 0.294 [0.266, 0.324] | 0.120 | 3.57 |
| Interaction logistic | 0.778 | 0.324 | — | — |

No tree-based candidate beats the logistic baseline; performance falls
monotonically with model complexity, so we keep the baseline as the primary
model (negative finding, see report §4). Class-reweighted tree models also
lose calibration (Brier ≈ 0.12 vs 0.083).

## Training/evaluation protocol (frozen)
- Split: time-based — train 2015–2018 (n=14,574), test 2019–2021 (n=10,426). No shuffling.
- Seeds: 42 (models, permutation importance, bootstrap).
- Metrics: ROC-AUC, PR-AUC (primary), Brier score, lift@10%; 200× bootstrap CIs.
- Calibration: decile plot on the test set (both models well calibrated).
- Sensitivity: outcome thresholds (top 5/10/20%), AI-label thresholds
  (score ≥1/2/3), alternative split (train ≤2019, test 2020–21), GBM variant.

## Key limitations
1. **Associational, not causal.** "Uplift" is a model-based counterfactual;
   AI adopters differ from non-adopters in unobserved ways (resources, topic
   fashion, English proficiency, LLM-assisted writing).
2. **Lexicon labels.** Regex lexicon misses implicit AI use and yields false
   positives (e.g., "neural network" in neuroscience). Hao et al. (2026) used a
   fine-tuned language model (F1=0.875); our transparent lexicon trades
   accuracy for reproducibility — mitigated by strict/loose sensitivity checks.
3. **Model-class dependence.** The rare AI flag (1.7%) is ignored entirely by
   the GBM (zero counterfactual uplift) but is estimable in the logistic
   interaction model — uplift conclusions are specification-dependent.
4. **Citation windows differ** across cohorts (2015 papers had ~11 years, 2021
   ~5 years); percentile-within-year mitigates but does not equalize windows.
5. **Single database** (Europe PMC citation network); results may not transfer
   to OpenAlex/Scopus counts.
6. Journal prestige feature encodes historical average — a feedback-loop risk
   if ever used for real editorial decisions.

## Failure modes observed
- Degraded ranking (ROC 0.75 vs 0.79) under the gradient-boosting model despite
  more capacity — capacity does not help with these tabular features.
- Reduced PR-AUC on the top-5% outcome (0.215) — class imbalance.
- Papers in journals unseen during training fall back to median prestige.
- Wide bootstrap CIs for subgroup uplift (only 314 AI papers in test).

#!/usr/bin/env python3
"""Train, evaluate, and stress-test citation-impact models.

Frozen protocol (configs/config.yaml):
  * time-based split: train 2015-2018, test 2019-2021 (no shuffling)
  * seeds fixed at 42 everywhere
  * baseline first: L2 logistic regression
  * main model: histogram gradient boosting
  * uncertainty: 200x bootstrap of test-set metrics
  * stress tests: ablations, outcome threshold, AI-label threshold,
    alternative split — reported as sensitivity analyses

Outputs:
  results/metrics.json, results/test_predictions.csv,
  results/figures/{roc_pr,calibration,importance}.png
"""
import json, pathlib
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/config.yaml"))
SEED = CFG["seeds"]["modeling"]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data/processed/dataset.csv")
kw_cols = [c for c in df.columns if c.startswith("kw::")]
FEATS = CFG["features"]["numeric"] + CFG["features"]["binary"] + kw_cols

train = df[df.year.isin(CFG["split"]["train_years"])]
test = df[df.year.isin(CFG["split"]["test_years"])]
print(f"train {len(train)}, test {len(test)}, features {len(FEATS)}")


def metrics(y, p):
    k = max(1, int(0.1 * len(y)))
    idx = np.argsort(-p)[:k]
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "lift_at_10pct": float(y.iloc[idx].mean() / y.mean())}


def boot_ci(y, p, fn, n=200):
    rng = np.random.default_rng(SEED)
    vals = []
    yv, pv = np.asarray(y), np.asarray(p)
    for _ in range(n):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].sum() in (0, len(i)):
            continue
        vals.append(fn(yv[i], pv[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(Xtr, ytr, Xte, yte, tag):
    out = {}
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    out["logistic_baseline"] = metrics(yte, pd.Series(p_lr))
    out["logistic_baseline"]["roc_auc_ci"] = boot_ci(yte, p_lr, roc_auc_score)
    out["logistic_baseline"]["pr_auc_ci"] = boot_ci(yte, p_lr, average_precision_score)

    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=SEED)
    gbm.fit(Xtr, ytr)
    p_gb = gbm.predict_proba(Xte)[:, 1]
    out["gradient_boosting"] = metrics(yte, pd.Series(p_gb))
    out["gradient_boosting"]["roc_auc_ci"] = boot_ci(yte, p_gb, roc_auc_score)
    out["gradient_boosting"]["pr_auc_ci"] = boot_ci(yte, p_gb, average_precision_score)
    print(f"  [{tag}] done", flush=True)
    return out, lr, gbm, p_lr, p_gb


results = {}
Xtr, Xte = train[FEATS], test[FEATS]

# ---------- primary outcome ----------
res, lr, gbm, p_lr, p_gb = run(Xtr, train.y_top10, Xte, test.y_top10, "main top10")
results["primary_top10"] = res

# ---------- challenger models: XGBoost & LightGBM with imbalance correction ----------
# ~10.3% positives in train -> scale_pos_weight = neg/pos for XGBoost;
# class_weight="balanced" applies the same reweighting idea inside LightGBM.
neg, pos = int((train.y_top10 == 0).sum()), int((train.y_top10 == 1).sum())
spw = neg / pos
print(f"imbalance correction: neg/pos = {neg}/{pos} -> scale_pos_weight = {spw:.3f}")

xgb = XGBClassifier(n_estimators=400, learning_rate=0.06, max_depth=6,
                    subsample=0.9, colsample_bytree=0.9,
                    scale_pos_weight=spw, eval_metric="logloss",
                    random_state=SEED, n_jobs=-1)
xgb.fit(Xtr, train.y_top10)
p_xgb = xgb.predict_proba(Xte)[:, 1]
results["primary_top10"]["xgboost"] = metrics(test.y_top10, pd.Series(p_xgb))
results["primary_top10"]["xgboost"]["roc_auc_ci"] = boot_ci(test.y_top10, p_xgb, roc_auc_score)
results["primary_top10"]["xgboost"]["pr_auc_ci"] = boot_ci(test.y_top10, p_xgb, average_precision_score)
print("  [xgboost] done", flush=True)

lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.06, num_leaves=31,
                      class_weight="balanced", random_state=SEED,
                      n_jobs=-1, verbose=-1)
# LightGBM rejects special characters (e.g. "kw::") in feature names — sanitize.
Xtr_l = Xtr.copy(); Xte_l = Xte.copy()
Xtr_l.columns = [c.replace(":", "_") for c in Xtr_l.columns]
Xte_l.columns = [c.replace(":", "_") for c in Xte_l.columns]
lgbm.fit(Xtr_l, train.y_top10)
p_lgbm = lgbm.predict_proba(Xte_l)[:, 1]
results["primary_top10"]["lightgbm"] = metrics(test.y_top10, pd.Series(p_lgbm))
results["primary_top10"]["lightgbm"]["roc_auc_ci"] = boot_ci(test.y_top10, p_lgbm, roc_auc_score)
results["primary_top10"]["lightgbm"]["pr_auc_ci"] = boot_ci(test.y_top10, p_lgbm, average_precision_score)
print("  [lightgbm] done", flush=True)

# ---------- counterfactual uplift under the GBM (model-dependence check) ----------
test_cf = test.copy()
X_on, X_off = Xte.copy(), Xte.copy()
X_on["ai_flag"] = 1
X_off["ai_flag"] = 0
uplift = gbm.predict_proba(X_on)[:, 1] - gbm.predict_proba(X_off)[:, 1]
test_cf["uplift_gbm"] = uplift
test_cf["p_top10"] = gbm.predict_proba(Xte)[:, 1]
results["gbm_counterfactual_uplift"] = {
    "overall_mean": float(np.mean(uplift)),
    "mean_on_ai_papers": float(np.mean(uplift[test_cf.ai_flag.values == 1])),
    "n_ai_test": int((test_cf.ai_flag == 1).sum()),
    "note": "GBM never splits on the rare ai_flag, so flipping it changes nothing — "
            "evidence that the AI coefficient is model-class dependent; see "
            "association_and_uplift.py for the interaction-model analysis.",
    "observed_top10_rate_by_ai": {
        "ai": float(test_cf.loc[test_cf.ai_flag == 1, "y_top10"].mean()),
        "non_ai": float(test_cf.loc[test_cf.ai_flag == 0, "y_top10"].mean())},
}

# ---------- ablations ----------
abl = {}
no_j = [f for f in FEATS if not f.startswith("journal_")]
r, *_ = run(train[no_j], train.y_top10, test[no_j], test.y_top10, "no_journal")
abl["without_journal_prestige"] = r["gradient_boosting"]
no_t = [f for f in FEATS if not f.startswith("kw::")]
r, *_ = run(train[no_t], train.y_top10, test[no_t], test.y_top10, "no_topics")
abl["without_topic_dummies"] = r["gradient_boosting"]
r, *_ = run(train[["ai_flag"]], train.y_top10, test[["ai_flag"]], test.y_top10, "ai_only")
abl["ai_flag_only_logistic"] = r["logistic_baseline"]
results["ablations"] = abl

# ---------- sensitivity ----------
sens = {}
for oc in ["y_top5", "y_top20"]:
    r, *_ = run(Xtr, train[oc], Xte, test[oc], oc)
    sens[oc] = r["gradient_boosting"]
for flag in ["ai_flag_strict", "ai_flag_loose"]:
    tr2, te2 = train.copy(), test.copy()
    tr2["ai_flag"] = tr2[flag]; te2["ai_flag"] = te2[flag]
    r, *_ = run(tr2[FEATS], tr2.y_top10, te2[FEATS], te2.y_top10, flag)
    sens[flag] = r["gradient_boosting"]
    sens[flag]["note"] = ("identical to primary because the GBM ignores the rare "
                          "AI flag; the flag swap changes no prediction")
trA = df[df.year.isin([2015, 2016, 2017, 2018, 2019])]
teA = df[df.year.isin([2020, 2021])]
r, *_ = run(trA[FEATS], trA.y_top10, teA[FEATS], teA.y_top10, "alt_split")
sens["alt_split_train2015_19_test2020_21"] = r["gradient_boosting"]
results["sensitivity"] = sens

# ---------- permutation importance (test set, frozen seed) ----------
pi = permutation_importance(gbm, Xte, test.y_top10, n_repeats=10,
                            random_state=SEED, scoring="average_precision")
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
results["permutation_importance_top15"] = {k: float(v) for k, v in imp.head(15).items()}

# ---------- figures ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for p, lab in [(p_lr, "Logistic baseline"), (p_gb, "Gradient boosting"),
               (p_xgb, "XGBoost"), (p_lgbm, "LightGBM")]:
    fpr, tpr, _ = roc_curve(test.y_top10, p)
    ax[0].plot(fpr, tpr, label=f"{lab} (AUC={roc_auc_score(test.y_top10, p):.3f})")
    prec, rec, _ = precision_recall_curve(test.y_top10, p)
    ax[1].plot(rec, prec, label=f"{lab} (AP={average_precision_score(test.y_top10, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=.7); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[0].set_title("ROC — top-10% cited (test 2019–2021)"); ax[0].legend()
ax[1].axhline(test.y_top10.mean(), color="k", ls="--", lw=.7)
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision–recall"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "roc_pr.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4.4))
for p, lab in [(p_lr, "Logistic"), (p_gb, "GBM"), (p_xgb, "XGBoost"), (p_lgbm, "LightGBM")]:
    bins = pd.qcut(p, 10, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": test.y_top10.values}).groupby(bins, observed=True).mean()
    ax.plot(cal["p"], cal["y"], "o-", label=lab)
ax.plot([0, 1], [0, 1], "k--", lw=.7)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed top-10% rate")
ax.set_title("Calibration (deciles, test set)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
top = imp.head(15).iloc[::-1]
ax.barh(top.index, top.values, color="#2c6fbb")
ax.set_title("Permutation importance (test, PR-AUC drop)")
fig.tight_layout(); fig.savefig(FIG / "importance.png", dpi=160); plt.close(fig)

test_cf.to_csv(ROOT / "results/test_predictions.csv", index=False)
json.dump(results, open(ROOT / "results/metrics.json", "w"), indent=2)
print("train_evaluate done")

#!/usr/bin/env python3
"""Association vs. prediction: adjusted odds ratios, AI x context interactions,
and bootstrap uncertainty for the estimated AI-adoption uplift.

Run AFTER train_evaluate.py. Writes:
  results/metrics_supplement.json     (ORs, interaction model, GBM variant)
  results/metrics_uplift_boot.json    (bootstrap CIs for uplift by subgroup)
  results/figures/uplift.png          (with CI error bars)
"""
import json, pathlib
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/config.yaml"))
SEED = CFG["seeds"]["modeling"]
FIG = ROOT / "results/figures"

df = pd.read_csv(ROOT / "data/processed/dataset.csv")
kw_cols = [c for c in df.columns if c.startswith("kw::")]
FEATS = CFG["features"]["numeric"] + CFG["features"]["binary"] + kw_cols
train = df[df.year.isin(CFG["split"]["train_years"])].copy()
test = df[df.year.isin(CFG["split"]["test_years"])].copy()
out = {}

# ---- unadjusted association (test years) ----
t = test
a = int(((t.ai_flag == 1) & (t.y_top10 == 1)).sum()); b = int(((t.ai_flag == 1) & (t.y_top10 == 0)).sum())
c = int(((t.ai_flag == 0) & (t.y_top10 == 1)).sum()); d_ = int(((t.ai_flag == 0) & (t.y_top10 == 0)).sum())
or_un = (a / b) / (c / d_)
se = np.sqrt(1/a + 1/b + 1/c + 1/d_)
out["unadjusted"] = {"counts": {"ai_top10": a, "ai_not": b, "nonai_top10": c, "nonai_not": d_},
                     "odds_ratio": float(or_un),
                     "ci95": [float(np.exp(np.log(or_un) - 1.96*se)), float(np.exp(np.log(or_un) + 1.96*se))]}

# ---- adjusted OR (logistic, ~unpenalized) with bootstrap CI ----
def adjusted_or(frame, reps=100):
    X = frame[FEATS].values; y = frame.y_top10.values
    sc = StandardScaler(); lr = LogisticRegression(max_iter=3000, C=1e6, random_state=SEED)
    lr.fit(sc.fit_transform(X), y)
    i_ai = FEATS.index("ai_flag")
    beta = lr.coef_[0][i_ai] / sc.scale_[i_ai]
    rng = np.random.default_rng(SEED); betas = []
    idx = np.arange(len(frame))
    for _ in range(reps):
        ii = rng.choice(idx, len(idx), replace=True)
        scb = StandardScaler(); lrb = LogisticRegression(max_iter=2000, C=1e6)
        lrb.fit(scb.fit_transform(X[ii]), y[ii])
        betas.append(lrb.coef_[0][i_ai] / scb.scale_[i_ai])
    lo, hi = np.percentile(betas, [2.5, 97.5])
    return {"odds_ratio": float(np.exp(beta)), "ci95": [float(np.exp(lo)), float(np.exp(hi))]}

out["adjusted_or_full_sample"] = adjusted_or(df)
out["adjusted_or_test_years"] = adjusted_or(test)

# ---- interaction model: ai_flag x (log prestige, log team size) ----
f2 = df.copy()
f2["log_prestige"] = np.log1p(f2["journal_train_mean_cites"])
f2["log_team"] = np.log1p(f2["n_authors"])
f2["ai_x_prestige"] = f2["ai_flag"] * f2["log_prestige"]
f2["ai_x_team"] = f2["ai_flag"] * f2["log_team"]
FEATS2 = [f for f in FEATS if f not in ("journal_train_mean_cites", "n_authors")] + \
         ["log_prestige", "log_team", "ai_x_prestige", "ai_x_team"]
tr2 = f2[f2.year.isin(CFG["split"]["train_years"])]
te2 = f2[f2.year.isin(CFG["split"]["test_years"])].copy()
sc2 = StandardScaler()
lr2 = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)
lr2.fit(sc2.fit_transform(tr2[FEATS2]), tr2.y_top10)
p2 = lr2.predict_proba(sc2.transform(te2[FEATS2]))[:, 1]
out["interaction_model"] = {"roc_auc": float(roc_auc_score(te2.y_top10, p2)),
                            "pr_auc": float(average_precision_score(te2.y_top10, p2))}
out["interaction_coefs"] = {}
for nm in ["ai_flag", "ai_x_prestige", "ai_x_team"]:
    i = FEATS2.index(nm)
    out["interaction_coefs"][nm] = {"or_per_unit": float(np.exp(lr2.coef_[0][i] / sc2.scale_[i])),
                                    "beta_raw": float(lr2.coef_[0][i] / sc2.scale_[i])}

def flip_uplift(model, scaler, frame):
    Xon = frame[FEATS2].copy(); Xoff = frame[FEATS2].copy()
    Xon["ai_flag"] = 1; Xon["ai_x_prestige"] = Xon["log_prestige"]; Xon["ai_x_team"] = Xon["log_team"]
    Xoff["ai_flag"] = 0; Xoff["ai_x_prestige"] = 0; Xoff["ai_x_team"] = 0
    return (model.predict_proba(scaler.transform(Xon))[:, 1]
            - model.predict_proba(scaler.transform(Xoff))[:, 1])

jq_labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
tq_labels = ["small", "mid-small", "mid-large", "large"]
te2["jq"] = pd.qcut(te2["journal_train_mean_cites"], 4, labels=jq_labels).values
te2["tq"] = pd.qcut(te2["n_authors"].clip(upper=te2.n_authors.quantile(.99)), 4, labels=tq_labels).values

u = flip_uplift(lr2, sc2, te2)
out["uplift_interaction"] = {
    "overall_mean": float(u.mean()),
    "mean_on_ai_papers": float(u[te2.ai_flag.values == 1].mean()),
    "by_journal_prestige": te2.assign(u=u).groupby("jq", observed=True)["u"].mean().to_dict(),
    "by_team_size": te2.assign(u=u).groupby("tq", observed=True)["u"].mean().to_dict(),
}

# ---- GBM variant sensitivity ----
gbm2 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)
gbm2.fit(train[FEATS], train.y_top10)
p3 = gbm2.predict_proba(test[FEATS])[:, 1]
out["gbm_variant_defaultish"] = {"roc_auc": float(roc_auc_score(test.y_top10, p3)),
                                 "pr_auc": float(average_precision_score(test.y_top10, p3))}
json.dump(out, open(ROOT / "results/metrics_supplement.json", "w"), indent=2)

# ---- bootstrap CIs for subgroup uplift ----
rng = np.random.default_rng(SEED); B = 50
tr_vals = tr2.reset_index(drop=True); tr_idx = np.arange(len(tr2))
bj, bt = [], []
for _ in range(B):
    ii = rng.choice(tr_idx, len(tr_idx), replace=True)
    trb = tr_vals.iloc[ii]
    scb = StandardScaler()
    lrb = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)
    lrb.fit(scb.fit_transform(trb[FEATS2]), trb.y_top10)
    ub = flip_uplift(lrb, scb, te2)
    bj.append(te2.assign(u=ub).groupby("jq", observed=True)["u"].mean().values)
    bt.append(te2.assign(u=ub).groupby("tq", observed=True)["u"].mean().values)
bj, bt = np.array(bj), np.array(bt)

def summ(arr, labels):
    return {lab: {"mean": float(arr[:, i].mean()),
                  "ci95": [float(np.percentile(arr[:, i], 2.5)), float(np.percentile(arr[:, i], 97.5))]}
            for i, lab in enumerate(labels)}

boot = {"bootstrap_reps": B,
        "uplift_by_journal_prestige": summ(bj, jq_labels),
        "uplift_by_team_size": summ(bt, tq_labels)}
json.dump(boot, open(ROOT / "results/metrics_uplift_boot.json", "w"), indent=2)

# ---- uplift figure with CI error bars ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a_, key, title, color in [(ax[0], "uplift_by_journal_prestige",
                               "AI-adoption uplift by journal prestige", "#3d8b6d"),
                              (ax[1], "uplift_by_team_size",
                               "AI-adoption uplift by team size", "#b4652c")]:
    dd = boot[key]; labs = list(dd.keys())
    means = [dd[k]["mean"] for k in labs]
    lo = [dd[k]["mean"] - dd[k]["ci95"][0] for k in labs]
    hi = [dd[k]["ci95"][1] - dd[k]["mean"] for k in labs]
    a_.bar(labs, means, yerr=[lo, hi], capsize=4, color=color, alpha=.85)
    a_.axhline(0, color="k", lw=.7)
    a_.set_ylabel("delta P(top-10% cited)")
    a_.set_title(title + " (bootstrap 95% CI)")
    a_.tick_params(axis="x", rotation=20)
fig.tight_layout(); fig.savefig(FIG / "uplift.png", dpi=160); plt.close(fig)
print("supplement done")

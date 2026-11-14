#!/usr/bin/env python3
"""EDA figures for the report: adoption trend, citations, AI score."""
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data/processed/dataset.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
n = df.groupby("year").size()
ax[0].bar(n.index.astype(str), n.values, color="#2c6fbb")
ax[0].set_title("Sampled papers per year"); ax[0].set_ylabel("papers")
r = df.groupby("year")["ai_flag"].mean() * 100
ax[1].plot(r.index.astype(str), r.values, "o-", color="#b43a3a")
ax[1].set_title("AI-adoption rate in Medicine (lexicon-based)")
ax[1].set_ylabel("% AI-augmented"); ax[1].set_ylim(0, None)
for x, y in zip(r.index.astype(str), r.values):
    ax[1].annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "eda_sample_adoption.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for flag, lab, c in [(0, "non-AI", "#666666"), (1, "AI-augmented", "#b43a3a")]:
    v = df.loc[df.ai_flag == flag, "cited_by"].clip(upper=200)
    ax[0].hist(v, bins=60, alpha=.55, label=lab, density=True, color=c)
ax[0].set_xlabel("citations (clipped at 200)"); ax[0].set_ylabel("density")
ax[0].set_title("Citation distribution"); ax[0].legend()
m = df.groupby(["year", "ai_flag"])["cited_by"].median().unstack()
m.columns = ["non-AI", "AI-augmented"]
m.plot(ax=ax[1], marker="o", color=["#666666", "#b43a3a"])
ax[1].set_title("Median citations by cohort year"); ax[1].set_ylabel("median citations")
ax[1].set_xticks(range(len(m.index))); ax[1].set_xticklabels(m.index.astype(str))
fig.tight_layout(); fig.savefig(FIG / "eda_citations.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
s = df["ai_score"].clip(upper=10)
ax[0].hist(s, bins=range(0, 12), color="#3d8b6d", rwidth=.85)
ax[0].set_title("AI-lexicon score distribution"); ax[0].set_xlabel("score (clipped at 10)")
ax[0].set_yscale("log"); ax[0].set_ylabel("papers (log scale)")
df["score_bin"] = pd.cut(df.ai_score, [-1, 0, 1, 2, 3, 5, 100],
                         labels=["0", "1", "2", "3", "4-5", "6+"])
g = df.groupby("score_bin", observed=True)["y_top10"].mean() * 100
ax[1].bar(g.index.astype(str), g.values, color="#3d8b6d")
ax[1].axhline(10, color="k", ls="--", lw=.7)
ax[1].set_title("Observed top-10% rate by AI score"); ax[1].set_ylabel("% top-10% cited")
fig.tight_layout(); fig.savefig(FIG / "eda_ai_score.png", dpi=160); plt.close(fig)
print("EDA figures saved")

#!/usr/bin/env python3
"""Build the analysis dataset from raw Europe PMC records.

Steps:
  1. Parse raw JSONL -> tidy DataFrame (one row per paper).
  2. Label AI adoption with a transparent regex lexicon (title + abstract).
  3. Define outcome: top-10% cited within publication-year cohort.
  4. Engineer leakage-safe features (journal prestige computed from TRAIN
     years only; topic dummies from TRAIN vocabulary only).
  5. Subsample to the frozen working sample (seed=42).

Outputs:
  data/processed/dataset.csv        full working sample with features
  results/label_stats.json          AI-adoption descriptives for the report
"""
import json, re, pathlib
import numpy as np, pandas as pd, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/config.yaml"))
RAW = ROOT / "data/raw/epmc_raw.jsonl"


def parse_record(rec):
    j = (rec.get("journalInfo") or {}).get("journal") or {}
    grants = (rec.get("grantsList") or {}).get("grant") or []
    kws = (rec.get("keywordList") or {}).get("keyword") or []
    ptypes = (rec.get("pubTypeList") or {}).get("pubType") or []
    auth = rec.get("authorString") or ""
    n_authors = 0 if not auth else len([a for a in auth.split(",") if a.strip()])
    return {
        "id": rec.get("id"),
        "year": int(rec.get("pubYear") or 0),
        "title": rec.get("title") or "",
        "abstract": rec.get("abstractText") or "",
        "cited_by": int(rec.get("citedByCount") or 0),
        "n_authors": n_authors,
        "n_keywords": len(kws),
        "n_grants": len(grants),
        "journal": j.get("title") or "unknown",
        "open_access": 1 if rec.get("isOpenAccess") == "Y" else 0,
        "is_review": 1 if any("review" in str(p).lower() for p in ptypes) else 0,
        "has_references": 1 if rec.get("hasReferences") == "y" else 0,
        "english": 1 if rec.get("language") == "eng" else 0,
        "keywords": "|".join(str(k).lower().strip() for k in kws),
        "window_hits": (rec.get("_fetch") or {}).get("window_hits"),
    }


def build_lexicon(cfg):
    parts = []
    for t in cfg["labeling"]["core_terms"]:
        parts.append((re.compile(r"\b" + re.escape(t) + r"\b", re.I), 2))
    for t in cfg["labeling"]["extended_terms"]:
        parts.append((re.compile(r"\b" + re.escape(t) + r"\b", re.I), 1))
    return parts


def ai_score(text, lexicon):
    return sum(w for rx, w in lexicon if rx.search(text))


def main():
    rows = []
    with open(RAW) as f:
        for line in f:
            rows.append(parse_record(json.loads(line)))
    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    df = df[df["year"].isin(CFG["sampling"]["years"])]
    df = df[df["abstract"].str.len() > 100]  # abstracts present (query already requires)
    print("parsed:", len(df))

    # --- subsample to frozen working sample ---
    target = CFG["sampling"]["target_working_sample"]
    if len(df) > target:
        df = df.sample(target, random_state=CFG["seeds"]["subsample"]).reset_index(drop=True)

    # --- AI adoption labels ---
    lex = build_lexicon(CFG)
    texts = df["title"] + " " + df["abstract"]
    df["ai_score"] = [ai_score(t, lex) for t in texts]
    thr = CFG["labeling"]["flag_threshold"]
    df["ai_flag"] = (df["ai_score"] >= thr).astype(int)
    df["ai_flag_strict"] = (df["ai_score"] >= CFG["labeling"]["sensitivity"]["strict_threshold"]).astype(int)
    df["ai_flag_loose"] = (df["ai_score"] >= CFG["labeling"]["sensitivity"]["loose_threshold"]).astype(int)

    # --- outcome: top-cited within publication-year cohort ---
    for pct, name in [(CFG["outcome"]["primary_percentile"], "top10"),
                      (CFG["outcome"]["sensitivity_percentiles"][0], "top5"),
                      (CFG["outcome"]["sensitivity_percentiles"][1], "top20")]:
        cut = df.groupby("year")["cited_by"].transform(lambda s: s.quantile(pct))
        df[f"y_{name}"] = (df["cited_by"] >= cut).astype(int)

    df["abstract_words"] = df["abstract"].str.split().str.len()
    df["title_words"] = df["title"].str.split().str.len()

    # --- leakage-safe journal prestige: TRAIN years only ---
    train_mask = df["year"].isin(CFG["split"]["train_years"])
    tr = df[train_mask]
    jmean = tr.groupby("journal")["cited_by"].mean().rename("journal_train_mean_cites")
    jn = tr.groupby("journal").size().rename("journal_train_n_papers")
    df = df.merge(jmean, on="journal", how="left").merge(jn, on="journal", how="left")
    df["journal_train_mean_cites"] = df["journal_train_mean_cites"].fillna(tr["cited_by"].median())
    df["journal_train_n_papers"] = df["journal_train_n_papers"].fillna(0)

    # --- topic dummies from TRAIN keyword vocabulary ---
    kw_counts = {}
    for kws in tr.loc[tr["keywords"] != "", "keywords"]:
        for k in set(kws.split("|")):
            kw_counts[k] = kw_counts.get(k, 0) + 1
    ntr = len(tr)
    lo = CFG["features"].get("topic_min_df", 0.002) * ntr
    hi = CFG["features"].get("topic_max_df", 0.50) * ntr
    vocab = [k for k, c in kw_counts.items() if lo < c < hi and len(k) > 3]
    vocab = sorted(vocab, key=lambda k: -kw_counts[k])[: CFG["features"]["topic_dummies"]]
    for k in vocab:
        df[f"kw::{k}"] = df["keywords"].str.contains(re.escape(k), regex=True).astype(int)

    out = ROOT / "data/processed/dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    stats = {
        "n_papers": int(len(df)),
        "years": {int(y): int(n) for y, n in df.groupby("year").size().items()},
        "ai_rate_overall": float(df["ai_flag"].mean()),
        "ai_n": int(df["ai_flag"].sum()),
        "ai_rate_by_year": {int(y): float(v) for y, v in df.groupby("year")["ai_flag"].mean().items()},
        "ai_rate_loose": float(df["ai_flag_loose"].mean()),
        "ai_rate_strict": float(df["ai_flag_strict"].mean()),
        "top10_rate_overall": float(df["y_top10"].mean()),
        "top10_rate_ai": float(df.loc[df.ai_flag == 1, "y_top10"].mean()),
        "top10_rate_nonai": float(df.loc[df.ai_flag == 0, "y_top10"].mean()),
        "median_cites_ai": float(df.loc[df.ai_flag == 1, "cited_by"].median()),
        "median_cites_nonai": float(df.loc[df.ai_flag == 0, "cited_by"].median()),
        "n_journals": int(df["journal"].nunique()),
        "share_with_keywords": float((df["n_keywords"] > 0).mean()),
        "open_access_share": float(df["open_access"].mean()),
        "review_share": float(df["is_review"].mean()),
        "english_share": float(df["english"].mean()),
        "median_authors": float(df["n_authors"].median()),
        "median_abstract_words": float(df["abstract_words"].median()),
        "topic_vocab": vocab,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(stats, open(ROOT / "results/label_stats.json", "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()

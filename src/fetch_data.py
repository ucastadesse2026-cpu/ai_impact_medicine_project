#!/usr/bin/env python3
"""Fetch a seeded random sample of Medicine journal articles from Europe PMC.

Sampling design (reproducible, see configs/config.yaml):
  For each year, draw `windows_per_year` random 2-day windows (seed=42),
  query Europe PMC for journal articles with abstracts published in the
  window, and keep the first `page_size` records sorted by publication date.
  Within-window ordering is arbitrary w.r.t. citations/AI use, so the
  resulting pool approximates a simple random sample of the year cohort.
  Parallelism only speeds up I/O; the window list is seed-deterministic.

Output: data/raw/epmc_raw.jsonl  (one JSON record per paper + fetch metadata)
"""
import json, time, random, sys, pathlib, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/config.yaml"))
OUT = ROOT / "data/raw/epmc_raw.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def get(url, params, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code}, retry {i+1}", flush=True)
        except requests.RequestException as e:
            print(f"  {e}, retry {i+1}", flush=True)
        time.sleep(2 ** i)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def random_windows(year, n, days, seed):
    rng = random.Random(seed + year)
    base = dt.date(year, 1, 1)
    max_off = 365 - days
    starts = []
    while len(starts) < n:
        off = rng.randint(0, max_off)
        s = base + dt.timedelta(days=off)
        e = s + dt.timedelta(days=days - 1)
        starts.append((s.isoformat(), e.isoformat()))
    return sorted(set(starts))


def fetch_window(year, start, end, s):
    q = s["query_template"].format(start=start, end=end)
    d = get(BASE, {"query": q, "resultType": "core",
                   "pageSize": s["page_size"], "format": "json",
                   "sort": "P_PDATE_D asc"})
    hits = d.get("hitCount", 0)
    recs = d.get("resultList", {}).get("result", [])
    for rec in recs:
        rec["_fetch"] = {"window": [start, end], "window_hits": hits,
                         "fetched_at": CFG["sampling"]["fetched_at"]}
    return year, start, recs


def main():
    s = CFG["sampling"]
    tasks = []
    for year in s["years"]:
        for start, end in random_windows(year, s["windows_per_year"],
                                         s["window_days"], CFG["seeds"]["sampling"]):
            tasks.append((year, start, end))
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_window, y, a, b, s) for y, a, b in tasks]
        for i, fut in enumerate(futs):
            y, a, recs = fut.result()
            results[(y, a)] = recs
            if (i + 1) % 12 == 0:
                print(f"{i+1}/{len(tasks)} windows", flush=True)
    n_written = 0
    with open(OUT, "w") as f:
        for (y, a) in sorted(results):
            for rec in results[(y, a)]:
                f.write(json.dumps(rec) + "\n")
                n_written += 1
    per_year = {}
    for (y, a), recs in results.items():
        per_year[y] = per_year.get(y, 0) + len(recs)
    for y in sorted(per_year):
        print(f"year {y}: {per_year[y]} records", flush=True)
    print(f"DONE: {n_written} raw records -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())

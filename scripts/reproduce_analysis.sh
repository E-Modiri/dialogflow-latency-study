#!/usr/bin/env bash
# Rebuilds every table, figure and number reported in the paper from the raw logs.
# Usage (from the repository root):  bash scripts/reproduce_analysis.sh
set -euo pipefail
rm -rf results_local
RAW=data/raw
S1=$RAW/session1_2026-11-05_weather_NL/attempts.csv
S2=$RAW/session2_2026-11-06_capital_joke_NL/attempts.csv
S3=$RAW/session3_2026-11-07_currency_capital-replication_US/attempts.csv

# Main results (Tables 3-8, Figures 2-6). Session 3's Capital City rows are the
# replication run and are excluded here; Capital City main results come from session 2.
python analysis/analysis.py "$S1" "$S2" "$S3" \
  --exclude "20261107T022933Z:capital:*" \
  --out results_local/main

# Capital City replication (Table 9, "Run 2"): session 3 only, Currency excluded.
python analysis/analysis.py "$S3" \
  --exclude "*:currency:*" \
  --out results_local/replication_capital_session3


# Table 9: original Capital City run (session 2) vs replication (session 3)
python - <<'PY'
import pandas as pd
rows = []
for run, vm, path in [("Run 1", "Groningen, NL", "results_local/main"),
                      ("Run 2 (replication)", "Salt Lake City, US", "results_local/replication_capital_session3")]:
    t = pd.read_csv(f"{path}/tables/sequential_paired_tests.csv")
    for _, r in t[t.domain == "capital"].iterrows():
        rows.append({"run": run, "vm_location": vm, "comparison": f"{r.a}-{r.b}", "n_pairs": r.n_pairs,
                     "mean_diff_ms": round(r.mean_diff_ms, 1), "ci_lo": round(r.ci_lo, 1), "ci_hi": round(r.ci_hi, 1)})
pd.DataFrame(rows).to_csv("results_local/main/tables/capital_replication.csv", index=False)
print("Table 9 written")
PY

echo "Done. Compare results_local/ with results/ (they should be identical)."

"""Supplementary numbers reported in the text of Sections 5.1, 5.4, 5.5 and 5.7 that are not
part of the tables produced by analysis/analysis.py:

  * time to eventual success of requests recovered by the single retry (Sections 5.1, 5.4, 5.5);
  * quota rejections in the sustained test at 3 requests/s: arrivals, the number of Dialogflow
    requests sent in the preceding 60 s at the first rejection, and the peak of that count (Section 5.5);
  * isolated transient failures (Section 5.7).

Usage:  python scripts/supplementary_analysis.py [data/raw]
The Capital City rows of session 3 (the replication run) are excluded, as in the main analysis.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "raw")
REPLICATION_RUN = "20261107T022933Z"

frames = [pd.read_csv(f, low_memory=False) for f in glob.glob(os.path.join(RAW, "*", "attempts.csv"))]
d = pd.concat(frames, ignore_index=True)
d = d[~((d.run_id == REPLICATION_RUN) & (d.domain == "capital"))]
d["ts"] = pd.to_datetime(d.ts_utc_start, utc=True)
d["t"] = (d.ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
m = d[d.phase == "measure"]

# ---- time to eventual success ------------------------------------------------------------
first = m[m.attempt == 1].set_index("request_uid")
retry = m[m.attempt == 2].set_index("request_uid")
j = first[["experiment", "domain", "arch", "t"]].join(
    retry[["t", "latency_ms", "success_e2e"]], rsuffix="_retry", how="inner")
j["time_to_eventual_success_ms"] = (j.t_retry - j.t) * 1000 + j.latency_ms
rec = j[j.success_e2e == True]  # noqa: E712
print("== Retries and time to eventual success (ms, first attempt start to end of successful retry)")
print(j.groupby("experiment").size().rename("retried").to_frame()
      .join(rec.groupby("experiment").size().rename("recovered")).fillna(0).astype(int).to_string())
for label, sel in [("burst, all recovered", rec.experiment == "burst"),
                   ("sustained, A recovered", (rec.experiment == "sustained") & (rec.arch == "A"))]:
    x = rec[sel].time_to_eventual_success_ms
    print(f"{label}: n={len(x)}, median={x.median():.0f}, min={x.min():.0f}, max={x.max():.0f}")

# ---- quota rejections in the sustained test at 3 requests/s ------------------------------
print("\n== Sustained test, A at 3 requests/s: quota rejections and rolling 60 s request counts")
rows = []
for dom in sorted(m.domain.unique()):
    a = d[(d.arch == "A") & (d.domain == dom)].sort_values("t")   # all Dialogflow requests of this agent
    t = a.t.to_numpy()
    a = a.assign(sent_last_60s=[int(((t > ti - 60) & (t <= ti)).sum()) for ti in t])
    s = a[(a.experiment == "sustained") & (a.phase == "measure") & (a.rate_rps == 3)]
    q = s[s.error_class == "quota"]
    rows.append({"domain": dom,
                 "arrivals": int((s.attempt == 1).sum()),
                 "quota_rejections": len(q),
                 "count_at_first_rejection": int(q.sort_values("t").sent_last_60s.iloc[0]) if len(q) else None,
                 "peak_count_in_60s": int(s.sent_last_60s.max())})
print(pd.DataFrame(rows).to_string(index=False))

# ---- isolated transient failures ---------------------------------------------------------
print("\n== First-attempt transport and webhook failures (measurement phase)")
tf = m[(m.attempt == 1) & m.error_class.isin(["transport", "webhook"])]
print(tf[["experiment", "domain", "arch", "batch_size", "rate_rps", "error_class", "latency_ms"]]
      .to_string(index=False))

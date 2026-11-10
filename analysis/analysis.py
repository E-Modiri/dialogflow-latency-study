"""
analysis.py — rebuilds every table, figure and in-text number from bench_harness logs.

    python analysis.py bench_out/attempts.csv [more.csv ...] --out results

Outputs
  results/tables/*.csv (+ .tex)   accounting, sequential paired tests, stage decomposition,
                                  burst summary, sustained summary, entity mismatches
  results/figures/*.png
  results/numbers.json            every number the manuscript text should quote

Statistical choices (reported in the Methods so reviewers can check them):
  * Unit of analysis for sequential tests = query (cluster). Each query is measured once per
    block, so a query contributes `blocks` paired observations. The paired difference d = A - B
    is taken within the same (block, query) cell, on the COMMON SET where every compared
    architecture succeeded on the first attempt.
  * Mean overhead CI: two-stage cluster bootstrap (resample queries, then blocks within query),
    5,000 replicates, percentile interval.
  * Wilcoxon signed-rank on per-query mean differences (n = number of queries), exact p when
    n <= 50 and there are no ties/zeros, else normal approximation. Effect size is the
    matched-pairs rank-biserial correlation r_rb = (W+ - W-) / (W+ + W-), range [-1, 1].
    With n = 5 categories (Joke) the smallest attainable two-sided p is 2/2^5 = 0.0625, so
    the category-level test cannot reach p < 0.05; we report the mixed model instead.
  * Mixed model: log(latency) ~ arch + block, random intercept per query (statsmodels MixedLM).
    exp(beta_arch) = multiplicative overhead, with 95% Wald CI.
  * Temporal dependence: lag-1 autocorrelation of the time-ordered paired differences, plus a
    moving-block bootstrap CI for the mean difference (block length = ceil(sqrt(n))).
  * Burst tests: the independent unit is the run. Mean-latency CI = t interval over run means
    (df = runs - 1). P90/P95 are pooled over all first-attempt requests. Exceedance of a
    threshold is the pooled proportion of requests above it (Wilson 95% CI), not an average
    of per-run P90s.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20261103)
N_BOOT = 5000
DOMAIN_ORDER = ["weather", "capital", "currency", "joke"]
DOMAIN_NAME = {"weather": "Weather", "capital": "Capital City", "currency": "Currency",
               "joke": "Joke"}
ARCH_NAME = {"A": "A: Dialogflow+tunnel+webhook", "C": "C: tunnel+webhook (control)",
             "L": "L: local webhook (control)", "B": "B: direct API"}
COMPARISONS = [("A", "B", "end-to-end overhead (A-B)"),
               ("A", "C", "Dialogflow + Google network legs (A-C)"),
               ("C", "L", "ngrok tunnel (C-L)"),
               ("L", "B", "webhook handling (L-B)")]
THRESHOLDS_MS = [600, 1000]


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------

def load(paths: List[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c in ["transport_ok", "intent_ok", "webhook_ok", "api_ok", "success_e2e"]:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1"])
    for c in ["latency_ms", "wh_handler_ms", "wh_api_ms", "wh_parse_ms", "block", "batch_size",
              "rate_rps", "order_index", "t_sched_s", "t_send_s", "attempt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts_utc_start"], utc=True, errors="coerce")
    df["error_class"] = df["error_class"].fillna("")
    return df


def requests_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse attempts to one row per logical request."""
    df = df.sort_values(["request_uid", "attempt"])
    g = df.groupby("request_uid", sort=False)
    first = g.head(1).set_index("request_uid")
    last = g.tail(1).set_index("request_uid")
    out = first[["run_id", "experiment", "phase", "domain", "arch", "block", "batch_size",
                 "rate_rps", "query_label", "order_index", "ts", "t_sched_s", "t_send_s",
                 "wh_handler_ms", "wh_api_ms", "entity_match", "extracted_entity",
                 "expected_entity", "matched_intent"]].copy()
    out["first_latency_ms"] = first["latency_ms"]
    out["first_success"] = first["success_e2e"]
    out["first_error"] = first["error_class"]
    out["n_attempts"] = g.size()
    out["eventual_success"] = g["success_e2e"].any()
    out["final_error"] = np.where(out["eventual_success"], "", last["error_class"])
    # time to eventual success = start of first attempt -> end of successful attempt
    succ = df[df["success_e2e"]].groupby("request_uid").head(1).set_index("request_uid")
    tts = (succ["ts"] - first.loc[succ.index, "ts"]).dt.total_seconds() * 1000 + succ["latency_ms"]
    out["time_to_success_ms"] = tts
    return out.reset_index()


# --------------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_boot_mean(d: pd.DataFrame, value: str, cluster: str, base: pd.Series = None):
    """Two-stage cluster bootstrap of a mean (and optional ratio to a base mean)."""
    groups = [g[value].to_numpy() for _, g in d.groupby(cluster)]
    bases = [g[base.name].to_numpy() for _, g in d.groupby(cluster)] if base is not None else None
    k = len(groups)
    means, ratios = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = RNG.integers(0, k, k)
        vals, bvals = [], []
        for i in idx:
            j = RNG.integers(0, len(groups[i]), len(groups[i]))
            vals.append(groups[i][j])
            if bases is not None:
                bvals.append(bases[i][j])
        v = np.concatenate(vals)
        means[b] = v.mean()
        if bases is not None:
            ratios[b] = v.mean() / np.concatenate(bvals).mean()
    ci = tuple(np.percentile(means, [2.5, 97.5]))
    rci = tuple(np.percentile(ratios, [2.5, 97.5])) if bases is not None else (math.nan, math.nan)
    return ci, rci


def moving_block_boot_mean(x: np.ndarray) -> tuple:
    n = len(x)
    if n < 4:
        return (math.nan, math.nan)
    L = int(math.ceil(math.sqrt(n)))
    starts = np.arange(n - L + 1)
    nb = int(math.ceil(n / L))
    m = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = RNG.choice(starts, nb)
        m[b] = np.concatenate([x[i:i + L] for i in s])[:n].mean()
    return tuple(np.percentile(m, [2.5, 97.5]))


def wilcoxon_rb(d: np.ndarray) -> Dict[str, float]:
    d = d[~np.isnan(d)]
    n = len(d)
    out = {"n": n, "W_plus": math.nan, "W_minus": math.nan, "p": math.nan, "r_rb": math.nan,
           "method": "", "min_attainable_p": 2 / 2 ** n if n else math.nan}
    if n < 2:
        return out
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    exact_ok = n <= 50 and len(nz) == n and len(np.unique(np.abs(nz))) == n
    method = "exact" if exact_ok else "approx"
    res = stats.wilcoxon(d, alternative="two-sided", method=method)
    out.update(W_plus=wp, W_minus=wm, p=float(res.pvalue), method=method,
               r_rb=(wp - wm) / (wp + wm) if (wp + wm) else math.nan)
    return out


def mixed_model(d: pd.DataFrame, a: str, b: str) -> Dict[str, float]:
    import statsmodels.formula.api as smf
    sub = d[d["arch"].isin([a, b])].copy()
    sub["loglat"] = np.log(sub["first_latency_ms"])
    sub["is_a"] = (sub["arch"] == a).astype(int)
    sub["block"] = sub["block"].astype(int)
    out = {"ratio": math.nan, "ratio_lo": math.nan, "ratio_hi": math.nan, "p": math.nan,
           "n_groups": sub["query_label"].nunique(), "converged": False}
    try:
        import warnings
        f = "loglat ~ is_a + C(block)" if sub["block"].nunique() > 1 else "loglat ~ is_a"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(f, sub, groups=sub["query_label"]).fit(reml=True)
        lo, hi = m.conf_int().loc["is_a"]
        out.update(ratio=math.exp(m.params["is_a"]), ratio_lo=math.exp(lo), ratio_hi=math.exp(hi),
                   p=float(m.pvalues["is_a"]), converged=bool(m.converged))
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:120]
    return out


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------

def accounting(att: pd.DataFrame, req: pd.DataFrame, paired_keys: Dict) -> pd.DataFrame:
    rows = []
    meas = req[req["phase"] == "measure"]
    for (exp, dom, arch), g in meas.groupby(["experiment", "domain", "arch"]):
        a = att[(att["phase"] == "measure") & (att["experiment"] == exp) &
                (att["domain"] == dom) & (att["arch"] == arch)]
        r = {"experiment": exp, "domain": dom, "arch": arch,
             "logical_requests": len(g), "attempts": len(a),
             "retries": int((g["n_attempts"] - 1).sum()),
             "first_attempt_success": int(g["first_success"].sum()),
             "eventual_success": int(g["eventual_success"].sum()),
             "success_after_retry": int((g["eventual_success"] & ~g["first_success"]).sum()),
             "entity_mismatch_but_api_ok": int(((g["entity_match"].astype(str) == "False")
                                                & g["first_success"]).sum())}
        for cls, n in g.loc[~g["first_success"], "first_error"].value_counts().items():
            r[f"first_fail_{cls or 'unknown'}"] = int(n)
        if exp == "sequential":
            r["in_common_paired_set"] = paired_keys.get(dom, 0)
            r["excluded_from_paired_set"] = len(g) - paired_keys.get(dom, 0)
        rows.append(r)
    return pd.DataFrame(rows).fillna(0)


def sequential_tables(req: pd.DataFrame, out: Dict) -> tuple:
    seq = req[(req["experiment"] == "sequential") & (req["phase"] == "measure")]
    test_rows, stage_rows, paired_keys = [], [], {}
    for dom in [d for d in DOMAIN_ORDER if d in seq["domain"].unique()]:
        s = seq[seq["domain"] == dom]
        archs = sorted(s["arch"].unique())
        keyed = s.drop_duplicates(["block", "query_label", "arch"]).set_index(
            ["block", "query_label", "arch"])
        wide_ok = keyed["first_success"].unstack("arch")
        wide_ms = keyed["first_latency_ms"].unstack("arch")
        wide_ts = keyed["ts"].unstack("arch")
        common = wide_ok[archs].fillna(False).astype(bool).all(axis=1)
        paired_keys[dom] = int(common.sum())
        ms = wide_ms[common]
        common_long = s.set_index(["block", "query_label"]).loc[ms.index].reset_index()
        out.setdefault("sequential", {})[dom] = {"common_cells": int(common.sum()),
                                                 "cells_total": int(len(common)),
                                                 "queries": int(s["query_label"].nunique()),
                                                 "blocks": int(s["block"].nunique()),
                                                 "by_arch": {}, "comparisons": {}}
        for a in archs:
            v = ms[a].to_numpy()
            out["sequential"][dom]["by_arch"][a] = {
                "n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "p95": float(np.percentile(v, 95))}
        for a, b, label in COMPARISONS:
            if a not in archs or b not in archs:
                continue
            dd = pd.DataFrame({"d": ms[a] - ms[b], "base": ms[b]}).reset_index()
            ci, rci = cluster_boot_mean(dd, "d", "query_label", base=dd["base"].rename("base"))
            perq = dd.groupby("query_label")["d"].mean().to_numpy()
            wx = wilcoxon_rb(perq)
            mm = mixed_model(common_long, a, b)
            order = wide_ts.loc[ms.index, a].sort_values().index
            dseq = (ms[a] - ms[b]).loc[order].to_numpy()
            ac1 = float(pd.Series(dseq).autocorr(1)) if len(dseq) > 3 else math.nan
            mbb = moving_block_boot_mean(dseq)
            row = {"domain": dom, "comparison": label, "a": a, "b": b,
                   "n_pairs": len(dd), "n_queries": wx["n"],
                   "mean_diff_ms": dd["d"].mean(), "ci_lo": ci[0], "ci_hi": ci[1],
                   "median_diff_ms": dd["d"].median(),
                   "pct_of_b": 100 * dd["d"].mean() / dd["base"].mean(),
                   "pct_ci_lo": 100 * rci[0], "pct_ci_hi": 100 * rci[1],
                   "wilcoxon_W_plus": wx["W_plus"], "wilcoxon_W_minus": wx["W_minus"],
                   "wilcoxon_p": wx["p"], "wilcoxon_method": wx["method"],
                   "min_attainable_p": wx["min_attainable_p"], "r_rb": wx["r_rb"],
                   "mm_ratio": mm["ratio"], "mm_ratio_lo": mm["ratio_lo"],
                   "mm_ratio_hi": mm["ratio_hi"], "mm_p": mm["p"],
                   "mm_converged": mm["converged"],
                   "lag1_autocorr": ac1, "mbb_ci_lo": mbb[0], "mbb_ci_hi": mbb[1]}
            test_rows.append(row)
            out["sequential"][dom]["comparisons"][f"{a}-{b}"] = row
        # stage decomposition (Architecture A and C rows carry server-side webhook timings)
        stage = {"domain": dom}
        cl = common_long.set_index(["block", "query_label", "arch"])
        for a in archs:
            sub = cl.xs(a, level="arch")
            stage[f"{a}_total_median"] = sub["first_latency_ms"].median()
            if a in ("A", "C", "L"):
                stage[f"{a}_webhook_handler_median"] = sub["wh_handler_ms"].median()
                stage[f"{a}_external_api_median"] = sub["wh_api_ms"].median()
                stage[f"{a}_outside_webhook_median"] = (sub["first_latency_ms"]
                                                        - sub["wh_handler_ms"]).median()
        if "B" in archs:
            stage["B_external_api_median"] = stage["B_total_median"]
        stage_rows.append(stage)
        out["sequential"][dom]["stages"] = stage
    return pd.DataFrame(test_rows), pd.DataFrame(stage_rows), paired_keys


def burst_tables(req: pd.DataFrame, out: Dict) -> pd.DataFrame:
    b = req[(req["experiment"] == "burst") & (req["phase"] == "measure")]
    rows = []
    for (dom, arch, size), g in b.groupby(["domain", "arch", "batch_size"]):
        ok = g[g["first_success"]]
        lat = ok["first_latency_ms"].to_numpy()
        run_means = ok.groupby("block")["first_latency_ms"].mean()
        k = len(run_means)
        t = stats.t.ppf(0.975, k - 1) if k > 1 else math.nan
        sem = run_means.std(ddof=1) / math.sqrt(k) if k > 1 else math.nan
        r = {"domain": dom, "arch": arch, "batch_size": int(size), "runs": k,
             "requests": len(g), "first_attempt_success": int(g["first_success"].sum()),
             "eventual_success": int(g["eventual_success"].sum()),
             "retries": int((g["n_attempts"] - 1).sum()),
             # mean of the run means, so that the mean, its t-interval and A-B are consistent
             "mean_ms": run_means.mean() if k else math.nan,
             "mean_pooled_ms": lat.mean() if len(lat) else math.nan,
             "mean_ci_lo": run_means.mean() - t * sem, "mean_ci_hi": run_means.mean() + t * sem,
             "between_run_sd": run_means.std(ddof=1) if k > 1 else math.nan,
             "median_ms": np.median(lat) if len(lat) else math.nan,
             "p90_pooled_ms": np.percentile(lat, 90) if len(lat) else math.nan,
             "p95_pooled_ms": np.percentile(lat, 95) if len(lat) else math.nan,
             "run_mean_trend_spearman": (stats.spearmanr(run_means.index, run_means.values)[0]
                                         if k > 2 else math.nan)}
        for thr in THRESHOLDS_MS:
            # failed first attempts count as "not answered within threshold"
            exceed = int((~g["first_success"] | (g["first_latency_ms"] > thr)).sum())
            lo, hi = wilson(exceed, len(g))
            r[f"exceed_{thr}_n"] = exceed
            r[f"exceed_{thr}_pct"] = 100 * exceed / len(g)
            r[f"exceed_{thr}_ci_lo"], r[f"exceed_{thr}_ci_hi"] = 100 * lo, 100 * hi
        rows.append(r)
    tab = pd.DataFrame(rows)
    # overhead per batch size, runs paired by run index
    oh_rows = []
    for (dom, size), g in b[b["first_success"]].groupby(["domain", "batch_size"]):
        rm = g.groupby(["block", "arch"])["first_latency_ms"].mean().unstack()
        if not {"A", "B"} <= set(rm.columns):
            continue
        d = (rm["A"] - rm["B"]).dropna()
        k = len(d)
        t = stats.t.ppf(0.975, k - 1) if k > 1 else math.nan
        sem = d.std(ddof=1) / math.sqrt(k) if k > 1 else math.nan
        oh_rows.append({"domain": dom, "batch_size": int(size), "runs_paired": k,
                        "overhead_ms": d.mean(), "overhead_ci_lo": d.mean() - t * sem,
                        "overhead_ci_hi": d.mean() + t * sem,
                        "overhead_pct": 100 * d.mean() / rm["B"].mean()})
    oh = pd.DataFrame(oh_rows)
    if len(tab) and len(oh):
        tab = tab.merge(oh, on=["domain", "batch_size"], how="left")
    out["burst"] = json.loads(tab.to_json(orient="records"))
    return tab


def sustained_tables(req: pd.DataFrame, out: Dict) -> pd.DataFrame:
    s = req[(req["experiment"] == "sustained") & (req["phase"] == "measure")]
    rows = []
    for (dom, arch, rate), g in s.groupby(["domain", "arch", "rate_rps"]):
        ok = g[g["first_success"]]
        span = g["t_send_s"].max() - g["t_send_s"].min()
        lag = (g["t_send_s"] - g["t_sched_s"]) * 1000
        rows.append({"domain": dom, "arch": arch, "offered_rps": rate, "requests": len(g),
                     "duration_s": span,
                     "achieved_rps": len(ok) / span if span > 0 else math.nan,
                     "first_attempt_error_pct": 100 * (1 - g["first_success"].mean()),
                     "eventual_error_pct": 100 * (1 - g["eventual_success"].mean()),
                     "p50_ms": ok["first_latency_ms"].median(),
                     "p90_ms": ok["first_latency_ms"].quantile(.9),
                     "p95_ms": ok["first_latency_ms"].quantile(.95),
                     "generator_lag_p95_ms": lag.quantile(.95)})
    tab = pd.DataFrame(rows)
    out["sustained"] = json.loads(tab.to_json(orient="records"))
    return tab


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------

def figures(req: pd.DataFrame, burst: pd.DataFrame, stages: pd.DataFrame, sus: pd.DataFrame,
            fig_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    col = {"A": "#4F46E5", "B": "#059669", "C": "#D97706", "L": "#6B7280"}
    seq = req[(req["experiment"] == "sequential") & (req["phase"] == "measure")
              & req["first_success"]]
    doms = [d for d in DOMAIN_ORDER if d in seq["domain"].unique()]

    # Fig: raw single-request distributions (log x so outliers stay visible)
    if doms:
        fig, axes = plt.subplots(len(doms), 1, figsize=(8, 2.6 * len(doms)), squeeze=False)
        for ax, dom in zip(axes[:, 0], doms):
            s = seq[seq["domain"] == dom]
            lo, hi = s["first_latency_ms"].min() * 0.8, s["first_latency_ms"].max() * 1.2
            bins = np.logspace(np.log10(lo), np.log10(hi), 50)
            for a in ["B", "L", "C", "A"]:
                v = s.loc[s["arch"] == a, "first_latency_ms"].to_numpy()
                if len(v) < 3:
                    continue
                # bar height = share of requests in the bin (comparable across architectures)
                ax.hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)), alpha=0.45,
                        color=col[a], label=f"{ARCH_NAME[a]} (n={len(v)})")
            ax.set_ylabel("% of requests per bin")
            ax.set_xscale("log")
            ax.set_title(f"{DOMAIN_NAME[dom]}: raw first-attempt latencies (successful)", fontsize=10)
            ax.set_xlabel("latency (ms, log scale)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_single_distributions.png"), dpi=200)
        plt.close(fig)

    # Fig: Joke category-level summary, shown separately from raw distribution
    if "joke" in doms:
        s = seq[seq["domain"] == "joke"]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        cats = sorted(s["query_label"].unique())
        for j, a in enumerate(sorted(s["arch"].unique())):
            for i, c in enumerate(cats):
                v = s[(s["arch"] == a) & (s["query_label"] == c)]["first_latency_ms"]
                x = i + (j - 1.5) * 0.18
                ax.scatter(np.full(len(v), x) + RNG.uniform(-.04, .04, len(v)), v, s=6,
                           color=col[a], alpha=.35)
                ax.scatter([x], [v.mean()], s=40, color=col[a], edgecolor="k",
                           label=ARCH_NAME[a] if i == 0 else None, zorder=3)
        ax.set_xticks(range(len(cats)), cats)
        ax.set_yscale("log")
        ax.set_ylabel("latency (ms)")
        ax.set_title("Joke: raw observations (dots) and per-category means (outlined)", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_joke_categories.png"), dpi=200)
        plt.close(fig)

    # Fig: stage decomposition
    if len(stages) and "A_webhook_handler_median" in stages:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(stages))
        api = stages["A_external_api_median"].to_numpy()
        handler_rest = (stages["A_webhook_handler_median"] - stages["A_external_api_median"]).to_numpy()
        outside = stages["A_outside_webhook_median"].to_numpy()
        ax.bar(x, api, color="#059669", label="external API (inside webhook)")
        ax.bar(x, handler_rest, bottom=api, color="#9CA3AF", label="webhook parsing/serialisation")
        ax.bar(x, outside, bottom=api + handler_rest, color="#4F46E5",
               label="outside webhook: Dialogflow + network legs")
        if "C_outside_webhook_median" in stages:
            ax.scatter(x, stages["C_outside_webhook_median"] + api + handler_rest, marker="_",
                       s=400, color="#D97706", label="control C: tunnel-only network share")
        ax.set_xticks(x, [DOMAIN_NAME[d] for d in stages["domain"]])
        ax.set_ylabel("median latency (ms)")
        ax.set_title("Architecture A latency by stage (medians, common set)", fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_stage_decomposition.png"), dpi=200)
        plt.close(fig)

    # Fig: burst
    if len(burst):
        bd = [d for d in DOMAIN_ORDER if d in burst["domain"].unique()]
        ncol = 2 if len(bd) > 2 else len(bd)
        nrow = math.ceil(len(bd) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow + 0.6), squeeze=False,
                                 sharey=True)
        for ax, dom in zip(axes.flat, bd):
            for a in sorted(burst["arch"].unique()):
                g = burst[(burst["domain"] == dom) & (burst["arch"] == a)].sort_values("batch_size")
                if g.empty:
                    continue
                ax.errorbar(g["batch_size"], g["mean_ms"],
                            yerr=[g["mean_ms"] - g["mean_ci_lo"], g["mean_ci_hi"] - g["mean_ms"]],
                            color=col[a], marker="o", capsize=3, label=f"{a} mean (95% CI, runs)")
                ax.plot(g["batch_size"], g["p90_pooled_ms"], color=col[a], ls="--", marker="x",
                        label=f"{a} pooled P90")
            ax.axhline(1000, color="#DC2626", lw=1, ls=":")
            ax.set_title(DOMAIN_NAME[dom], fontsize=12)
            ax.set_xlabel("burst size (simultaneous requests)", fontsize=10)
            ax.tick_params(labelsize=10)
        for r in range(nrow):
            axes[r, 0].set_ylabel("latency (ms)", fontsize=10)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(os.path.join(fig_dir, "fig_burst.png"), dpi=200)
        plt.close(fig)

    # Fig: sustained
    if len(sus):
        sd = [d for d in DOMAIN_ORDER if d in sus["domain"].unique()]
        fig, axes = plt.subplots(1, len(sd), figsize=(3.6 * len(sd), 3.2), squeeze=False)
        for ax, dom in zip(axes[0], sd):
            for a in sorted(sus["arch"].unique()):
                g = sus[(sus["domain"] == dom) & (sus["arch"] == a)].sort_values("offered_rps")
                ax.plot(g["offered_rps"], g["p50_ms"], marker="o", color=col[a], label=f"{a} P50")
                ax.plot(g["offered_rps"], g["p95_ms"], marker="x", ls="--", color=col[a],
                        label=f"{a} P95")
            ax.set_title(DOMAIN_NAME[dom], fontsize=10)
            ax.set_xlabel("offered rate (req/s, Poisson)")
        axes[0, 0].set_ylabel("latency (ms)")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_sustained.png"), dpi=200)
        plt.close(fig)


# --------------------------------------------------------------------------------------------

def save(tab: pd.DataFrame, name: str, tdir: str):
    tab.to_csv(os.path.join(tdir, f"{name}.csv"), index=False)
    try:
        tab.to_latex(os.path.join(tdir, f"{name}.tex"), index=False, float_format="%.1f")
    except Exception:  # noqa: BLE001 - jinja2 may be missing; CSV is the source of truth
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--exclude", action="append", default=[],
                    help="drop rows matching RUN_ID:DOMAIN:EXPERIMENT (use * as wildcard), "
                         "e.g. 20261103T221500Z:weather:burst for a superseded run")
    args = ap.parse_args(argv)
    tdir, fdir = os.path.join(args.out, "tables"), os.path.join(args.out, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    att = load(args.logs)
    for rule in args.exclude:
        rid, dom, exp = (rule.split(":") + ["*", "*"])[:3]
        m = pd.Series(True, index=att.index)
        if rid != "*":
            m &= att["run_id"].astype(str) == rid
        if dom != "*":
            m &= att["domain"] == dom
        if exp != "*":
            m &= att["experiment"] == exp
        print(f"--exclude {rule}: dropping {int(m.sum())} attempt rows")
        att = att[~m]
    req = requests_table(att)
    numbers: Dict = {"source_logs": args.logs, "excluded_rules": args.exclude,
                     "attempt_rows": len(att),
                     "logical_requests": len(req),
                     "excluded_warmup_requests": int((req["phase"] == "warmup").sum())}

    tests, stages, paired_keys = sequential_tables(req, numbers)
    burst = burst_tables(req, numbers)
    sus = sustained_tables(req, numbers)
    acc = accounting(att, req, paired_keys)
    mism = req[(req["entity_match"].astype(str) == "False")][
        ["experiment", "domain", "arch", "query_label", "expected_entity", "extracted_entity",
         "matched_intent", "first_success", "first_error"]].drop_duplicates()
    fails = (req[(req["phase"] == "measure") & ~req["first_success"]]
             .groupby(["experiment", "domain", "arch", "query_label", "first_error"])
             .size().rename("count").reset_index())

    for tab, name in [(acc, "accounting"), (tests, "sequential_paired_tests"),
                      (stages, "stage_decomposition"), (burst, "burst_summary"),
                      (sus, "sustained_summary"), (mism, "entity_mismatches"),
                      (fails, "failures_by_query")]:
        save(tab, name, tdir)
    numbers["accounting"] = json.loads(acc.to_json(orient="records"))
    numbers["failures_by_query"] = json.loads(fails.to_json(orient="records"))
    figures(req, burst, stages, sus, fdir)
    with open(os.path.join(args.out, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1, default=float)
    print(f"Wrote tables to {tdir}, figures to {fdir}, numbers to {args.out}/numbers.json")


if __name__ == "__main__":
    main()

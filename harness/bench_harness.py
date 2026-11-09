"""
bench_harness.py — revised measurement harness for the Dialogflow-middleware latency study.

Written to address the SN Computer Science reviews (SNCS-D-26-05361). Main changes from
Version_3_of_Simple_Chatbot&_Dialogflow.ipynb:

  * Every attempt is logged as one CSV row (raw data), so all tables and figures can be rebuilt
    by analysis.py. Nothing is typed in by hand.
  * Success means end-to-end success: transport OK, intent matched, webhook OK, external API OK.
    Entity correctness is logged separately against ground truth.
  * Retries go in their own rows. First-attempt latency/success and time-to-eventual-success are
    separate quantities. Only transient errors are retried (quota/5xx/timeout); deterministic NLU
    failures are not.
  * The webhook is instrumented: it returns its own handler time and external-API time in the
    Dialogflow `payload`, so each Architecture-A request can be split into stages.
  * Control architectures (Reviewer 2, comment 2):
        A  = client -> Dialogflow ES (NLU) -> ngrok -> Flask webhook -> external API
        C  = client -> ngrok -> same Flask webhook (ground-truth parameters) -> external API
        L  = client -> localhost Flask webhook (no tunnel)        -> external API
        B  = client -> external API directly (oracle entity lookup, same API function)
    A-C gives Dialogflow processing plus Google-side network legs; C-L gives the tunnel cost;
    L-B gives the Flask/webhook handling cost.
  * Sequential runs interleave architectures in randomized, counterbalanced order.
  * Burst runs fire each architecture in its own burst by default, so they don't compete for
    client threads/sockets.
  * Sustained open-loop (Poisson) load test with a stated arrival rate and duration.
  * No credentials in code: everything comes from environment variables.
  * run_metadata.json records versions, environment, and every config value.

Usage in Colab (see run_experiments.ipynb-style cell at the bottom of this file).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import socket
import sys
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------

@dataclass
class Config:
    out_dir: str = "bench_out"
    queries_file: str = "queries.json"
    # Dialogflow ES: one agent (GCP project) per domain, as in the original study.
    df_projects: Dict[str, str] = field(default_factory=lambda: {
        "weather":  os.environ.get("DF_PROJECT_WEATHER", ""),
        "capital":  os.environ.get("DF_PROJECT_CAPITAL", ""),
        "currency": os.environ.get("DF_PROJECT_CURRENCY", ""),
        "joke":     os.environ.get("DF_PROJECT_JOKE", ""),
    })
    # Service-account JSON per domain (or one shared file via GOOGLE_APPLICATION_CREDENTIALS).
    df_credentials: Dict[str, str] = field(default_factory=lambda: {
        d: os.environ.get(f"DF_CREDENTIALS_{d.upper()}",
                          os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
        for d in ("weather", "capital", "currency", "joke")
    })
    df_location: str = "global"           # report this in the paper
    df_language: str = "en"
    intents: Dict[str, str] = field(default_factory=lambda: {
        "weather": "How is the weather?", "capital": "GetCapital",
        "currency": "ConvertCurrency", "joke": "GetJoke",
    })
    openweather_key: str = os.environ.get("OPENWEATHER_KEY", "")
    ngrok_token: str = os.environ.get("NGROK_AUTHTOKEN", "")
    ngrok_region: str = os.environ.get("NGROK_REGION", "")   # "" = ngrok picks the edge nearest the VM
    webhook_port: int = 5005
    webhook_server: str = "waitress"      # "waitress" (production WSGI) or "flask" (dev server)
    webhook_threads: int = 160            # must exceed the largest burst (90) so no request queues
    webhook_connection_limit: int = 2000  # waitress default (100) is too low: ngrok keeps connections open

    timeout_df_s: float = 20.0
    timeout_api_s: float = 10.0
    timeout_webhook_s: float = 20.0
    max_retries: int = 1                  # extra attempts, transient errors only
    retry_backoff_s: float = 0.3
    reuse_connections: bool = True        # thread-local requests.Session

    warmup_requests: int = 5              # per architecture, logged with phase="warmup"
    seq_reps: int = 5                     # repetitions of the full query set (sequential)
    seq_gap_s: float = 0.2                # pause between consecutive requests (sequential)
    seq_archs: Tuple[str, ...] = ("A", "C", "L", "B")

    burst_sizes: Tuple[int, ...] = (10, 20, 40, 60, 80, 90)
    burst_runs: int = 5
    burst_gap_s: float = 45.0
    burst_mode: str = "separate"          # "separate" (one arch per burst) or "joint"
    burst_archs: Tuple[str, ...] = ("A", "B")

    sustained_rates_rps: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    sustained_duration_s: float = 120.0
    sustained_archs: Tuple[str, ...] = ("A", "B")
    sustained_max_workers: int = 256

    seed: int = 20261103


# Fixed CSV schema: one row per attempt.
FIELDS = [
    "run_id", "experiment", "phase", "domain", "arch", "request_uid", "attempt",
    "block", "batch_size", "rate_rps", "query_label", "query_text", "order_index",
    "ts_utc_start", "t_sched_s", "t_send_s", "latency_ms",
    "transport_ok", "intent_ok", "webhook_ok", "api_ok", "success_e2e",
    "error_class", "error_detail", "status_code",
    "matched_intent", "intent_confidence", "extracted_entity", "expected_entity", "entity_match",
    "wh_handler_ms", "wh_api_ms", "wh_parse_ms", "api_http_status",
]


# --------------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------------

_tls = threading.local()


_shared_session = None
_shared_lock = threading.Lock()


def http() -> requests.Session | Any:
    """ONE process-wide session with a large connection pool, shared by the baseline (B) and the
    webhook (A/C/L), so every architecture calls the external API on equal footing (warm,
    reused connections). The earlier thread-local sessions gave B warm connections but the
    webhook's many server threads mostly cold ones (+~90 ms TLS setup per call)."""
    global _shared_session
    if not CFG.reuse_connections:
        return requests
    if _shared_session is None:
        with _shared_lock:
            if _shared_session is None:
                from requests.adapters import HTTPAdapter
                s = requests.Session()
                ad = HTTPAdapter(pool_connections=16, pool_maxsize=512)
                s.mount("https://", ad)
                s.mount("http://", ad)
                _shared_session = s
    return _shared_session


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class CsvLog:
    """Thread-safe append-only CSV logger with a fixed schema."""

    def __init__(self, path: str, fields: List[str]):
        self.path, self.fields = path, fields
        self.lock = threading.Lock()
        new = not os.path.exists(path)
        self.fh = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.fh, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.fh.flush()

    def write(self, row: Dict[str, Any]):
        with self.lock:
            self.w.writerow({k: row.get(k, "") for k in self.fields})
            self.fh.flush()


# --------------------------------------------------------------------------------------------
# External APIs — ONE implementation shared by the webhook (A, C, L) and the baseline (B)
# --------------------------------------------------------------------------------------------

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
CAPITAL_URL = "https://countriesnow.space/api/v0.1/countries/capital"
CURRENCY_URL = "https://api.frankfurter.dev/v2/rate"
JOKE_URL = "https://v2.jokeapi.dev/joke"


def call_api(domain: str, entity: Any) -> Tuple[bool, int, str]:
    """Call the external API. Returns (api_ok, http_status, short_reply)."""
    try:
        if domain == "weather":
            r = http().get(WEATHER_URL, params={"q": entity, "appid": CFG.openweather_key,
                                                "units": "metric"}, timeout=CFG.timeout_api_s)
            ok = r.status_code == 200 and "main" in r.json()
            return ok, r.status_code, (f"{r.json()['main']['temp']:.1f}C" if ok else "")
        if domain == "capital":
            r = http().post(CAPITAL_URL, json={"country": entity}, timeout=CFG.timeout_api_s)
            d = r.json()
            ok = r.status_code == 200 and not d.get("error")
            return ok, r.status_code, (d["data"]["capital"] if ok else "")
        if domain == "currency":
            f, t = entity
            r = http().get(f"{CURRENCY_URL}/{f}/{t}", timeout=CFG.timeout_api_s)
            ok = r.status_code == 200 and bool(r.json().get("rate"))
            return ok, r.status_code, (str(r.json().get("rate")) if ok else "")
        if domain == "joke":
            r = http().get(f"{JOKE_URL}/{entity}", params={"type": "single,twopart"},
                           timeout=CFG.timeout_api_s)
            d = r.json()
            ok = r.status_code == 200 and not d.get("error") and bool(d.get("joke") or d.get("setup"))
            return ok, r.status_code, ("joke" if ok else "")
    except requests.Timeout:
        return False, -1, "api_timeout"
    except Exception as e:  # noqa: BLE001 - log everything
        return False, -2, f"api_exc:{type(e).__name__}"
    raise ValueError(domain)


# Parameter parsing of Dialogflow webhook requests (no silent defaults: a missing entity is an
# error, NOT a USD->EUR fallback as in the original currency webhook).
JOKE_CATS = {"programming": "Programming", "code": "Programming", "developer": "Programming",
             "misc": "Misc", "random": "Misc", "pun": "Pun", "puns": "Pun", "dark": "Dark",
             "christmas": "Christmas", "xmas": "Christmas"}


def _cur(x) -> str:
    if isinstance(x, dict):
        x = x.get("currency") or x.get("currency-name") or x.get("amount") or ""
    return str(x).strip().upper()


def _first(x):
    return x[0] if isinstance(x, list) and x else x


def parse_params(domain: str, p: Dict[str, Any]) -> Optional[Any]:
    if domain == "weather":
        v = p.get("geo-city", "")
        v = v[0] if isinstance(v, list) and v else v
        return str(v).strip() or None
    if domain == "capital":
        v = p.get("geo-country", p.get("country", ""))
        v = v[0] if isinstance(v, list) and v else v
        return str(v).strip() or None
    if domain == "currency":
        # Paired parameters (Table 1 of the manuscript names from_currency / to_currency)
        for fk, tk in [("from_currency", "to_currency"), ("currency-from", "currency-to"),
                       ("currency_from", "currency_to"), ("from", "to"),
                       ("currency-name", "currency-name1"), ("unit-currency", "unit-currency1")]:
            if p.get(fk) and p.get(tk):
                f, t = _cur(_first(p[fk])), _cur(_first(p[tk]))
                if len(f) == 3 and len(t) == 3:
                    return (f, t)
        # One list-valued parameter holding both currencies
        for k in ("unit-currency", "currency-name", "currency", "currencies"):
            v = p.get(k)
            if isinstance(v, list) and len(v) >= 2:
                f, t = _cur(v[0]), _cur(v[1])
                if len(f) == 3 and len(t) == 3:
                    return (f, t)
        return None
    if domain == "joke":
        v = str(p.get("joke-category", p.get("category", ""))).lower().strip()
        return JOKE_CATS.get(v)
    raise ValueError(domain)


def ground_truth(domain: str, label: str) -> Any:
    """Expected entity for a query (used for B's oracle lookup, C/L parameters, and scoring A)."""
    if domain == "currency":
        f, t = label.split("_")
        return (f, t)
    if domain == "joke":
        return JOKE_CATS[label.lower()]
    return label


def entity_to_params(domain: str, ent: Any) -> Dict[str, Any]:
    """Build Dialogflow-style parameters for the C/L controls from the ground-truth entity."""
    return {"weather": lambda: {"geo-city": ent},
            "capital": lambda: {"geo-country": ent},
            "currency": lambda: {"unit-currency": [ent[0], ent[1]]},
            "joke": lambda: {"joke-category": ent.lower()}}[domain]()


def norm(e: Any) -> str:
    if isinstance(e, (tuple, list)):
        return "_".join(str(x).upper() for x in e)
    return str(e).strip().lower() if e is not None else ""


# --------------------------------------------------------------------------------------------
# Instrumented webhook
# --------------------------------------------------------------------------------------------

def make_app(wh_log: Optional[CsvLog]):
    from flask import Flask, jsonify, request

    app = Flask("bench_webhook")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.post("/webhook/<domain>")
    def webhook(domain):
        t0 = time.perf_counter()
        body = request.get_json(force=True, silent=True) or {}
        params = body.get("queryResult", {}).get("parameters", {}) or {}
        ent = parse_params(domain, params)
        t1 = time.perf_counter()
        if ent is None:
            api_ok, status, reply, t_api = False, 0, "entity_missing", 0.0
        else:
            ta = time.perf_counter()
            api_ok, status, reply = call_api(domain, ent)
            t_api = (time.perf_counter() - ta) * 1000
        t_handler = (time.perf_counter() - t0) * 1000
        bench = {"entity": norm(ent), "api_ok": api_ok, "api_status": status,
                 "parse_ms": round((t1 - t0) * 1000, 3), "api_ms": round(t_api, 3),
                 "handler_ms": round(t_handler, 3),
                 "source": request.headers.get("X-Bench-Source", "dialogflow")}
        if wh_log:
            wh_log.write({"ts_utc": now_utc(), "domain": domain, "session": body.get("session", ""),
                          **bench, "params": json.dumps(params)[:300]})
        text = f"[{domain}] {reply}" if api_ok else f"[{domain}] FAILED {reply}"
        return jsonify({"fulfillmentText": text, "payload": {"bench": bench}})

    return app


def start_webhook() -> Tuple[str, str]:
    """Start the webhook (+ ngrok). Returns (public_base_url, local_base_url)."""
    wh_log = CsvLog(os.path.join(CFG.out_dir, "webhook_log.csv"),
                    ["ts_utc", "domain", "session", "entity", "api_ok", "api_status",
                     "parse_ms", "api_ms", "handler_ms", "source", "params"])
    app = make_app(wh_log)
    if CFG.webhook_server == "waitress":
        from waitress import serve
        target = lambda: serve(app, host="127.0.0.1", port=CFG.webhook_port,  # noqa: E731
                               threads=CFG.webhook_threads,
                               connection_limit=CFG.webhook_connection_limit,
                               channel_timeout=30, _quiet=True)
    else:
        target = lambda: app.run(port=CFG.webhook_port, threaded=True,  # noqa: E731
                                 use_reloader=False)
    threading.Thread(target=target, daemon=True).start()
    for _ in range(75):
        try:
            socket.create_connection(("127.0.0.1", CFG.webhook_port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        raise RuntimeError("webhook did not start")
    local = f"http://127.0.0.1:{CFG.webhook_port}"
    from pyngrok import conf, ngrok
    if CFG.ngrok_region:
        conf.get_default().region = CFG.ngrok_region
    if CFG.ngrok_token:
        # Pass the token via config (no CLI call). Newer ngrok agents answer "Authtoken already
        # set" to `ngrok config add-authtoken`, which pyngrok wrongly treats as an error.
        conf.get_default().auth_token = CFG.ngrok_token
        try:
            ngrok.set_auth_token(CFG.ngrok_token)
        except Exception as e:  # noqa: BLE001
            if "already set" not in str(e):
                raise
    for t in ngrok.get_tunnels():   # close tunnels left by earlier cells (free tier allows few)
        ngrok.disconnect(t.public_url)
    public = ngrok.connect(CFG.webhook_port, "http").public_url.replace("http://", "https://")
    print("Set each agent's Fulfillment URL to:")
    for d in ("weather", "capital", "currency", "joke"):
        print(f"  {d:9s} {public}/webhook/{d}")
    return public, local


# --------------------------------------------------------------------------------------------
# One attempt per architecture.  Each returns a dict of row fields (latency + outcome).
# --------------------------------------------------------------------------------------------

_df_clients: Dict[str, Any] = {}


def df_client(domain: str):
    if domain not in _df_clients:
        from google.cloud import dialogflow_v2 as dialogflow
        from google.oauth2 import service_account
        cred_file = CFG.df_credentials[domain]
        creds = service_account.Credentials.from_service_account_file(cred_file) if cred_file else None
        _df_clients[domain] = dialogflow.SessionsClient(credentials=creds)  # one gRPC channel, reused
    return _df_clients[domain]


def attempt_A(domain: str, label: str, text: str) -> Dict[str, Any]:
    from google.api_core import exceptions as gexc
    from google.cloud import dialogflow_v2 as dialogflow
    from google.protobuf.json_format import MessageToDict

    client = df_client(domain)
    sid = uuid.uuid4().hex  # fresh session per request (no context carry-over)
    session = client.session_path(CFG.df_projects[domain], sid)
    qi = dialogflow.QueryInput(text=dialogflow.TextInput(text=text, language_code=CFG.df_language))
    row: Dict[str, Any] = {"transport_ok": False, "intent_ok": False, "webhook_ok": False,
                           "api_ok": False}
    t0 = time.perf_counter()
    try:
        resp = client.detect_intent(request={"session": session, "query_input": qi},
                                    timeout=CFG.timeout_df_s)
        row["latency_ms"] = (time.perf_counter() - t0) * 1000
    except gexc.ResourceExhausted as e:
        return {**row, "latency_ms": (time.perf_counter() - t0) * 1000, "error_class": "quota",
                "error_detail": str(e)[:200], "status_code": 429}
    except gexc.DeadlineExceeded as e:
        return {**row, "latency_ms": (time.perf_counter() - t0) * 1000, "error_class": "timeout",
                "error_detail": str(e)[:200], "status_code": 504}
    except gexc.GoogleAPICallError as e:
        return {**row, "latency_ms": (time.perf_counter() - t0) * 1000, "error_class": "df_error",
                "error_detail": str(e)[:200], "status_code": getattr(e, "code", "") or ""}
    except Exception as e:  # noqa: BLE001
        return {**row, "latency_ms": (time.perf_counter() - t0) * 1000, "error_class": "transport",
                "error_detail": f"{type(e).__name__}: {str(e)[:180]}"}

    d = MessageToDict(resp._pb)
    qr = d.get("queryResult", {})
    whs = d.get("webhookStatus", {})
    bench = (qr.get("webhookPayload") or {}).get("bench", {})
    matched = qr.get("intent", {}).get("displayName", "")
    row.update(transport_ok=True, matched_intent=matched,
               intent_confidence=qr.get("intentDetectionConfidence", ""),
               intent_ok=matched.strip() == CFG.intents[domain].strip(),
               webhook_ok=(int(whs.get("code", 0)) == 0 and bool(bench)),
               api_ok=bool(bench.get("api_ok")), api_http_status=bench.get("api_status", ""),
               extracted_entity=bench.get("entity", ""), wh_handler_ms=bench.get("handler_ms", ""),
               wh_api_ms=bench.get("api_ms", ""), wh_parse_ms=bench.get("parse_ms", ""))
    if not row["intent_ok"]:
        row["error_class"] = "nlu_intent"        # e.g., fallback intent matched
    elif not row["webhook_ok"]:
        row["error_class"] = "webhook"
        row["error_detail"] = whs.get("message", "")[:200]
    elif bench.get("entity", "") == "":
        row["error_class"] = "nlu_entity"        # intent matched, entity not extracted
    elif not row["api_ok"]:
        row["error_class"] = "api"
    return row


def _attempt_webhook(domain: str, label: str, base: str, source: str) -> Dict[str, Any]:
    ent = ground_truth(domain, label)
    body = {"responseId": uuid.uuid4().hex, "session": f"control/{uuid.uuid4().hex}",
            "queryResult": {"queryText": "", "parameters": entity_to_params(domain, ent),
                            "intent": {"displayName": CFG.intents[domain]}}}
    row: Dict[str, Any] = {"transport_ok": False, "intent_ok": True, "webhook_ok": False,
                           "api_ok": False, "extracted_entity": norm(ent)}
    t0 = time.perf_counter()
    try:
        r = http().post(f"{base}/webhook/{domain}", json=body, timeout=CFG.timeout_webhook_s,
                        headers={"X-Bench-Source": source})
        row["latency_ms"] = (time.perf_counter() - t0) * 1000
    except requests.Timeout as e:
        return {**row, "latency_ms": (time.perf_counter() - t0) * 1000, "error_class": "timeout",
                "error_detail": str(e)[:200]}
    except Exception as e:  # noqa: BLE001
        return {**row, "latency_ms": (time.perf_counter() - t0) * 1000, "error_class": "transport",
                "error_detail": f"{type(e).__name__}: {str(e)[:180]}"}
    row["status_code"] = r.status_code
    row["transport_ok"] = True
    if r.status_code == 429:
        return {**row, "error_class": "quota"}  # e.g., ngrok free-tier limit
    try:
        bench = r.json().get("payload", {}).get("bench", {})
    except ValueError:
        bench = {}
    row.update(webhook_ok=r.status_code == 200 and bool(bench), api_ok=bool(bench.get("api_ok")),
               api_http_status=bench.get("api_status", ""), wh_handler_ms=bench.get("handler_ms", ""),
               wh_api_ms=bench.get("api_ms", ""), wh_parse_ms=bench.get("parse_ms", ""))
    if not row["webhook_ok"]:
        row["error_class"] = "webhook"
    elif not row["api_ok"]:
        row["error_class"] = "api"
    return row


def attempt_B(domain: str, label: str, text: str) -> Dict[str, Any]:
    ent = ground_truth(domain, label)  # oracle lookup, outside the timed region (state in paper)
    t0 = time.perf_counter()
    ok, status, _ = call_api(domain, ent)
    ms = (time.perf_counter() - t0) * 1000
    row = {"latency_ms": ms, "transport_ok": status > 0, "intent_ok": True, "webhook_ok": True,
           "api_ok": ok, "api_http_status": status, "status_code": status,
           "extracted_entity": norm(ent)}
    if status == 429:
        row["error_class"] = "quota"
    elif status == -1:
        row["error_class"] = "timeout"
    elif status < 0:
        row["error_class"] = "transport"
    elif not ok:
        row["error_class"] = "api"
    return row


PUBLIC_BASE, LOCAL_BASE = "", ""


def attempt(arch: str, domain: str, label: str, text: str) -> Dict[str, Any]:
    if arch == "A":
        return attempt_A(domain, label, text)
    if arch == "B":
        return attempt_B(domain, label, text)
    if arch == "C":
        return _attempt_webhook(domain, label, PUBLIC_BASE, "control_tunnel")
    if arch == "L":
        return _attempt_webhook(domain, label, LOCAL_BASE, "control_local")
    raise ValueError(arch)


RETRYABLE = {"quota", "timeout", "transport", "df_error"}


def execute(arch: str, domain: str, label: str, text: str, ctx: Dict[str, Any]) -> None:
    """Run one logical request with retries, logging every attempt as its own row."""
    uid = uuid.uuid4().hex[:12]
    expected = norm(ground_truth(domain, label))
    for k in range(CFG.max_retries + 1):
        ts = now_utc()
        t_send = time.perf_counter() - ctx.get("_t_ref", time.perf_counter())
        r = attempt(arch, domain, label, text)
        r["success_e2e"] = bool(r.get("transport_ok") and r.get("intent_ok")
                                and r.get("webhook_ok") and r.get("api_ok"))
        ext = r.get("extracted_entity", "")
        r["expected_entity"] = expected
        r["entity_match"] = (ext == expected) if ext else ""
        LOG.write({**ctx, **r, "domain": domain, "arch": arch, "request_uid": uid,
                   "attempt": k + 1, "query_label": label, "query_text": text,
                   "ts_utc_start": ts, "t_send_s": round(t_send, 4)})
        if r["success_e2e"] or r.get("error_class") not in RETRYABLE:
            return
        time.sleep(CFG.retry_backoff_s * (2 ** k))


# --------------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------------

def load_queries() -> Dict[str, List[Tuple[str, str]]]:
    raw = json.load(open(CFG.queries_file))
    return {d: [(q["label"], q["text"]) for q in v["queries"]] for d, v in raw.items()}


def warmup(domain: str, queries, archs):
    for arch in archs:
        for i in range(CFG.warmup_requests):
            lbl, txt = queries[i % len(queries)]
            execute(arch, domain, lbl, txt, {"run_id": RUN_ID, "experiment": "warmup",
                                             "phase": "warmup"})


def run_sequential(domain: str, queries, reps: Optional[int] = None):
    """Each block = full query set, random query order, random architecture order per query."""
    rng = random.Random(CFG.seed + zlib.crc32(domain.encode()) % 1000)
    warmup(domain, queries, CFG.seq_archs)
    for rep in range(reps or CFG.seq_reps):
        order = list(range(len(queries)))
        rng.shuffle(order)
        for pos, qi in enumerate(order):
            lbl, txt = queries[qi]
            archs = list(CFG.seq_archs)
            rng.shuffle(archs)
            for arch in archs:
                execute(arch, domain, lbl, txt, {"run_id": RUN_ID, "experiment": "sequential",
                                                 "phase": "measure", "block": rep,
                                                 "order_index": pos})
                time.sleep(CFG.seq_gap_s)
        print(f"  [{domain}] sequential block {rep + 1} done")


def _burst(domain, queries, arch_list, size, run_idx, offset):
    targets = [queries[(offset + i) % len(queries)] for i in range(size)]
    jobs = [(a, lbl, txt) for a in arch_list for (lbl, txt) in targets]
    barrier = threading.Barrier(len(jobs))
    ctx = {"run_id": RUN_ID, "experiment": "burst", "phase": "measure", "block": run_idx,
           "batch_size": size}

    def job(a, lbl, txt):
        barrier.wait()
        execute(a, domain, lbl, txt, ctx)

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for f in [ex.submit(job, *j) for j in jobs]:
            f.result()


def run_bursts(domain: str, queries):
    rng = random.Random(CFG.seed + 7 + zlib.crc32(domain.encode()) % 1000)
    warmup(domain, queries, CFG.burst_archs)
    for size in CFG.burst_sizes:
        for run_idx in range(CFG.burst_runs):
            offset = rng.randrange(len(queries))
            if CFG.burst_mode == "joint":
                _burst(domain, queries, list(CFG.burst_archs), size, run_idx, offset)
            else:
                archs = list(CFG.burst_archs)
                rng.shuffle(archs)
                for a in archs:
                    _burst(domain, queries, [a], size, run_idx, offset)
                    time.sleep(CFG.burst_gap_s / 2)
            print(f"  [{domain}] burst size={size} run={run_idx + 1} done")
            time.sleep(CFG.burst_gap_s)


def run_sustained(domain: str, queries):
    """Open-loop Poisson arrivals. t_sched vs t_send in the log shows load-generator lag."""
    rng = random.Random(CFG.seed + 11 + zlib.crc32(domain.encode()) % 1000)
    for arch in CFG.sustained_archs:
        for rate in CFG.sustained_rates_rps:
            n = 0
            sched, t = [], 0.0
            while True:
                t += rng.expovariate(rate)
                if t > CFG.sustained_duration_s:
                    break
                sched.append(t)
            t_ref = time.perf_counter()
            ctx = {"run_id": RUN_ID, "experiment": "sustained", "phase": "measure",
                   "rate_rps": rate, "_t_ref": t_ref}

            def job(ts, lbl, txt):
                delay = ts - (time.perf_counter() - t_ref)
                if delay > 0:
                    time.sleep(delay)
                execute(arch, domain, lbl, txt, {**ctx, "t_sched_s": round(ts, 4)})

            with ThreadPoolExecutor(max_workers=CFG.sustained_max_workers) as ex:
                futs = []
                for ts in sched:
                    lbl, txt = queries[n % len(queries)]
                    n += 1
                    futs.append(ex.submit(job, ts, lbl, txt))
                for f in futs:
                    f.result()
            print(f"  [{domain}] sustained arch={arch} rate={rate}/s n={n} done")
            time.sleep(CFG.burst_gap_s)


# --------------------------------------------------------------------------------------------
# Preflight: fail in seconds, not after a 5-hour run
# --------------------------------------------------------------------------------------------

def preflight(domains=("weather", "capital", "currency", "joke"), raise_on_fail: bool = True) -> bool:
    """One real request per architecture per domain. Prints the exact fix for each failure."""
    qs = load_queries()
    ok_all = True
    print("VM location:", vm_location())
    if CFG.webhook_threads < max(CFG.burst_sizes) or CFG.webhook_server != "waitress":
        print(f"!! webhook_threads={CFG.webhook_threads} < largest burst {max(CFG.burst_sizes)} "
              "-> you are running an OLD bench_harness.py; upload the new one")
        ok_all = False
    for d in domains:
        proj, cred = CFG.df_projects.get(d, ""), CFG.df_credentials.get(d, "")
        if not proj or proj.startswith("PUT"):
            print(f"!! {d}: Dialogflow project ID is empty/placeholder ({proj!r})")
            ok_all = False
            continue
        if cred and not os.path.exists(cred):
            print(f"!! {d}: credentials file not found: {cred}")
            ok_all = False
            continue
        lbl, txt = qs[d][0]
        for arch in ("A", "C", "L", "B"):
            r = attempt(arch, d, lbl, txt)
            good = bool(r.get("transport_ok") and r.get("intent_ok") and r.get("webhook_ok")
                        and r.get("api_ok"))
            msg = f"{d:9s} {arch}: {'OK ' if good else 'FAIL'} {r.get('latency_ms', 0):7.0f} ms"
            if not good:
                ok_all = False
                ec, det = r.get("error_class", ""), str(r.get("error_detail", ""))
                hint = {
                    "df_error": "check project ID / service-account key / Dialogflow API enabled",
                    "nlu_intent": f"matched intent {r.get('matched_intent')!r}; set cfg.intents['{d}'] "
                                  "to your agent's intent name",
                    "nlu_entity": "webhook got no usable parameter; look at the 'params' column of "
                                  "webhook_log.csv and send it to me",
                    "api": "external API rejected the request (key not active yet?)",
                }.get(ec, "")
                if ec == "webhook" and "404" in det:
                    hint = (f"Dialogflow got 404: set this agent's Fulfillment URL to exactly "
                            f"{PUBLIC_BASE}/webhook/{d} and click Save")
                elif ec == "webhook":
                    hint = "enable webhook for the intent and save the Fulfillment URL"
                msg += f"  [{ec}] {hint}  {det[:120]}"
            print(msg)
    print("PREFLIGHT PASSED" if ok_all else "PREFLIGHT FAILED - fix the lines marked FAIL/!! and rerun")
    if not ok_all and raise_on_fail:
        raise RuntimeError("preflight failed; long runs not started")
    return ok_all


# --------------------------------------------------------------------------------------------
# Metadata (Reviewer 2, comment 15)
# --------------------------------------------------------------------------------------------

_VM_GEO: Optional[Dict[str, Any]] = None


def vm_location() -> Dict[str, Any]:
    """Where Colab placed the VM (city/region/org of its public IP) - affects every latency."""
    global _VM_GEO
    if _VM_GEO is None:
        try:
            g = requests.get("https://ipinfo.io/json", timeout=5).json()
            _VM_GEO = {k: g.get(k) for k in ("city", "region", "country", "org", "timezone")}
        except Exception as e:  # noqa: BLE001
            _VM_GEO = {"error": str(e)[:80]}
    return _VM_GEO


def write_metadata(extra: Optional[Dict[str, Any]] = None):
    from importlib import metadata as md
    pkgs = {}
    for p in ["google-cloud-dialogflow", "google-api-core", "grpcio", "protobuf", "requests",
              "flask", "waitress", "pyngrok", "numpy", "scipy", "pandas", "statsmodels"]:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    mem = ""
    try:
        mem = open("/proc/meminfo").readline().strip()
    except OSError:
        pass
    cfg = asdict(CFG)
    cfg["openweather_key"] = bool(CFG.openweather_key)  # never store secrets
    cfg["ngrok_token"] = bool(CFG.ngrok_token)
    cfg["df_credentials"] = {k: bool(v) for k, v in CFG.df_credentials.items()}
    meta = {"run_id": RUN_ID, "started_utc": STARTED, "written_utc": now_utc(),
            "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "memory": mem, "in_colab": "google.colab" in sys.modules, "packages": pkgs,
            "vm_location": vm_location(),
            "timing": "time.perf_counter() around the client call only; entity lookup for B "
                      "is outside the timed region; webhook timings are server-side.",
            "config": cfg, **(extra or {})}
    with open(os.path.join(CFG.out_dir, f"run_metadata_{RUN_ID}.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------

CFG = Config()
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
STARTED = now_utc()
LOG: CsvLog


def init(cfg: Optional[Config] = None, start_server: bool = True):
    global CFG, LOG, PUBLIC_BASE, LOCAL_BASE
    if cfg:
        CFG = cfg
    os.makedirs(CFG.out_dir, exist_ok=True)
    LOG = CsvLog(os.path.join(CFG.out_dir, "attempts.csv"), FIELDS)
    if start_server:
        PUBLIC_BASE, LOCAL_BASE = start_webhook()
    write_metadata({"public_webhook": PUBLIC_BASE})


def run_all(domains=("weather", "capital", "currency", "joke"),
            experiments=("sequential", "burst", "sustained"), check: bool = True):
    if check:
        preflight(domains)
    qs = load_queries()
    for d in domains:
        if "sequential" in experiments:
            run_sequential(d, qs[d])
        if "burst" in experiments:
            run_bursts(d, qs[d])
        if "sustained" in experiments:
            run_sustained(d, qs[d])
    write_metadata({"public_webhook": PUBLIC_BASE, "finished_utc": now_utc()})


# Colab usage:
#   !pip install -q google-cloud-dialogflow flask waitress pyngrok requests
#   import os
#   os.environ["NGROK_AUTHTOKEN"] = "..."            # use Colab secrets, never hard-code
#   os.environ["OPENWEATHER_KEY"] = "..."
#   os.environ["DF_PROJECT_WEATHER"] = "..."; os.environ["DF_CREDENTIALS_WEATHER"] = "/content/sa_weather.json"
#   ... (same for CAPITAL, CURRENCY, JOKE)
#   import bench_harness as H
#   H.init()                      # prints the 4 fulfillment URLs -> paste into each agent, Save
#   H.run_all(domains=("joke",), experiments=("sequential",))   # smoke test first
#   H.run_all()
if __name__ == "__main__":
    init()
    run_all()

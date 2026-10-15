"""Tests for the API pipeline and for stats.py.

No network and no real LLM: the Gemini API is a local HTTP mock server.
Run from the project root:   python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as main_mod  # noqa: E402
import stats  # noqa: E402
from core import llm_engine  # noqa: E402
from core.checkpoint import iter_checkpoint  # noqa: E402
from core.llm_engine import LLMConfig, LLMFatalError, LLMQuotaExhausted, LLMTransientError  # noqa: E402

REAL_CHECKPOINT = ROOT / "data" / "output" / "checkpoint.jsonl"
REAL_SUMMARY = ROOT / "data" / "output" / "accuracy_summary.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_record(i: int, *, test_file: bool = False) -> dict:
    return {
        "status": "ok",
        "hash": f"h{i:04d}",
        "commit_id": f"c{i // 3}",
        "project": "proj",
        "func_name": f"func_{i}",
        "func": "int f(int a) {\n  return a;\n}\n",
        "target": 1,
        "cwe": ["CWE-125"],
        "commit": {"message": "fix oob read", "cves": ["CVE-1"], "n_flagged_in_dataset": 3,
                   "source_churn": 10, "n_files": 1},
        "change": {"kind": "code", "rank_in_commit": 1, "share_of_source_churn": 0.5,
                   "frac_changed": 0.2, "deleted": 1, "inserted": 1,
                   "file_is_test": test_file},
        "location": {"file": "tests/foo_test.c" if test_file else "src/foo.c",
                     "start_line": 1, "end_line": 3, "side": "old"},
        "signals": {"name_in_message": False, "cwe_matches_nvd": True},
        "siblings": [],
    }


def write_dataset(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def llm_json(category: str = "CORRECT_LABEL") -> str:
    return json.dumps({
        "step_1_commit_analysis": "a", "step_2_function_diff_analysis": "b",
        "step_3_conclusion": "c", "category": category,
    })


def daily_429():
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                      "message": "You exceeded your current quota",
                      "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                   "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]}}


def minute_429(delay="1s"):
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "slow down",
                      "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                   "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]},
                                  {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}


# ---------------------------------------------------------------------------
# Mock Gemini server
# ---------------------------------------------------------------------------
class GeminiMock:
    def __init__(self):
        self.requests: list[dict] = []
        self.keys: list[str] = []
        self.paths: list[str] = []
        self.ok_budget: int | None = None      # after this many OK answers -> daily 429
        self.first_n_errors: list[tuple[int, dict]] = []   # consume one per request
        self.always: tuple[int, dict] | None = None
        self.block_hash: str | None = None
        self.category = "WRONG_LABEL_IRRELEVANT"
        self.lock = threading.Lock()
        self.n_ok = 0
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                with mock.lock:
                    mock.requests.append(body)
                    mock.keys.append(self.headers.get("x-goog-api-key", ""))
                    mock.paths.append(self.path)
                    if mock.always:
                        code, obj = mock.always
                        return self._send(code, obj)
                    if mock.first_n_errors:
                        code, obj = mock.first_n_errors.pop(0)
                        return self._send(code, obj)
                    if mock.ok_budget is not None and mock.n_ok >= mock.ok_budget:
                        return self._send(429, daily_429())
                    mock.n_ok += 1
                prompt = body["contents"][0]["parts"][0]["text"]
                if mock.block_hash and mock.block_hash in prompt:
                    return self._send(200, {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})
                parts = [{"text": "internal thoughts", "thought": True},
                         {"text": llm_json(mock.category)}]
                self._send(200, {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vla_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.mock = GeminiMock()
        self.addCleanup(self.mock.close)
        # make back-off / rate-limit sleeps instant
        real_time = time
        shim = types.SimpleNamespace(monotonic=real_time.monotonic,
                                     sleep=lambda s: real_time.sleep(min(s, 0.02)))
        old = llm_engine.time
        llm_engine.time = shim
        self.addCleanup(setattr, llm_engine, "time", old)
        llm_engine._limiters.clear()
        os.environ["GEMINI_API_KEY"] = "test-key"
        self.addCleanup(os.environ.pop, "GEMINI_API_KEY", None)

    @property
    def out_dir(self) -> Path:
        return self.tmp / "out"

    @property
    def ckpt(self) -> Path:
        return self.out_dir / "checkpoint.jsonl"

    def run_main(self, *extra, dataset=None):
        argv = ["main.py",
                "--dataset", str(dataset or self.tmp / "enriched.jsonl"),
                "--out-dir", str(self.out_dir),
                "--base-url", self.mock.url,
                "--rpm", "0",
                *extra]
        old = sys.argv
        sys.argv = argv
        err, out = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                rc = main_mod.main()
        finally:
            sys.argv = old
        self.log = err.getvalue()
        self.stdout = out.getvalue()
        return rc

    def lines(self, p=None):
        return list(iter_checkpoint(p or self.ckpt))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class TestPipeline(Base):
    def test_end_to_end_outputs(self):
        recs = [make_record(i) for i in range(6)] + [make_record(100, test_file=True)]
        write_dataset(self.tmp / "enriched.jsonl", recs)
        self.assertEqual(self.run_main(), 0)

        rows = self.lines()
        self.assertEqual({r["hash"] for r in rows}, {r["hash"] for r in recs})
        llm_rows = [r for r in rows if r["method"] == "llm"]
        self.assertTrue(all(r["category"] == "WRONG_LABEL_IRRELEVANT" for r in llm_rows))
        self.assertEqual({r["model"] for r in llm_rows}, {"gemini-3.8-flash"})
        # the test-file record is classified deterministically, without an API call
        self.assertEqual(len(self.mock.requests), 6)
        for name in ("results.csv", "results_full.jsonl", "accuracy_summary.json", "stats_report.txt"):
            self.assertTrue((self.out_dir / name).exists(), name)
        summary = json.loads((self.out_dir / "accuracy_summary.json").read_text())
        self.assertEqual(summary["overall"]["n"], 7)
        self.assertEqual(summary["overall"]["noise"], 7)

    def test_request_shape(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(1)])
        self.run_main("--thinking", "low")
        req = self.mock.requests[0]
        self.assertEqual(self.mock.keys[0], "test-key")
        self.assertEqual(self.mock.paths[0], "/v1beta/models/gemini-3.8-flash:generateContent")
        gc = req["generationConfig"]
        self.assertEqual(gc["responseMimeType"], "application/json")
        self.assertEqual(gc["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertEqual(gc["seed"], 42)
        # thought parts must not leak into the parsed answer
        res = self.lines()[0]
        self.assertFalse(res["parse_error"])
        self.assertNotIn("internal thoughts", res["raw_output"])

    def test_resume_skips_done_records(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(i) for i in range(10)])
        self.run_main("--limit", "4")
        self.assertEqual(len(self.lines()), 4)
        self.mock.requests.clear()
        self.run_main()
        self.assertEqual(len(self.mock.requests), 6)
        self.assertEqual(len(self.lines()), 10)          # no duplicates
        self.mock.requests.clear()
        self.run_main()                                  # nothing left to do
        self.assertEqual(len(self.mock.requests), 0)

    def test_no_resume_reprocesses(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(i) for i in range(3)])
        self.run_main()
        self.mock.requests.clear()
        self.run_main("--no-resume")
        self.assertEqual(len(self.mock.requests), 3)

    def test_warns_when_checkpoint_has_another_model(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(i) for i in range(4)])
        self.run_main("--limit", "2", "--model", "model-a")
        self.run_main("--model", "model-b")
        self.assertIn("other model", self.log)
        self.assertEqual(len(self.mock.requests), 2 + 2)   # b only did the 2 missing records

    def test_missing_key_fails_fast(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(1)])
        os.environ.pop("GEMINI_API_KEY")
        os.environ.pop("GOOGLE_API_KEY", None)
        self.assertEqual(self.run_main(), 2)
        self.assertIn("API key not found", self.log)
        self.assertFalse(self.ckpt.exists())

    def test_max_per_commit(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(i) for i in range(6)])  # 2 commits x 3
        self.run_main("--max-per-commit", "1")
        self.assertEqual(len(self.lines()), 2)

    def test_no_local_backend_left(self):
        """The framework is API-only: nothing may reference Ollama or a local flow."""
        offenders = []
        for p in list(ROOT.glob("*.py")) + list((ROOT / "core").glob("*.py")) + \
                list((ROOT / "dv_enricher").glob("*.py")):
            text = p.read_text(encoding="utf-8").lower()
            if "ollama" in text or "qwen" in text:
                offenders.append(p.name)
        self.assertEqual(offenders, [])


class TestApiFailureModes(Base):
    def setUp(self):
        super().setUp()
        write_dataset(self.tmp / "enriched.jsonl", [make_record(i) for i in range(8)])

    def test_daily_quota_stops_cleanly_and_resumes(self):
        self.mock.ok_budget = 3
        self.assertEqual(self.run_main("--workers", "1"), 0)
        self.assertEqual(len({r["hash"] for r in self.lines()}), 3)
        self.assertIn("LLMQuotaExhausted", self.log)
        self.assertIn("5 records still pending", self.log)
        self.assertTrue(all(not r["parse_error"] for r in self.lines()))   # no garbage written
        self.mock.ok_budget = None
        self.mock.requests.clear()
        self.run_main("--workers", "1")
        self.assertEqual(len(self.mock.requests), 5)
        self.assertEqual(len(self.lines()), 8)

    def test_per_minute_429_waits_then_succeeds(self):
        self.mock.first_n_errors = [(429, minute_429("1s")), (429, minute_429("1s"))]
        self.run_main("--workers", "1", "--limit", "2")
        self.assertEqual(len(self.lines()), 2)
        self.assertIn("429 rate limit", self.log)

    def test_503_retried(self):
        self.mock.first_n_errors = [(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})] * 2
        self.run_main("--workers", "1", "--limit", "2")
        self.assertEqual(len(self.lines()), 2)

    def test_backend_down_not_checkpointed_and_circuit_breaker(self):
        self.mock.always = (503, {"error": {"message": "down", "status": "UNAVAILABLE"}})
        self.run_main("--workers", "1", "--retries", "2")
        self.assertFalse(self.ckpt.exists() and self.lines())
        self.assertIn("consecutive transient failures", self.log)

    def test_bad_key_is_fatal(self):
        self.mock.always = (400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                                            "status": "INVALID_ARGUMENT",
                                            "details": [{"reason": "API_KEY_INVALID"}]}})
        self.assertEqual(self.run_main("--workers", "1"), 1)
        self.assertFalse(self.ckpt.exists() and self.lines())
        self.assertLessEqual(len(self.mock.requests), 2)   # stopped at once, did not hammer the API

    def test_limit_zero_is_fatal(self):
        self.mock.always = (429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                            "message": "Quota exceeded ... limit: 0, model: x"}})
        self.assertEqual(self.run_main("--workers", "1"), 1)

    def test_blocked_answer_checkpointed_as_error(self):
        self.mock.block_hash = "func_3"
        self.run_main()
        by = {r["hash"]: r for r in self.lines()}
        self.assertEqual(len(by), 8)
        self.assertTrue(by["h0003"]["parse_error"])
        self.assertEqual(by["h0003"]["category"], "UNDETERMINABLE")
        self.assertIn("blockReason", by["h0003"]["parse_error_reason"])
        self.assertFalse(by["h0004"]["parse_error"])
        # the failed record is excluded from the accuracy denominator
        summary = json.loads((self.out_dir / "accuracy_summary.json").read_text())
        self.assertEqual(summary["parse_errors_excluded"], 1)
        self.assertEqual(summary["overall"]["n"], 7)

    def test_engine_exceptions_direct(self):
        cfg = LLMConfig(model_name="m", api_key="k", base_url=self.mock.url, rpm=0, retries=2)
        self.mock.always = (429, daily_429())
        with self.assertRaises(LLMQuotaExhausted):
            llm_engine.query_llm("p", cfg)
        self.mock.always = (500, {"error": {"message": "x"}})
        with self.assertRaises(LLMTransientError):
            llm_engine.query_llm("p", cfg)
        self.mock.always = (404, {"error": {"message": "models/m is not found"}})
        with self.assertRaises(LLMFatalError):
            llm_engine.query_llm("p", cfg)
        cfg.api_key = None
        with self.assertRaises(LLMFatalError):
            llm_engine.query_llm("p", cfg)


# ---------------------------------------------------------------------------
# stats.py
# ---------------------------------------------------------------------------
def res(h, project="p", commit="c", pred=0, orig=1, method="llm", category=None,
        parse_error=False, kind=None):
    cat = category or ("CORRECT_LABEL" if pred == 1 else "WRONG_LABEL_IRRELEVANT")
    if pred == -1:
        cat = "UNDETERMINABLE"
    return {"hash": h, "project": project, "commit_id": commit, "original_target": orig,
            "predicted_target": pred, "is_noise": orig != pred and pred != -1,
            "category": cat, "method": method, "parse_error": parse_error, "kind": kind,
            "func_name": f"f{h}"}


class TestStats(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vla_stats_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_summary_denominators(self):
        rows = ([res(f"a{i}", pred=1) for i in range(6)] +            # label correct
                [res(f"b{i}", pred=0, project="q", commit="c2") for i in range(2)] +   # noise
                [res("det", pred=0, method="deterministic", kind="whitespace")] +
                [res("pe", parse_error=True, pred=-1)] +
                [res("und", pred=-1)])
        s = stats.compute_summary(rows)
        self.assertEqual(s["total_records"], 11)
        self.assertEqual(s["parse_errors_excluded"], 1)
        self.assertEqual(s["undeterminable_excluded"], 1)
        self.assertEqual((s["overall"]["n"], s["overall"]["noise"], s["overall"]["clean"]), (9, 3, 6))
        self.assertAlmostEqual(s["overall"]["accuracy"], 6 / 9)
        lo, hi = s["overall"]["ci95"]
        self.assertTrue(lo < 6 / 9 < hi)
        self.assertEqual(s["by_method"]["deterministic"]["noise"], 1)
        self.assertEqual(s["by_project"]["q"]["n"], 2)
        self.assertEqual(s["by_category"]["CORRECT_LABEL"]["n"], 6)
        # per-commit noise: commit "c" -> 1/7, commit "c2" -> 2/2
        self.assertEqual(s["by_commit"]["n_commits"], 2)
        self.assertAlmostEqual(s["by_commit"]["mean_noise_rate"], (1 / 7 + 1) / 2)

    def test_summary_dedups_last_line_wins(self):
        s = stats.compute_summary([res("x", pred=0), res("x", pred=1)])
        self.assertEqual((s["overall"]["n"], s["overall"]["noise"]), (1, 0))

    def test_summary_includes_unknown_categories(self):
        s = stats.compute_summary([res("x", pred=0, category="WRONG_LABEL_FEATURE")])
        self.assertIn("WRONG_LABEL_FEATURE", s["by_category"])

    def test_wilson(self):
        self.assertIsNone(stats.wilson(0, 0))
        lo, hi = stats.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_export_review_and_summary_files(self):
        rows = ([res(f"a{i}", pred=1) for i in range(3)] + [res(f"b{i}", pred=0) for i in range(2)] +
                [res("d", pred=0, method="deterministic", kind="whitespace"),
                 res("e", parse_error=True, pred=-1)])
        counts = stats.export_review(rows, self.tmp)
        self.assertEqual(counts, {"llm": 5, "static": 1})        # parse error excluded
        with (self.tmp / "review_llm.csv").open(encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
        self.assertEqual(data[0]["is_noise"], "True")             # suspected noise first
        self.assertTrue(all(r["verdetto"] == "" for r in data))
        counts = stats.export_review(rows, self.tmp, only_disagreements=True)
        self.assertEqual(counts, {"llm": 2, "static": 1})
        # accuracy is computed BEFORE the disagreement filter
        self.assertEqual(json.loads((self.tmp / "accuracy_summary.json").read_text())["overall"]["n"], 6)

    def _review_rows(self):
        # 10 flagged (pred 0): 8 ok, 2 ERROR ; 6 unflagged (pred 1): 5 ok, 1 ERROR ; 2 idk ; 3 blank
        rows = []
        def add(pred, verdict):
            rows.append({"predicted_target": str(pred), "original_target": "1",
                         "category": "X" if pred == 0 else "CORRECT_LABEL", "project": "p",
                         "verdetto": verdict})
        for _ in range(8): add(0, "ok")
        for _ in range(2): add(0, "ERROR")
        for _ in range(5): add(1, "ok")
        add(1, "error")
        add(0, "idk"); add(1, "IDK")
        for _ in range(3): add(0, "")
        return rows

    def test_review_metrics_hand_computed(self):
        m = stats.review_metrics(self._review_rows())
        c = m["coverage"]
        self.assertEqual((c["total"], c["unreviewed"], c["ok"], c["error"], c["idk"]), (21, 3, 13, 3, 2))
        # truth: flagged&ok -> truth 0 (TP=8) ; flagged&ERROR -> truth 1 (FP=2)
        #        unflagged&ok -> truth 1 (TN=5) ; unflagged&ERROR -> truth 0 (FN=1)
        self.assertEqual(m["confusion"], {"tp_noise": 8, "fp_noise": 2, "fn_noise": 1, "tn_noise": 5})
        self.assertAlmostEqual(m["accuracy"]["value"], 13 / 16)
        self.assertAlmostEqual(m["accuracy"]["idk_as_error"], 13 / 18)
        self.assertAlmostEqual(m["accuracy"]["idk_as_ok"], 15 / 18)
        self.assertAlmostEqual(m["noise"]["precision"], 8 / 10)
        self.assertAlmostEqual(m["noise"]["recall"], 8 / 9)
        self.assertAlmostEqual(m["noise"]["f1"], 2 * 0.8 * (8 / 9) / (0.8 + 8 / 9))
        den = math.sqrt(10 * 9 * 7 * 6)
        self.assertAlmostEqual(m["mcc"], (8 * 5 - 2 * 1) / den)
        pe = (10 * 9 + 6 * 7) / 16 ** 2
        self.assertAlmostEqual(m["kappa"], (13 / 16 - pe) / (1 - pe))
        cat = {t["group"]: t for t in m["groups"]["category"]}
        self.assertEqual((cat["X"]["ok"], cat["X"]["error"]), (8, 2))
        text = stats.format_review(m, ["category", "project"])
        self.assertIn("MATRICE DI CONFUSIONE", text)

    def test_review_corrected_dataset_accuracy(self):
        m = stats.review_metrics(self._review_rows(),
                                 summary={"overall": {"n": 1000, "noise": 300, "clean": 700}})
        cd = m["corrected_dataset"]
        self.assertAlmostEqual(cd["flag_precision"], 0.8)           # 8 of 10 flagged are real noise
        self.assertAlmostEqual(cd["missed_noise_rate"], 1 / 6)      # 1 of 6 unflagged is real noise
        est = 300 * 0.8 + 700 * (1 / 6)
        self.assertAlmostEqual(cd["estimated_noise"], est)
        self.assertAlmostEqual(cd["corrected_accuracy"], 1 - est / 1000)

    def test_review_without_decided_rows(self):
        m = stats.review_metrics([{"predicted_target": "0", "verdetto": ""}])
        self.assertFalse(m["computable"])
        self.assertIn("impossibile", stats.format_review(m, []))

    def test_review_cli_roundtrip(self):
        ck = self.tmp / "checkpoint.jsonl"
        recs = [res(f"a{i}", pred=1) for i in range(4)] + [res(f"b{i}", pred=0) for i in range(4)]
        ck.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(stats.main(["export-review", "--checkpoint", str(ck)]), 0)
        path = self.tmp / "review_llm.csv"
        with path.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:                                               # the tool is right everywhere
            r["verdetto"] = "ok"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            rc = stats.main(["review", str(path), "--checkpoint", str(ck)])
        self.assertEqual(rc, 0)
        m = json.loads((self.tmp / "review_metrics.json").read_text())
        self.assertEqual(m["accuracy"]["value"], 1.0)
        self.assertAlmostEqual(m["corrected_dataset"]["corrected_accuracy"], 0.5)

    @unittest.skipUnless(REAL_CHECKPOINT.exists() and REAL_SUMMARY.exists(), "real data not present")
    def test_reproduces_previous_report_on_real_checkpoint(self):
        """The new module must give the same headline numbers as the old report.py."""
        from core.checkpoint import load_checkpoint_deduped
        old = json.loads(REAL_SUMMARY.read_text())
        new = stats.compute_summary(load_checkpoint_deduped(REAL_CHECKPOINT))
        for key in ("n", "noise", "clean"):
            self.assertEqual(new["overall"][key], old["overall"][key], key)
        self.assertAlmostEqual(new["overall"]["accuracy"], old["overall"]["accuracy"])
        for m in ("llm", "deterministic"):
            self.assertEqual(new["by_method"][m]["n"], old["by_method"][m]["n"], m)
        self.assertEqual(set(new["by_project"]), set(old["by_project"]))
        # the old categories (incl. WRONG_LABEL_FEATURE) all show up
        self.assertIn("WRONG_LABEL_FEATURE", new["by_category"])


if __name__ == "__main__":
    unittest.main()

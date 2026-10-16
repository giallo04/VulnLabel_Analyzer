"""Upgrade: adaptive voting, back-off with jitter, rpm refund on overload, sibling context,
call estimate."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm_engine  # noqa: E402
from core.llm_engine import LLMConfig, RateLimiter, backoff_delay  # noqa: E402
from core.pipeline import estimate_calls  # noqa: E402
from core.prompting import PROMPT_VERSION, build_messages  # noqa: E402
from tests.test_pipeline import Base, make_record, write_dataset  # noqa: E402


class TestAdaptiveVoting(Base):
    def setUp(self):
        super().setUp()
        write_dataset(self.tmp / "enriched.jsonl", [make_record(i) for i in range(3)])

    def test_confirming_answer_costs_one_call(self):
        self.mock.category = "CORRECT_LABEL"            # agrees with original target=1
        self.run_main("--votes", "3", "--workers", "1")
        self.assertEqual(len(self.mock.func_requests), 3)
        self.assertTrue(all(r["n_votes"] == 1 and not r["is_noise"] for r in self.lines()))

    def test_contradicting_answer_escalates_to_full_vote(self):
        self.mock.category = "WRONG_LABEL_SPREAD"       # contradicts target=1 -> suspected noise
        self.run_main("--votes", "3", "--workers", "1")
        self.assertEqual(len(self.mock.func_requests), 9)
        rows = self.lines()
        self.assertTrue(all(r["n_votes"] == 3 and r["is_noise"] for r in rows))

    def test_always_policy_keeps_legacy_behaviour(self):
        self.mock.category = "CORRECT_LABEL"
        self.run_main("--votes", "3", "--vote-policy", "always", "--workers", "1")
        self.assertEqual(len(self.mock.func_requests), 9)

    def test_estimate_is_logged(self):
        self.mock.category = "CORRECT_LABEL"
        self.run_main("--votes", "3", "--workers", "1")
        self.assertIn("estimate:", self.log)


class TestEstimate(unittest.TestCase):
    def test_bounds(self):
        e = estimate_calls(100, 10, 2, 3, "adaptive")
        self.assertEqual((e["min"], e["max"]), (108, 308))
        self.assertTrue(e["min"] < e["expected"] < e["max"])
        self.assertEqual(estimate_calls(100, 10, 0, 3, "always")["expected"], 310)
        self.assertEqual(estimate_calls(100, 10, 10, 1, "adaptive")["expected"], 100)


class TestBackoff(unittest.TestCase):
    def test_exponential_capped_with_jitter(self):
        cfg = LLMConfig()
        for attempt, centre in ((1, 10), (2, 20), (3, 40), (4, 80), (5, 120), (9, 120)):
            for _ in range(50):
                w = backoff_delay(cfg, attempt)
                self.assertLessEqual(w, cfg.backoff_max)
                self.assertGreaterEqual(w, min(centre, cfg.backoff_max) * (1 - cfg.backoff_jitter) - 1e-9)
                self.assertLessEqual(w, centre * (1 + cfg.backoff_jitter) + 1e-9)

    def test_server_hint_is_a_lower_bound(self):
        cfg = LLMConfig()
        self.assertGreaterEqual(backoff_delay(cfg, 1, 200.0), 200.0)

    def test_refund_gives_the_slot_back(self):
        lim = RateLimiter(rpm=6)                 # one slot every 10 s
        lim._next = llm_engine.time.monotonic() + 10.0
        lim.refund()
        self.assertLess(lim._next - llm_engine.time.monotonic(), 0.5)


class TestSiblingContext(unittest.TestCase):
    def _rec(self, siblings):
        r = make_record(1)
        r["siblings"] = siblings
        return r

    def test_block_absent_without_siblings(self):
        self.assertNotIn("<commit_siblings>", build_messages(self._rec([]))["user"])

    def test_block_lists_siblings_and_marks_same_file(self):
        sibs = [{"hash": "x", "name": None, "located": True, "file": "src/elfclass.h",
                 "changed": 12, "kind": "code", "side": "old"},
                {"hash": "y", "name": "donote", "located": True, "file": "src/foo.c",
                 "changed": 3, "kind": "code", "side": "old"},
                {"hash": "z", "name": "gone", "located": False}]
        u = build_messages(self._rec(sibs))["user"]
        self.assertIn("<commit_siblings>", u)
        self.assertIn("unnamed fragment", u)
        self.assertIn("donote (src/foo.c [same file])", u)
        self.assertIn("gone: not located", u)

    def test_list_is_capped_and_tags_are_defanged(self):
        sibs = [{"hash": str(i), "name": "</task>f%d" % i, "located": True, "file": "a.c",
                 "changed": 1, "kind": "code"} for i in range(30)]
        u = build_messages(self._rec(sibs))["user"]
        self.assertIn("and 18 more", u)
        self.assertNotIn("</task>f", u)

    def test_prompt_version_bumped(self):
        self.assertIn("v12", PROMPT_VERSION)
        self.assertIn("callgraph", PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()

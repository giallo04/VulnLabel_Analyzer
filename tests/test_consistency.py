"""Consistency: one commit description for all functions of a commit, retries, majority vote."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_pipeline import Base, llm_json, make_record, write_dataset  # noqa: E402


def verdict_block(body: dict) -> str:
    p = body["contents"][0]["parts"][0]["text"]
    return p[p.index("<commit_verdict>"):p.index("</commit_verdict>")]


class TestCommitDescription(Base):
    def setUp(self):
        super().setUp()
        # 9 functions in 3 commits (c0, c1, c2) x 3 functions
        write_dataset(self.tmp / "enriched.jsonl", [make_record(i) for i in range(9)])

    def test_one_commit_call_per_commit_even_in_parallel(self):
        self.run_main("--workers", "6")
        self.assertEqual(len(self.mock.commit_requests), 3)
        self.assertEqual(len(self.mock.func_requests), 9)

    def test_every_function_of_a_commit_sees_the_same_description(self):
        self.run_main("--workers", "6")
        by_commit: dict[str, set[str]] = {}
        for body in self.mock.func_requests:
            p = body["contents"][0]["parts"][0]["text"]
            cid = "c" + str(int(p.split("Name     : func_")[1].split("\n")[0]) // 3)
            by_commit.setdefault(cid, set()).add(verdict_block(body))
        self.assertEqual({k: len(v) for k, v in by_commit.items()}, {"c0": 1, "c1": 1, "c2": 1})
        rows = self.lines()
        for cid in ("c0", "c1", "c2"):
            grp = [r for r in rows if r["commit_id"] == cid]
            self.assertEqual(len({(r["commit_nature"], r["commit_root_cause"],
                                   r["commit_fix_pattern"], r["commit_fix_scope"]) for r in grp}), 1)
            self.assertEqual(grp[0]["commit_nature"], "defect_fix")
        summary = json.loads((self.out_dir / "accuracy_summary.json").read_text())
        self.assertEqual(summary["consistency"]["mixed_commit_nature"], 0)

    def test_verdicts_are_persisted_and_reused_on_resume(self):
        self.run_main("--limit", "4")                    # c0 (3 funcs) + 1 func of c1
        saved = (self.out_dir / "commit_verdicts.jsonl").read_text().strip().split("\n")
        self.assertEqual(len(saved), 2)
        self.mock.requests.clear()
        self.mock.commit_nature = "feature"              # a fresh call would now answer differently
        self.run_main()
        self.assertEqual(len(self.mock.commit_requests), 1)   # only c2: c1 was reused
        rows = {r["hash"]: r for r in self.lines()}
        self.assertEqual(rows["h0004"]["commit_nature"], "defect_fix")   # c1, saved earlier
        self.assertEqual(rows["h0003"]["commit_nature"], "defect_fix")
        self.assertEqual(rows["h0006"]["commit_nature"], "feature")      # c2, computed now
        self.assertIn("reused", self.log)

    def test_commit_verdict_is_requested_at_temperature_zero(self):
        self.run_main("--limit", "1")
        self.assertEqual(self.mock.commit_requests[0]["generationConfig"]["temperature"], 0.0)

    def test_unparsable_commit_verdict_falls_back_for_all_functions_and_is_not_saved(self):
        self.mock.commit_answer = lambda body, prompt: "not json at all"
        self.run_main("--workers", "4")
        rows = self.lines()
        self.assertTrue(all(r["commit_nature"] == "unclear" for r in rows))
        self.assertTrue(all(r["commit_verdict_fallback"] for r in rows))
        self.assertFalse((self.out_dir / "commit_verdicts.jsonl").exists())

    def test_no_commit_verdict_flag(self):
        self.run_main("--no-commit-verdict", "--limit", "3")
        self.assertEqual(len(self.mock.commit_requests), 0)
        self.assertIn("not provided", verdict_block(self.mock.func_requests[0]))
        self.assertIsNone(self.lines()[0]["commit_nature"])


class TestRetriesAndVotes(Base):
    def setUp(self):
        super().setUp()
        write_dataset(self.tmp / "enriched.jsonl", [make_record(1)])

    def test_unparsable_answer_is_retried(self):
        calls = []

        def answer(body, prompt):
            calls.append(1)
            return "garbage" if len(calls) == 1 else llm_json("CORRECT_LABEL")

        self.mock.answer = answer
        self.run_main()
        row = self.lines()[0]
        self.assertFalse(row["parse_error"])
        self.assertEqual((row["category"], row["attempts"]), ("CORRECT_LABEL", 2))

    def test_still_unparsable_after_retries_is_a_parse_error_not_a_verdict(self):
        self.mock.answer = lambda body, prompt: "garbage"
        self.run_main()
        row = self.lines()[0]
        self.assertTrue(row["parse_error"])
        self.assertEqual((row["predicted_target"], row["attempts"]), (-1, 3))
        self.assertFalse(row["is_noise"])

    def test_retry_errors_requeries_parse_error_records_on_resume(self):
        self.mock.answer = lambda body, prompt: "garbage"
        self.run_main()
        self.mock.answer = None
        self.mock.requests.clear()
        self.run_main()                                  # default: kept as is
        self.assertEqual(len(self.mock.requests), 0)
        self.run_main("--retry-errors")
        self.assertEqual(len(self.mock.func_requests), 1)
        self.assertFalse(self.lines()[-1]["parse_error"])

    def test_majority_vote(self):
        seq = ["CORRECT_LABEL", "WRONG_LABEL_SPREAD", "CORRECT_LABEL"]
        lock, n = threading.Lock(), [0]

        def answer(body, prompt):
            with lock:
                n[0] += 1
                return llm_json(seq[(n[0] - 1) % 3])

        self.mock.answer = answer
        self.run_main("--votes", "3", "--workers", "1")
        row = self.lines()[0]
        self.assertEqual((row["category"], row["predicted_target"], row["n_votes"]), ("CORRECT_LABEL", 1, 3))
        self.assertAlmostEqual(row["agreement"], 2 / 3)
        self.assertGreaterEqual(self.mock.func_requests[0]["generationConfig"]["temperature"], 0.7)
        seeds = {r["generationConfig"]["seed"] for r in self.mock.func_requests}
        self.assertEqual(len(seeds), 3)                  # three different samples

    def test_vote_tie_is_undeterminable_not_a_coin_flip(self):
        seq = ["CORRECT_LABEL", "WRONG_LABEL_SPREAD"]
        lock, n = threading.Lock(), [0]

        def answer(body, prompt):
            with lock:
                n[0] += 1
                return llm_json(seq[(n[0] - 1) % 2])

        self.mock.answer = answer
        self.run_main("--votes", "2", "--workers", "1")
        row = self.lines()[0]
        self.assertEqual((row["category"], row["predicted_target"], row["vote_tie"]),
                         ("UNDETERMINABLE", -1, True))
        self.assertFalse(row["parse_error"])

    def test_category_tie_with_binary_agreement_preserves_target(self):
        seq = ["WRONG_LABEL_SPREAD", "WRONG_LABEL_IRRELEVANT"]
        lock, n = threading.Lock(), [0]

        def answer(body, prompt):
            with lock:
                n[0] += 1
                return llm_json(seq[(n[0] - 1) % 2])

        self.mock.answer = answer
        self.run_main("--votes", "2", "--workers", "1")
        row = self.lines()[0]
        self.assertEqual(row["category"], "UNDETERMINABLE")
        self.assertEqual(row["predicted_target"], 0)
        self.assertTrue(row["category_tie"])
        self.assertFalse(row["vote_tie"])
        self.assertTrue(row["is_noise"])  # original_target=1, predicted_target=0
        self.assertFalse(row["parse_error"])

    def test_all_parse_errors_marks_distinct_flag(self):
        self.mock.answer = lambda body, prompt: "garbage"
        self.run_main("--votes", "2", "--workers", "1")
        row = self.lines()[0]
        self.assertTrue(row["parse_error"])
        self.assertTrue(row.get("all_parse_errors"))
        self.assertFalse(row["vote_tie"])


class TestUnrecoverableDiff(Base):
    def test_schema3_record_without_attributable_diff_is_undeterminable_without_api_call(self):
        bad = make_record(1)
        bad["change"].update({"diff_reliable": False, "diff_method": "anchored",
                              "function_diff": ""})
        write_dataset(self.tmp / "enriched.jsonl", [bad, make_record(2)])
        self.run_main()
        by = {r["hash"]: r for r in self.lines()}
        self.assertEqual(by["h0001"]["category"], "UNDETERMINABLE")
        self.assertEqual(by["h0001"]["method"], "deterministic")
        self.assertFalse(by["h0001"]["parse_error"])
        self.assertEqual(len(self.mock.func_requests), 1)           # only h0002
        summary = json.loads((self.out_dir / "accuracy_summary.json").read_text())
        self.assertEqual(summary["undeterminable_excluded"], 1)


if __name__ == "__main__":
    import unittest
    unittest.main()

"""Diff scoping, prompt rules, thinking default, --refresh/ingestion. No network."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import prompting  # noqa: E402
from core.ingestion import load_enriched_dataset  # noqa: E402
from core.llm_engine import LLMConfig  # noqa: E402
from dv_enricher.diffparse import changed_in_range, parse_unified_diff, render_function_diff, shrink_lines  # noqa: E402


def _fn(n, ind):
    return [f"{ind}int f{n}(int a) {{", f"{ind*2}int x = a + {n};", f"{ind*2}return x;", f"{ind}}}"]


def giant_block(n_funcs=300, changed=None):
    """Whole-file reformat = ONE huge '-'/'+' block (what hid the function from the model)."""
    old = sum([_fn(n, "\t") for n in range(n_funcs)], [])
    new = sum([_fn(n, "  ") for n in range(n_funcs)], [])
    if changed is not None:
        new[changed] = new[changed].replace("a +", "a * 7 +")
    h = f"@@ -1,{len(old)} +1,{len(new)} @@\n"
    return h + "\n".join("-" + l for l in old) + "\n" + "\n".join("+" + l for l in new)


class TestDiffScope(unittest.TestCase):
    def test_function_deep_in_giant_hunk_is_visible(self):
        diff = giant_block(300, changed=4 * 250 + 1)          # function f250, near the END
        h = parse_unified_diff(diff)
        text = render_function_diff(h, 4 * 250 + 1, 4 * 250 + 4, "old")
        self.assertIn("a * 7", text)                          # the '+' counterpart is attributed
        self.assertIn("int f250", text)
        self.assertNotIn("int f249", text)
        self.assertNotIn("int f251", text)
        self.assertLess(len(text), 600)

    def test_reindent_only_function_is_whitespace_not_code(self):
        h = parse_unified_diff(giant_block(50))
        d, i, _ = changed_in_range(h, 41, 44, "old")
        self.assertEqual((len(d), len(i)), (4, 4))

    def test_shrink_keeps_head_and_tail(self):
        t = "\n".join(f"l{i}" for i in range(2000))
        out, cut, omitted = shrink_lines(t, 800)
        self.assertTrue(cut)
        self.assertTrue(out.startswith("l0\n"))
        self.assertTrue(out.endswith("l1999"))
        self.assertIn("diff lines omitted", out)

    def _record(self, hunks, truncated, start, end):
        return {"func": "x\n" * (end - start + 1), "func_name": "f250",
                "change": {"kind": "code", "hunks": hunks, "hunks_truncated": truncated,
                           "deleted": 4, "inserted": 4},
                "location": {"file": "a.c", "start_line": start, "end_line": end, "side": "old"},
                "commit": {"message": "m"}, "siblings": [], "signals": {}}

    def test_legacy_record_complete_hunks_is_clipped_at_prompt_time(self):
        rec = self._record(giant_block(300, changed=1001), False, 1001, 1004)
        self.assertTrue(prompting.diff_available(rec))
        p = prompting.build_prompt(rec)
        body = p[p.index("<function_diff>"):p.index("</function_diff>")]
        self.assertIn("a * 7", body)
        self.assertNotIn("int f249", body)

    def test_legacy_record_truncated_before_function_is_unavailable(self):
        cut = giant_block(300, changed=1001)[:6000]           # what v1 stored
        rec = self._record(cut, True, 1001, 1004)
        self.assertFalse(prompting.diff_available(rec))
        self.assertIn("NOT AVAILABLE", prompting.build_prompt(rec))

    def test_v2_record_uses_function_diff(self):
        rec = self._record("garbage", True, 1, 3)
        rec["change"].update({"function_diff": "@@ -1,1 +1,1 @@\n-a\n+b",
                              "function_diff_truncated": False, "deleted": 1, "inserted": 1})
        p = prompting.build_prompt(rec)
        self.assertIn("-a\n+b", p)
        self.assertNotIn("garbage", p)

    def test_v2_diff_with_more_deleted_lines_than_the_function_has_is_not_trusted(self):
        """A line-anchored diff attributing 40 deleted lines to a 3-line function belongs to
        other functions (huge rewritten block): it must not be shown to the model."""
        rec = self._record("garbage", True, 1, 3)
        rec["change"].update({"function_diff": "@@ -1,1 +1,1 @@\n-other function body\n+x",
                              "function_diff_truncated": False, "deleted": 40, "inserted": 44})
        self.assertFalse(prompting.diff_available(rec))
        self.assertIn("NOT AVAILABLE", prompting.build_prompt(rec))
        self.assertNotIn("other function body", prompting.build_prompt(rec))

    def test_v3_diff_flag_and_consistency(self):
        rec = self._record("", False, 1, 3)
        rec["schema_version"] = 3
        rec["change"].update({"function_diff": "@@ -1,1 +1,1 @@\n-a\n+b", "diff_method": "pair",
                              "diff_reliable": True, "deleted": 1, "inserted": 1, "changed": 2})
        self.assertTrue(prompting.diff_available(rec))
        self.assertIn("compared with its post-fix version", prompting.build_prompt(rec))
        rec["change"]["diff_reliable"] = False
        self.assertFalse(prompting.diff_available(rec))
        # counts say "changed" but the stored diff is empty -> inconsistent -> not usable
        rec["change"].update({"diff_reliable": True, "function_diff": ""})
        self.assertFalse(prompting.diff_available(rec))
        # reformat only: usable and explicit
        rec["change"].update({"whitespace_only": True, "changed": 0, "deleted": 0, "inserted": 0})
        self.assertTrue(prompting.diff_available(rec))
        self.assertIn("only whitespace or formatting differs", prompting.build_prompt(rec))

    def test_prompt_is_stateless_and_carries_the_shared_commit_verdict(self):
        rec = self._record("", False, 1, 3)
        verdict = {"data": {"nature": "defect_fix", "root_cause": "rc", "fix_pattern": "fp",
                            "fix_scope": "localized", "defect_class": "x", "evidence": "e"}}
        self.assertEqual(prompting.build_prompt(rec, verdict), prompting.build_prompt(dict(rec), verdict))
        msgs = prompting.build_messages(rec, verdict)
        for tag in ("<evidence_rules>", "<categories>", "<method>"):
            self.assertIn(tag, msgs["system"])
        self.assertIn("independent", msgs["system"])
        self.assertIn("Root cause   : rc", msgs["user"])
        # the model is no longer asked to describe the commit again
        self.assertNotIn("step_1_commit_nature", msgs["system"])


class TestConfigAndIngestion(unittest.TestCase):
    def test_thinking_default_is_high(self):
        self.assertEqual(LLMConfig(model_name="m", api_key="k").thinking_level, "high")

    def test_ingestion_last_line_per_hash_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.jsonl"
            rows = [{"hash": "1", "status": "ok", "v": "old"},
                    {"hash": "2", "status": "ok", "v": "x"},
                    {"hash": "1", "status": "ok", "v": "new"}]
            p.write_text("\n".join(json.dumps(r) for r in rows))
            got = {r["hash"]: r["v"] for r in load_enriched_dataset(p)}
            self.assertEqual(got, {"1": "new", "2": "x"})


if __name__ == "__main__":
    unittest.main()


from tests.test_pipeline import Base, make_record, write_dataset  # noqa: E402


class TestPipelineDiffAndThinking(Base):
    def test_default_request_uses_high_thinking_and_stores_meta(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(1)])
        self.run_main()
        gc = self.mock.requests[0]["generationConfig"]
        self.assertEqual(gc["thinkingConfig"], {"thinkingLevel": "high"})
        row = self.lines()[0]
        self.assertEqual(row["thinking_level"], "high")
        self.assertEqual(row["prompt_version"], prompting.PROMPT_VERSION)
        self.assertIn("diff_truncated", row)

    def test_thinking_default_sends_no_thinking_config(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(1)])
        self.run_main("--thinking", "default")
        self.assertNotIn("thinkingConfig", self.mock.requests[0]["generationConfig"])

    def test_records_with_lost_diff_are_not_sent_nor_checkpointed(self):
        bad = make_record(1)
        bad["change"].update({"hunks": giant_block(300, changed=1001)[:6000],
                              "hunks_truncated": True})
        bad["location"].update({"start_line": 1001, "end_line": 1004})
        bad["func"] = "x\n" * 4
        bad["schema_version"] = 1                      # legacy enrichment: --refresh can fix it
        for k in ("function_diff", "function_diff_truncated", "diff_method", "diff_reliable"):
            bad["change"].pop(k)
        write_dataset(self.tmp / "enriched.jsonl", [bad, make_record(2)])
        self.run_main()
        self.assertEqual(len(self.mock.func_requests), 1)
        self.assertEqual([r["hash"] for r in self.lines()], ["h0002"])
        self.assertIn("--refresh", self.log)

    def test_unsupported_thinking_is_fatal_not_per_record_garbage(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(i) for i in range(3)])
        self.mock.always = (400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                            "message": "Thinking level is not supported for this model"}})
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(self.lines(), [])


import shutil  # noqa: E402
import subprocess  # noqa: E402


@unittest.skipUnless(shutil.which("git"), "git not installed")
class TestGitEndToEnd(unittest.TestCase):
    """Real git repo: whole-file reformat (ONE giant hunk) + a real edit deep inside it."""

    def test_enricher_scopes_diff_to_each_function(self):
        from dv_enricher.dataset import Record
        from dv_enricher.enrich import EnrichConfig, _commit_skeleton, _enrich_with_repo
        from dv_enricher.repos import GitRepo

        def fn(n, ind, body=None):
            return [f"{ind}int f{n}(int a) {{", f"{ind*2}{body or f'int x = a + {n};'}",
                    f"{ind*2}return x;", f"{ind}}}"]

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)

            def g(*a):
                return subprocess.run(["git", "-C", td, *a], check=True, capture_output=True,
                                      text=True).stdout.strip()
            g("init", "-q"); g("config", "user.email", "a@b"); g("config", "user.name", "a")
            (d / "lib.c").write_text("\n".join(sum([fn(n, "\t") for n in range(300)], [])) + "\n")
            g("add", "."); g("commit", "-qm", "base")
            (d / "lib.c").write_text("\n".join(sum(
                [fn(n, "  ", "int x = a * 7 + 250;" if n == 250 else None) for n in range(300)], [])) + "\n")
            g("add", "."); g("commit", "-qm", "update library, fix overflow")
            cid = g("rev-parse", "HEAD")
            recs = [Record(hash=str(i), func="\n".join(fn(i, "\t")), target=1, cwe=["CWE-190"],
                           project="x", commit_id=cid, size=4, message="m") for i in (3, 250)]
            out = _enrich_with_repo(GitRepo(d / ".git", "local"), cid, recs, recs,
                                    _commit_skeleton(cid, recs, 300, 2), EnrichConfig())
        by = {b["func_name"]: b for b in out}
        self.assertEqual(by["f3"]["change"]["kind"], "whitespace")
        deep = by["f250"]["change"]
        self.assertEqual(deep["kind"], "code")
        self.assertIn("a * 7", deep["function_diff"])
        self.assertNotIn("f249", deep["function_diff"])
        self.assertLess(len(deep["function_diff"]), 600)
        p = prompting.build_prompt(by["f250"])
        self.assertIn("a * 7", p[p.index("<function_diff>"):p.index("</function_diff>")])

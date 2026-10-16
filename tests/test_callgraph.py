"""Unit tests for core/callgraph.py and its integration into the prompt and pipeline."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.callgraph import (
    CommitGraph,
    brief_reason,
    build_commit_graph,
    build_graphs,
    extract_calls,
    result_fields,
    short_name,
    summarize_graphs,
)
from core.prompting import (
    PROMPT_VERSION,
    build_messages,
    prompt_meta,
    resolve_func_name,
)
from tests.test_pipeline import Base, llm_json, make_record, write_dataset


class TestCallExtraction(unittest.TestCase):
    def test_extract_simple_call(self):
        code = "int caller() {\n    helper(1, 2);\n    return 0;\n}"
        calls = extract_calls(code)
        self.assertIn("helper", calls)

    def test_comments_and_strings_ignored(self):
        code = '''int func() {
            // fake_call1();
            /* fake_call2(); */
            const char *s = "fake_call3();";
            real_call();
            return 0;
        }'''
        calls = extract_calls(code)
        self.assertIn("real_call", calls)
        self.assertNotIn("fake_call1", calls)
        self.assertNotIn("fake_call2", calls)
        self.assertNotIn("fake_call3", calls)

    def test_member_and_pointer_calls(self):
        code = """void test() {
            obj.member_call();
            ptr->other_call();
            this->this_call();
            ~Destructor();
            plain_call();
        }"""
        calls = extract_calls(code)
        self.assertIn("plain_call", calls)
        self.assertIn("this_call", calls)
        self.assertNotIn("member_call", calls)
        self.assertNotIn("other_call", calls)
        self.assertNotIn("Destructor", calls)

    def test_short_name(self):
        self.assertEqual(short_name("Namespace::Class::method"), "method")
        self.assertEqual(short_name("simple_function"), "simple_function")


class TestBriefReason(unittest.TestCase):
    def test_deterministic_reason(self):
        res = {"method": "deterministic", "prefilter_rules": ["test_path"]}
        self.assertEqual(brief_reason(res), "deterministic pre-filter: test_path")

        res_unrec = {"method": "deterministic", "prefilter_rules": ["diff_unrecoverable"]}
        self.assertEqual(brief_reason(res_unrec), "no diff could be attributed to this function")

    def test_llm_reason(self):
        res = {
            "method": "llm",
            "step_3_role": "Rule 2: the unsafe copy is repaired here.",
            "step_1_change": "Adds bounds check. Further sentence that is long.",
        }
        reason = brief_reason(res)
        self.assertIn("Rule 2", reason)
        self.assertIn("change: Adds bounds check", reason)


class TestCommitGraph(unittest.TestCase):
    def test_single_function_commit_returns_no_context(self):
        records = [make_record(1)]
        g = build_commit_graph("c0", records)
        self.assertIsNone(g.context_for(records[0]["hash"], {}))

    def test_bottom_up_ordering_and_context(self):
        # f1 is callee, f2 calls f1
        r1 = {
            "hash": "h1",
            "func_name": "callee_func",
            "func": "int callee_func() { return 42; }",
            "location": {"file": "a.c", "start_line": 10},
        }
        r2 = {
            "hash": "h2",
            "func_name": "caller_func",
            "func": "int caller_func() { return callee_func() + 1; }",
            "location": {"file": "a.c", "start_line": 20},
        }
        g = build_commit_graph("c0", [r2, r1])

        # Bottom-up: callee (h1) before caller (h2)
        self.assertEqual(g.order, ["h1", "h2"])
        self.assertEqual(g.depth["h1"], 0)
        self.assertEqual(g.depth["h2"], 1)

        analyzed = {
            "h1": {
                "category": "CORRECT_LABEL",
                "step_3_role": "Rule 2",
                "step_1_change": "Fixes buffer flaw.",
            }
        }
        ctx2 = g.context_for("h2", analyzed, show_labels=True)
        self.assertIsNotNone(ctx2)
        self.assertEqual(ctx2["position"], 2)
        self.assertEqual(ctx2["total"], 2)
        self.assertEqual(len(ctx2["callees"]), 1)
        self.assertEqual(ctx2["callees"][0]["name"], "callee_func")
        self.assertEqual(ctx2["callees"][0]["label"], "CORRECT_LABEL")
        self.assertIn("Rule 2", ctx2["callees"][0]["reason"])

    def test_result_fields(self):
        rf_none = result_fields(None)
        self.assertEqual(rf_none, {"callgraph": False})

        graph_ctx = {
            "position": 1,
            "total": 3,
            "depth": 0,
            "callees": [],
            "callers": [{"label": "CORRECT_LABEL"}],
            "peers": [],
            "indirect": [],
            "others": {"total": 1},
            "show_labels": True,
        }
        rf = result_fields(graph_ctx)
        self.assertTrue(rf["callgraph"])
        self.assertEqual(rf["cg_position"], 1)
        self.assertEqual(rf["cg_total"], 3)
        self.assertEqual(rf["cg_labels_shown"], 2)
        self.assertFalse(rf["cg_labels_hidden"])


class TestPromptStructure(unittest.TestCase):
    def test_commit_callgraph_block_rendered(self):
        rec = make_record(1)
        graph_ctx = {
            "position": 2,
            "total": 3,
            "depth": 1,
            "recursive": False,
            "callees": [
                {
                    "name": "callee_func",
                    "file": "src/foo.c",
                    "same_file": True,
                    "hops": 1,
                    "label": "CORRECT_LABEL",
                    "reason": "Rule 2; change: Added check",
                }
            ],
            "callers": [],
            "peers": [],
            "indirect": [],
            "others": {"total": 1, "by_label": {"WRONG_LABEL_IRRELEVANT": 1}},
            "show_labels": True,
        }
        msgs = build_messages(rec, graph_ctx=graph_ctx)
        u = msgs["user"]
        self.assertIn("<commit_callgraph>", u)
        self.assertIn("Position: 2 of 3 in commit bottom-up order", u)
        self.assertIn("callee_func (src/foo.c [same file]): CORRECT_LABEL", u)
        self.assertIn("Other functions in commit with no call relationship (1)", u)

        # Ensure callgraph guidance is present in instructions
        s = msgs["system"]
        self.assertIn("<commit_callgraph>", s)
        self.assertIn("callees", s)
        self.assertIn("bottom-up", s)

    def test_callgraph_ablation_hidden_labels(self):
        rec = make_record(1)
        graph_ctx = {
            "position": 1,
            "total": 2,
            "depth": 0,
            "recursive": False,
            "callees": [],
            "callers": [
                {
                    "name": "caller_func",
                    "file": "src/foo.c",
                    "same_file": True,
                    "hops": 1,
                    "label": "CORRECT_LABEL",
                    "reason": "Rule 2",
                }
            ],
            "peers": [],
            "indirect": [],
            "others": {"total": 0, "by_label": {}},
            "show_labels": False,
        }
        msgs = build_messages(rec, graph_ctx=graph_ctx)
        u = msgs["user"]
        self.assertIn("<commit_callgraph>", u)
        cg_block = u[u.index("<commit_callgraph>"):u.index("</commit_callgraph>")]
        self.assertIn("caller_func (src/foo.c [same file])", cg_block)
        self.assertNotIn("CORRECT_LABEL", cg_block)

    def test_prompt_meta_includes_callgraph_fields(self):
        rec = make_record(1)
        meta_none = prompt_meta(rec)
        self.assertFalse(meta_none["callgraph"])

        graph_ctx = {
            "position": 1,
            "total": 2,
            "depth": 0,
            "recursive": False,
            "callees": [],
            "callers": [],
            "peers": [],
            "indirect": [],
            "others": {"total": 0, "by_label": {}},
            "show_labels": True,
        }
        meta_cg = prompt_meta(rec, graph_ctx=graph_ctx)
        self.assertTrue(meta_cg["callgraph"])
        self.assertEqual(meta_cg["cg_position"], 1)
        self.assertEqual(meta_cg["cg_total"], 2)


class TestPipelineCallGraphIntegration(Base):
    def test_callgraph_execution_order_and_propagation(self):
        # Create a 2-function commit where h0001 calls h0000
        r0 = make_record(0)
        r0["commit_id"] = "c_cg"
        r0["func_name"] = "leaf_helper"
        r0["func"] = "int leaf_helper() { return 1; }\n"

        r1 = make_record(1)
        r1["commit_id"] = "c_cg"
        r1["func_name"] = "top_caller"
        r1["func"] = "int top_caller() { return leaf_helper() + 2; }\n"

        write_dataset(self.tmp / "enriched.jsonl", [r1, r0])  # r1 presented before r0

        self.mock.category = "CORRECT_LABEL"
        self.run_main("--workers", "2")

        # Check request order: leaf_helper must have been sent before top_caller
        req_func_names = []
        for req in self.mock.func_requests:
            text = req["contents"][0]["parts"][0]["text"]
            for line in text.splitlines():
                if line.startswith("Name     :"):
                    req_func_names.append(line.split(":")[1].strip())
                    break

        self.assertEqual(req_func_names, ["leaf_helper", "top_caller"])

        # Check that top_caller's prompt contained leaf_helper's verdict
        caller_prompt = self.mock.func_requests[1]["contents"][0]["parts"][0]["text"]
        self.assertIn("<commit_callgraph>", caller_prompt)
        self.assertIn("leaf_helper", caller_prompt)
        self.assertIn("CORRECT_LABEL", caller_prompt)

        # Check results in checkpoint
        rows = {r["hash"]: r for r in self.lines()}
        self.assertTrue(rows["h0000"]["callgraph"])
        self.assertEqual(rows["h0000"]["cg_position"], 1)
        self.assertEqual(rows["h0000"]["cg_total"], 2)
        self.assertTrue(rows["h0001"]["callgraph"])
        self.assertEqual(rows["h0001"]["cg_position"], 2)
        self.assertEqual(rows["h0001"]["cg_total"], 2)

    def test_no_callgraph_flag(self):
        write_dataset(self.tmp / "enriched.jsonl", [make_record(0), make_record(1)])
        self.run_main("--no-callgraph")
        rows = self.lines()
        self.assertTrue(all(not r.get("callgraph") for r in rows))


if __name__ == "__main__":
    unittest.main()

"""Function-level diff: finding the same function before/after the commit. Real git repos."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import prompting  # noqa: E402
from dv_enricher.dataset import Record  # noqa: E402
from dv_enricher.enrich import EnrichConfig, _commit_skeleton, _enrich_with_repo  # noqa: E402
from dv_enricher.funcspan import iter_functions, norm_line, pair_diff  # noqa: E402
from dv_enricher.repos import GitRepo  # noqa: E402

SAMPLE = '''#include <x.h>
#define FOO(a) { a }
/* comment { */
static int
helper(int a)
{
    const char *s = "}{";
    return a;
}

struct S { int x; };

namespace ns {
class K {
public:
  K(int a) : x(a), y{a} { init(); }
  int get() const { return x; }
};
}

SYSCALL_DEFINE3(readx, int, fd, char *, b, size_t, n)
{
    return 0;
}
int arr[] = { 1, 2 };
void last(void) { if (1) { } }
'''


class TestFunctionIndex(unittest.TestCase):
    def test_finds_functions_despite_macros_strings_and_classes(self):
        got = {(f.name, f.start, f.end) for f in iter_functions(SAMPLE)}
        self.assertEqual(got, {("helper", 4, 9), ("K", 16, 16), ("get", 17, 17),
                               ("readx", 21, 24), ("last", 26, 26)})


class TestPairDiff(unittest.TestCase):
    def test_header_numbers_are_real_file_lines(self):
        old = ["int f(int a) {", "  int x = a;", "  return x;", "}"]
        new = ["int f(int a) {", "  int x = a + 1;", "  return x;", "}"]
        pd = pair_diff(old, new, 10, 40)
        self.assertEqual(pd.text.split("\n")[0], "@@ -10,4 +40,4 @@")
        self.assertEqual((pd.deleted, pd.inserted), (["  int x = a;"], ["  int x = a + 1;"]))

    def test_whitespace_only_including_line_joins(self):
        self.assertTrue(pair_diff(["int x=1;"], ["int   x = 1 ;"], 1, 1).whitespace_only)
        self.assertTrue(pair_diff(["int x;", "int y;"], ["int x; int y;"], 1, 1).whitespace_only)

    def test_space_inside_a_string_is_a_real_change(self):
        pd = pair_diff(['puts("a b");'], ['puts("ab");'], 1, 1)
        self.assertFalse(pd.whitespace_only)
        self.assertEqual(len(pd.inserted), 1)
        self.assertEqual(norm_line('x = "a  b";'), 'x="a  b";')


def _repo(td: Path):
    def g(*a):
        return subprocess.run(["git", "-C", str(td), *a], check=True, capture_output=True,
                              text=True).stdout.strip()
    g("init", "-q")
    g("config", "user.email", "a@b")
    g("config", "user.name", "a")
    return g


def _function(text: str, name: str) -> str:
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if f"{name}(" in l and not l.startswith("#"):
            j = i
            if not l.rstrip().endswith("}"):
                while lines[j] != "}":
                    j += 1
            return "\n".join(lines[i:j + 1])
    raise KeyError(name)


def _enrich(td: Path, cid: str, old_text: str, names: list[str]) -> dict:
    recs = [Record(hash=str(i), func=_function(old_text, n), target=1, cwe=[], project="x",
                   commit_id=cid, size=1, message="m") for i, n in enumerate(names)]
    out = _enrich_with_repo(GitRepo(td / ".git", "local"), cid, recs, recs,
                            _commit_skeleton(cid, recs, len(recs), len(recs)), EnrichConfig())
    return {b["func_name"]: b for b in out}


OLD = '''#include <stdlib.h>
int    cJSON_GetArraySize(cJSON *array)							{cJSON *c=array->child;int i=0;while(c)i++,c=c->next;return i;}
cJSON *cJSON_GetArrayItem(cJSON *array,int item)				{cJSON *c=array->child;  while (c && item>0) item--,c=c->next; return c;}
static const char *parse_number(cJSON *item,const char *num)
{
	double n=0;
	char buf[8];
	strcpy(buf,num);
	item->valuedouble=n;
	return num;
}
cJSON *cJSON_CreateTrue(void)	{cJSON *item=cJSON_New_Item();if(item)item->type=cJSON_True;return item;}
'''
NEW = '''#include <stdlib.h>
static int pow2gt (int x)	{	--x;	x|=x>>1;	x|=x>>2;	x|=x>>4;	return x+1;	}
typedef struct {char *buffer; int length; int offset; } printbuffer;
static char* ensure(printbuffer *p,int needed)
{
    char *newbuffer;int newsize;
    if (!p || !p->buffer) return 0;
    needed+=p->offset;
    return p->buffer+p->offset;
}
cJSON *cJSON_CreateTrue(void)
{
    cJSON *item = cJSON_New_Item();
    if (item)
        item->type = cJSON_True;
    return item;
}
static const char *parse_number(cJSON *item, const char *num)
{
    double n = 0;
    char buf[8];
    strncpy(buf, num, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    item->valuedouble = n;
    return num;
}
int cJSON_GetArraySize(cJSON *array)
{
    cJSON *c = array->child;
    int i = 0;
    while (c)
        i++, c = c->next;
    return i;
}
'''


@unittest.skipUnless(shutil.which("git"), "git not installed")
class TestRewrittenLibrary(unittest.TestCase):
    """The failure that motivated this module: a library file reformatted, reordered and extended
    in one commit. Line-anchored attribution handed the model the body of OTHER functions
    ('the diff shows cJSON_Minify instead of cJSON_CreateTrue', frac_changed > 1)."""

    def test_each_function_gets_its_own_diff(self):
        with tempfile.TemporaryDirectory() as t:
            td = Path(t)
            g = _repo(td)
            (td / "cJSON.c").write_text(OLD)
            g("add", ".")
            g("commit", "-qm", "base")
            (td / "cJSON.c").write_text(NEW)
            g("add", ".")
            g("commit", "-qm", "update cJSON, CVE-2016-4303")
            by = _enrich(td, g("rev-parse", "HEAD"), OLD,
                         ["cJSON_GetArraySize", "parse_number", "cJSON_CreateTrue"])
        pn = by["parse_number"]["change"]
        self.assertEqual(pn["diff_method"], "pair")
        self.assertTrue(pn["diff_reliable"])
        self.assertEqual(pn["kind"], "code")
        self.assertIn("strncpy(buf, num, sizeof(buf) - 1);", pn["function_diff"])
        self.assertIn("-\tstrcpy(buf,num);", pn["function_diff"])
        for foreign in ("ensure", "pow2gt", "cJSON_CreateTrue", "cJSON_GetArraySize"):
            self.assertNotIn(foreign, pn["function_diff"])
        self.assertLessEqual(pn["frac_changed"], 1.0)
        # the one-liners were only re-laid out: no semantic change, and no diff to mislead
        for name in ("cJSON_GetArraySize", "cJSON_CreateTrue"):
            c = by[name]["change"]
            self.assertEqual((c["kind"], c["whitespace_only"], c["function_diff"]), ("whitespace", True, ""))
            self.assertTrue(prompting.diff_available(by[name]))
            self.assertIn("only whitespace or formatting differs", prompting.build_prompt(by[name]))

    def test_moved_file_is_followed(self):
        old = "int add(int a, int b)\n{\n    return a + b;\n}\n"
        new = "int add(int a, int b)\n{\n    if (a > 1000) return -1;\n    return a + b;\n}\n"
        with tempfile.TemporaryDirectory() as t:
            td = Path(t)
            g = _repo(td)
            (td / "old.c").write_text(old + "\n" + "int other(void) { return 1; }\n")
            g("add", ".")
            g("commit", "-qm", "base")
            g("mv", "old.c", "new.c")
            (td / "new.c").write_text(new + "\n" + "int other(void) { return 1; }\n")
            g("add", ".")
            g("commit", "-qm", "move and fix")
            by = _enrich(td, g("rev-parse", "HEAD"), old, ["add"])
        c = by["add"]["change"]
        self.assertEqual(c["diff_method"], "pair_moved")
        self.assertEqual(c["counterpart"]["path"], "new.c")
        self.assertIn("+    if (a > 1000) return -1;", c["function_diff"])

    def test_removed_function_is_all_deletions(self):
        old = "int keep(void) { return 1; }\n\nint gone(int a)\n{\n    return a * 2;\n}\n"
        new = "int keep(void) { return 1; }\n"
        with tempfile.TemporaryDirectory() as t:
            td = Path(t)
            g = _repo(td)
            (td / "a.c").write_text(old)
            g("add", ".")
            g("commit", "-qm", "base")
            (td / "a.c").write_text(new)
            g("add", ".")
            g("commit", "-qm", "drop gone()")
            by = _enrich(td, g("rev-parse", "HEAD"), old, ["gone"])
        c = by["gone"]["change"]
        self.assertTrue(c["diff_reliable"])
        self.assertEqual(c["inserted"], 0)
        self.assertEqual(c["deleted"], 4)
        self.assertIn("-    return a * 2;", c["function_diff"])

    def test_same_name_in_two_files_picks_the_right_counterpart(self):
        a_old = "static int init(int a)\n{\n    return a + 1;\n}\n"
        b_old = "static int init(int a)\n{\n    return a * 100;\n}\n"
        a_new = "static int init(int a)\n{\n    return a + 2;\n}\n"
        with tempfile.TemporaryDirectory() as t:
            td = Path(t)
            g = _repo(td)
            (td / "a.c").write_text(a_old)
            (td / "b.c").write_text(b_old)
            g("add", ".")
            g("commit", "-qm", "base")
            (td / "a.c").write_text(a_new)
            g("add", ".")
            g("commit", "-qm", "fix a")
            by = _enrich(td, g("rev-parse", "HEAD"), a_old, ["init"])
        c = by["init"]["change"]
        self.assertIn("-    return a + 1;", c["function_diff"])
        self.assertIn("+    return a + 2;", c["function_diff"])
        self.assertNotIn("100", c["function_diff"])


if __name__ == "__main__":
    unittest.main()

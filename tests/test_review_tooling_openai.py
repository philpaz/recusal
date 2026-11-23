"""The numbers ``recusal-openai``'s README and ``docs/REVIEW.md`` state about its review
are the numbers the scripts under ``tools/review/openai/`` compute.

A page that says "44,148 cases" is a claim. These tests import each script, count what it
defines, and fail if a page states a number no script produces, names a test that does
not exist, or counts a review round the tests and the changelog do not carry. The counts
need no SDK, so they are checked on every lane.

The results (no finding, every mutation caught) need the OpenAI SDK, so those tests are
skipped where it is not installed, as the adapter's own tests are. Where it is, they run
the whole keyword matrix, the corpus's single forms, one case in ``GRID_SLICE`` of its
grid, and one mutation in each of the three files. The full corpus and the full mutation
run take minutes and are the commands the pages give; the pages say that this file checks
a part.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "tools" / "review" / "openai"
GATE_TESTS = ROOT / "tests" / "test_openai_responses_gate.py"
AUDIT_TESTS = ROOT / "tests" / "test_openai_responses_audit.py"
GRID_SLICE = 89  # docs/REVIEW.md states this number; the test below holds it to this one

needs_sdk = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI SDK requires Python 3.10+"
)


def _script(name):
    spec = importlib.util.spec_from_file_location(
        f"recusal_review_openai_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corpus():
    return _script("corpus")


@pytest.fixture(scope="module")
def matrix():
    return _script("keyword_matrix")


@pytest.fixture(scope="module")
def mutate():
    return _script("mutate")


def _readme_section() -> str:
    page = (ROOT / "adapters" / "openai" / "README.md").read_text(encoding="utf-8")
    return page.split("\n## How it was tested\n", 1)[1].split("\n## ", 1)[0]


def _review() -> str:
    return (ROOT / "docs" / "REVIEW.md").read_text(encoding="utf-8")


def _review_section() -> str:
    return _review().split("\n## `recusal-openai`\n", 1)[1].split("\n## ", 1)[0]


def _numbers(text: str) -> list:
    """Every number the prose states: code, link targets and version strings set aside."""
    text = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.S | re.M)
    text = re.sub(r"`[^`\n]*`", "", text)
    text = re.sub(r"\]\([^)\s]+\)", "]", text)
    text = re.sub(r"\b\d+(?:\.\d+)+\b", "", text)
    return [int(n.replace(",", "")) for n in re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", text)]


# --- the counts, on every lane ---------------------------------------------------------------


def test_every_number_the_pages_state_is_one_a_script_computes(corpus, matrix, mutate):
    cases, combinations, mutations = corpus.counts(), matrix.counts(), mutate.counts()
    assert cases["cases"] == cases["forms"] + cases["grid"]
    assert combinations["combinations"] == (
        combinations["gate_function_calls"]
        + combinations["gate_tool_calls"]
        + combinations["judge_and_run"]
    )
    assert mutations["mutations"] == (
        mutations["in_gate"] + mutations["in_audit"] + mutations["in_pyproject"]
    )
    assert mutations["not_applied"] == 0
    headline = {
        cases["cases"],
        cases["forms"],
        cases["grid"],
        combinations["combinations"],
        mutations["mutations"],
    }
    detail = {
        combinations["gate_function_calls"],
        combinations["judge_and_run"],
        mutations["in_gate"],
        mutations["in_audit"],
        mutations["in_pyproject"],
        GRID_SLICE,
    }
    readme = set(_numbers(_readme_section()))
    assert readme == headline, sorted(readme ^ headline)
    review = set(_numbers(_review_section()))
    assert review == headline | detail, sorted(review ^ (headline | detail))
    # and each is written the way a reader will search for it
    for page in (_readme_section(), _review_section()):
        for number in headline:
            assert f"{number:,}" in page, number


def test_the_page_states_numbers_only_in_a_packages_own_section():
    """The general part of ``docs/REVIEW.md`` describes methods. A number there would be
    one no package's script is held to: only the steps of a list are numbered."""
    general = _review().split("\n## `recusal-", 1)[0]
    assert [n for n in _numbers(general) if n > 3] == []


def test_a_package_has_a_section_exactly_when_its_scripts_are_in_the_repository():
    directories = sorted(p.name for p in (ROOT / "tools" / "review").iterdir() if p.is_dir())
    sections = re.findall(r"^## `recusal-([a-z-]+)`$", _review(), re.M)
    assert sections == directories == ["openai"]
    status = (ROOT / "README.md").read_text(encoding="utf-8")
    status = status.split("\n## Project status\n", 1)[1].split("\n## ", 1)[0]
    assert "https://github.com/philpaz/recusal/blob/main/docs/REVIEW.md" in status
    named = status.split("in this repository. Today that is ", 1)[1].split(".\n", 1)[0]
    assert re.findall(r"`recusal-([a-z-]+)`", named) == directories
    # the paragraph that points there quotes no count from a review
    pointer = status.split("[What a review consisted of]", 1)[1].split("\n\n", 1)[0]
    assert _numbers(pointer) == []


def test_no_two_cases_of_a_set_are_the_same_input(corpus):
    grid = list(corpus._grid())
    assert len({corpus._fingerprint(case) for case in grid}) == len(grid)
    for name in corpus.SETS:
        kept = list(corpus.cases(name))
        assert len({corpus._fingerprint(case) for case in kept}) == len(kept) > 0
    assert sum(1 for _ in corpus.cases()) == corpus.counts()["cases"]
    # every name the grid calls known is a key of the table the cases are run against
    tools, _policy, _resets = corpus._grid_table([])
    assert tuple(tools) == corpus.GRID_KNOWN
    assert set(corpus.FORMS_NAMES) == set(corpus._forms_table([])[0])


def test_every_mutation_is_one_edit_that_still_applies_to_the_source(mutate):
    assert mutate.not_applied() == []
    labels = [mutation.label for mutation in mutate.MUTATIONS]
    assert len(set(labels)) == len(labels)
    for mutation in mutate.MUTATIONS:
        assert mutation.file in (mutate.GATE, mutate.AUDIT, mutate.PYPROJECT), mutation.label
        assert mutation.old != mutation.new and mutation.old, mutation.label
        assert 0 <= mutation.occurrence < mutation.count, mutation.label
        text = (ROOT / mutation.file).read_text(encoding="utf-8")
        assert mutate._mutated(text, mutation) != text, mutation.label


def test_negative_control_a_mutation_whose_text_is_gone_is_reported(mutate, monkeypatch):
    gone = mutate.Mutation(mutate.GATE, "gone", "text that is not in the source", "x")
    monkeypatch.setattr(mutate, "MUTATIONS", [*mutate.MUTATIONS, gone])
    assert mutate.not_applied() == ["gone"]
    assert mutate.counts()["not_applied"] == 1


def test_negative_control_the_corpus_check_sees_each_kind_of_finding(corpus):
    case = corpus.Case(
        "grid",
        "made up",
        "responses",
        [("c0", "t", "{}", {}), ("c1", "nope", "{}", {})],
        {},
        None,
        {},
        False,
        {},
    )
    tools = {"t": len}
    entry = {"action": {"tool_use_id": "c0"}}
    other = {"action": {"tool_use_id": "c1"}}

    def outcome(**changes):
        base = dict(
            status="answered",
            tool_raised=False,
            ran=[("t",)],
            texts=["ok", "Recusal refused: unknown tool 'nope'"],
            ids=["c0", "c1"],
            entries=[],
        )
        base.update(changes)
        return corpus.Outcome(**base)

    def kinds(plain, audited):
        return [kind for kind, _ in corpus.check(case, tools, plain, audited)]

    refusal = "Recusal refused `nope` [FAIL]: unknown tool"
    good = outcome(texts=["ok", refusal], entries=[entry, other])
    assert kinds(outcome(), good) == []
    assert kinds(outcome(ran=[]), good) == ["divergence"]
    assert kinds(outcome(), outcome(status="raised TypeError", entries=[])) == ["divergence"]
    assert kinds(outcome(), good._replace(entries=[entry])) == ["entry_count"]
    assert kinds(outcome(), good._replace(entries=[other, entry])) == ["entry_ids"]
    assert kinds(outcome(ids=["c1", "c0"]), good) == ["answer_ids"]
    assert kinds(outcome(), good._replace(texts=["ok", "something else"])) == ["text"]
    twice = [("t",), ("t",)]
    assert kinds(outcome(ran=twice), good._replace(ran=twice)) == ["unknown_ran"] * 2
    long = "x" * (corpus.LONG_ANSWER + 1)
    assert (
        kinds(outcome(texts=[long, long]), good._replace(texts=[long, long])) == ["long_answer"] * 4
    )
    leak = [corpus.MARKER, corpus.MARKER]
    assert set(kinds(outcome(texts=leak), good._replace(texts=leak))) == {"marker"}
    big = dict(entry, reasons=["x" * corpus.LONG_ENTRIES])
    assert kinds(outcome(), good._replace(entries=[big, other])) == ["long_entry"]
    raised = outcome(status="raised ValueError", texts=[], ids=[])
    assert kinds(raised._replace(ran=[]), raised._replace(ran=[])) == []
    assert kinds(raised, raised) == ["raised_after_run"]
    assert kinds(raised._replace(ran=[]), raised._replace(ran=[], entries=[entry])) == [
        "entry_count"
    ]
    by_tool = raised._replace(tool_raised=True)
    assert kinds(by_tool, by_tool._replace(entries=[entry])) == []
    assert kinds(by_tool, by_tool) == ["entry_count"]


def test_negative_control_the_matrix_check_sees_each_kind_of_finding(matrix):
    check = matrix.check
    assert check("gate_tool_calls", "answered", 2, 2, 2, True) == []
    assert check("gate_tool_calls", "answered", 2, 0, 0, False) == []
    assert check("judge_and_run", "answered", 1, 1, 1, True) == []
    assert check("judge_and_run", "raised TypeError", 0, 0, 0, True) == []
    assert check("judge_and_run", "raised TypeError", 1, 0, 0, True) == ["raised_after_run"]
    assert check("judge_and_run", "raised ValueError", 0, 1, 0, True) == ["raised_but_recorded"]
    assert check("judge_and_run", "raised ValueError", 0, 0, 1, True) == ["raised_but_recorded"]
    assert check("gate_function_calls", "answered", 0, 2, 2, True) == ["answered_not_run"]
    assert check("gate_function_calls", "answered", 2, 1, 2, True) == ["entries"]
    assert check("gate_function_calls", "answered", 2, 2, 0, True) == ["entries"]
    # a keyword left out is not a value: no list holds the text that stands for it
    for values in (matrix.SURFACE, matrix.ACTOR, matrix.CONTROL, matrix.CALL_ID_JUDGE):
        assert values[0] == matrix.ABSENT and matrix.ABSENT not in values[1:]
        assert matrix.LOG not in values
        assert len({repr(value) for value in values}) == len(values)


@pytest.mark.parametrize("name", ["corpus", "keyword_matrix", "mutate"])
def test_each_script_explains_itself_and_counts_without_the_sdk(name, corpus, matrix, mutate):
    module = {"corpus": corpus, "keyword_matrix": matrix, "mutate": mutate}[name]
    doc = module.__doc__
    assert f"python tools/review/openai/{name}.py --summary" in doc
    for part in ("What a finding is", "What it does not show", "Exit status: 0"):
        if name == "mutate" and part == "What a finding is":
            part = 'What "caught" means'
        assert part in doc, part
    assert str(ROOT) not in (SCRIPTS / f"{name}.py").read_text(encoding="utf-8")
    script = str(SCRIPTS / f"{name}.py")
    helped = subprocess.run(
        [sys.executable, script, "--help"], capture_output=True, text=True, check=False
    )
    assert helped.returncode == 0 and "--summary" in helped.stdout and "--count" in helped.stdout
    counted = subprocess.run(
        [sys.executable, script, "--count"], capture_output=True, text=True, check=False
    )
    assert counted.returncode == 0, counted.stderr
    lines = counted.stdout.strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0]) == module.counts()
    assert module.counts()["tool"] == f"tools/review/openai/{name}.py"


def test_every_test_the_pages_name_is_a_test_in_the_adapters_files():
    defined = set()
    for path in (GATE_TESTS, AUDIT_TESTS):
        defined |= set(re.findall(r"^\s*def (test_\w+)\(", path.read_text("utf-8"), re.M))
    for page in (_readme_section(), _review_section()):
        named = set(re.findall(r"`(test_\w+)`", page))
        assert len(named) >= 6
        assert named <= defined, sorted(named - defined)
    # the README names a subset of what the review page names
    readme = set(re.findall(r"`(test_\w+)`", _readme_section()))
    assert readme <= set(re.findall(r"`(test_\w+)`", _review_section()))


def test_the_rounds_the_pages_count_are_the_rounds_the_tests_and_the_changelog_carry():
    """Four rounds: the adapter's tests are labeled up to round four and no further, the
    findings the pages quote from rounds one to three are in the changelog entry, and the
    pages say that round four's two tests are not."""
    tests = GATE_TESTS.read_text(encoding="utf-8") + AUDIT_TESTS.read_text(encoding="utf-8")
    for label in ("round two", "round three", "round four"):
        assert label in tests, label
    assert not re.search(r"round (?:five|six|5|6)", tests)
    assert "was four review rounds" in _readme_section()
    assert "was four rounds" in _review_section()
    for page in (_readme_section(), _review_section()):
        rounds = re.findall(r"^- (?:\*\*)?Round (\w+)[.:]", page, re.M)
        assert rounds == ["one", "two", "three", "four"], rounds
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = changelog.split("- **`recusal-openai` 0.2.2: ", 1)[1].split("\n- **", 1)[0]
    entry = " ".join(entry.split("\n## ", 1)[0].split())
    for recorded in (
        'let `{"amount": NaN}` and `{"amount": -Infinity}` run in the plain loop',
        "so a call to `delete` in namespace `prod_db` ran the application's `delete`",
        "A call that was not finished ran",
        'refused every call as a "policy error" although no policy had run',
        "A model-chosen tool name was written raw",
        "raised `TypeError` out of the turn",
        "The package allowed `recusal>=0.14.0`",
        'refused every call as a "policy error" and wrote no entry',
        "A model-chosen call id had no bound",
    ):
        assert recorded in entry, recorded
    for unrecorded in ("RecursionError", "judge_and_run called directly"):
        assert unrecorded not in entry
    assert "Round four: no changelog item." in _readme_section()
    assert "Neither is in the changelog entry" in _review_section()


def test_the_pages_claim_no_more_than_the_scripts_show():
    scripts = "".join(p.read_text(encoding="utf-8") for p in sorted(SCRIPTS.glob("*.py")))
    new = {"README section": _readme_section(), "docs/REVIEW.md": _review(), "scripts": scripts}
    for name, text in new.items():
        assert "\u2014" not in text and "\u2013" not in text, name
        flat = " ".join(text.split()).lower()
        for word in (
            "secure",
            "hardened",
            "proven",
            "comprehensive",
            "battle-tested",
            "independent",
            "third-party",
            "third party",
            "external",
            "certify",
            "guarantee",
        ):
            assert not re.search(rf"\b{word}", flat), (name, word)
    assert "the\nproject's own work" in _readme_section()
    assert "the project's own\nwork" in _review()
    assert "not a\nfinished or certified security product" in _review()
    assert "a large number of generated cases is not a proof" in _readme_section()
    assert "**A count of cases is not a proof.**" in _review()
    target = "adapters/openai/README.md#what-the-gate-refuses-and-what-it-does-not-see)"
    assert target in _readme_section() and target in _review_section()
    assert "docs/REVIEW.md#recusal-openai)" in _readme_section()
    page = (ROOT / "adapters" / "openai" / "README.md").read_text(encoding="utf-8")
    assert "\n## What the gate refuses, and what it does not see\n" in page
    assert page.index("\n## Status\n") < page.index("\n## How it was tested\n")
    # each command the README gives is a script that is there, and each count has its command
    commands = re.findall(r"`python (tools/review/openai/\w+\.py) --summary`", _readme_section())
    assert sorted(commands) == sorted(
        f"tools/review/openai/{name}.py" for name in ("corpus", "keyword_matrix", "mutate")
    )
    assert f"one case in {GRID_SLICE} of the grid" in " ".join(_review_section().split())
    assert "requirements/adapter-openai.txt" in _readme_section()
    assert (ROOT / "requirements" / "adapter-openai.txt").is_file()


def test_both_openai_lanes_run_this_file():
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_review_tooling_openai.py") == 2, workflow


# --- the results, where the SDK is installed -------------------------------------------------


@needs_sdk
def test_the_whole_keyword_matrix_has_no_finding(matrix):
    pytest.importorskip("openai", reason="the openai package is not installed")
    summary = matrix.run()
    assert summary["combinations"] == matrix.counts()["combinations"]
    assert summary["answered"] + summary["raised"] == summary["combinations"]
    assert summary["answered"] > 0 and summary["raised"] > 0
    assert summary["findings"] == 0, summary
    assert all(summary[kind] == 0 for kind in matrix.KINDS)


@needs_sdk
def test_the_single_forms_and_a_slice_of_the_grid_have_no_finding(corpus):
    pytest.importorskip("openai", reason="the openai package is not installed")
    counts = corpus.counts()
    forms = corpus.run("forms")
    assert forms["cases"] == forms["forms"] == counts["forms"]
    assert forms["findings"] == 0, forms
    part = corpus.run("grid", every=GRID_SLICE)
    assert part["cases"] == part["grid"] == -(-counts["grid"] // GRID_SLICE)
    assert part["loop_runs"] == 2 * part["cases"]
    assert part["answered"] > 0 and part["raised_by_a_tool"] > 0
    assert part["findings"] == 0, part
    assert all(part[kind] == 0 for kind in corpus.KINDS)


@needs_sdk
def test_one_mutation_in_each_file_is_caught_and_an_edit_no_test_sees_survives(mutate, monkeypatch):
    """A slice of the run ``python tools/review/openai/mutate.py`` makes: the first
    mutation of each file is caught. A comment-only edit added for this test survives,
    which is what the run reports when no test notices an edit, so "caught" is not what
    the runner says of everything."""
    pytest.importorskip("openai", reason="the openai package is not installed")
    before = {path: (ROOT / path).read_bytes() for path in (mutate.GATE, mutate.AUDIT)}
    unseen = mutate.Mutation(
        mutate.GATE,
        "an edit to a comment",
        "# nothing nested deeper is fingerprinted, copied or handed to a policy",
        "# nothing nested deeper is fingerprinted or handed to a policy",
    )
    monkeypatch.setattr(mutate, "MUTATIONS", [*mutate.MUTATIONS, unseen])
    assert mutate.not_applied() == []
    files = [mutation.file for mutation in mutate.MUTATIONS]
    chosen = [files.index(path) for path in (mutate.GATE, mutate.AUDIT, mutate.PYPROJECT)]
    summary = mutate.run([*chosen, len(mutate.MUTATIONS) - 1])
    assert summary["baseline"] == "passed", summary
    assert summary["selected"] == 4 and summary["caught"] == 3, summary
    assert summary["survivors"] == ["an edit to a comment"] and summary["survived"] == 1
    assert summary["not_applied"] == 0 and summary["timed_out"] == 0
    assert summary["working_tree_unchanged"] is True
    assert before == {path: (ROOT / path).read_bytes() for path in before}

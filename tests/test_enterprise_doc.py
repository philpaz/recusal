"""``docs/ENTERPRISE.md`` is internally consistent, and still says what it is.

The document is a specification nobody has built, so no test here can show it is right.
What can be held is that it does not contradict itself: every scenario it cites exists,
every section it points to exists, every state it names is one of the seven it defines,
each locking transaction states the same lock order, and the two controls that stop work
are named and scoped the same way wherever they appear. Those are the properties a review
found broken, so they are the ones locked. Links are followed by
``tests/test_markdown_links.py``.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "ENTERPRISE.md").read_text(encoding="utf-8")
STATES = (
    "PENDING_AUDIT",
    "READY",
    "DISPATCHING",
    "SUCCEEDED",
    "FAILED",
    "OUTCOME_UNKNOWN",
    "CANCELED",
)


def _section(number: str) -> str:
    """The text of one numbered section, up to the next heading of the same or higher level."""
    level = "###" if "." in number else "##"
    start = re.search(rf"^{level} {re.escape(number)} .*$", DOC, re.M)
    assert start is not None, f"no section {number}"
    rest = DOC[start.end() :]
    stop = re.search(r"^#{2,3} \d" if level == "###" else r"^## \d", rest, re.M)
    return rest[: stop.start()] if stop else rest


def _scenarios() -> dict:
    # each scenario's whole row: what is done, and the outcome required
    return dict(re.findall(r"^\| (A\d\d) \| (.*) \|$", DOC, re.M))


def test_it_still_says_it_is_a_position_and_nothing_is_built():
    opening = DOC.split("## 1 ", 1)[0]
    assert "**This is a reference architecture, and a position.**" in opening
    assert "Nothing in\nit is built, deployed or independently validated today" in opening
    assert "None of it has started." in _section("11")


def test_the_scenarios_are_numbered_without_a_gap_or_a_repeat():
    listed = re.findall(r"^\| (A\d\d) \|", DOC, re.M)
    assert listed == [f"A{n:02d}" for n in range(1, len(listed) + 1)]
    assert len(listed) >= 45


def test_every_scenario_cited_in_the_text_exists():
    cited = set(re.findall(r"\bA\d\d\b", DOC))
    assert cited <= set(_scenarios()), sorted(cited - set(_scenarios()))
    # a range such as "A41 to A45" cites its ends, and both exist
    for first, last in re.findall(r"\b(A\d\d) to (A\d\d)\b", DOC):
        assert first in _scenarios() and last in _scenarios() and first < last


def test_every_section_the_text_points_to_exists():
    headings = set(re.findall(r"^#{2,3} (\d+(?:\.\d+)?) ", DOC, re.M))
    cited = set(re.findall(r"\bsections? (\d+(?:\.\d+)?)", DOC))
    cited |= set(
        re.findall(r"\bsections? \d+(?:\.\d+)?(?:,? (?:and|to) | and | to |, )(\d+(?:\.\d+)?)", DOC)
    )
    assert cited <= headings, sorted(cited - headings)
    for wanted in ("6.8", "6.9", "6.10"):
        assert wanted in headings


def test_every_state_named_is_one_the_document_defines():
    table = _section("6.4")
    defined = re.findall(r"^\| `([A-Z_]+)` \|", table, re.M)
    assert defined == list(STATES)
    used = set(re.findall(r"`([A-Z]+(?:_[A-Z]+)*)`", DOC))
    state_like = {name for name in used if name in STATES or name.endswith(("_AUDIT", "_UNKNOWN"))}
    assert state_like <= set(STATES), sorted(state_like - set(STATES))


def test_no_transition_returns_an_action_to_ready():
    """The one promise recovery rests on: once past ``READY`` an action never goes back."""
    for number in ("6.4", "6.5", "6.10"):
        text = _section(number)
        assert not re.search(r"(back|returns?|moved?) to `READY`", text.replace("never", "")) or (
            "never back to `READY`" in text or "never returns to `READY`" in text
        ), number
    assert "the state never returns to `READY`" in _section("6.4")
    assert "never returns an action to `READY`" in _section("6.5")


def test_each_locking_transaction_states_the_same_lock_order():
    """Active policy, then grant, then action. A transaction may skip the grant; none may
    take them in another order, and each one that changes state checks the pause."""
    authorization, promotion, dispatch = _section("6.1"), _section("6.2"), _section("6.3")
    resend = _section("6.5")
    assert re.search(
        r"Lock the shared active-policy row.*then\s+the grant row", authorization, re.S
    )
    assert re.search(
        r"locks the shared\s+active-policy row for share, then the action row for update", promotion
    )
    assert "takes no\ngrant lock" in promotion
    assert re.search(
        r"Lock the shared active-policy row for share, then the grant row for update,\s+"
        r"then the action row for update, in that order",
        dispatch,
    )
    assert re.search(
        r"Lock the shared active-policy row for share, then the grant row for update, then\s+"
        r"the action row for update, in that order",
        resend,
    )
    for name, text in (
        ("6.1", authorization),
        ("6.2", promotion),
        ("6.3", dispatch),
        ("6.5", resend),
    ):
        assert "global\n" in text or "global pause" in text, f"{name} does not check the pause"
        assert "pause" in text, name


def test_the_resend_is_a_guarded_state_change_with_every_dispatch_check():
    resend = _section("6.5")
    assert "change `OUTCOME_UNKNOWN` to `DISPATCHING`" in resend
    assert "the claim ID on the row is the one this recovery\n     worker read" in resend
    for check in (
        "the grant is not revoked",
        "`not_after` have not\n     passed",
        "epochs and digests match the active record",
        "fingerprint recomputed from the stored request",
        "retention\n     window for the action-derived key is still open",
    ):
        assert check in resend, check
    assert "it is not\n  canceled, because the original request may have been applied" in resend


def test_the_two_controls_that_stop_work_have_one_name_and_one_scope_each():
    pause = _section("6.9")
    assert "**The global pause stops the whole deployment.**" in pause
    assert "**Admission suspension stops one grant.**" in pause
    # the old single name, which meant both, is gone from the whole document
    assert "admission pause" not in DOC.lower()
    stops = pause.split("**What it stops, for every grant.**", 1)[1].split(
        "- **What it does not", 1
    )[0]
    for section in ("section 6.1", "section 6.2", "section 6.3", "section 6.5"):
        assert section in stops, section
    assert "no other grant is touched in\n  any way" in pause
    assert "A revocation is never held up by a pause." in pause
    assert "can therefore stop the\n  deployment and cannot start it" in pause
    actors = _section("3")
    assert "| Admission controller | Setting the global pause" in actors
    assert "Clearing the pause" in actors.split("| Admission controller |", 1)[1].split("\n", 1)[0]


def test_the_scenarios_agree_with_the_pause_semantics():
    scenarios = _scenarios()
    assert (
        "the global pause is not\nset" in scenarios["A35"]
        or "the global pause is not set" in scenarios["A35"]
    )
    assert "every other grant authorizes, is promoted and dispatches as before" in scenarios["A35"]
    a43 = scenarios["A43"]
    assert "no grant authorizes" in a43 and "no action of any grant is promoted to `READY`" in a43
    assert "none is canceled by the pause" in a43
    assert "unaffected" not in a43, "the old wording promised other grants were unaffected"
    assert "can set and cannot clear" in scenarios["A45"]


def test_promotion_needs_completeness_and_a_failed_proof_is_a_restore():
    failover = _section("6.8")
    assert "n - k + 1" in failover and "pg_last_wal_receive_lsn()" in failover
    assert "An\n     asynchronous standby is never promoted by this procedure." in failover
    assert "nothing is promoted by this rule" in failover
    assert failover.count("section 6.10") >= 2 or failover.count("6.10") >= 3
    assert (
        "every authorization-intent event the audit destination has\n     acknowledged must be present"
        in failover
    )
    assert "(A44)" in failover and "n - k + 1" in _scenarios()["A44"]


def test_a_restore_invalidates_authority_and_accounts_for_every_action():
    restore = _section("6.10")
    steps = re.findall(r"^(\d)\. \*\*", restore, re.M)
    assert steps == ["1", "2", "3", "4", "5", "6"]
    for phrase in (
        "revocations, budget and nonce\nconsumption, policy activations",
        "Advance the epoch past every epoch anything was authorized under",
        "every\n   restored grant is without authority",
        "The action is not in the restored database.",
        "It holds\n     no arguments",
        "Nothing from before the restore is sent again.",
        "an operator clears the pause",
        "Every grant outstanding at the time of a restore dies",
    ):
        assert phrase in restore, phrase
    a26 = _scenarios()["A26"]
    for case in ("a grant's revocation", "a budget", "a policy activation", "an action's dispatch"):
        assert case in a26, case
    assert "refuses to\nresume" in a26 or "refuses to resume" in a26
    assert "section 6.10" in _section("7")


def test_the_quotations_from_postgresql_are_the_ones_the_rules_use():
    sources = _section("12")
    quorum = "replicated to at least the requested number of synchronous standbys in the list"
    receive = "received and synced to disk by streaming replication"
    assert quorum in sources and receive in sources
    failover = " ".join(_section("6.8").split())
    assert (
        quorum.replace("replicated to", "replicated to") in failover
        or "requested number of synchronous standbys in the list" in failover
    )
    assert receive in failover
    assert "The arithmetic is this document's, not PostgreSQL's." in sources
    assert "none is evidence that the procedures" in " ".join(sources.split())

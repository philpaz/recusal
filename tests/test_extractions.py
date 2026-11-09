"""Two facade-preserving extractions, held to the facades and to a golden record.

The neutral evaluator (``decide`` and the one audit-record path) moved from
``recusal/claude_code.py`` to ``recusal/_adjudicate.py``; the runtime manifest policy moved
from ``recusal/mcp.py`` to ``recusal/_manifest_policy.py``. Nothing public moved: each
original module re-exports the moved names, so every import path that worked before works
now and resolves to the very same object. This file pins that, pins the module boundaries
the extraction was for, and holds ``decide`` to an audit record captured before the move.
"""

import inspect
import json
import sys

import pytest

import recusal
from recusal import Finding
from recusal.audit import AuditLog

# --- the facades -----------------------------------------------------------------------------


def test_every_moved_evaluator_name_is_the_same_object_through_claude_code():
    import recusal._adjudicate as inner
    import recusal.claude_code as facade

    for name in (
        "Policy",
        "_adjudicate",
        "_record",
        "_control_identity",
        "_reset_control_identity",
        "_input_fingerprint",
        "_bind_input",
        "_AUTHORITATIVE_CONTROL_KEYS",
        "_HOOK_SURFACE",
    ):
        assert getattr(facade, name) is getattr(inner, name), name
    # what stayed: the hook and its event parsing, the allowlist
    for name in ("run_pretooluse_hook", "_runtime_context", "allowlist_policy"):
        assert getattr(facade, name).__module__ == "recusal.claude_code", name
    # the public callable stays in the facade: same module, a wrapper over the moved one
    assert facade.decide.__module__ == "recusal.claude_code"
    assert facade.decide.__wrapped__ is inner.decide
    assert facade.decide.__doc__ == inner.decide.__doc__
    assert inspect.signature(facade.decide) == inspect.signature(inner.decide)


def test_manifest_policy_is_the_same_object_through_mcp():
    import recusal._manifest_policy as inner
    import recusal.mcp as facade

    assert facade.manifest_policy.__module__ == "recusal.mcp"
    assert facade.manifest_policy.__wrapped__ is inner.manifest_policy
    assert facade.manifest_policy.__doc__ == inner.manifest_policy.__doc__
    assert inspect.signature(facade.manifest_policy) == inspect.signature(inner.manifest_policy)
    # the artifact half stayed where it was
    for name in ("build_manifest", "diff_manifest", "load_manifest", "_validate_manifest"):
        assert getattr(facade, name).__module__ == "recusal.mcp", name


def _shape(func):
    """A signature's shape, independent of how a Python version renders annotations."""
    return [
        (
            p.name,
            p.kind.name,
            p.default if p.default is not inspect.Parameter.empty else "<required>",
        )
        for p in inspect.signature(func).parameters.values()
    ]


def test_the_public_signatures_did_not_move():
    from recusal.claude_code import decide
    from recusal.mcp import manifest_policy

    assert _shape(decide) == [
        ("tool_name", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("tool_input", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("policy", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("allow_on_pass", "KEYWORD_ONLY", False),
        ("fail_closed", "KEYWORD_ONLY", True),
        ("audit", "KEYWORD_ONLY", None),
        ("surface", "KEYWORD_ONLY", None),
        ("tool_use_id", "KEYWORD_ONLY", None),
        ("actor", "KEYWORD_ONLY", None),
        ("control", "KEYWORD_ONLY", None),
        # 0.14.0, additive: keyword-only, last, and off by default, so every existing
        # call means what it meant
        ("safe_errors", "KEYWORD_ONLY", False),
    ]
    assert _shape(manifest_policy) == [
        ("manifest_path", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("policy", "KEYWORD_ONLY", None),
    ]
    # the annotations are the moved function's own, compared as objects, never as the
    # strings a given Python version renders them to
    import recusal._adjudicate as adjudicate
    import recusal._manifest_policy as manifest

    for facade, inner in ((decide, adjudicate.decide), (manifest_policy, manifest.manifest_policy)):
        assert inspect.signature(facade) == inspect.signature(inner)
        assert facade.__annotations__ == inner.__annotations__  # 3.9-safe, no get_annotations


# --- the boundaries the extraction was for ------------------------------------------------------


def _imports_of(module_name: str) -> set:
    module = sys.modules[module_name]
    return {
        name
        for name, value in vars(module).items()
        if inspect.ismodule(value) or (hasattr(value, "__module__") and not name.startswith("__"))
    }


def test_the_evaluator_imports_no_adapter_persistence_or_mcp_code():
    import recusal._adjudicate as inner

    origins = {
        getattr(value, "__module__", None)
        for name, value in vars(inner).items()
        if not name.startswith("__") and not inspect.ismodule(value)
    }
    origins.discard(None)
    assert not any(o.startswith("recusal.claude_code") for o in origins)
    assert not any(o.startswith("recusal.mcp") for o in origins)
    assert not any(o.startswith("recusal.audit") for o in origins)  # typing-only
    assert not any(o.startswith("recusal.__main__") for o in origins)
    assert {o for o in origins if o.startswith("recusal.")} <= {
        "recusal._adjudicate",
        "recusal.evidence",
        "recusal.authorization",
    }


def test_the_manifest_policy_module_imports_mcp_only_at_call_time():
    import recusal._manifest_policy as inner

    assert "mcp" not in {name for name, value in vars(inner).items() if inspect.ismodule(value)}
    origins = {
        getattr(value, "__module__", None)
        for name, value in vars(inner).items()
        if not name.startswith("__") and not inspect.ismodule(value)
    }
    # the evaluator's return-contract helper is the one module-load dependency beyond the
    # kernel: recusal._adjudicate imports only recusal.evidence and recusal.authorization,
    # never recusal.mcp, so the deferred validator import above stays the only cycle risk
    assert {o for o in origins if o and o.startswith("recusal.")} <= {
        "recusal._manifest_policy",
        "recusal._adjudicate",
        "recusal.evidence",
    }


def test_importing_either_private_module_first_does_not_cycle():
    import subprocess

    for first in ("recusal._manifest_policy", "recusal._adjudicate"):
        code = (
            f"import {first}; import recusal.mcp, recusal.claude_code; "
            f"print(recusal.mcp.manifest_policy.__module__, recusal.claude_code.decide.__module__)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["recusal.mcp", "recusal.claude_code"]


# --- the golden record: decide writes exactly what it wrote before the move --------------------


def _policy(name, args):
    if args.get("sql", "").startswith("DELETE"):
        return [Finding.fail("x", severity="CRITICAL", message="no")]
    return []


GOLDEN_DENY = {
    "control": {"policy_id": "p", "recusal_version": recusal.__version__},
    "decision": "deny",
    "input_sha256": "bfbfb925d6f70a7522b96fd7a7a9d9e4c248d6fa3beddce9fdc0bc12f31a1025",
    "reason": "Recusal refused `run_sql` [FAIL]: no",
    "surface": "golden",
    "tool": "run_sql",
    "tool_use_id": "t1",
}
GOLDEN_DEFER = {
    "control": {"recusal_version": recusal.__version__},
    "decision": "defer",
    "input_sha256": "f25189374bfbf48658282bc196f4f8b6d09c7e4aa745f5b2351c827a8ef5aad9",
    "reason": "Passed with 0 warning(s) and 0 metric(s).",
    "surface": "golden",
    "tool": "run_sql",
}


@pytest.mark.parametrize("path", ["recusal.claude_code", "recusal._adjudicate"])
def test_decide_writes_the_record_captured_before_the_extraction(tmp_path, path):
    import importlib

    decide = importlib.import_module(path).decide
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    assert decide(
        "run_sql",
        {"sql": "DELETE FROM t"},
        _policy,
        audit=log,
        surface="golden",
        tool_use_id="t1",
        actor="me",
        control={"policy_id": "p", "recusal_version": "9.9.9"},
    ) == ("deny", "Recusal refused `run_sql` [FAIL]: no")
    assert decide("run_sql", {"sql": "SELECT 1"}, _policy, audit=log, surface="golden") == (
        "defer",
        "Passed with 0 warning(s) and 0 metric(s).",
    )
    entries = [
        json.loads(line) for line in (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()
    ]
    assert entries[0]["action"] == GOLDEN_DENY
    assert entries[1]["action"] == GOLDEN_DEFER
    assert entries[0]["actor"] == "me"
    assert sorted(entries[0]) == [
        "action",
        "actor",
        "decision",
        "failures",
        "hash",
        "highest_severity",
        "prev_hash",
        "reasons",
        "seq",
        "timestamp",
    ]


# --- identity discovery is one contract -------------------------------------------------------


def test_identity_getter_prefers_the_method_then_the_legacy_attribute_then_none():
    from recusal.claude_code import _control_identity, _identity_getter

    class Both:
        last_manifest_digest = "sha256:" + "1" * 64

        def get_control_identity(self):
            return {"manifest_sha256": "sha256:" + "2" * 64}

    class Attr:
        last_manifest_digest = "sha256:" + "3" * 64

    class Nothing:
        pass

    assert _identity_getter(Both())() == {"manifest_sha256": "sha256:" + "2" * 64}
    assert _identity_getter(Attr())() == {"manifest_sha256": "sha256:" + "3" * 64}
    assert _identity_getter(Nothing()) is None
    # and the evaluator's record reads through the same contract
    assert _control_identity(Both(), None)["manifest_sha256"] == "sha256:" + "2" * 64
    assert _control_identity(Attr(), None)["manifest_sha256"] == "sha256:" + "3" * 64
    assert "manifest_sha256" not in _control_identity(Nothing(), None)

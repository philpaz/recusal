"""Strict coercion: a loose evidence dict with no outcome must not silently pass."""

import pytest

from recusal import Finding, compute_verdict


def test_lenient_default_still_passes_a_statusless_dict():
    # documented footgun, kept for back-compat: no status/passed -> treated as a pass.
    assert compute_verdict([{"severity": "CRITICAL", "message": "no status"}]).passed


def test_strict_rejects_a_statusless_dict():
    with pytest.raises(ValueError):
        compute_verdict([{"severity": "CRITICAL", "message": "no status"}], strict=True)
    with pytest.raises(ValueError):
        Finding.coerce({"severity": "CRITICAL", "message": "no status"}, strict=True)


def test_explicit_passed_key_is_honored_in_both_modes():
    assert compute_verdict([{"severity": "CRITICAL", "passed": False}]).refused
    assert compute_verdict([{"severity": "CRITICAL", "passed": False}], strict=True).refused
    assert compute_verdict([{"severity": "CRITICAL", "passed": True}], strict=True).passed


@pytest.mark.parametrize(
    "status",
    [
        "fail",
        "failed",
        "failure",
        "false",
        "0",
        "no",
        "n",
        "off",
        "error",
        "denied",
        "rejected",
        "blocked",
        "fatal",
        "warn",
        "",
        "unknown",
        "borked",
    ],
)
def test_status_field_fails_closed_on_any_non_pass_token(status):
    # Regression: the `status` branch once used a hardcoded {"fail","error","warn"}
    # blocklist, so a CRITICAL finding with status "failed"/"false"/"denied"/… coerced
    # to PASS through every strict adapter (the exact bool("false")==True footgun this
    # library exists to prevent, on the status path). It is now a pass ALLOWLIST: any
    # token not affirmatively "passed" fails closed.
    assert not Finding.coerce({"severity": "CRITICAL", "status": status}).passed
    assert compute_verdict([{"severity": "CRITICAL", "status": status}], strict=True).refused


@pytest.mark.parametrize("status", ["pass", "passed", "ok", "okay", "success", "PASS", " Pass "])
def test_status_field_still_passes_affirmative_tokens(status):
    assert Finding.coerce({"severity": "INFO", "status": status}).passed


@pytest.mark.parametrize(
    "value",
    ["maybe", "unknown", "borked", "false", "no", "n", "0", "off", "fail", "failed", "", "   "],
)
def test_passed_string_fails_closed_on_any_non_affirmative_token(value):
    # Regression: a string `passed` was once read against a false-token BLOCKLIST, so an
    # unrecognized token like "maybe" coerced to PASS while status="maybe" failed closed.
    # Both fields now share the allowlist posture: unrecognized -> failure.
    assert not Finding.coerce({"severity": "CRITICAL", "passed": value}).passed
    assert compute_verdict([{"severity": "CRITICAL", "passed": value}], strict=True).refused


@pytest.mark.parametrize(
    "value", ["true", "TRUE", " Yes ", "1", "y", "on", "pass", "passed", "ok", "success"]
)
def test_passed_string_still_passes_affirmative_tokens(value):
    assert Finding.coerce({"severity": "INFO", "passed": value}).passed
    assert compute_verdict([{"severity": "CRITICAL", "passed": value}], strict=True).passed

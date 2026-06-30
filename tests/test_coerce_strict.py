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

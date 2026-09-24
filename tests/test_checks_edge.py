"""Heavy edge-case tests for the built-in checks, nulls, NaN, boundaries, row shapes."""

from datetime import date, datetime, timezone

from recusal import Severity
from recusal.checks import (
    date_range,
    in_range,
    in_set,
    null_rate,
    referential_integrity,
    required_keys,
    row_count,
)


def test_null_rate_empty_rows_passes_vacuously():
    assert null_rate([], "x").passed


def test_null_rate_counts_none_nan_and_empty_string():
    rows = [{"x": None}, {"x": float("nan")}, {"x": ""}, {"x": "ok"}]
    f = null_rate(rows, "x", max_rate=0.5)  # 3/4 = 75% > 50%
    assert not f.passed
    assert abs(f.context["null_rate"] - 0.75) < 1e-9


def test_null_rate_at_threshold_passes():
    rows = [{"x": None}, {"x": "a"}]  # 50%
    assert null_rate(rows, "x", max_rate=0.5).passed  # not > 0.5


def test_in_range_non_numeric_is_a_violation():
    f = in_range([{"v": "abc"}, {"v": 5}], "v", 0, 10)
    assert not f.passed and f.context["violation_count"] == 1


def test_in_range_boundaries_inclusive():
    assert in_range([{"v": 0}, {"v": 10}], "v", 0, 10).passed


def test_in_range_ignores_nulls():
    assert in_range([{"v": None}, {"v": 5}], "v", 0, 10).passed


def test_row_count_exact_threshold():
    assert row_count([{"a": 1}], min_rows=1).passed
    assert not row_count([], min_rows=1).passed


def test_referential_integrity_ignores_null_fk():
    assert referential_integrity([{"fk": None}, {"fk": 1}], [{"id": 1}], fk="fk", pk="id").passed


def test_referential_integrity_reports_orphan_sample():
    f = referential_integrity([{"fk": 9}], [{"id": 1}], fk="fk", pk="id")
    assert not f.passed and f.context["orphan_count"] == 1


def test_in_set_ignores_nulls():
    assert in_set([{"t": None}, {"t": "a"}], "t", allowed=["a"]).passed


def test_required_keys_on_attribute_rows():
    class Row:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getitem__(self, k):
            return getattr(self, k)

    rows = [Row(id=1, name="x"), Row(id=2)]
    f = required_keys(rows, keys=["id", "name"])
    assert not f.passed and "name" in f.context["missing_keys"]


def test_checks_work_on_dict_subclass_rows():
    class Series(dict):
        pass

    rows = [Series(email="a@x.com"), Series(email="")]
    f = null_rate(rows, "email", max_rate=0.4)  # 1/2 = 50% > 40%
    assert not f.passed


def test_severity_parameter_overrides_default():
    assert row_count([], min_rows=1, severity="WARNING").severity is Severity.WARNING


def test_date_range_empty_rows_passes_vacuously():
    assert date_range([], "d", "2026-01-01", "2026-01-31").passed


def test_date_range_all_null_column_passes_vacuously():
    rows = [{"d": None}, {"d": float("nan")}, {"d": ""}]
    assert date_range(rows, "d", "2026-01-01", "2026-01-31").passed


def test_date_range_boundaries_inclusive():
    rows = [{"d": "2026-01-01"}, {"d": "2026-01-31"}]
    assert date_range(rows, "d", "2026-01-01", "2026-01-31").passed
    assert not date_range([{"d": "2025-12-31"}], "d", "2026-01-01", "2026-01-31").passed
    assert not date_range([{"d": "2026-02-01"}], "d", "2026-01-01", "2026-01-31").passed


def test_date_range_naive_versus_aware_datetime_mix():
    aware_min = datetime(2026, 1, 1, tzinfo=timezone.utc)
    aware_max = datetime(2026, 1, 31, tzinfo=timezone.utc)
    naive_rows = [{"d": datetime(2026, 1, 15)}]
    f = date_range(naive_rows, "d", aware_min, aware_max)
    assert not f.passed
    assert f.context["violation_count"] == 1

    # reverse: naive bounds with aware row
    aware_rows = [{"d": datetime(2026, 1, 15, tzinfo=timezone.utc)}]
    f2 = date_range(aware_rows, "d", datetime(2026, 1, 1), datetime(2026, 1, 31))
    assert not f2.passed
    assert f2.context["violation_count"] == 1


def test_date_range_non_date_value_is_a_violation():
    f = date_range([{"d": "not-a-date"}, {"d": 12345}], "d", "2026-01-01", "2026-01-31")
    assert not f.passed
    assert f.context["violation_count"] == 2


def test_date_range_invalid_boundary_fails_cleanly():
    f = date_range([{"d": "2026-01-15"}], "d", min_date="invalid", max_date="2026-01-31")
    assert not f.passed
    assert "invalid date_range window" in f.message
    assert "violation_count" not in f.context


def test_date_range_rejects_version_dependent_formats():
    # date.fromisoformat and datetime.fromisoformat accept these on 3.11+ but not on 3.9/3.10.
    # Recusal parses against an explicit grammar so all Pythons agree.
    for bad in ["20260615", "2026-06-15T10:00:00.5", "2026-W24-1"]:
        f = date_range([{"d": bad}], "d", "2026-01-01", "2026-12-31")
        assert not f.passed
        assert f.context["violation_count"] == 1


def test_date_range_mixed_boundary_kinds_fails_as_invalid_window():
    f = date_range(
        [{"d": "2026-06-01"}], "d", min_date="2026-01-01", max_date="2026-12-31T23:59:59"
    )
    assert not f.passed
    assert "invalid date_range window" in f.message
    assert "mixed boundary kinds" in f.message
    assert "violation_count" not in f.context


def test_date_range_inverted_boundaries_fails_as_invalid_window():
    f = date_range([{"d": "2026-06-01"}], "d", min_date="2026-12-31", max_date="2026-01-01")
    assert not f.passed
    assert "invalid date_range window" in f.message
    assert "is after max_date" in f.message
    assert "violation_count" not in f.context


def test_date_range_date_and_datetime_objects():
    rows = [{"d": date(2026, 1, 15)}, {"d": datetime(2026, 1, 20, 10, 30)}]
    assert date_range(rows, "d", date(2026, 1, 1), date(2026, 1, 31)).passed


def test_date_range_grammar_is_exact_ascii_with_no_trailing_characters():
    # the whole string must be the grammar: no trailing newline, no non-ASCII digits
    for value in ("2026-01-15\n", "2026-01-15T10:00:00\n", "\u0662\u0660\u0662\u0666-01-15"):
        f = date_range([{"d": value}], "d", "2026-01-01", "2026-01-31")
        assert not f.passed and f.context["violation_count"] == 1, repr(value)
    boundary = date_range([{"d": "2026-01-15"}], "d", "2026-01-01\n", "2026-01-31")
    assert not boundary.passed and "invalid date_range window" in boundary.message

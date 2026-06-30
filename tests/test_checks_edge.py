"""Heavy edge-case tests for the built-in checks — nulls, NaN, boundaries, row shapes."""

from recusal import Severity
from recusal.checks import (
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

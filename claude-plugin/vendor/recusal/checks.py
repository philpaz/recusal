"""
Built-in deterministic checks, turn data into evidence.

The verdict kernel (``compute_verdict``) decides; these functions do the tedious
part that earns the decision: inspect real data and emit findings. Each check
takes a materialized sequence of dict-like rows (a ``list`` of dicts, JSON records,
or pandas rows, anything with ``[]`` access and a length). Iterators such as a
generator or ``csv.DictReader`` must be wrapped first: ``list(csv.DictReader(f))``.
Each check returns a single ``Finding`` ready to hand to ``compute_verdict``.

Severity is a parameter, not a hardcode, *you* decide whether a high null rate
is a CRITICAL stop or a WARNING. Pure logic, standard library only, no pandas
required.

Typical use::

    from recusal import compute_verdict
    from recusal.checks import date_range, null_rate, referential_integrity, row_count

    findings = [
        row_count(users, min_rows=1),
        null_rate(users, "email", max_rate=0.10),
        referential_integrity(orders, users, fk="user_id", pk="id"),
        date_range(orders, "created_at", min_date="2026-01-01", max_date="2026-12-31"),
    ]
    verdict = compute_verdict(findings)   # PASS / RETRY / FAIL

"""

from datetime import date, datetime
from typing import Any, Iterable, Sequence

from .evidence import Finding, RuleSeverity

Rows = Sequence[Any]  # each row supports row["column"]


def _passed(check_type: str, severity: str, message: str, **context: Any) -> Finding:
    return Finding.ok(check_type, severity=severity, message=message, **context)


def _failed(check_type: str, severity: str, message: str, **context: Any) -> Finding:
    return Finding.fail(check_type, severity=severity, message=message, **context)


def row_count(
    rows: Rows,
    min_rows: int = 1,
    severity: str = RuleSeverity.CRITICAL.value,
    label: str = "dataset",
) -> Finding:
    """Fail if there are fewer than ``min_rows`` rows (empty data usually means
    generation silently failed)."""
    n = len(rows)
    if n < min_rows:
        return _failed(
            "row_count",
            severity,
            f"{label}: {n} rows < required {min_rows}.",
            actual=n,
            min_rows=min_rows,
            label=label,
        )
    return _passed("row_count", severity, f"{label}: {n} rows.", actual=n, label=label)


def null_rate(
    rows: Rows,
    column: str,
    max_rate: float = 0.15,
    severity: str = RuleSeverity.ERROR.value,
) -> Finding:
    """Fail if the fraction of null/empty values in ``column`` exceeds ``max_rate``."""
    total = len(rows)
    if total == 0:
        return _passed("null_rate", severity, f"{column}: no rows to check.", column=column)
    nulls = sum(1 for r in rows if _is_null(_get(r, column)))
    rate = nulls / total
    if rate > max_rate:
        return _failed(
            "null_rate",
            severity,
            f"{column}: null rate {rate:.1%} > max {max_rate:.1%} ({nulls}/{total}).",
            column=column,
            null_rate=rate,
            max_rate=max_rate,
        )
    return _passed(
        "null_rate",
        severity,
        f"{column}: null rate {rate:.1%} within {max_rate:.1%}.",
        column=column,
        null_rate=rate,
    )


def referential_integrity(
    child_rows: Rows,
    parent_rows: Rows,
    fk: str,
    pk: str,
    severity: str = RuleSeverity.CRITICAL.value,
) -> Finding:
    """Fail if any child row's foreign key has no matching parent primary key
    (orphans = broken relationships)."""
    parent_keys = {_get(r, pk) for r in parent_rows}
    orphans = [
        _get(r, fk)
        for r in child_rows
        if not _is_null(_get(r, fk)) and _get(r, fk) not in parent_keys
    ]
    if orphans:
        sample = ", ".join(str(o) for o in orphans[:5])
        return _failed(
            "referential_integrity",
            severity,
            f"{len(orphans)} orphan(s) in {fk} -> {pk} (e.g. {sample}).",
            fk=fk,
            pk=pk,
            orphan_count=len(orphans),
        )
    return _passed(
        "referential_integrity",
        severity,
        f"{fk} -> {pk}: no orphans.",
        fk=fk,
        pk=pk,
        orphan_count=0,
    )


def in_set(
    rows: Rows,
    column: str,
    allowed: Iterable[Any],
    severity: str = RuleSeverity.ERROR.value,
) -> Finding:
    """Fail if any non-null value in ``column`` is outside the ``allowed`` set."""
    allowed_set = set(allowed)
    bad = [
        _get(r, column)
        for r in rows
        if not _is_null(_get(r, column)) and _get(r, column) not in allowed_set
    ]
    if bad:
        sample = ", ".join(str(b) for b in sorted(set(map(str, bad)))[:5])
        return _failed(
            "in_set",
            severity,
            f"{column}: {len(bad)} value(s) not in approved set (e.g. {sample}).",
            column=column,
            violation_count=len(bad),
        )
    return _passed("in_set", severity, f"{column}: all values in approved set.", column=column)


def in_range(
    rows: Rows,
    column: str,
    min_value: float,
    max_value: float,
    severity: str = RuleSeverity.ERROR.value,
) -> Finding:
    """Fail if any numeric value in ``column`` falls outside [min_value, max_value]."""
    violations = 0
    for r in rows:
        v = _get(r, column)
        if _is_null(v):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            violations += 1
            continue
        if fv < min_value or fv > max_value:
            violations += 1
    if violations:
        return _failed(
            "in_range",
            severity,
            f"{column}: {violations} value(s) outside [{min_value}, {max_value}].",
            column=column,
            violation_count=violations,
        )
    return _passed(
        "in_range",
        severity,
        f"{column}: all values within [{min_value}, {max_value}].",
        column=column,
    )


def date_range(
    rows: Rows,
    column: str,
    min_date: Any,
    max_date: Any,
    severity: str = RuleSeverity.ERROR.value,
) -> Finding:
    """Fail if any date or datetime in ``column`` falls outside [min_date, max_date].

    Values and boundaries may be ``datetime.date``, ``datetime.datetime``, or
    ISO-formatted strings. Null and empty values are ignored (use ``null_rate``
    to enforce presence). Values that cannot be parsed as dates, or that trigger
    offset-naive versus offset-aware comparison mismatches, are counted as
    violations.

    What this check does NOT establish:
    - It does not establish that dates are monotonically ordered or sorted.
    - It does not establish that there are no gaps or missing days in the series.
    - It does not verify business day / holiday calendar validity.
    - It does not establish timezone semantic correctness or daylight-saving integrity.
    - It does not establish non-nullness or dataset completeness.
    """
    try:
        min_d = _parse_date(min_date)
        max_d = _parse_date(max_date)
    except (TypeError, ValueError) as err:
        return _failed(
            "date_range",
            severity,
            f"{column}: invalid date_range boundary: {err}.",
            column=column,
            min_date=str(min_date),
            max_date=str(max_date),
            violation_count=len(rows),
        )

    violations = 0
    for r in rows:
        v = _get(r, column)
        if _is_null(v):
            continue
        try:
            dv = _parse_date(v)
            if isinstance(dv, datetime) and type(min_d) is date and type(max_d) is date:
                dv = dv.date()
            elif type(dv) is date and isinstance(min_d, datetime) and isinstance(max_d, datetime):
                if min_d.tzinfo is None and max_d.tzinfo is None:
                    dv = datetime.combine(dv, datetime.min.time())
            if dv < min_d or dv > max_d:
                violations += 1
        except (TypeError, ValueError):
            violations += 1
            continue

    if violations:
        return _failed(
            "date_range",
            severity,
            f"{column}: {violations} value(s) outside [{min_date}, {max_date}].",
            column=column,
            violation_count=violations,
        )
    return _passed(
        "date_range",
        severity,
        f"{column}: all values within [{min_date}, {max_date}].",
        column=column,
    )


def required_keys(
    rows: Rows,
    keys: Iterable[str],
    severity: str = RuleSeverity.CRITICAL.value,
) -> Finding:
    """Fail if any row is missing one of the required ``keys`` (schema drift)."""
    required = list(keys)
    missing_rows = 0
    seen_missing: set = set()
    for r in rows:
        absent = [k for k in required if not _has(r, k)]
        if absent:
            missing_rows += 1
            seen_missing.update(absent)
    if missing_rows:
        cols = ", ".join(sorted(seen_missing))
        return _failed(
            "required_keys",
            severity,
            f"{missing_rows} row(s) missing required key(s): {cols}.",
            missing_keys=sorted(seen_missing),
            affected_rows=missing_rows,
        )
    return _passed(
        "required_keys", severity, f"all rows have required keys: {', '.join(required)}."
    )


# ── tiny accessors that work for dicts and pandas-like rows ────────────────────


def _get(row: Any, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return getattr(row, key, None)


def _has(row: Any, key: str) -> bool:
    try:
        row[key]
        return True
    except (KeyError, IndexError, TypeError, AttributeError):
        return hasattr(row, key)


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    # NaN is the only value not equal to itself.
    return value != value


def _parse_date(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("empty date string")
        if s.endswith("Z") or s.endswith("z"):
            s = s[:-1] + "+00:00"
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            try:
                return date.fromisoformat(s)
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            pass
        try:
            return date.fromisoformat(s)
        except ValueError:
            pass
        raise ValueError(f"unrecognized date format: {value!r}")
    raise TypeError(f"expected date, datetime, or ISO string, got {type(value).__name__}")

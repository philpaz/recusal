"""
Built-in deterministic checks — turn data into evidence.

The verdict kernel (``compute_verdict``) decides; these functions do the tedious
part that earns the decision: inspect real data and emit findings. Each check
takes plain Python (a list of dict-like rows — anything with ``[]`` access, so
``csv.DictReader`` rows, JSON records, or pandas rows all work) and returns a
single **finding** dict ready to hand to ``compute_verdict``:

    {"type": ..., "severity": "CRITICAL"|"ERROR"|"WARNING"|"INFO",
     "status": "pass"|"fail", "message": ..., ...context...}

Severity is a parameter, not a hardcode — *you* decide whether a high null rate
is a CRITICAL stop or a WARNING. Pure logic, standard library only, no pandas
required.

Typical use::

    from recusal import compute_verdict
    from recusal.checks import row_count, null_rate, referential_integrity

    findings = [
        row_count(members, min_rows=1),
        null_rate(members, "email", max_rate=0.10),
        referential_integrity(accounts, members, fk="member_id", pk="id"),
    ]
    verdict = compute_verdict(findings)   # PASS / RETRY / FAIL
"""

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
) -> dict:
    """Fail if there are fewer than ``min_rows`` rows (empty data usually means
    generation silently failed)."""
    n = len(rows)
    if n < min_rows:
        return _failed(
            "row_count", severity,
            f"{label}: {n} rows < required {min_rows}.",
            actual=n, min_rows=min_rows, label=label,
        )
    return _passed("row_count", severity, f"{label}: {n} rows.", actual=n, label=label)


def null_rate(
    rows: Rows,
    column: str,
    max_rate: float = 0.15,
    severity: str = RuleSeverity.ERROR.value,
) -> dict:
    """Fail if the fraction of null/empty values in ``column`` exceeds ``max_rate``."""
    total = len(rows)
    if total == 0:
        return _passed("null_rate", severity, f"{column}: no rows to check.", column=column)
    nulls = sum(1 for r in rows if _is_null(_get(r, column)))
    rate = nulls / total
    if rate > max_rate:
        return _failed(
            "null_rate", severity,
            f"{column}: null rate {rate:.1%} > max {max_rate:.1%} ({nulls}/{total}).",
            column=column, null_rate=rate, max_rate=max_rate,
        )
    return _passed(
        "null_rate", severity,
        f"{column}: null rate {rate:.1%} within {max_rate:.1%}.",
        column=column, null_rate=rate,
    )


def referential_integrity(
    child_rows: Rows,
    parent_rows: Rows,
    fk: str,
    pk: str,
    severity: str = RuleSeverity.CRITICAL.value,
) -> dict:
    """Fail if any child row's foreign key has no matching parent primary key
    (orphans = broken relationships)."""
    parent_keys = {_get(r, pk) for r in parent_rows}
    orphans = [
        _get(r, fk) for r in child_rows
        if not _is_null(_get(r, fk)) and _get(r, fk) not in parent_keys
    ]
    if orphans:
        sample = ", ".join(str(o) for o in orphans[:5])
        return _failed(
            "referential_integrity", severity,
            f"{len(orphans)} orphan(s) in {fk} -> {pk} (e.g. {sample}).",
            fk=fk, pk=pk, orphan_count=len(orphans),
        )
    return _passed(
        "referential_integrity", severity,
        f"{fk} -> {pk}: no orphans.", fk=fk, pk=pk, orphan_count=0,
    )


def in_set(
    rows: Rows,
    column: str,
    allowed: Iterable[Any],
    severity: str = RuleSeverity.ERROR.value,
) -> dict:
    """Fail if any non-null value in ``column`` is outside the ``allowed`` set."""
    allowed_set = set(allowed)
    bad = [
        _get(r, column) for r in rows
        if not _is_null(_get(r, column)) and _get(r, column) not in allowed_set
    ]
    if bad:
        sample = ", ".join(str(b) for b in sorted(set(map(str, bad)))[:5])
        return _failed(
            "in_set", severity,
            f"{column}: {len(bad)} value(s) not in approved set (e.g. {sample}).",
            column=column, violation_count=len(bad),
        )
    return _passed("in_set", severity, f"{column}: all values in approved set.", column=column)


def in_range(
    rows: Rows,
    column: str,
    min_value: float,
    max_value: float,
    severity: str = RuleSeverity.ERROR.value,
) -> dict:
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
            "in_range", severity,
            f"{column}: {violations} value(s) outside [{min_value}, {max_value}].",
            column=column, violation_count=violations,
        )
    return _passed(
        "in_range", severity,
        f"{column}: all values within [{min_value}, {max_value}].", column=column,
    )


def required_keys(
    rows: Rows,
    keys: Iterable[str],
    severity: str = RuleSeverity.CRITICAL.value,
) -> dict:
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
            "required_keys", severity,
            f"{missing_rows} row(s) missing required key(s): {cols}.",
            missing_keys=sorted(seen_missing), affected_rows=missing_rows,
        )
    return _passed("required_keys", severity, f"all rows have required keys: {', '.join(required)}.")


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

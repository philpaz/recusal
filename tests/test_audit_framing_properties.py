"""Generated file endings preserve audit records or refuse without changing bytes."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from recusal import AuditFramingError, AuditLog, compute_verdict

DETERMINISTIC = settings(derandomize=True, max_examples=150)
VERDICT = compute_verdict([])


def _record(label):
    entry = AuditLog().append(VERDICT, action={"label": label}, timestamp="2026-01-01T00:00:00")
    return json.dumps(entry, ensure_ascii=True).encode("utf-8")


def _parsed_lines(raw):
    parsed = []
    for line in raw.split(b"\n"):
        try:
            parsed.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            pass
    return parsed


def _assert_append_boundary(before, resume, *, must_succeed=False):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "audit.jsonl"
        path.write_bytes(before)
        log = AuditLog(str(path), resume=resume)
        try:
            appended = log.append(VERDICT, action={"label": "new"})
        except AuditFramingError:
            assert not must_succeed
            assert path.read_bytes() == before
            return

        after = path.read_bytes()
        assert after.startswith(before)
        assert after.endswith(b"\n")
        separator = os.linesep.encode("ascii") if before and not before.endswith(b"\n") else b""
        added = after[len(before) :]
        assert added.startswith(separator)
        new_line = added[len(separator) :]
        assert len(new_line.splitlines()) == 1
        assert json.loads(new_line.decode("utf-8")) == appended
        assert _parsed_lines(after) == _parsed_lines(before) + [appended]


@DETERMINISTIC
@example(tail=b"\xff", terminated=False, crlf=False, resume="tail")
@example(tail=b" \t\r", terminated=False, crlf=True, resume="full")
@example(tail=b"{" + b"x" * 65536, terminated=False, crlf=False, resume="tail")
@example(tail=b"x" * 65537, terminated=True, crlf=True, resume="full")
@given(
    tail=st.binary(max_size=1024),
    terminated=st.booleans(),
    crlf=st.booleans(),
    resume=st.sampled_from(["full", "tail"]),
)
def test_arbitrary_endings_never_fuse_or_rewrite_existing_records(tail, terminated, crlf, resume):
    newline = b"\r\n" if crlf else b"\n"
    before = _record("existing") + newline + tail + (newline if terminated else b"")
    _assert_append_boundary(before, resume)


@DETERMINISTIC
@given(
    label=st.text(max_size=100),
    newline=st.sampled_from([b"", b"\n", b"\r\n"]),
    resume=st.sampled_from(["full", "tail"]),
)
def test_complete_records_with_any_terminator_remain_readable(label, newline, resume):
    _assert_append_boundary(_record(label) + newline, resume, must_succeed=True)


@st.composite
def truncated_records(draw):
    record = _record(draw(st.text(max_size=100)))
    # The empty prefix is a valid empty file, not an interrupted record.
    offset = draw(st.integers(min_value=1, max_value=len(record) - 1))
    return record[:offset]


@DETERMINISTIC
@given(fragment=truncated_records(), resume=st.sampled_from(["full", "tail"]))
def test_every_nonempty_partial_record_refuses_without_a_write(fragment, resume):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "audit.jsonl"
        before = _record("existing") + b"\n" + fragment
        path.write_bytes(before)
        log = AuditLog(str(path), resume=resume)
        with pytest.raises(AuditFramingError) as raised:
            log.append(VERDICT)
        assert path.read_bytes() == before
        assert raised.value.offset == len(before) - len(fragment)
        assert raised.value.length == len(fragment)


@DETERMINISTIC
@given(resume=st.sampled_from(["full", "tail"]))
def test_complete_record_larger_than_the_tail_window_is_terminated(resume):
    _assert_append_boundary(_record("x" * 65537), resume, must_succeed=True)


@DETERMINISTIC
@given(resume=st.sampled_from(["full", "tail"]))
def test_empty_file_appends_without_a_spurious_separator(resume):
    _assert_append_boundary(b"", resume, must_succeed=True)

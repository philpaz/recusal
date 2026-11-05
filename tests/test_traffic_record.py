"""tools/traffic_record.py keeps a per-day traffic record past GitHub's 14-day window.

The pure functions are tested directly. The capture is tested through a fake request
function that plays the GitHub API: no network, no token.
"""

import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tr():
    spec = importlib.util.spec_from_file_location(
        "traffic_record", ROOT / "tools" / "traffic_record.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(kind, rows):
    return {
        "count": sum(c for _, c, _ in rows),
        "uniques": sum(u for _, _, u in rows),
        kind: [{"timestamp": f"{d}T00:00:00Z", "count": c, "uniques": u} for d, c, u in rows],
    }


# --- pure ----------------------------------------------------------------------------------


def test_csv_round_trips_and_sorts(tr):
    table = {"2026-11-05": (3, 2), "2026-11-03": (10, 4)}
    text = tr.render(table)
    assert text == "date,count,uniques\n2026-11-03,10,4\n2026-11-05,3,2\n"
    assert tr.parse_csv(text) == table
    assert tr.parse_csv("") == {} and tr.parse_csv("date,count,uniques\n") == {}


def test_the_window_wins_for_the_days_it_covers_and_older_days_are_kept(tr):
    existing = {"2026-10-20": (7, 7), "2026-11-04": (100, 9)}  # 11-04 captured while partial
    window = tr.window_rows(
        _payload("clones", [("2026-11-04", 240, 12), ("2026-11-05", 5, 1)]), "clones"
    )
    merged = tr.merge(existing, window)
    assert merged == {"2026-10-20": (7, 7), "2026-11-04": (240, 12), "2026-11-05": (5, 1)}
    assert existing == {"2026-10-20": (7, 7), "2026-11-04": (100, 9)}  # merge copies


def test_summary_states_totals_and_calls_summed_uniques_an_upper_bound(tr):
    text = tr.summary({"clones": {"2026-11-04": (240, 12), "2026-11-05": (5, 1)}, "views": {}})
    assert "**clones**: 245 over 2 days, 2026-11-04 to 2026-11-05" in text
    assert "sum of daily uniques 13 (an upper bound" in text
    assert "**views**: no days on record" in text


# --- capture through a fake API --------------------------------------------------------------


class FakeGitHub:
    def __init__(self, clones, views, files=None):
        self.windows = {"clones": _payload("clones", clones), "views": _payload("views", views)}
        self.files = dict(files or {})  # path -> text
        self.puts = []

    def __call__(self, method, path, body=None):
        import base64

        if method == "GET" and "/traffic/" in path:
            return 200, self.windows[path.rsplit("/", 1)[1]]
        if method == "GET" and "/contents/" in path:
            name = path.split("/contents/", 1)[1].split("?", 1)[0]
            if name not in self.files:
                return 404, {"message": "Not Found"}
            return 200, {
                "content": base64.b64encode(self.files[name].encode()).decode(),
                "sha": f"sha-{name}",
            }
        if method == "PUT" and "/contents/" in path:
            name = path.split("/contents/", 1)[1]
            assert body["branch"] == "traffic"
            if name in self.files:
                assert body["sha"] == f"sha-{name}", "an update must carry the blob sha"
            else:
                assert "sha" not in body, "a creation carries no sha"
            self.files[name] = base64.b64decode(body["content"]).decode()
            self.puts.append((name, body["message"]))
            return (200 if name in self.files else 201), {}
        raise AssertionError(f"unexpected call {method} {path}")


def test_first_capture_creates_both_csvs_and_the_summary(tr):
    gh = FakeGitHub(
        clones=[("2026-11-04", 240, 12), ("2026-11-05", 5, 1)], views=[("2026-11-05", 3, 2)]
    )
    out = io.StringIO()
    written = tr.capture(gh, "o/r", "traffic", dry_run=False, out=out)
    assert written == 3
    assert gh.files["clones.csv"] == "date,count,uniques\n2026-11-04,240,12\n2026-11-05,5,1\n"
    assert gh.files["views.csv"] == "date,count,uniques\n2026-11-05,3,2\n"
    assert "245 over 2 days" in gh.files["SUMMARY.md"]
    assert [m for _, m in gh.puts] == [
        "traffic: clones through 2026-11-05",
        "traffic: views through 2026-11-05",
        "traffic: totals through 2026-11-05",
    ]


def test_a_later_capture_keeps_days_that_left_the_window_and_revises_recent_ones(tr):
    gh = FakeGitHub(
        clones=[("2026-11-05", 300, 20)],  # 11-04 has left the window; 11-05 is now complete
        views=[("2026-11-05", 3, 2)],
        files={
            "clones.csv": "date,count,uniques\n2026-11-04,240,12\n2026-11-05,5,1\n",
            "views.csv": "date,count,uniques\n2026-11-05,3,2\n",
            "SUMMARY.md": "old",
        },
    )
    written = tr.capture(gh, "o/r", "traffic", dry_run=False, out=io.StringIO())
    assert written == 2  # clones changed, views did not, summary rewritten
    assert gh.files["clones.csv"] == "date,count,uniques\n2026-11-04,240,12\n2026-11-05,300,20\n"
    assert [n for n, _ in gh.puts] == ["clones.csv", "SUMMARY.md"]


def test_an_unchanged_window_writes_nothing(tr):
    gh = FakeGitHub(
        clones=[("2026-11-05", 5, 1)],
        views=[("2026-11-05", 3, 2)],
        files={
            "clones.csv": "date,count,uniques\n2026-11-05,5,1\n",
            "views.csv": "date,count,uniques\n2026-11-05,3,2\n",
        },
    )
    assert tr.capture(gh, "o/r", "traffic", dry_run=False, out=io.StringIO()) == 0
    assert gh.puts == []


def test_a_dry_run_prints_the_files_and_writes_nothing(tr):
    gh = FakeGitHub(clones=[("2026-11-05", 5, 1)], views=[])
    out = io.StringIO()
    assert tr.capture(gh, "o/r", "traffic", dry_run=True, out=out) == 0
    assert "clones.csv (dry run)" in out.getvalue() and gh.puts == []


def test_a_forbidden_traffic_read_names_the_token_requirement(tr):
    def forbidden(method, path, body=None):
        return 403, {"message": "Resource not accessible by integration"}

    with pytest.raises(RuntimeError, match="push access"):
        tr.capture(forbidden, "o/r", "traffic", dry_run=True, out=io.StringIO())


def test_main_refuses_without_a_token(tr, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert tr.main(["--repo", "o/r"]) == 2
    assert "GITHUB_TOKEN" in capsys.readouterr().err

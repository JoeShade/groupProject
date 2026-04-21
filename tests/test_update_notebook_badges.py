"""Unit tests for the notebook badge refresh helper.

These tests cover the git-hours summary and rendered badge HTML.
They focus on the helper in isolation and avoid touching blankTemplate.ipynb.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_SCRIPT = REPO_ROOT / "scripts" / "update_notebook_badges.py"


def load_helper_module():
    spec = importlib.util.spec_from_file_location("update_notebook_badges", HELPER_SCRIPT)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_hours_value_falls_back_to_zero_when_git_history_is_unavailable(
    monkeypatch,
) -> None:
    helper_module = load_helper_module()

    def raise_git_error(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="git log")

    monkeypatch.setattr(helper_module.subprocess, "check_output", raise_git_error)

    assert helper_module.compute_hours_value(REPO_ROOT) == "0h 00m"


def test_compute_hours_value_uses_first_to_last_commit_window_per_day(monkeypatch) -> None:
    helper_module = load_helper_module()

    raw_history = "\n".join(
        [
            "2026-04-21T18:00:00+00:00",
            "2026-04-21T14:30:00+00:00",
            "2026-04-21T09:00:00+00:00",
            "2026-04-20T11:15:00+00:00",
            "2026-04-20T10:00:00+00:00",
        ]
    )

    monkeypatch.setattr(
        helper_module.subprocess,
        "check_output",
        lambda *args, **kwargs: raw_history,
    )

    assert helper_module.compute_hours_value(REPO_ROOT) == "10h 15m"


def test_render_badges_includes_hours_badge_and_member_links(monkeypatch) -> None:
    helper_module = load_helper_module()
    monkeypatch.setattr(helper_module, "compute_hours_value", lambda repo_root: "3h 15m")

    badge_markup = helper_module.render_badges(REPO_ROOT)

    assert 'alt="Hours"' in badge_markup
    assert "3h%2015m" in badge_markup
    assert "https://github.com/JoeShade" in badge_markup
    assert "https://github.com/SimonAndreou" in badge_markup

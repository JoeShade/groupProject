from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote


MARKER = "<!-- generated-team-badges: run `python scripts/update_notebook_badges.py` to refresh -->"
NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "blankTemplate.ipynb"
MEMBERS = [
    ("James Barlow", "https://github.com/jamesbarlow1812", "7C3AED"),
    ("Joe Shade", "https://github.com/JoeShade", "D97706"),
    ("Lena Kraemer", "https://github.com/Nomo2001", "059669"),
    ("Simon Andreou", "https://github.com/SimonAndreou", "DC2626"),
]


def badge_url(label: str, message: str, color: str, label_color: str, logo: str) -> str:
    return (
        f"https://img.shields.io/badge/{quote(label)}-{quote(message)}-{color}"
        f"?style=flat-square&logo={logo}&logoColor=white&labelColor={label_color}"
    )


def compute_hours_value(repo_root: Path) -> str:
    try:
        raw = subprocess.check_output(
            ["git", "log", "--date=iso-strict", "--pretty=format:%cI"],
            cwd=repo_root,
            text=True,
        )
    except Exception:
        return "0h 00m"

    by_day: dict[datetime.date, list[datetime]] = defaultdict(list)
    for line in raw.splitlines():
        if not line.strip():
            continue
        stamp = datetime.fromisoformat(line.strip())
        by_day[stamp.date()].append(stamp)

    total = timedelta()
    for times in by_day.values():
        total += max(times) - min(times)

    total_minutes = int(total.total_seconds() // 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes:02d}m"


def render_badges(repo_root: Path) -> str:
    hours_value = compute_hours_value(repo_root)
    badges = [
        f'<img alt="Hours" src="{badge_url("Hours", hours_value, "4A4A4A", "1F6FEB", "git")}" />'
    ]
    for name, link, color in MEMBERS:
        badges.append(
            f'<a href="{link}"><img alt="{name}" src="{badge_url(" ", name, "4A4A4A", color, "github")}" /></a>'
        )
    return f'<div align="center">{" ".join(badges)}</div>'


def update_notebook(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    badge_cell = None
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if MARKER in source:
            badge_cell = cell
            break

    if badge_cell is None:
        raise RuntimeError(f"Could not find badge marker in {notebook_path.name}")

    badge_cell["source"] = [
        f"{MARKER}\n",
        f"{render_badges(notebook_path.parent)}\n",
    ]

    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    update_notebook(NOTEBOOK_PATH)

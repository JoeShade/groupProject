"""Export notebook figures and tables as PNG assets.

This helper reads saved notebook outputs rather than executing notebook cells.
It extracts embedded PNG figures and renders HTML tables into PNG files for use
in the README, slides, or other presentation material.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import textwrap
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK_PATH = REPO_ROOT / "LungCancerML.ipynb"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "notebook_exports"

DATA_IMAGE_PATTERN = re.compile(
    r"<img\b[^>]*\bsrc=[\"']data:image/png;base64,(?P<data>[^\"']+)[\"'][^>]*>",
    flags=re.IGNORECASE | re.DOTALL,
)
TABLE_PATTERN = re.compile(r"<table\b.*?</table>", flags=re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
LABEL_PATTERN_TEMPLATE = r"\b({kind}\s+\d+[\.,]?\s*.*?)(?=\s+(?:Figure|Table)\s+\d+\b|$)"


@dataclass(frozen=True)
class NotebookAsset:
    kind: str
    cell_index: int
    ordinal: int
    label: str
    caption: str
    image_data: bytes | None = None
    table_rows: tuple[tuple[str, ...], ...] | None = None


class HTMLTableParser(HTMLParser):
    """Minimal HTML table parser for notebook-generated tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            return
        if not self._in_table:
            return
        if tag == "tr":
            self._in_row = True
            self._current_row = []
            return
        if tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell = []
            return
        if tag == "br" and self._in_cell:
            self._current_cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._in_cell:
            cell_text = normalise_whitespace("".join(self._current_cell))
            self._current_row.append(cell_text)
            self._current_cell = []
            self._in_cell = False
            return
        if tag == "tr" and self._in_row:
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = []
            self._in_row = False
            return
        if tag == "table":
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)


def load_notebook(notebook_path: Path) -> dict:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict):
        raise ValueError("Notebook root must be a JSON object.")
    if not isinstance(notebook.get("cells"), list):
        raise ValueError("Notebook JSON must contain a 'cells' list.")
    return notebook


def normalise_source(source: object) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        if not all(isinstance(line, str) for line in source):
            raise TypeError("Notebook source lines must be strings.")
        return "".join(source)
    raise TypeError("Notebook source must be a string or list of strings.")


def normalise_output_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not all(isinstance(line, str) for line in value):
            raise TypeError("Notebook output value lines must be strings.")
        return "".join(value)
    raise TypeError("Notebook output value must be a string or list of strings.")


def normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def strip_html(value: str) -> str:
    return normalise_whitespace(TAG_PATTERN.sub(" ", value))


def find_label_caption(text: str, kind: str, fallback: str) -> tuple[str, str]:
    pattern = LABEL_PATTERN_TEMPLATE.format(kind=kind)
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return fallback, fallback

    caption = normalise_whitespace(match.group(1))
    caption = caption[:220].rstrip(" ,;:-")

    label_match = re.match(rf"({kind}\s+\d+)", caption, flags=re.IGNORECASE)
    label = label_match.group(1).title() if label_match else fallback
    return label, caption


def parse_table_rows(table_html: str) -> tuple[tuple[str, ...], ...]:
    parser = HTMLTableParser()
    parser.feed(table_html)

    if not parser.rows:
        return tuple()

    width = max(len(row) for row in parser.rows)
    padded_rows = [row + [""] * (width - len(row)) for row in parser.rows]
    return tuple(tuple(cell for cell in row) for row in padded_rows)


def iter_html_tables(html: str, cell_index: int) -> Iterable[NotebookAsset]:
    for ordinal, match in enumerate(TABLE_PATTERN.finditer(html), start=1):
        table_html = match.group(0)
        rows = parse_table_rows(table_html)
        if not rows:
            continue

        context_end = next_context_boundary(html, match.end())
        context_text = strip_html(html[match.end() : context_end])
        fallback = f"Table cell {cell_index} item {ordinal}"
        label, caption = find_label_caption(context_text, "Table", fallback)

        yield NotebookAsset(
            kind="table",
            cell_index=cell_index,
            ordinal=ordinal,
            label=label,
            caption=caption,
            table_rows=rows,
        )


def iter_html_images(html: str, cell_index: int) -> Iterable[NotebookAsset]:
    for ordinal, match in enumerate(DATA_IMAGE_PATTERN.finditer(html), start=1):
        image_data = base64.b64decode(match.group("data"))
        context_end = next_context_boundary(html, match.end())
        context_text = strip_html(html[match.end() : context_end])
        fallback = f"Figure cell {cell_index} item {ordinal}"
        label, caption = find_label_caption(context_text, "Figure", fallback)

        yield NotebookAsset(
            kind="figure",
            cell_index=cell_index,
            ordinal=ordinal,
            label=label,
            caption=caption,
            image_data=image_data,
        )


def next_context_boundary(html: str, start_index: int) -> int:
    boundary_match = re.search(
        r"(<table\b|<img\b|data:image/png)",
        html[start_index:],
        flags=re.IGNORECASE,
    )
    if boundary_match:
        return start_index + boundary_match.start()
    return min(len(html), start_index + 1200)


def iter_output_html(output: dict) -> Iterable[str]:
    data = output.get("data")
    if not isinstance(data, dict):
        return
    html_value = data.get("text/html")
    if html_value is not None:
        yield normalise_output_value(html_value)


def iter_output_pngs(
    output: dict,
    cell_index: int,
    fallback_label: str | None = None,
    fallback_caption: str | None = None,
) -> Iterable[NotebookAsset]:
    data = output.get("data")
    if not isinstance(data, dict):
        return
    png_value = data.get("image/png")
    if png_value is None:
        return

    image_data = base64.b64decode(normalise_output_value(png_value))
    label = fallback_label or f"Figure cell {cell_index}"
    caption = fallback_caption or label
    yield NotebookAsset(
        kind="figure",
        cell_index=cell_index,
        ordinal=1,
        label=label,
        caption=caption,
        image_data=image_data,
    )


def following_markdown_caption(
    cells: list[object],
    current_index: int,
    kind: str,
    lookahead: int = 2,
) -> tuple[str, str] | None:
    for next_index in range(current_index + 1, min(len(cells), current_index + 1 + lookahead)):
        next_cell = cells[next_index]
        if not isinstance(next_cell, dict):
            continue
        if next_cell.get("cell_type") != "markdown":
            continue

        source = normalise_source(next_cell.get("source", []))
        text = strip_html(source)
        label, caption = find_label_caption(text, kind, fallback="")
        if label and caption:
            return label, caption

    return None


def collect_notebook_assets(notebook: dict) -> list[NotebookAsset]:
    assets: list[NotebookAsset] = []
    cells = notebook.get("cells", [])

    for zero_based_index, cell in enumerate(cells):
        cell_index = zero_based_index + 1
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            source = normalise_source(cell.get("source", []))
            assets.extend(iter_html_tables(source, cell_index))

        if cell_type != "code":
            continue

        for output in cell.get("outputs") or []:
            if not isinstance(output, dict):
                continue

            nearby_caption = following_markdown_caption(
                cells,
                zero_based_index,
                kind="Figure",
            )
            if nearby_caption is None:
                assets.extend(iter_output_pngs(output, cell_index))
            else:
                label, caption = nearby_caption
                assets.extend(
                    iter_output_pngs(
                        output,
                        cell_index,
                        fallback_label=label,
                        fallback_caption=caption,
                    )
                )
            for html in iter_output_html(output):
                assets.extend(iter_html_images(html, cell_index))
                assets.extend(iter_html_tables(html, cell_index))

    return assets


def slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        slug = fallback
    return slug[:90].strip("-")


def label_number(label: str) -> int | None:
    match = re.search(r"\d+", label)
    if not match:
        return None
    return int(match.group(0))


def filename_for_asset(asset: NotebookAsset, used_names: set[str]) -> str:
    number = label_number(asset.label)
    caption_slug = slugify(asset.caption, f"cell-{asset.cell_index}-{asset.ordinal}")

    if number is None:
        stem = f"{asset.kind}-cell-{asset.cell_index:03d}-{asset.ordinal:02d}-{caption_slug}"
    else:
        stem = f"{asset.kind}-{number:03d}-{caption_slug}"

    candidate = f"{stem}.png"
    suffix = 2
    while candidate in used_names:
        candidate = f"{stem}-{suffix}.png"
        suffix += 1

    used_names.add(candidate)
    return candidate


def wrapped_rows(
    rows: tuple[tuple[str, ...], ...],
    max_cell_chars: int,
) -> list[list[str]]:
    wrapped: list[list[str]] = []
    for row in rows:
        wrapped_row = [
            "\n".join(textwrap.wrap(cell, width=max_cell_chars)) if cell else ""
            for cell in row
        ]
        wrapped.append(wrapped_row)
    return wrapped


def render_table_png(
    rows: tuple[tuple[str, ...], ...],
    output_path: Path,
    caption: str,
    dpi: int,
    max_cell_chars: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("Cannot render an empty table.")

    wrapped = wrapped_rows(rows, max_cell_chars=max_cell_chars)
    headers = wrapped[0]
    body = wrapped[1:] if len(wrapped) > 1 else [[""] * len(headers)]

    column_count = max(1, len(headers))
    row_line_counts = [
        max(1, max((cell.count("\n") + 1 for cell in row), default=1))
        for row in wrapped
    ]
    wrapped_caption = textwrap.fill(caption, width=120) if caption else ""
    caption_line_count = max(1, wrapped_caption.count("\n") + 1) if wrapped_caption else 0
    fig_width = min(18.0, max(7.0, column_count * 1.85))
    table_height = sum(0.2 * count + 0.18 for count in row_line_counts)
    fig_height = min(36.0, max(1.8, table_height + 0.42 + caption_line_count * 0.2))

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    caption_space = 0.065 + caption_line_count * 0.02 if wrapped_caption else 0.03
    ax = fig.add_axes([0.01, caption_space + 0.015, 0.98, 0.98 - caption_space - 0.015])
    ax.axis("off")
    table = ax.table(
        cellText=body,
        colLabels=headers,
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D8D8")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#ECEAEA")
            cell.set_text_props(weight="bold", color="#000000")
        elif row_index % 2:
            cell.set_facecolor("#FBFBFB")
        else:
            cell.set_facecolor("#F4F2F2")

    if wrapped_caption:
        fig.text(
            0.5,
            0.018,
            wrapped_caption,
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
            color="#333333",
        )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def write_asset(asset: NotebookAsset, output_path: Path, dpi: int, max_cell_chars: int) -> None:
    if asset.kind == "figure":
        if asset.image_data is None:
            raise ValueError(f"Figure asset {asset.label!r} has no image data.")
        output_path.write_bytes(asset.image_data)
        return

    if asset.kind == "table":
        if asset.table_rows is None:
            raise ValueError(f"Table asset {asset.label!r} has no table rows.")
        render_table_png(
            asset.table_rows,
            output_path,
            caption=asset.caption,
            dpi=dpi,
            max_cell_chars=max_cell_chars,
        )
        return

    raise ValueError(f"Unsupported asset kind: {asset.kind}")


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.png"):
        path.unlink()
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        manifest_path.unlink()


def write_manifest(output_dir: Path, exported_rows: list[dict[str, str]]) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "label", "caption", "cell_index", "filename"],
        )
        writer.writeheader()
        writer.writerows(exported_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract saved notebook figures and tables as PNG assets."
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK_PATH,
        help=f"Notebook to read. Defaults to {DEFAULT_NOTEBOOK_PATH.name}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for exported PNG files. Defaults to "
            f"{DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT).as_posix()}."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing PNG files and manifest.csv from the output directory first.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI used when rendering table PNG files. Defaults to 200.",
    )
    parser.add_argument(
        "--max-cell-chars",
        type=int,
        default=26,
        help="Maximum characters per line when wrapping table cells. Defaults to 26.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook_path = args.notebook.resolve()
    output_dir = args.output_dir.resolve()

    if args.clean:
        clean_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    notebook = load_notebook(notebook_path)
    assets = collect_notebook_assets(notebook)

    if not assets:
        print(f"No figures or tables found in {notebook_path}.")
        return 0

    used_names: set[str] = set()
    exported_rows: list[dict[str, str]] = []

    for asset in assets:
        filename = filename_for_asset(asset, used_names)
        output_path = output_dir / filename
        write_asset(
            asset,
            output_path,
            dpi=args.dpi,
            max_cell_chars=args.max_cell_chars,
        )
        exported_rows.append(
            {
                "kind": asset.kind,
                "label": asset.label,
                "caption": asset.caption,
                "cell_index": str(asset.cell_index),
                "filename": filename,
            }
        )

    write_manifest(output_dir, exported_rows)
    print(
        f"Exported {len(exported_rows)} notebook assets "
        f"from {notebook_path} to {output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

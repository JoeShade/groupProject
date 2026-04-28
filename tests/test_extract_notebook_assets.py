"""Unit tests for the notebook asset extraction helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_SCRIPT = REPO_ROOT / "scripts" / "extract_notebook_assets.py"
ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
    "DwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def load_helper_module():
    spec = importlib.util.spec_from_file_location("extract_notebook_assets", HELPER_SCRIPT)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_notebook_assets_finds_markdown_tables_html_tables_and_images() -> None:
    extractor = load_helper_module()
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "<table><tr><th>Column</th><th>Meaning</th></tr>"
                    "<tr><td><code>A</code></td><td>Alpha</td></tr></table>"
                    "<p><em>Table 1, Markdown table caption</em></p>"
                ],
            },
            {
                "cell_type": "code",
                "outputs": [
                    {"data": {"image/png": ONE_PIXEL_PNG}},
                ],
            },
            {
                "cell_type": "markdown",
                "source": ["<p><em>Figure 1. Direct output graph.</em></p>"],
            },
            {
                "cell_type": "code",
                "outputs": [
                    {
                        "data": {
                            "text/html": [
                                f"<img src='data:image/png;base64,{ONE_PIXEL_PNG}'>"
                                "<p><em>Figure 2. Example graph.</em></p>"
                                "<table><tr><th>Metric</th><th>Value</th></tr>"
                                "<tr><td>Recall</td><td>0.95</td></tr></table>"
                                "<p><em>Table 3. Output table caption.</em></p>"
                            ]
                        }
                    }
                ],
            },
        ]
    }

    assets = extractor.collect_notebook_assets(notebook)

    assert [asset.kind for asset in assets] == ["table", "figure", "figure", "table"]
    assert assets[0].label == "Table 1"
    assert assets[0].table_rows == (("Column", "Meaning"), ("A", "Alpha"))
    assert assets[1].label == "Figure 1"
    assert assets[1].caption == "Figure 1. Direct output graph."
    assert assets[1].image_data is not None
    assert assets[2].label == "Figure 2"
    assert assets[2].caption == "Figure 2. Example graph."
    assert assets[2].image_data is not None
    assert assets[3].label == "Table 3"
    assert assets[3].table_rows == (("Metric", "Value"), ("Recall", "0.95"))


def test_filename_for_asset_deduplicates_repeated_labels() -> None:
    extractor = load_helper_module()
    used_names: set[str] = set()
    first = extractor.NotebookAsset(
        kind="figure",
        cell_index=1,
        ordinal=1,
        label="Figure 4",
        caption="Figure 4. Same caption.",
        image_data=b"png",
    )
    second = extractor.NotebookAsset(
        kind="figure",
        cell_index=2,
        ordinal=1,
        label="Figure 4",
        caption="Figure 4. Same caption.",
        image_data=b"png",
    )

    first_name = extractor.filename_for_asset(first, used_names)
    second_name = extractor.filename_for_asset(second, used_names)

    assert first_name == "figure-004-figure-4-same-caption.png"
    assert second_name == "figure-004-figure-4-same-caption-2.png"

from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from matplotlib.colors import LinearSegmentedColormap

from knime2py.nodes.scatter_plot import generate_py_body, parse_scatter_plot_settings


matplotlib.use("Agg")

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "INZ_visa_decisions_model" / "Scatter Plot (#1572)"


def test_scatter_plot_parses_inz_settings() -> None:
    settings = parse_scatter_plot_settings(FIXTURE_DIR)

    assert settings.x_col == "Prediction (Approval rate)"
    assert settings.y_col == "Approval rate"
    assert settings.color_col == "Applications_log10"
    assert settings.width == 800
    assert settings.height == 600
    assert settings.image_format == "SVG"


def test_scatter_plot_writes_svg_with_green_blue_palette(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "Prediction (Approval rate)": [0.2, 0.7],
            "Approval rate": [0.1, 0.8],
            "Applications_log10": [1.0, 3.0],
        }
    )
    context = {"SRC:1": df}
    body = generate_py_body("SP", str(FIXTURE_DIR), [("SRC", "1")], [])

    monkeypatch.chdir(tmp_path)
    exec(
        "\n".join(body),
        {
            "context": context,
            "pd": pd,
            "np": np,
            "plt": plt,
            "Path": Path,
            "tempfile": tempfile,
            "LinearSegmentedColormap": LinearSegmentedColormap,
        },
    )

    image = tmp_path / "scatter_SP.svg"
    assert image.exists()
    text = image.read_text(encoding="utf-8")
    assert "knime2py_green_blue" in text or "#2ca25f" in text or "#2b8cbe" in text

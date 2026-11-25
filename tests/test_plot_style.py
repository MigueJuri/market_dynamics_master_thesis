"""Tests del estilo de figuras ``styles/thesis.mplstyle``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

STYLE = Path(__file__).resolve().parents[1] / "styles" / "thesis.mplstyle"
TEXTWIDTH_IN = 155 / 25.4  # \textwidth de la tesis


@pytest.fixture
def thesis_style():
    with plt.style.context(STYLE):
        yield plt.rcParams


def test_style_file_exists():
    assert STYLE.is_file()


def test_figure_width_matches_textwidth(thesis_style):
    assert thesis_style["figure.figsize"][0] == pytest.approx(TEXTWIDTH_IN, abs=1e-3)


def test_fonts_match_body(thesis_style):
    assert thesis_style["text.usetex"] is True
    assert thesis_style["font.family"] == ["serif"]
    for key in (
        "font.size",
        "axes.labelsize",
        "axes.titlesize",
        "legend.fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
    ):
        assert float(thesis_style[key]) == 12.0, key


def test_saved_size_is_not_cropped(thesis_style, tmp_path):
    """Sin ``bbox_inches='tight'`` el PDF conserva el ancho del texto."""
    assert thesis_style["savefig.bbox"] in (None, "standard")
    assert thesis_style["figure.constrained_layout.use"] is True
    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "fig.pdf"
        fig.savefig(out)
        plt.close(fig)
    assert out.stat().st_size > 0
    w, _ = fig.get_size_inches()
    assert w == pytest.approx(TEXTWIDTH_IN, abs=1e-3)

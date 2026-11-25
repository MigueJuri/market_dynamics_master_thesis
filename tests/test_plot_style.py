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


def _toy_backtest(n: int = 60):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=n)
    ret = rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "strategy_ret": ret,
            "gross_return": ret,
            "equity": np.exp(np.cumsum(ret)),
            "drawdown": np.zeros(n),
        },
        index=idx,
    )


def test_src_plots_follow_style_width(thesis_style):
    """Las funciones de ``src/`` no fijan el ancho: lo toman de ``rcParams``."""
    import numpy as np
    import pandas as pd

    from thesis.plotting import (
        plot_equity_diagnostics,
        plot_sector_average_heatmap,
        plot_spread,
    )
    from thesis.portfolio import plot_equity_drawdown

    idx = pd.bdate_range("2020-01-01", periods=60)
    spread = pd.Series(np.sin(np.linspace(0.0, 6.0, 60)), index=idx)
    matrix = pd.DataFrame(np.eye(3), index=list("abc"), columns=list("abc"))
    rets = pd.DataFrame({"1/N": spread * 0.01, "1/σ": spread * 0.02}, index=idx)

    with plt.rc_context({"text.usetex": False}):
        figs = [
            plot_spread(spread),
            plot_sector_average_heatmap(matrix),
            plot_equity_diagnostics(_toy_backtest(), rolling_window=10),
            plot_equity_drawdown(rets),
        ]
    for fig in figs:
        assert fig is not None
        w, _ = fig.get_size_inches()
        assert w == pytest.approx(TEXTWIDTH_IN, abs=1e-3)
        plt.close(fig)

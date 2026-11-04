"""Tests para ``thesis.btc``.

Todos los tests son sintéticos: no requieren red ni datos reales.  La
descarga se prueba con una sesión HTTP falsa que devuelve velas al estilo
de la API de Bitstamp.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from thesis.btc import (
    BTC_CACHE_FILENAME,
    BTC_OHLCV_COLUMNS,
    detect_market_cycles,
    fetch_btc_ohlcv,
    fit_power_law_quantiles,
    plot_btc_quantreg,
    prepare_power_law_frame,
    validate_btc_ohlcv,
)

_SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    """Cierra las figuras abiertas al terminar cada prueba."""
    yield
    plt.close("all")


def _make_ohlcv(n: int = 400, start: str = "2011-08-18") -> pd.DataFrame:
    """Serie OHLCV diaria sintética con precio creciente."""
    index = pd.date_range(start, periods=n, freq="D")
    close = 10.0 ** (1.0 + 2.0 * np.log10(np.arange(1, n + 1) + 300.0))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "volume": np.full(n, 1_000.0),
        },
        index=index,
    )


def _make_cycle_series(n: int = 1600, period: int = 600, amplitude: float = 0.6):
    """Serie diaria con picos y valles conocidos (sinuso en log10)."""
    index = pd.date_range("2011-08-18", periods=n, freq="D")
    t = np.arange(n)
    close = 10.0 ** (2.0 + amplitude * np.sin(2.0 * np.pi * t / period))
    df = pd.DataFrame(
        {
            "open": close,
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": np.full(n, 1_000.0),
        },
        index=index,
    )
    return df, index


class _FakeResponse:
    """Respuesta mínima con la interfaz que usa el cliente de Bitstamp."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Sesión HTTP falsa que devuelve páginas predefinidas de velas."""

    def __init__(self, pages: list[dict]) -> None:
        self._pages = list(pages)
        self.calls: list[dict] = []

    def get(self, url: str, params: dict | None = None) -> _FakeResponse:
        self.calls.append({"url": url, "params": params})
        payload = self._pages.pop(0) if self._pages else {"data": {"ohlc": []}}
        return _FakeResponse(payload)


class _ExplodingSession:
    """Sesión que falla si se usa: sirve para probar la ruta de caché."""

    def get(self, url: str, params: dict | None = None) -> None:
        raise AssertionError("no debe haber peticiones HTTP con la caché presente")


def _bitstamp_page(timestamps: list[int], price: float = 100.0) -> dict:
    """Construye una página de la API de Bitstamp (valores como texto)."""
    ohlc = [
        {
            "timestamp": str(ts),
            "open": f"{price:.2f}",
            "close": f"{price * 1.01:.2f}",
            "high": f"{price * 1.02:.2f}",
            "low": f"{price * 0.99:.2f}",
            "volume": "1234.5",
        }
        for ts in timestamps
    ]
    return {"data": {"ohlc": ohlc}}


def test_validate_btc_ohlcv_ok() -> None:
    df = _make_ohlcv(50)
    cleaned = validate_btc_ohlcv(df)
    assert list(cleaned.columns) == list(BTC_OHLCV_COLUMNS)
    assert cleaned.index.is_monotonic_increasing
    assert (cleaned[["open", "close", "high", "low"]] > 0).all().all()


def test_validate_btc_ohlcv_rejects_missing_columns() -> None:
    df = _make_ohlcv(20).drop(columns=["volume"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_btc_ohlcv(df)


def test_validate_btc_ohlcv_rejects_non_positive_price() -> None:
    df = _make_ohlcv(20)
    df.iloc[5, df.columns.get_loc("close")] = -1.0
    with pytest.raises(ValueError, match="no positivo"):
        validate_btc_ohlcv(df)


def test_validate_btc_ohlcv_rejects_nan() -> None:
    df = _make_ohlcv(20)
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_btc_ohlcv(df)


def test_validate_btc_ohlcv_rejects_non_monotonic_index() -> None:
    df = _make_ohlcv(20)
    shuffled = df.iloc[::-1]
    with pytest.raises(ValueError, match="monótono"):
        validate_btc_ohlcv(shuffled)


def test_prepare_power_law_frame_filtra_dias_y_precios() -> None:
    df = _make_ohlcv(30)
    df.iloc[0, df.columns.get_loc("close")] = 0.0
    frame = prepare_power_law_frame(df)
    assert (frame["days"] > 0).all()
    assert (frame["close"] > 0).all()
    assert frame["log_days"].notna().all()
    assert frame["log_price"].notna().all()


def test_prepare_power_law_frame_es_causal() -> None:
    """Truncar la serie no altera las filas ya calculadas."""
    df = _make_ohlcv(300)
    full = prepare_power_law_frame(df)
    truncated = prepare_power_law_frame(df.iloc[:200])
    assert list(truncated.index) == list(full.index[: len(truncated)])
    pd.testing.assert_frame_equal(truncated, full.iloc[: len(truncated)])


def test_fit_power_law_quantiles_recupera_el_exponente() -> None:
    """El slope de la regresión recupera el exponente del power-law."""
    days = np.arange(100, 5000)
    index = pd.Timestamp("2009-01-03") + pd.to_timedelta(days, unit="D")
    close = 10.0 ** (1.5 + 2.3 * np.log10(days))
    df = pd.DataFrame(
        {
            "open": close,
            "close": close,
            "high": close,
            "low": close,
            "volume": np.full(len(days), 1.0),
        },
        index=index,
    )

    fit = fit_power_law_quantiles(prepare_power_law_frame(df))
    assert set(fit) == {0.10, 0.50, 0.90}
    for res in fit.values():
        assert res["params"]["log_days"] == pytest.approx(2.3, abs=1e-3)
        assert res["params"]["Intercept"] == pytest.approx(1.5, abs=1e-3)
        assert len(res["prediction"]) == len(df)


def test_plot_btc_quantreg_devuelve_dos_paneles() -> None:
    df = _make_ohlcv(200)
    fig = plot_btc_quantreg(df)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_yscale() == "log"
    assert fig.axes[1].get_xscale() == "log"


def test_plot_btc_quantreg_acepta_anotaciones() -> None:
    df = _make_ohlcv(200)
    fig = plot_btc_quantreg(df, halvings=["2020-05-11"], peaks=[], bottoms=[])
    assert len(fig.axes) == 2


def test_plot_btc_quantreg_respeta_los_cuantiles() -> None:
    """La figura dibuja exactamente los cuantiles solicitados."""
    df = _make_ohlcv(200)
    fig = plot_btc_quantreg(df, quantiles=(0.50,))
    labels = [
        line.get_label()
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_label().endswith("Percentile Reg")
    ]
    assert labels == ["50th Percentile Reg", "50th Percentile Reg"]


def test_detect_market_cycles_recupera_picos_y_valles() -> None:
    df, index = _make_cycle_series()
    peaks, bottoms = detect_market_cycles(df)

    expected_peaks = {index[150], index[750], index[1350]}
    expected_bottoms = {index[450], index[1050]}

    assert expected_peaks.issubset(set(peaks.index))
    assert expected_bottoms.issubset(set(bottoms.index))
    assert "Days_Since_Last_Peak" in peaks.columns
    assert "Days_Since_Last_Bottom" in bottoms.columns


def test_detect_market_cycles_inyecta_el_pico_de_mtgox() -> None:
    df, _ = _make_cycle_series()
    peaks, _ = detect_market_cycles(df)
    assert pd.Timestamp("2011-06-08") in peaks.index
    assert peaks.loc[pd.Timestamp("2011-06-08"), "close"] == pytest.approx(31.90)


def test_fetch_btc_ohlcv_descarga_y_guarda_cache(tmp_path: Path) -> None:
    start = int(pd.Timestamp("2011-08-18").timestamp())
    session = _FakeSession([_bitstamp_page([start, start + 86400])])

    df = fetch_btc_ohlcv(cache_dir=tmp_path, session=session)

    assert list(df.columns) == list(BTC_OHLCV_COLUMNS)
    assert df.index.is_monotonic_increasing
    assert len(df) == 2
    assert len(session.calls) >= 1
    assert session.calls[0]["params"]["step"] == 86400

    cache = tmp_path / BTC_CACHE_FILENAME
    assert cache.exists()

    # La segunda llamada lee de la caché y no vuelve a pedir datos.
    cached = fetch_btc_ohlcv(cache_dir=tmp_path, session=_ExplodingSession())
    assert len(cached) == len(df)


def test_fetch_btc_ohlcv_rechaza_respuesta_vacia(tmp_path: Path) -> None:
    session = _FakeSession([])
    with pytest.raises(ValueError, match="Bitstamp"):
        fetch_btc_ohlcv(cache_dir=tmp_path, session=session)


def test_importar_thesis_btc_no_descarga_ni_dibuja(tmp_path: Path) -> None:
    """Importar el módulo no hace red, no dibuja y no escribe en disco."""
    code = (
        "import socket\n"
        "def _blocked(*args, **kwargs):\n"
        "    raise AssertionError('no network access allowed')\n"
        "socket.socket.connect = _blocked\n"
        "socket.create_connection = _blocked\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import thesis.btc\n"
        "assert not plt.get_fignums(), 'import drew a figure'\n"
        "print('ok')\n"
    )
    env = {
        **os.environ,
        "THESIS_DATA_DIR": str(tmp_path),
        "PYTHONPATH": str(_SRC),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert list(tmp_path.iterdir()) == []

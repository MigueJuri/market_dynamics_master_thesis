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
    BTC_CYCLE_SMOOTH_WINDOW,
    BTC_OHLCV_COLUMNS,
    detect_market_cycles,
    detrend_power_law_quantile,
    fetch_btc_ohlcv,
    find_local_extrema,
    fit_power_law_quantiles,
    idealized_cycle,
    plot_btc_quantreg,
    prepare_power_law_frame,
    select_extrema,
    smooth_cycle_residual,
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


def _make_detrend_frame(n: int = 1200, period: int = 300) -> pd.DataFrame:
    """Precio power-law con una oscilación logarítmica de periodo ``period``."""
    days = np.arange(1, n + 1)
    index = pd.Timestamp("2009-01-03") + pd.to_timedelta(days, unit="D")
    log_price = 1.0 + 2.0 * np.log10(days) + 0.5 * np.sin(2.0 * np.pi * days / period)
    close = 10.0**log_price
    return pd.DataFrame(
        {
            "open": close,
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": np.full(n, 1_000.0),
        },
        index=index,
    )


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


def test_plot_btc_quantreg_devuelve_tres_figuras_independientes() -> None:
    df = _make_ohlcv(200)
    figures = plot_btc_quantreg(df)
    assert len(figures) == 3
    linear_fig, temporal_fig, power_law_fig = figures
    assert len(linear_fig.axes) == 1
    assert len(temporal_fig.axes) == 1
    assert len(power_law_fig.axes) == 1
    assert linear_fig.axes[0].get_yscale() == "linear"
    assert temporal_fig.axes[0].get_yscale() == "log"
    assert power_law_fig.axes[0].get_xscale() == "log"


def test_plot_btc_quantreg_acepta_anotaciones() -> None:
    df = _make_ohlcv(200)
    figures = plot_btc_quantreg(
        df, halvings=["2020-05-11"], peaks=[], bottoms=[]
    )
    assert all(len(fig.axes) == 1 for fig in figures)


def _count_series_lines(fig: plt.Figure, n_points: int) -> int:
    """Número de trazos con ``n_points`` datos (series, no anotaciones)."""
    return sum(
        1
        for ax in fig.axes
        for line in ax.get_lines()
        if len(line.get_xdata()) == n_points
    )


def test_plot_btc_quantreg_respeta_los_cuantiles() -> None:
    """La figura dibuja exactamente los cuantiles solicitados.

    Se cuentan las series dibujadas y no el texto de las etiquetas: los
    rótulos de los gráficos se pueden cambiar libremente.
    """
    df = _make_ohlcv(200)
    n_points = len(df)

    linear_fig, temporal_fig, power_law_fig = plot_btc_quantreg(
        df, quantiles=(0.50,)
    )
    # Panel lineal: sólo la serie de precio.
    assert _count_series_lines(linear_fig, n_points) == 1
    # Paneles temporal y log-log: precio + una recta de cuantil.
    assert _count_series_lines(temporal_fig, n_points) == 2
    assert _count_series_lines(power_law_fig, n_points) == 2

    linear_fig, temporal_fig, power_law_fig = plot_btc_quantreg(df)
    # Con los tres cuantiles por defecto hay una recta más en cada panel.
    assert _count_series_lines(temporal_fig, n_points) == 4
    assert _count_series_lines(power_law_fig, n_points) == 4


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


def test_detrend_power_law_quantile_conserva_la_banda_ajustada() -> None:
    """La banda restada es exactamente la predicción del cuantil pedido."""
    frame = prepare_power_law_frame(_make_detrend_frame())
    fit = fit_power_law_quantiles(frame, quantiles=(0.10,))
    detrended = detrend_power_law_quantile(frame, fit, quantile=0.10)

    assert "band" in detrended.columns
    np.testing.assert_allclose(
        detrended["band"].to_numpy(),
        fit[0.10]["prediction"].reindex(frame.index).to_numpy(),
    )
    np.testing.assert_allclose(
        detrended["resid_log"].to_numpy(),
        (detrended["log_price"] - detrended["log_band"]).to_numpy(),
    )
    np.testing.assert_allclose(
        detrended["ratio"].to_numpy(),
        10.0 ** detrended["resid_log"].to_numpy(),
    )


def test_detrend_power_law_quantile_acerca_los_minimos_al_cero() -> None:
    """Tras restar la banda q10, los mínimos locales quedan cerca de cero."""
    frame = prepare_power_law_frame(_make_detrend_frame())
    detrended = detrend_power_law_quantile(frame, quantile=0.10)

    bottoms, _ = find_local_extrema(
        detrended["resid_log"], distance=200, prominence=0.2
    )
    assert len(bottoms) > 0
    assert bottoms["value"].abs().max() < 0.25


def test_detrend_power_law_quantile_rechaza_cuantil_ausente() -> None:
    frame = prepare_power_law_frame(_make_detrend_frame(60))
    fit = fit_power_law_quantiles(frame, quantiles=(0.50,))
    with pytest.raises(ValueError, match="cuantil"):
        detrend_power_law_quantile(frame, fit, quantile=0.10)


def test_find_local_extrema_detecta_extremos_y_nombra_ids() -> None:
    index = pd.date_range("2020-01-01", periods=1200, freq="D")
    values = pd.Series(
        np.sin(2.0 * np.pi * np.arange(1200) / 300.0), index=index
    )
    bottoms, peaks = find_local_extrema(values, distance=200, prominence=0.5)

    assert list(bottoms["id"]) == ["B1", "B2", "B3", "B4"]
    assert list(peaks["id"]) == ["P1", "P2", "P3", "P4"]
    assert bottoms.index[0] == index[225]
    assert peaks.index[0] == index[75]
    assert bottoms["value"].iloc[0] == pytest.approx(-1.0, abs=1e-9)
    assert peaks["value"].iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_smooth_cycle_residual_es_una_media_movil_trailing() -> None:
    """La media móvil usa sólo datos pasados y deja NaN al principio."""
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    values = pd.Series(np.linspace(0.0, 1.0, 200), index=index)

    smoothed = smooth_cycle_residual(values, window=10)

    assert smoothed.iloc[:9].isna().all()
    assert smoothed.iloc[9:].notna().all()
    assert smoothed.iloc[9] == pytest.approx(values.iloc[:10].mean())
    assert smoothed.iloc[-1] == pytest.approx(values.iloc[-10:].mean())

    # Es causal: truncar la serie no altera los valores ya calculados.
    truncado = smooth_cycle_residual(values.iloc[:100], window=10)
    pd.testing.assert_series_equal(truncado, smoothed.iloc[:100])


def test_smooth_cycle_residual_centrada_usa_ventana_centrada() -> None:
    """Con ``centered=True`` la ventana se centra en cada fecha."""
    index = pd.date_range("2020-01-01", periods=21, freq="D")
    values = pd.Series(np.arange(21.0), index=index)

    smoothed = smooth_cycle_residual(values, window=5, centered=True)

    # En el centro la media usa los cinco puntos que rodean a la fecha.
    assert smoothed.iloc[10] == pytest.approx(values.iloc[8:13].mean())
    # Con la ventana completa los NaN quedan en los bordes (~window // 2).
    assert smoothed.iloc[:2].isna().all()
    assert smoothed.iloc[-2:].isna().all()

    # Con min_periods parcial la serie cubre todo el histórico.
    completa = smooth_cycle_residual(
        values, window=5, min_periods=3, centered=True
    )
    assert completa.notna().all()
    assert completa.iloc[10] == pytest.approx(values.iloc[8:13].mean())


def test_smooth_cycle_residual_rechaza_ventanas_invalidas() -> None:
    values = pd.Series(np.arange(50.0))
    with pytest.raises(ValueError, match="ventana"):
        smooth_cycle_residual(values, window=0)
    with pytest.raises(ValueError, match="min_periods"):
        smooth_cycle_residual(values, window=10, min_periods=0)


def test_find_local_extrema_ignora_los_nan_de_la_media_movil() -> None:
    """Los NaN iniciales de la serie suavizada no son candidatos."""
    index = pd.date_range("2020-01-01", periods=1200, freq="D")
    values = pd.Series(np.sin(2.0 * np.pi * np.arange(1200) / 300.0), index=index)
    suavizada = smooth_cycle_residual(values, window=BTC_CYCLE_SMOOTH_WINDOW)

    bottoms, peaks = find_local_extrema(suavizada, distance=200, prominence=0.5)
    bottoms_limpios, peaks_limpios = find_local_extrema(
        suavizada.dropna(), distance=200, prominence=0.5
    )

    assert list(bottoms["id"]) == list(bottoms_limpios["id"])
    assert list(peaks["id"]) == list(peaks_limpios["id"])
    assert bottoms.index.min() >= index[BTC_CYCLE_SMOOTH_WINDOW - 1]
    assert peaks.index.min() >= index[BTC_CYCLE_SMOOTH_WINDOW - 1]
    np.testing.assert_allclose(
        bottoms["value"].to_numpy(),
        suavizada.reindex(bottoms.index).to_numpy(),
    )


def test_ciclo_idealizado_se_construye_sobre_la_serie_suavizada() -> None:
    """Pipeline de la Etapa 5: detrender, suavizar, detectar y ajustar."""
    frame = prepare_power_law_frame(_make_detrend_frame())
    detrended = detrend_power_law_quantile(frame, quantile=0.10)
    suavizada = smooth_cycle_residual(
        detrended["resid_log"], window=BTC_CYCLE_SMOOTH_WINDOW
    )

    bottoms, peaks = find_local_extrema(suavizada, distance=200, prominence=0.2)
    assert len(bottoms) > 0
    assert len(peaks) > 0

    # El ciclo une un valle y el pico que lo sigue cronológicamente.
    valley = bottoms.iloc[[0]]
    peak = peaks[peaks.index > bottoms.index[0]].iloc[[0]]
    ideal = idealized_cycle(suavizada, valley, peak)
    # El valle elegido queda anclado en cero.
    assert ideal.loc[valley.index[0]] == pytest.approx(0.0)
    # El pico queda a la amplitud de su ciclo (residuo suavizado).
    assert ideal.loc[peak.index[0]] == pytest.approx(
        suavizada.loc[peak.index[0]] - suavizada.loc[valley.index[0]]
    )


def test_select_extrema_resuelve_ids_fechas_y_orden() -> None:
    index = pd.date_range("2020-01-01", periods=1200, freq="D")
    values = pd.Series(
        np.sin(2.0 * np.pi * np.arange(1200) / 300.0), index=index
    )
    bottoms, _ = find_local_extrema(values, distance=200, prominence=0.5)

    por_id = select_extrema(bottoms, ["B2"])
    por_fecha = select_extrema(bottoms, [str(bottoms.index[1].date())])
    por_orden = select_extrema(bottoms, [2])

    assert por_id.index.tolist() == [bottoms.index[1]]
    assert por_fecha.index.tolist() == por_id.index.tolist()
    assert por_orden.index.tolist() == por_id.index.tolist()


def test_select_extrema_rechaza_id_o_fecha_desconocidos() -> None:
    index = pd.date_range("2020-01-01", periods=1200, freq="D")
    values = pd.Series(
        np.sin(2.0 * np.pi * np.arange(1200) / 300.0), index=index
    )
    bottoms, _ = find_local_extrema(values, distance=200, prominence=0.5)

    with pytest.raises(ValueError, match="B99"):
        select_extrema(bottoms, ["B99"])
    with pytest.raises(ValueError, match="no está en la tabla"):
        select_extrema(bottoms, ["2030-01-01"])


def test_idealized_cycle_ancora_los_valles_en_cero() -> None:
    """Los valles quedan en 0 y los picos a la amplitud de su ciclo."""
    index = pd.date_range("2020-01-01", periods=1200, freq="D")
    values = pd.Series(
        np.sin(2.0 * np.pi * np.arange(1200) / 300.0), index=index
    )
    bottoms, peaks = find_local_extrema(values, distance=200, prominence=0.5)

    ideal = idealized_cycle(values, bottoms.iloc[[0, 1]], peaks.iloc[[1]])

    assert ideal.loc[index[225]] == pytest.approx(0.0)
    assert ideal.loc[index[375]] == pytest.approx(2.0)
    assert ideal.loc[index[525]] == pytest.approx(0.0)
    # Interpolación lineal entre vértices.
    assert ideal.loc[index[300]] == pytest.approx(1.0)
    # Fuera del rango cubierto no hay ciclo idealizado.
    assert np.isnan(ideal.loc[index[100]])
    assert np.isnan(ideal.loc[index[1100]])


def test_idealized_cycle_exige_dos_extremos() -> None:
    index = pd.date_range("2020-01-01", periods=1200, freq="D")
    values = pd.Series(
        np.sin(2.0 * np.pi * np.arange(1200) / 300.0), index=index
    )
    bottoms, peaks = find_local_extrema(values, distance=200, prominence=0.5)

    with pytest.raises(ValueError, match="al menos dos"):
        idealized_cycle(values, bottoms.iloc[[0]], peaks.iloc[:0])


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

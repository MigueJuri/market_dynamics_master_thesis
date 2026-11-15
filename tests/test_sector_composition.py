"""Tests para ``thesis.sector_composition``.

Los tests unitarios son **sintéticos**: no requieren red ni datos reales.
La descarga de precios y el scraping de Wikipedia se prueban con dobles
 (``monkeypatch``) y las pruebas que sí necesitan red están marcadas con
``integration`` (se omiten por defecto).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from thesis.sector_composition import (
    GICS_SECTOR_ES,
    MINOR_SECTOR_THRESHOLD,
    SP500_WIKI_URL,
    aggregate_small_sectors,
    build_sector_weights,
    company_hhi,
    fetch_spy_sector_weights,
    get_sp500_constituents,
    sector_entropy,
)

_SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    """Cierra las figuras abiertas al terminar cada prueba."""
    yield
    plt.close("all")


def _make_weights() -> pd.Series:
    """Pesos sectoriales sintéticos en fracción (suman 1)."""
    return pd.Series(
        {
            "technology": 0.30,
            "healthcare": 0.20,
            "financial_services": 0.15,
            "energy": 0.05,
            "utilities": 0.02,
            "realestate": 0.01,
        },
        name="weight",
    )


def _make_prices() -> pd.DataFrame:
    """Cierres sintéticos en formato ancho (fecha × ticker)."""
    index = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 120.0],
            "MSFT": [50.0, 55.0, 60.0],
            "XOM": [20.0, 21.0, 22.0],
            "SINE": [np.nan, np.nan, np.nan],
        },
        index=index,
    )


def _make_shares() -> pd.Series:
    """Acciones en circulación sintéticas por ticker."""
    return pd.Series(
        {
            "AAPL": 1_000.0,
            "MSFT": 500.0,
            "XOM": 200.0,
            "SINE": 100.0,
        },
        name="SharesOutstanding",
    )


def _make_sector_map() -> pd.Series:
    """Sector GICS sintético por ticker."""
    return pd.Series(
        {
            "AAPL": "Information Technology",
            "MSFT": "Information Technology",
            "XOM": "Energy",
            "SINE": "Energy",
        },
        name="Sector",
    )


# ----------------------------------------------------------------------
# aggregate_small_sectors
# ----------------------------------------------------------------------


def test_aggregate_small_sectors_agrupa_menores_que_el_umbral() -> None:
    """Los sectores por debajo del umbral se suman en "Otros"."""
    out = aggregate_small_sectors(_make_weights())

    assert "Otros" in out.index
    # energy (0,05) + utilities (0,02) + realestate (0,01)
    assert out["Otros"] == pytest.approx(0.08)
    assert "energy" not in out.index
    assert "utilities" not in out.index
    assert "realestate" not in out.index


def test_aggregate_small_sectors_traduce_las_etiquetas() -> None:
    """Las etiquetas grandes se traducen al español."""
    out = aggregate_small_sectors(_make_weights())

    assert "Tecnología" in out.index
    assert "Salud" in out.index
    assert out["Tecnología"] == pytest.approx(0.30)


def test_aggregate_small_sectors_ordena_descendente() -> None:
    """El resultado queda ordenado de mayor a menor peso."""
    out = aggregate_small_sectors(_make_weights())

    assert list(out.index) == ["Tecnología", "Salud", "Servicios Financieros", "Otros"]
    assert out.is_monotonic_decreasing


def test_aggregate_small_sectors_conserva_la_suma() -> None:
    """La suma de los pesos no cambia al agrupar."""
    weights = _make_weights()
    out = aggregate_small_sectors(weights)

    assert float(out.sum()) == pytest.approx(float(weights.sum()))


def test_aggregate_small_sectors_umbral_exacto_no_se_agrupa() -> None:
    """Un sector con peso exactamente igual al umbral no se agrupa."""
    weights = pd.Series({"technology": MINOR_SECTOR_THRESHOLD})
    out = aggregate_small_sectors(weights)

    assert "Otros" not in out.index
    assert out["Tecnología"] == pytest.approx(MINOR_SECTOR_THRESHOLD)


def test_aggregate_small_sectors_umbral_personalizado() -> None:
    """El umbral es configurable."""
    out = aggregate_small_sectors(_make_weights(), threshold=0.25)

    assert "Otros" in out.index
    # Todo salvo technology (0,30): 0,20 + 0,15 + 0,05 + 0,02 + 0,01
    assert out["Otros"] == pytest.approx(0.43)
    assert "Tecnología" in out.index


def test_aggregate_small_sectors_traduce_sin_traduccion() -> None:
    """Sin traducción, el nombre del sector se hace legible."""
    out = aggregate_small_sectors(pd.Series({"some_new_sector": 0.50}), translation={})

    assert out.index[0] == "Some New Sector"


def test_aggregate_small_sectors_vacio() -> None:
    """Una serie vacía devuelve una serie vacía."""
    out = aggregate_small_sectors(pd.Series(dtype=float))

    assert out.empty


# ----------------------------------------------------------------------
# build_sector_weights
# ----------------------------------------------------------------------


def test_build_sector_weights_filas_suman_100() -> None:
    """Cada fila de pesos suma 100 (porcentaje)."""
    weights = build_sector_weights(_make_prices(), _make_shares(), _make_sector_map())

    assert (weights.sum(axis=1) - 100.0).abs().max() < 1e-9
    assert set(weights.columns) == {"Information Technology", "Energy"}


def test_build_sector_weights_descarta_tickers_sin_datos() -> None:
    """Un ticker sin precios (todo NaN) no aparece en el resultado."""
    weights = build_sector_weights(_make_prices(), _make_shares(), _make_sector_map())

    assert "Energy" in weights.columns
    # XOM aporta 20·200 = 4 000; AAPL+MSFT aportan 100 000 + 25 000.
    assert weights["Energy"].iloc[0] == pytest.approx(4000.0 / 129000.0 * 100.0)


def test_build_sector_weights_descarta_ticker_sin_acciones() -> None:
    """Un ticker con precio pero sin acciones se descarta."""
    shares = _make_shares().copy()
    shares["XOM"] = np.nan

    weights = build_sector_weights(_make_prices(), shares, _make_sector_map())

    assert "Energy" not in weights.columns
    assert (weights.sum(axis=1) - 100.0).abs().max() < 1e-9


def test_build_sector_weights_sector_map_incompleto_falla() -> None:
    """Si ``sector_map`` no cubre un ticker con datos, falla rápido."""
    sector_map = _make_sector_map().drop("XOM")

    with pytest.raises(ValueError, match="sector_map"):
        build_sector_weights(_make_prices(), _make_shares(), sector_map)


def test_build_sector_weights_sin_datos_falla() -> None:
    """Sin precios ni acciones utilizables no hay pesos que calcular."""
    prices = _make_prices().iloc[:, :0]

    with pytest.raises(ValueError):
        build_sector_weights(prices, _make_shares(), _make_sector_map())


def test_build_sector_weights_precios_negativos_falla() -> None:
    """Los precios negativos (datos corruptos) producen pesos negativos."""
    prices = _make_prices()
    prices.iloc[0, 0] = -100.0

    with pytest.raises(ValueError, match="negativos"):
        build_sector_weights(prices, _make_shares(), _make_sector_map())


# ----------------------------------------------------------------------
# company_hhi
# ----------------------------------------------------------------------


def test_company_hhi_uniforme_es_uno_sobre_n() -> None:
    """Capitalización uniforme sobre N empresas: HHI = 1/N."""
    n = 5
    index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    prices = pd.DataFrame({f"T{i}": [10.0, 20.0] for i in range(n)}, index=index)
    shares = pd.Series({f"T{i}": 1000.0 for i in range(n)}, name="SharesOutstanding")

    hhi = company_hhi(prices, shares)

    assert hhi.iloc[0] == pytest.approx(1.0 / n)
    assert hhi.iloc[1] == pytest.approx(1.0 / n)


def test_company_hhi_una_empresa_con_todo_el_peso() -> None:
    """Una empresa con toda la capitalización: HHI ≈ 1."""
    prices = pd.DataFrame(
        {"BIG": [1000.0], "SMALL": [1.0]},
        index=pd.to_datetime(["2020-01-31"]),
    )
    shares = pd.Series({"BIG": 1000.0, "SMALL": 1.0})

    hhi = company_hhi(prices, shares)

    caps = np.array([1_000_000.0, 1.0])
    weights = caps / caps.sum()
    assert hhi.iloc[0] == pytest.approx(float((weights**2).sum()))
    assert hhi.iloc[0] == pytest.approx(1.0, abs=1e-3)


def test_company_hhi_coincide_con_calculo_manual() -> None:
    """El HHI coincide con el cálculo manual ``Σ w_i²``."""
    prices = _make_prices()
    shares = _make_shares()

    hhi = company_hhi(prices, shares)

    caps = prices.mul(shares.reindex(prices.columns), axis=1)
    for date, value in hhi.items():
        row = caps.loc[date].dropna()
        weights = row.to_numpy(dtype=float) / float(row.sum())
        assert value == pytest.approx(float((weights**2).sum()))


def test_company_hhi_rango_y_tipo() -> None:
    """El HHI está en [1/N, 1], es finito y conserva nombre e índice."""
    prices = _make_prices()
    shares = _make_shares()

    hhi = company_hhi(prices, shares)

    assert isinstance(hhi, pd.Series)
    assert hhi.name == "hhi"
    assert hhi.index.equals(prices.index)
    assert np.isfinite(hhi.to_numpy(dtype=float)).all()
    # AAPL, MSFT y XOM tienen datos (SINE se descarta): N = 3.
    n = 3
    assert (hhi >= 1.0 / n - 1e-12).all()
    assert (hhi <= 1.0 + 1e-12).all()


def test_company_hhi_descarta_tickers_sin_datos() -> None:
    """Un ticker sin precios o sin acciones se descarta sin romper el cálculo."""
    prices = _make_prices()
    shares = _make_shares().copy()
    shares["XOM"] = np.nan

    hhi = company_hhi(prices, shares)

    # Sólo quedan AAPL y MSFT con datos: N = 2.
    n = 2
    assert hhi.index.equals(prices.index)
    assert np.isfinite(hhi.to_numpy(dtype=float)).all()
    assert (hhi >= 1.0 / n - 1e-12).all()


def test_company_hhi_descarta_filas_sin_capitalizacion() -> None:
    """Las filas con capitalización total 0 se descartan (no hay 0/0)."""
    prices = pd.DataFrame(
        {"A": [10.0, 0.0], "B": [20.0, 0.0]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    shares = pd.Series({"A": 100.0, "B": 100.0})

    hhi = company_hhi(prices, shares)

    assert len(hhi) == 1
    assert hhi.index[0] == pd.Timestamp("2020-01-31")


def test_company_hhi_precios_negativos_falla() -> None:
    """Los precios negativos (datos corruptos) producen un error explícito."""
    prices = _make_prices()
    prices.iloc[0, 0] = -100.0

    with pytest.raises(ValueError, match="negativos"):
        company_hhi(prices, _make_shares())


def test_company_hhi_es_puntual() -> None:
    """Truncar la serie no altera las filas ya calculadas."""
    prices = _make_prices()
    shares = _make_shares()

    full = company_hhi(prices, shares)
    truncated = company_hhi(prices.iloc[:2], shares)

    pd.testing.assert_series_equal(truncated, full.iloc[:2])


# ----------------------------------------------------------------------
# sector_entropy
# ----------------------------------------------------------------------


def test_sector_entropy_uniforme_es_log_n() -> None:
    """Una distribución uniforme sobre n sectores da ln(n)."""
    n = 4
    weights = pd.DataFrame(
        {f"s{i}": [100.0 / n] for i in range(n)},
        index=pd.to_datetime(["2020-01-31"]),
    )

    entropy = sector_entropy(weights)

    assert entropy.iloc[0] == pytest.approx(np.log(n))


def test_sector_entropy_p_cero_no_da_infinito() -> None:
    """Un sector con peso 0 no produce -inf (límite p·ln(p) → 0)."""
    weights = pd.DataFrame(
        {"a": [50.0], "b": [50.0], "c": [0.0]},
        index=pd.to_datetime(["2020-01-31"]),
    )

    entropy = sector_entropy(weights)

    assert np.isfinite(entropy.iloc[0])
    assert entropy.iloc[0] == pytest.approx(np.log(2.0))


def test_sector_entropy_coincide_con_calculo_manual() -> None:
    """La entropía coincide con el cálculo manual ``-Σ p·ln(p)``."""
    weights = pd.DataFrame(
        {"a": [20.0, 40.0], "b": [30.0, 60.0], "c": [50.0, 0.0]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )

    entropy = sector_entropy(weights)

    for i, row in weights.iterrows():
        p = row.to_numpy(dtype=float) / 100.0
        manual = -np.sum(p[p > 0] * np.log(p[p > 0]))
        assert entropy.loc[i] == pytest.approx(manual)


def test_sector_entropy_es_puntual() -> None:
    """Truncar la serie no altera las filas ya calculadas."""
    weights = pd.DataFrame(
        {"a": [10.0, 20.0, 30.0], "b": [90.0, 80.0, 70.0]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )

    full = sector_entropy(weights)
    truncated = sector_entropy(weights.iloc[:2])

    pd.testing.assert_series_equal(truncated, full.iloc[:2])


def test_sector_entropy_nombre_y_tipo() -> None:
    """La entropía es una serie con el mismo índice que la entrada."""
    weights = pd.DataFrame(
        {"a": [50.0], "b": [50.0]}, index=pd.to_datetime(["2020-01-31"])
    )

    entropy = sector_entropy(weights)

    assert isinstance(entropy, pd.Series)
    assert entropy.index.equals(weights.index)


# ----------------------------------------------------------------------
# Descargas (dobles)
# ----------------------------------------------------------------------


class _FakeResponse:
    """Respuesta HTTP mínima para el scraping de Wikipedia."""

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    """Sesión HTTP falsa que devuelve una tabla HTML fija."""

    HTML = """
    <table>
      <thead><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr></thead>
      <tbody>
        <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
        <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
      </tbody>
    </table>
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append((url, kwargs))
        return _FakeResponse(self.HTML)


def test_get_sp500_constituents_parsea_tabla() -> None:
    """El scraping devuelve las columnas canónicas y ``BRK.B → BRK-B``."""
    session = _FakeSession()

    df = get_sp500_constituents(session=session)  # type: ignore[arg-type]

    assert list(df.columns) == ["Symbol", "Security", "Sector"]
    assert df["Symbol"].tolist() == ["AAPL", "BRK-B"]
    assert df["Sector"].tolist() == ["Information Technology", "Financials"]
    assert session.calls[0][0] == SP500_WIKI_URL


def test_download_price_history_usa_lotes(monkeypatch: pytest.MonkeyPatch) -> None:
    """La descarga se hace por lotes y devuelve un frame ancho."""
    import thesis.sector_composition as sc

    llamadas: list[list[str]] = []

    def fake_download(batch, **kwargs):
        llamadas.append(list(batch))
        index = pd.to_datetime(["2020-01-02", "2020-01-03"])
        campos = {
            field: pd.DataFrame({ticker: [1.0, 2.0] for ticker in batch}, index=index)
            for field in ("Open", "High", "Low", "Close", "Volume")
        }
        return pd.concat(campos, axis=1)

    monkeypatch.setattr(sc.yf, "download", fake_download)

    prices = sc.download_price_history(
        ["A", "B", "C"], start="2020-01-01", end="2020-02-01", batch_size=2, sleep=0.0
    )

    assert llamadas == [["A", "B"], ["C"]]
    assert list(prices.columns) == ["A", "B", "C"]
    assert prices.shape == (2, 3)


def test_download_price_history_ticker_unico(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con un único ticker, yfinance devuelve las columnas planas."""
    import thesis.sector_composition as sc

    def fake_download(batch, **kwargs):
        return pd.DataFrame(
            {"Close": [1.0, 2.0]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )

    monkeypatch.setattr(sc.yf, "download", fake_download)

    prices = sc.download_price_history(["A"], start="2020-01-01", sleep=0.0)

    assert list(prices.columns) == ["A"]
    assert prices.shape == (2, 1)


def test_download_price_history_vacio() -> None:
    """Sin tickers se devuelve un frame vacío."""
    import thesis.sector_composition as sc

    prices = sc.download_price_history([], start="2020-01-01")

    assert prices.empty


# ----------------------------------------------------------------------
# Caché de precios
# ----------------------------------------------------------------------


def _fake_download_factory(llamadas: list[list[str]]):
    """Doble de ``yf.download`` que registra los lotes y devuelve cierres."""

    def fake_download(batch, **kwargs):
        llamadas.append(list(batch))
        index = pd.to_datetime(["2020-01-02", "2020-01-03"])
        campos = {
            field: pd.DataFrame({ticker: [1.0, 2.0] for ticker in batch}, index=index)
            for field in ("Open", "High", "Low", "Close", "Volume")
        }
        return pd.concat(campos, axis=1)

    return fake_download


def test_download_price_history_escribe_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """La descarga se persiste en un CSV con la ventana ``start:end``."""
    import thesis.sector_composition as sc

    llamadas: list[list[str]] = []
    monkeypatch.setattr(sc.yf, "download", _fake_download_factory(llamadas))

    sc.download_price_history(
        ["A", "B"],
        start="2020-01-01",
        end="2020-02-01",
        sleep=0.0,
        cache_dir=tmp_path,
    )

    cache = tmp_path / "sp500_prices_2020-01-01_2020-02-01.csv"
    assert cache.exists()
    assert list(tmp_path.iterdir()) == [cache]


def test_download_price_history_usa_cache_sin_red(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Si el CSV existe, yfinance no se invoca y el frame coincide."""
    import thesis.sector_composition as sc

    llamadas: list[list[str]] = []
    monkeypatch.setattr(sc.yf, "download", _fake_download_factory(llamadas))

    primero = sc.download_price_history(
        ["A", "B"],
        start="2020-01-01",
        end="2020-02-01",
        sleep=0.0,
        cache_dir=tmp_path,
    )
    assert len(llamadas) == 1

    segundo = sc.download_price_history(
        ["A", "B"],
        start="2020-01-01",
        end="2020-02-01",
        sleep=0.0,
        cache_dir=tmp_path,
    )

    assert len(llamadas) == 1  # no se vuelve a tocar la red
    pd.testing.assert_frame_equal(segundo, primero)


def test_download_price_history_refresh_ignora_la_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Con ``refresh=True`` se vuelve a descargar y se sobrescribe el CSV."""
    import thesis.sector_composition as sc

    llamadas: list[list[str]] = []
    monkeypatch.setattr(sc.yf, "download", _fake_download_factory(llamadas))

    for _ in range(2):
        sc.download_price_history(
            ["A"],
            start="2020-01-01",
            end="2020-02-01",
            sleep=0.0,
            cache_dir=tmp_path,
            refresh=True,
        )

    assert len(llamadas) == 2


def test_download_price_history_sin_end_no_cachea(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sin ``end`` explícito no hay caché (la ventana no se puede congelar)."""
    import thesis.sector_composition as sc

    llamadas: list[list[str]] = []
    monkeypatch.setattr(sc.yf, "download", _fake_download_factory(llamadas))

    sc.download_price_history(
        ["A"], start="2020-01-01", end=None, sleep=0.0, cache_dir=tmp_path
    )

    assert list(tmp_path.iterdir()) == []


def test_download_price_history_cache_de_otro_universo_se_descarta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Si la caché no cubre los tickers pedidos, se vuelve a descargar."""
    import thesis.sector_composition as sc

    cache = tmp_path / "sp500_prices_2020-01-01_2020-02-01.csv"
    pd.DataFrame(
        {"A": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"])
    ).to_csv(cache)

    llamadas: list[list[str]] = []
    monkeypatch.setattr(sc.yf, "download", _fake_download_factory(llamadas))

    prices = sc.download_price_history(
        ["A", "B"],
        start="2020-01-01",
        end="2020-02-01",
        sleep=0.0,
        cache_dir=tmp_path,
    )

    assert len(llamadas) == 1
    assert list(prices.columns) == ["A", "B"]


def test_download_price_history_cache_guarda_todos_los_tickers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """La caché incluye todos los tickers pedidos (los fallidos en NaN)."""
    import thesis.sector_composition as sc

    def fake_download(batch, **kwargs):
        index = pd.to_datetime(["2020-01-02", "2020-01-03"])
        # Sólo el primer ticker del lote devuelve datos.
        campos = {
            field: pd.DataFrame({batch[0]: [1.0, 2.0]}, index=index)
            for field in ("Open", "High", "Low", "Close", "Volume")
        }
        return pd.concat(campos, axis=1)

    monkeypatch.setattr(sc.yf, "download", fake_download)

    sc.download_price_history(
        ["A", "B"],
        start="2020-01-01",
        end="2020-02-01",
        sleep=0.0,
        cache_dir=tmp_path,
    )

    cache = tmp_path / "sp500_prices_2020-01-01_2020-02-01.csv"
    cached = pd.read_csv(cache, index_col=0, parse_dates=True)

    assert list(cached.columns) == ["A", "B"]
    assert cached["B"].isna().all()


def test_get_shares_outstanding_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los tickers que fallan quedan con ``NaN`` y no rompen la serie."""
    import thesis.sector_composition as sc

    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def fast_info(self):
            if self.symbol == "BAD":
                raise RuntimeError("boom")
            return {"shares": 123.0}

    monkeypatch.setattr(sc.yf, "Ticker", _FakeTicker)

    shares = sc.get_shares_outstanding(["OK", "BAD"])

    assert shares["OK"] == pytest.approx(123.0)
    assert np.isnan(shares["BAD"])


def test_get_shares_outstanding_escribe_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Las acciones en circulación se cachean en ``sp500_shares.csv``."""
    import thesis.sector_composition as sc

    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def fast_info(self):
            return {"shares": 123.0}

    monkeypatch.setattr(sc.yf, "Ticker", _FakeTicker)

    sc.get_shares_outstanding(["OK", "BAD"], cache_dir=tmp_path)

    cache = tmp_path / "sp500_shares.csv"
    assert cache.exists()
    assert list(tmp_path.iterdir()) == [cache]


def test_get_shares_outstanding_usa_cache_sin_red(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Con el CSV en caché, yfinance no se consulta."""
    import thesis.sector_composition as sc

    class _FakeTicker:
        instancias = 0

        def __init__(self, symbol: str) -> None:
            type(self).instancias += 1
            self.symbol = symbol

        @property
        def fast_info(self):
            return {"shares": 123.0}

    monkeypatch.setattr(sc.yf, "Ticker", _FakeTicker)

    primero = sc.get_shares_outstanding(["OK", "BAD"], cache_dir=tmp_path)
    consultas = _FakeTicker.instancias
    assert consultas == 2

    segundo = sc.get_shares_outstanding(["OK", "BAD"], cache_dir=tmp_path)

    assert _FakeTicker.instancias == consultas  # no se vuelve a consultar
    pd.testing.assert_series_equal(segundo, primero)


def test_fetch_spy_sector_weights_error_si_no_hay_datos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si yfinance no devuelve composición sectorial, se avisa con claridad."""
    import thesis.sector_composition as sc

    class _FakeTicker:
        funds_data = None

    monkeypatch.setattr(sc.yf, "Ticker", lambda symbol: _FakeTicker())

    with pytest.raises(ValueError, match="composición sectorial"):
        sc.fetch_spy_sector_weights("SPY")


def test_fetch_spy_sector_weights_normaliza(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los pesos de yfinance se devuelven como serie en fracción."""
    import thesis.sector_composition as sc

    class _FakeFunds:
        sector_weightings: ClassVar[dict[str, float]] = {
            "technology": 0.3,
            "energy": 0.05,
        }

    class _FakeTicker:
        funds_data = _FakeFunds()

    monkeypatch.setattr(sc.yf, "Ticker", lambda symbol: _FakeTicker())

    weights = sc.fetch_spy_sector_weights("SPY")

    assert weights["technology"] == pytest.approx(0.3)
    assert float(weights.sum()) == pytest.approx(0.35)


def test_constantes_publicas() -> None:
    """Las constantes del módulo tienen los valores documentados."""
    assert MINOR_SECTOR_THRESHOLD == 0.08
    assert GICS_SECTOR_ES["technology"] == "Tecnología"
    assert SP500_WIKI_URL.startswith("https://en.wikipedia.org/wiki/")


# ----------------------------------------------------------------------
# Importación sin efectos
# ----------------------------------------------------------------------


def test_importar_thesis_sector_composition_no_descarga_ni_dibuja(
    tmp_path: Path,
) -> None:
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
        "import thesis.sector_composition\n"
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


# ----------------------------------------------------------------------
# Integración (requiere red; se omiten por defecto)
# ----------------------------------------------------------------------


@pytest.mark.integration
def test_get_sp500_constituents_integracion() -> None:
    """Wikipedia devuelve la lista de constituyentes del S&P 500."""
    df = get_sp500_constituents()

    assert list(df.columns) == ["Symbol", "Security", "Sector"]
    assert len(df) > 400
    assert df["Symbol"].is_unique


@pytest.mark.integration
def test_fetch_spy_sector_weights_integracion() -> None:
    """Los pesos sectoriales de SPY suman aproximadamente 1."""
    weights = fetch_spy_sector_weights("SPY")

    assert not weights.empty
    assert float(weights.sum()) == pytest.approx(1.0, abs=0.01)
    assert (weights > 0).all()

"""Pruebas de los contratos públicos de :mod:`thesis.pair_trading`.

El foco está en los dataclasses de resultado: si un ``@dataclass`` declara
una documentación de atributos pero ninguna anotación, el constructor
generado acepta cero argumentos y la función que lo instancia falla en
tiempo de ejecución con ``TypeError``.  Estos tests fijan ese contrato.

Además se cubren las funciones numéricas principales con datos
sintéticos sin acceso a red.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from thesis.pair_trading import (
    GICS_SECTOR_ES,
    ADFResult,
    ClusterSectorReport,
    DatasetReport,
    HedgeResult,
    OUParameters,
    PairSelectionUniverse,
    adf_test,
    align_cluster_sector_labels,
    cluster_sector_purity,
    compute_hedge_ratio,
    correlation_distance,
    evaluate_cluster_sector_alignment,
    export_pair_selection_table,
    filter_by_history,
    fit_ou_process,
    gaussian_mi_matrix,
    hierarchical_clustering,
    ksg_mi_matrix,
    load_log_prices,
    load_log_returns,
    load_returns_panel,
    mi_distance,
    prepare_pair_selection_universe,
    print_dataset_report,
    sector_average_matrix,
    select_table_columns,
    simulate_gbm,
    simulate_ou_process,
    top_pairs,
    translate_gics_sectors,
    unmapped_gics_sectors,
)

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PFE", "KO", "WMT"]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    """Generador aleatorio reproducible."""
    return np.random.default_rng(42)


@pytest.fixture
def cointegrated_pair(rng):
    """Construye dos log-precios cointegrados sin descargar datos.

    ``log_a`` es una transformación lineal de ``log_b`` más un residual
    OU estacionario pequeño, por lo que la regresión log-lineal tiene
    R² alto y el spread resultante es estacionario.
    """
    n = 252
    log_b = np.cumsum(rng.normal(0, 0.01, size=n))
    spread = simulate_ou_process(
        theta=15.0, mu=0.0, sigma=0.05, x0=0.0, T=1.0, dt=1 / 252
    )
    log_a = 0.5 + 0.8 * log_b + spread
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(log_a, index=idx, name="A"), pd.Series(log_b, index=idx, name="B")


@pytest.fixture
def returns_synthetic(rng):
    """DataFrame de retornos sintéticos para tests de matrices."""
    n = 120
    data = rng.multivariate_normal(
        mean=np.zeros(4),
        cov=[
            [1.0, 0.8, 0.1, 0.0],
            [0.8, 1.0, 0.0, 0.1],
            [0.1, 0.0, 1.0, 0.2],
            [0.0, 0.1, 0.2, 1.0],
        ],
        size=n,
    )
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C", "D"])


@pytest.fixture
def sector_labels():
    """Sectores GICS de ejemplo para los tickers sintéticos."""
    return pd.Series(
        ["Tech", "Tech", "Health", "Health", "Cons", "Cons", "Fin", "Fin"],
        index=TICKERS,
        name="sector",
    )


@pytest.fixture
def precios_sinteticos():
    """Precios ajustados con un hueco en ``B`` y un ticker casi vacío en ``C``."""
    idx = pd.date_range("2023-01-02", periods=6, freq="B")
    return pd.DataFrame(
        {
            "A": [100.0, 101.0, 99.5, 102.0, 103.0, 104.0],
            "B": [50.0, 51.0, np.nan, 52.0, 53.0, 54.0],
            "C": [10.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=idx,
    )


@pytest.fixture
def panel_con_ipo_reciente():
    """Serie larga de dos activos y un tercero que acaba de cotizar.

    Reproduce el fallo que motivates :func:`load_returns_panel`: sin filtro
    de historia, el ``dropna`` global reduce el panel a las 30 sesiones del
    activo recién salgado a bolsa.
    """
    idx = pd.date_range("2019-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    precios = pd.DataFrame(
        {
            "A": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
            "B": 50.0 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx)))),
        },
        index=idx,
    )
    precios["C"] = np.nan
    precios.loc[idx[-30:], "C"] = 20.0 * np.exp(
        np.cumsum(rng.normal(0, 0.02, 30))
    )
    return precios


# =============================================================================
# Carga de precios y retornos
# =============================================================================


def _patch_prices(monkeypatch, precios: pd.DataFrame) -> None:
    """Sustituye la descarga de precios por un frame determinista."""
    monkeypatch.setattr(
        "thesis.pair_trading.load_multiple_prices",
        lambda *args, **kwargs: precios.copy(),
    )


def test_load_log_prices_es_log_de_los_precios(
    monkeypatch, precios_sinteticos
) -> None:
    """``load_log_prices`` es el logaritmo de los precios comunes."""
    _patch_prices(monkeypatch, precios_sinteticos)

    log_precios = load_log_prices(["A", "B"])

    esperado = np.log(precios_sinteticos[["A", "B"]].dropna())
    pd.testing.assert_frame_equal(log_precios, esperado)


def test_load_log_prices_conserva_niveles_reales(
    monkeypatch, precios_sinteticos
) -> None:
    """Los niveles no son reconstruidos con ``cumsum`` de log-retornos.

    Reconstruir ``log P`` sumando log-retornos tras eliminar las filas con
    ``NaN`` introduce una constante por activo y pierde las fechas en las
    que el otro activo no cotizó; el ADF sobre esa serie no coincide con
    el de ``PairTrading.py``.
    """
    _patch_prices(monkeypatch, precios_sinteticos)

    log_precios = load_log_prices(["A", "B"])
    fila_inicial = precios_sinteticos.index[0]

    assert log_precios.loc[fila_inicial, "A"] == pytest.approx(np.log(100.0))
    assert log_precios.loc[fila_inicial, "B"] == pytest.approx(np.log(50.0))
    # La fecha en la que ``B`` no cotizó se descarta para ambos activos.
    assert precios_sinteticos.index[2] not in log_precios.index
    assert len(log_precios) == 5


def test_load_log_prices_descarta_ticker_sin_datos(
    monkeypatch, precios_sinteticos
) -> None:
    """Los tickers por debajo de ``min_valid_obs`` se eliminan."""
    _patch_prices(monkeypatch, precios_sinteticos)

    log_precios = load_log_prices(["A", "B", "C"])

    assert "C" not in log_precios.columns


def test_load_log_prices_exige_dos_tickers_validos(
    monkeypatch, precios_sinteticos
) -> None:
    """Si sólo sobrevive un ticker se lanza ``ValueError``."""
    _patch_prices(monkeypatch, precios_sinteticos[["A", "C"]])

    with pytest.raises(ValueError, match="al menos 2 tickers"):
        load_log_prices(["A", "C"])


def test_load_log_prices_ignora_precios_no_positivos(
    monkeypatch, precios_sinteticos
) -> None:
    """Un precio no positivo se descarta en lugar de producir ``-inf``."""
    precios = precios_sinteticos.copy()
    precios.loc[precios.index[3], "B"] = 0.0
    _patch_prices(monkeypatch, precios)

    log_precios = load_log_prices(["A", "B"])

    assert np.isfinite(log_precios.to_numpy()).all()
    assert precios.index[3] not in log_precios.index


def test_load_log_returns_es_log_retorno_de_precios(
    monkeypatch, precios_sinteticos
) -> None:
    """``load_log_returns`` devuelve la log-retorno de los precios comunes."""
    _patch_prices(monkeypatch, precios_sinteticos)

    with pytest.warns(UserWarning, match="degradado"):
        retornos = load_log_returns(["A", "B"])

    # La retorno se calcula sobre la serie completa y el ``NaN`` del hueco
    # de ``B`` se descarta después, de modo que el retorno de ``B`` el día
    # en que vuelve a cotizar abarca los dos días.
    esperado = np.log(
        precios_sinteticos[["A", "B"]] / precios_sinteticos[["A", "B"]].shift(1)
    ).dropna(how="any")
    pd.testing.assert_frame_equal(retornos, esperado)


# =============================================================================
# Filtrado de historia y diagnóstico del panel
# =============================================================================


def test_filter_by_history_descarta_ticker_reciente(precios_sinteticos) -> None:
    """``filter_by_history`` separa conservados y descartados con su motivo."""
    panel, dropped = filter_by_history(precios_sinteticos, min_history_obs=5)

    assert list(panel.columns) == ["A", "B"]
    assert list(dropped.index) == ["C"]
    assert dropped.loc["C", "n_obs"] == 1
    assert dropped.loc["C", "motivo"] == "historia_insuficiente"
    assert dropped.loc["C", "primera_fecha"] == precios_sinteticos.index[0]


def test_filter_by_history_motivo_sin_datos(precios_sinteticos) -> None:
    """Sin umbral de historia, el descarte se atribuye a la falta de datos."""
    _, dropped = filter_by_history(precios_sinteticos, min_valid_obs=3)

    assert dropped.loc["C", "motivo"] == "sin_datos"


def test_filter_by_history_informa_de_historia_insuficiente() -> None:
    """Un ticker con datos pero pocos queda marcado como historia corta."""
    idx = pd.date_range("2023-01-02", periods=10, freq="B")
    precios = pd.DataFrame(
        {"A": np.arange(1.0, 11.0), "B": [np.nan] * 8 + [5.0, 6.0]}, index=idx
    )

    _, dropped = filter_by_history(precios, min_history_obs=5, min_valid_obs=3)

    assert dropped.loc["B", "motivo"] == "historia_insuficiente"
    assert dropped.loc["B", "n_obs"] == 2
    assert dropped.loc["B", "primera_fecha"] == idx[8]


def test_filter_by_history_sin_descartes_devuelve_tabla_vacia() -> None:
    """Si nadie queda fuera, la tabla de descarte conserva sus columnas."""
    idx = pd.date_range("2023-01-02", periods=10, freq="B")
    precios = pd.DataFrame({"A": np.arange(1.0, 11.0), "B": np.arange(2.0, 12.0)},
                           index=idx)

    panel, dropped = filter_by_history(precios, min_history_obs=5)

    assert list(panel.columns) == ["A", "B"]
    assert dropped.empty
    assert list(dropped.columns) == [
        "n_obs", "primera_fecha", "ultima_fecha", "motivo",
    ]


def test_panel_conserva_historia_tras_filtrar(
    monkeypatch, panel_con_ipo_reciente
) -> None:
    """La IPO reciente se descarta sin truncar la ventana del panel.

    Sin el filtro previo al alineado, ``dropna(how="any")`` conservaría
    únicamente las 30 sesiones en las que también cotiza ``C``.
    """
    _patch_prices(monkeypatch, panel_con_ipo_reciente)

    retornos, _ = load_returns_panel(
        ["A", "B", "C"],
        "2019-01-01",
        "2020-08-01",
        min_history_obs=250,
    )

    assert retornos.shape[1] == 2
    assert len(retornos) == 399
    assert retornos.index[0] == panel_con_ipo_reciente.index[1]


def test_load_returns_panel_diagnostica_la_cobertura(
    monkeypatch, panel_con_ipo_reciente
) -> None:
    """El informe expone sesiones, activos, cobertura y ritmo anual."""
    _patch_prices(monkeypatch, panel_con_ipo_reciente)

    retornos, report = load_returns_panel(
        ["A", "B", "C"],
        "2019-01-01",
        "2020-08-01",
        min_history_obs=250,
    )

    assert isinstance(report, DatasetReport)
    assert report.requested_start == "2019-01-01"
    assert report.requested_end == "2020-08-01"
    assert report.n_sessions == len(retornos) == 399
    assert report.n_assets == 2
    assert report.n_assets_requested == 3
    assert report.first_date == retornos.index[0]
    assert report.last_date == retornos.index[-1]
    assert report.calendar_days == (
        retornos.index[-1] - retornos.index[0]
    ).days
    # El ritmo anual confirma que el índice es de sesiones, no de días.
    assert 200.0 <= report.sessions_per_year <= 270.0
    # La cobertura se mide contra el calendario realmente disponible, no
    # contra la ventana pedida: así se ve cuánto pierde la intersección.
    assert report.union_sessions == 400
    assert report.coverage_pct == pytest.approx(399 / 400)
    assert 0.0 < report.window_pct <= 1.0
    assert report.history_per_asset.to_dict() == {"A": 400, "B": 400}


def test_load_returns_panel_informa_de_los_descartes(
    monkeypatch, panel_con_ipo_reciente
) -> None:
    """Los activos descartados quedan listados con su conteo y su motivo."""
    _patch_prices(monkeypatch, panel_con_ipo_reciente)

    _, report = load_returns_panel(
        ["A", "B", "C"],
        "2019-01-01",
        "2020-08-01",
        min_history_obs=250,
    )

    assert list(report.dropped_assets.index) == ["C"]
    assert report.dropped_assets.loc["C", "n_obs"] == 30
    assert report.dropped_assets.loc["C", "motivo"] == "historia_insuficiente"


def test_load_returns_panel_advierte_si_la_muestra_es_corta(
    monkeypatch, panel_con_ipo_reciente
) -> None:
    """Sin filtro de historia el panel se degrada y se advierte."""
    _patch_prices(monkeypatch, panel_con_ipo_reciente)

    with pytest.warns(UserWarning, match="sesiones"):
        retornos, report = load_returns_panel(
            ["A", "B", "C"], "2019-01-01", "2020-08-01",
        )

    # Sin el filtro sobreviven los tres tickers, pero sólo las 30 sesiones
    # en las que también cotiza ``C``.
    assert report.n_sessions == 29
    assert report.n_assets == 3
    assert len(retornos.columns) == 3
    # La intersección usa el 7% del calendario disponible.
    assert report.union_sessions == 400
    assert report.coverage_pct == pytest.approx(29 / 400)


def test_load_returns_panel_coincide_con_load_log_returns(
    monkeypatch, panel_con_ipo_reciente
) -> None:
    """El atajo ``load_log_returns`` devuelve el mismo panel."""
    _patch_prices(monkeypatch, panel_con_ipo_reciente)

    con_informe, _ = load_returns_panel(
        ["A", "B", "C"], "2019-01-01", "2020-08-01", min_history_obs=250,
    )
    atajo = load_log_returns(
        ["A", "B", "C"], "2019-01-01", "2020-08-01", min_history_obs=250,
    )

    pd.testing.assert_frame_equal(con_informe, atajo)


# =============================================================================
# Universo compartido por los cuadernos de selección
# =============================================================================


@pytest.fixture
def constituyentes_sin_duplicados():
    """Constituyentes con emisores distintos (el emisor se deriva de ``Security``)."""
    return pd.DataFrame(
        {
            "Symbol": ["A", "B", "D"],
            "Security": ["Alpha", "Beta", "Delta"],
            "GICS Sector": ["Tech", "Tech", "Fin"],
        }
    )


@pytest.fixture
def constituyentes_con_duplicados():
    """Dos clases de acción por emisor, con nombres de empresa reales.

    ``filter_duplicate_issuers`` deriva el emisor de ``Security``, así que la
    agrupación sólo ocurre si ambas filas comparten la raíz de la empresa.
    """
    return pd.DataFrame(
        {
            "Symbol": ["GOOGL", "GOOG", "FOXA", "FOX"],
            "Security": [
                "Alphabet Inc. Class A",
                "Alphabet Inc. Class C",
                "Fox Corporation Class A",
                "Fox Corporation Class B",
            ],
            "GICS Sector": ["Tech", "Tech", "Cons", "Cons"],
        }
    )


def test_prepare_universo_filtra_por_historia(
    monkeypatch, panel_con_ipo_reciente, constituyentes_sin_duplicados
) -> None:
    """El universo descarta la IPO reciente antes de alinear el panel."""
    _patch_prices(monkeypatch, panel_con_ipo_reciente.rename(
        columns={"C": "D"}
    ))

    universo = prepare_pair_selection_universe(
        constituents=constituyentes_sin_duplicados,
        start="2019-01-01",
        end="2020-08-01",
        min_history_obs=250,
        verbose=False,
    )

    assert isinstance(universo, PairSelectionUniverse)
    assert list(universo.returns.columns) == ["A", "B"]
    assert universo.report.n_sessions == 399
    assert list(universo.report.dropped_assets.index) == ["D"]
    assert universo.report.dropped_assets.loc["D", "motivo"] == (
        "historia_insuficiente"
    )


def test_prepare_universo_alinea_sectores_y_emisores(
    monkeypatch, panel_con_ipo_reciente, constituyentes_sin_duplicados
) -> None:
    """Sectores y emisores quedan indexados como las columnas del panel."""
    _patch_prices(monkeypatch, panel_con_ipo_reciente.rename(
        columns={"C": "D"}
    ))

    universo = prepare_pair_selection_universe(
        constituents=constituyentes_sin_duplicados,
        start="2019-01-01",
        end="2020-08-01",
        min_history_obs=250,
        verbose=False,
    )

    assert list(universo.sectors.index) == list(universo.returns.columns)
    assert list(universo.issuers.index) == list(universo.returns.columns)
    assert universo.sectors.to_dict() == {"A": "Tech", "B": "Tech"}
    # El emisor se deriva de ``Security``, no de una columna previa.
    assert universo.issuers.to_dict() == {"A": "alpha", "B": "beta"}
    assert universo.n_sectors == 1


def test_prepare_universo_descarta_clases_del_mismo_emisor(
    monkeypatch, panel_con_ipo_reciente, constituyentes_con_duplicados
) -> None:
    """Cada emisor aporta un único símbolo antes de tocar los precios."""
    precios = panel_con_ipo_reciente.rename(
        columns={"A": "GOOGL", "B": "FOXA", "C": "FOX"}
    )
    _patch_prices(monkeypatch, precios)

    universo = prepare_pair_selection_universe(
        constituents=constituyentes_con_duplicados,
        start="2019-01-01",
        end="2020-08-01",
        min_history_obs=250,
        verbose=False,
    )

    assert sorted(universo.removed_duplicate_issuers["Symbol"]) == ["FOX", "GOOG"]
    assert list(universo.returns.columns) == ["GOOGL", "FOXA"]
    assert universo.issuers.to_dict() == {"GOOGL": "alphabet", "FOXA": "fox"}


def test_prepare_universo_imprime_diagnostico(
    monkeypatch, panel_con_ipo_reciente, constituyentes_con_duplicados, capsys
) -> None:
    """Con ``verbose`` se imprimen el universo y el diagnóstico de cobertura."""
    precios = panel_con_ipo_reciente.rename(
        columns={"A": "GOOGL", "B": "FOXA", "C": "FOX"}
    )
    _patch_prices(monkeypatch, precios)

    prepare_pair_selection_universe(
        constituents=constituyentes_con_duplicados,
        start="2019-01-01",
        end="2020-08-01",
        min_history_obs=250,
    )

    salida = capsys.readouterr().out
    assert "activos únicos" in salida
    assert "Ventana efectiva" in salida
    assert "Sesiones" in salida
    assert "Cobertura" in salida
    assert "Sectores GICS presentes" in salida


def test_print_dataset_report_resume_y_mitiga_descarte(capsys) -> None:
    """El informe resume la cobertura y recorta la lista de descartes."""
    report = DatasetReport(
        requested_start="2019-01-01",
        requested_end="2020-08-01",
        first_date=pd.Timestamp("2019-01-02"),
        last_date=pd.Timestamp("2020-07-31"),
        n_sessions=400,
        n_assets=2,
        n_assets_requested=6,
        min_history_obs=250,
        union_sessions=402,
        coverage_pct=400 / 402,
        window_pct=0.9,
        expected_sessions=410,
        calendar_days=576,
        sessions_per_year=253.0,
        history_per_asset=pd.Series({"A": 402, "B": 402}),
        dropped_assets=pd.DataFrame(
            {
                "n_obs": [10, 20, 30, 40],
                "motivo": ["historia_insuficiente"] * 4,
            },
            index=["C", "D", "E", "F"],
        ),
    )

    print_dataset_report(report, min_history_obs=250, max_dropped=2)
    salida = capsys.readouterr().out

    assert "exige ≥ 250 sesiones" in salida
    assert "99.5%" in salida
    assert "y 2 más" in salida
    # Con ``max_dropped=2`` sólo se listan C y D.
    assert "\nC" in salida and "\nD" in salida
    assert "\nE" not in salida and "\nF" not in salida


# =============================================================================
# Contrato de los dataclasses de resultado
# =============================================================================

RESULT_DATACLASSES = [
    ClusterSectorReport,
    HedgeResult,
    ADFResult,
    OUParameters,
]

DOCUMENTED_FIELDS = {
    ClusterSectorReport: (
        "contingency",
        "per_cluster",
        "weighted_purity",
        "normalized_entropy",
        "mean_hhi",
        "ari",
        "ami",
        "homogeneity",
        "completeness",
        "v_measure",
    ),
    HedgeResult: ("alpha", "beta", "r_squared", "model"),
    ADFResult: ("statistic", "pvalue", "is_stationary"),
    OUParameters: (
        "theta",
        "mu",
        "sigma",
        "half_life",
        "sigma_eq",
        "r_squared",
    ),
}


@pytest.mark.parametrize("cls", RESULT_DATACLASSES)
def test_dataclass_declara_campos(cls: type) -> None:
    """Cada dataclass de resultado declara los campos de su documentación."""
    declared = tuple(f.name for f in dataclasses.fields(cls))
    assert declared == DOCUMENTED_FIELDS[cls]


@pytest.mark.parametrize("cls", RESULT_DATACLASSES)
def test_dataclass_es_inmutable(cls: type) -> None:
    """Los dataclasses de resultado son ``frozen``."""
    assert cls.__dataclass_params__.frozen is True


# =============================================================================
# Hedge ratio
# =============================================================================


def test_compute_hedge_ratio_con_cointegracion(cointegrated_pair) -> None:
    """Sobre log-precios cointegrados, beta es finito y R² alto."""
    log_a, log_b = cointegrated_pair
    hedge = compute_hedge_ratio(log_a, log_b)

    assert isinstance(hedge, HedgeResult)
    assert np.isfinite(hedge.beta)
    assert np.isfinite(hedge.alpha)
    assert 0.0 <= hedge.r_squared <= 1.0
    assert hedge.r_squared > 0.7


def test_compute_hedge_ratio_series_sin_superposicion() -> None:
    """Si no hay observaciones comunes se lanza ``ValueError``."""
    a = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2023-01-01", periods=3))
    b = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3))

    with pytest.raises(ValueError, match="observaciones comunes"):
        compute_hedge_ratio(a, b)


def test_compute_hedge_ratio_con_nan() -> None:
    """Los NaN se eliminan por pares antes de ajustar la OLS."""
    idx = pd.date_range("2023-01-01", periods=10)
    a = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx
    )
    b = pd.Series([2.0, 4.0, 5.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)

    hedge = compute_hedge_ratio(a, b)
    assert np.isfinite(hedge.beta)


# =============================================================================
# ADF
# =============================================================================


def test_adf_detecta_estacionariedad(cointegrated_pair) -> None:
    """Sobre el spread cointegrado el ADF rechaza la raíz unitaria."""
    log_a, log_b = cointegrated_pair
    hedge = compute_hedge_ratio(log_a, log_b)
    spread = log_a - (hedge.alpha + hedge.beta * log_b)
    adf = adf_test(spread)

    assert isinstance(adf, ADFResult)
    assert adf.is_stationary
    assert adf.pvalue < 0.05
    assert np.isfinite(adf.statistic)


def test_adf_sobre_ruido_blanco_no_estacionario(rng) -> None:
    """Sobre una caminata aleatoria el ADF no rechaza la raíz unitaria."""
    walk = np.cumsum(rng.normal(0, 0.01, size=252))
    spread = pd.Series(walk, index=pd.date_range("2023-01-01", periods=252))

    adf = adf_test(spread)
    assert not adf.is_stationary


def test_adf_spread_vacio() -> None:
    """Un spread vacío lanza ``ValueError``."""
    spread = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="observaciones válidas"):
        adf_test(spread)


# =============================================================================
# Proceso OU
# =============================================================================


def test_fit_ou_process_recupera_parametros(rng) -> None:
    """Ajustar OU sobre una trayectoria simulada da theta y sigma positivos."""
    theta_true, mu_true, sigma_true = 5.0, 0.5, 0.3
    x = simulate_ou_process(
        theta_true, mu_true, sigma_true, x0=mu_true, T=5.0, dt=1 / 252
    )
    spread = pd.Series(x, index=pd.date_range("2023-01-01", periods=len(x)))

    ou = fit_ou_process(spread)

    assert isinstance(ou, OUParameters)
    assert ou.theta > 0.0
    assert ou.sigma > 0.0
    assert ou.half_life > 0.0
    assert ou.sigma_eq > 0.0
    assert 0.0 <= ou.r_squared <= 1.0


def test_fit_ou_process_muestra_insuficiente() -> None:
    """Con menos de 3 observaciones se lanza ``ValueError``."""
    spread = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="3 observaciones"):
        fit_ou_process(spread)


def test_simulate_ou_process_forma_y_media(rng) -> None:
    """La simulación OU devuelve el número de puntos esperado."""
    x = simulate_ou_process(theta=2.0, mu=1.0, sigma=0.1, x0=1.0, T=1.0, dt=1 / 252)
    assert len(x) == 252
    assert np.isfinite(x).all()


def test_simulate_gbm_positivo(rng) -> None:
    """El GBM simulado siempre es positivo."""
    prices = simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, T=1.0, dt=1 / 252)
    assert len(prices) == 252
    assert (prices > 0).all()


# =============================================================================
# Mutual Information
# =============================================================================


def test_gaussian_mi_matrix_diagonal_nan(returns_synthetic) -> None:
    """La diagonal de la MI gaussiana es NaN."""
    mi = gaussian_mi_matrix(returns_synthetic)
    assert np.isnan(np.diag(mi.to_numpy())).all()
    assert (mi.to_numpy()[~np.eye(mi.shape[0], dtype=bool)] >= 0).all()


def test_gaussian_mi_matrix_activos_independientes(rng) -> None:
    """Para activos independientes la MI gaussiana es cercana a cero."""
    data = rng.normal(0, 1, size=(500, 2))
    returns = pd.DataFrame(data, columns=["A", "B"])
    mi = gaussian_mi_matrix(returns)
    assert mi.loc["A", "B"] < 0.1


def test_ksg_mi_matrix_diagonal_nan(returns_synthetic) -> None:
    """La diagonal de la MI KSG es NaN y pares correlacionados tienen MI alta."""
    mi = ksg_mi_matrix(returns_synthetic, k_neighbors=3, seed=42)
    assert np.isnan(np.diag(mi.to_numpy())).all()
    assert mi.loc["A", "B"] > mi.loc["A", "C"]


def test_ksg_mi_matrix_dataframe_vacio() -> None:
    """Un DataFrame vacío lanza ``ValueError``."""
    with pytest.raises(ValueError, match="vacío"):
        ksg_mi_matrix(pd.DataFrame())


def test_ksg_mi_matrix_series_varianza_cero() -> None:
    """Una serie de varianza cero deja NaN en su fila/columna."""
    data = np.column_stack([np.ones(50), np.arange(50)])
    returns = pd.DataFrame(data, columns=["A", "B"])
    mi = ksg_mi_matrix(returns, k_neighbors=3, seed=42)
    assert np.isnan(mi.loc["A", "B"]) or mi.loc["A", "B"] == 0.0


# =============================================================================
# Selección de pares
# =============================================================================


def test_top_pairs_devuelve_n(returns_synthetic) -> None:
    """top_pairs devuelve a lo sumo n pares ordenados."""
    mi = gaussian_mi_matrix(returns_synthetic)
    pairs = top_pairs(mi, n=3)

    assert len(pairs) == 3
    for a, b, val in pairs:
        assert a in mi.columns
        assert b in mi.columns
        assert a != b
        assert np.isfinite(val)
    # Deben estar ordenados de mayor a menor
    vals = [val for _, _, val in pairs]
    assert vals == sorted(vals, reverse=True)


def test_top_pairs_filtra_mismo_emisor() -> None:
    """Con ``issuer_map`` se descartan pares del mismo emisor."""
    mi = pd.DataFrame(
        [[np.nan, 1.0, 0.5], [1.0, np.nan, 0.4], [0.5, 0.4, np.nan]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    issuer_map = pd.Series({"A": "Alpha", "B": "Alpha", "C": "Beta"})
    pairs = top_pairs(mi, n=5, issuer_map=issuer_map)

    assert all({a, b} != {"A", "B"} for a, b, _ in pairs)


def test_top_pairs_ascendente(returns_synthetic) -> None:
    """ascending=True devuelve los pares con menor valor."""
    mi = gaussian_mi_matrix(returns_synthetic)
    pairs = top_pairs(mi, n=3, ascending=True)
    vals = [val for _, _, val in pairs]
    assert vals == sorted(vals)


# =============================================================================
# Distancias y clustering
# =============================================================================


def test_correlation_distance_propiedades(returns_synthetic) -> None:
    """correlation_distance devuelve una matriz simétrica con diagonal 0."""
    corr = returns_synthetic.corr(method="pearson")
    dist = correlation_distance(corr, use_sqrt=True)

    assert dist.shape == (corr.shape[0], corr.shape[1])
    assert np.allclose(np.diag(dist), 0.0)
    assert np.allclose(dist, dist.T)
    assert (dist >= 0).all()
    assert (dist <= np.sqrt(2)).all()


def test_mi_distance_propiedades(returns_synthetic) -> None:
    """mi_distance cumple autodistancia nula y simetría."""
    mi = gaussian_mi_matrix(returns_synthetic)
    dist = mi_distance(mi)

    assert dist.shape == (mi.shape[0], mi.shape[1])
    assert np.allclose(np.diag(dist), 0.0)
    assert np.allclose(dist, dist.T)
    assert (dist >= 0).all()


def test_hierarchical_clustering_devuelve_etiquetas(returns_synthetic) -> None:
    """hierarchical_clustering devuelve tantas etiquetas como activos."""
    corr = returns_synthetic.corr(method="pearson")
    dist = correlation_distance(corr, use_sqrt=True)
    labels, order = hierarchical_clustering(dist, n_clusters=2)

    assert len(labels) == len(returns_synthetic.columns)
    assert set(labels) <= {1, 2}
    assert len(order) == len(returns_synthetic.columns)


# =============================================================================
# Evaluación sectorial de clusters
# =============================================================================


def test_evaluacion_con_alineacion_perfecta() -> None:
    """Si los clusters coinciden con los sectores, todas las métricas valen 1."""
    clusters, sectors = _labels(
        [0, 0, 1, 1, 2, 2, 3, 3],
        ["Info", "Info", "Fin", "Fin", "Health", "Health", "Cons", "Cons"],
    )

    report = evaluate_cluster_sector_alignment(clusters, sectors)

    assert isinstance(report, ClusterSectorReport)
    assert report.weighted_purity == pytest.approx(1.0)
    assert report.normalized_entropy == pytest.approx(0.0)
    assert report.mean_hhi == pytest.approx(1.0)
    assert report.ari == pytest.approx(1.0)
    assert report.ami == pytest.approx(1.0)
    assert report.homogeneity == pytest.approx(1.0)
    assert report.completeness == pytest.approx(1.0)
    assert report.v_measure == pytest.approx(1.0)


def test_evaluacion_con_clusters_mixtos() -> None:
    """Con clusters mezclados las métricas caen dentro de su rango y son
    consistentes con la tabla de contingencia."""
    clusters, sectors = _labels(
        [0, 0, 0, 1, 1, 2, 2, 2],
        ["Info", "Info", "Tech", "Tech", "Tech", "Fin", "Fin", "Info"],
    )

    report = evaluate_cluster_sector_alignment(clusters, sectors)

    assert 0.0 < report.weighted_purity < 1.0
    assert 0.0 <= report.normalized_entropy <= 1.0
    assert 0.0 < report.mean_hhi <= 1.0
    assert -1.0 <= report.ari <= 1.0
    assert 0.0 <= report.ami <= 1.0
    assert 0.0 <= report.v_measure <= 1.0

    assert list(report.per_cluster.columns) == [
        "size",
        "dominant_sector",
        "purity",
        "norm_entropy",
        "hhi",
    ]
    assert report.per_cluster["size"].sum() == len(clusters)
    assert report.per_cluster.index.tolist() == [0, 1, 2]

    # La purity global es la media ponderada de la purity por cluster.
    expected = (
        report.per_cluster["purity"] * report.per_cluster["size"]
    ).sum() / report.per_cluster["size"].sum()
    assert report.weighted_purity == pytest.approx(expected)


def test_contingency_incluye_filas_de_resumen() -> None:
    """La contingencia sector × cluster añade ``cluster_size`` y
    ``cluster_purity`` como filas."""
    clusters, sectors = _labels(
        [0, 0, 1, 1, 2, 2],
        ["Info", "Info", "Tech", "Tech", "Health", "Health"],
    )

    report = evaluate_cluster_sector_alignment(clusters, sectors)
    contingency = report.contingency

    assert {"cluster_size", "cluster_purity"} <= set(contingency.index)
    assert set(contingency.index) >= set(sectors)
    assert contingency.loc["cluster_size"].sum() == len(clusters)
    for cid in contingency.columns:
        assert contingency.loc["cluster_purity", cid] <= 1.0


def test_evaluacion_descarta_tickers_no_comunes() -> None:
    """Sólo se evalúan los tickers presentes en ambas series."""
    clusters, sectors = _labels([0, 0, 1, 1], ["Info", "Info", "Tech", "Tech"])
    sectors = sectors.drop(index="XOM")

    report = evaluate_cluster_sector_alignment(clusters, sectors)

    assert report.per_cluster["size"].sum() == 3


def test_evaluacion_sin_tickers_comunes_falla() -> None:
    """Sin tickers comunes la función lanza ``ValueError``."""
    clusters = pd.Series([0, 1], index=["AAA", "BBB"], name="cluster")
    sectors = pd.Series(["Info", "Tech"], index=["CCC", "DDD"], name="sector")

    with pytest.raises(ValueError, match="tickers comunes"):
        evaluate_cluster_sector_alignment(clusters, sectors)


def test_align_cluster_sector_labels_elimina_nan() -> None:
    """La alineación descarta etiquetas NaN en cualquiera de las series."""
    clusters = pd.Series([0, 1, np.nan, 2], index=TICKERS[:4], name="cluster")
    sectors = pd.Series(
        ["Tech", np.nan, "Fin", "Health"], index=TICKERS[:4], name="sector"
    )

    cl, _se = align_cluster_sector_labels(clusters, sectors)
    assert len(cl) == 2
    assert "MSFT" not in cl.index


def test_cluster_sector_purity_diagonal() -> None:
    """Con clusters iguales a sectores, la purity de cada cluster es 1."""
    clusters, sectors = _labels(
        [0, 0, 1, 1, 2, 2],
        ["Tech", "Tech", "Fin", "Fin", "Health", "Health"],
    )
    contingency = cluster_sector_purity(clusters, sectors)
    purity = contingency.loc["cluster_purity"]
    assert (purity == 1.0).all()


# =============================================================================
# Agregación sectorial
# =============================================================================


def test_sector_average_matrix_forma_y_valores(
    returns_synthetic, sector_labels
) -> None:
    """sector_average_matrix reduce a una matriz sector por sector."""
    mi = gaussian_mi_matrix(returns_synthetic)
    # Asignamos sectores a las columnas del fixture
    sectors = pd.Series(
        ["Tech", "Tech", "Health", "Health"],
        index=returns_synthetic.columns,
        name="sector",
    )
    avg = sector_average_matrix(mi, sectors, operation="mean")

    assert avg.shape == (2, 2)
    assert set(avg.index) == {"Tech", "Health"}
    assert np.isfinite(avg.to_numpy()).all()


def test_sector_average_matrix_sin_tickers_comunes() -> None:
    """Sin tickers comunes entre matriz y sectores se lanza ``ValueError``."""
    matrix = pd.DataFrame(
        [[np.nan, 1.0], [1.0, np.nan]], index=["A", "B"], columns=["A", "B"]
    )
    sectors = pd.Series({"C": "Tech", "D": "Health"})

    with pytest.raises(ValueError, match="tickers comunes"):
        sector_average_matrix(matrix, sectors)


# =============================================================================
# Datos vacíos, NaN y muestras insuficientes
# =============================================================================


def test_compute_hedge_ratio_datos_vacios() -> None:
    """Series vacías deben lanzar ``ValueError``."""
    a = pd.Series(dtype=float)
    b = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="observaciones comunes"):
        compute_hedge_ratio(a, b)


def test_adf_test_spread_todo_nan() -> None:
    """Un spread compuesto solo por NaN debe lanzar ``ValueError``."""
    spread = pd.Series([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="observaciones válidas"):
        adf_test(spread)


def test_fit_ou_process_spread_corto() -> None:
    """Un spread con menos de 3 puntos no permite ajustar OU."""
    spread = pd.Series([1.0, 1.1])
    with pytest.raises(ValueError, match="3 observaciones"):
        fit_ou_process(spread)


def test_gaussian_mi_matrix_vacia() -> None:
    """Un DataFrame vacío lanza ``ValueError``."""
    with pytest.raises(ValueError, match="vacío"):
        gaussian_mi_matrix(pd.DataFrame())


def test_top_pairs_matriz_vacia() -> None:
    """top_pairs sobre una matriz sin valores finitos devuelve lista vacia."""
    mi = pd.DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    pairs = top_pairs(mi, n=5)
    assert pairs == []


# =============================================================================
# Exportación de tablas
# =============================================================================


@pytest.fixture
def tabla_de_pares():
    """Tabla de pares con valores ausentes, booleanos y decimales."""
    return pd.DataFrame(
        {
            "ticker_a": ["AAPL", "XOM"],
            "ticker_b": ["MSFT", "CVX"],
            "sector_a": ["Info Tech", "Energy"],
            "correlacion": [0.8976, np.nan],
            "n_sesiones": [5047, 8307],
            "estacionario": [True, False],
        }
    )


@pytest.fixture
def reporte_de_ejemplo():
    """Diagnóstico de panel para las pruebas de exportación."""
    return DatasetReport(
        requested_start="1993-01-01",
        requested_end="2026-01-01",
        first_date=pd.Timestamp("2005-12-06"),
        last_date=pd.Timestamp("2025-12-31"),
        n_sessions=5047,
        n_assets=388,
        n_assets_requested=497,
        min_history_obs=5040,
        union_sessions=8307,
        coverage_pct=5047 / 8307,
        window_pct=0.586,
        expected_sessions=8609,
        calendar_days=7300,
        sessions_per_year=251.0,
        history_per_asset=pd.Series({"A": 8307, "B": 8307}),
        dropped_assets=pd.DataFrame(
            {
                "n_obs": [46, 222],
                "motivo": ["historia_insuficiente"] * 2,
            },
            index=["Q", "SNDK"],
        ),
    )


def test_export_escribe_csv_tex_y_json(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """La exportación produce los tres ficheros con el mismo nombre base."""
    rutas = export_pair_selection_table(
        tabla_de_pares,
        reporte_de_ejemplo,
        output_dir=tmp_path,
        stem="prueba",
        method="correlación de Pearson",
        label="tab:prueba",
    )

    assert set(rutas) == {"csv", "tex", "json"}
    for path in rutas.values():
        assert path.exists() and path.stat().st_size > 0
        assert path.parent == tmp_path
    assert rutas["csv"].name == "prueba.csv"
    assert rutas["tex"].name == "prueba.tex"
    assert rutas["json"].name == "prueba.json"


def test_export_anade_ranking_al_csv(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """El CSV incluye el ranking como primera columna."""
    rutas = export_pair_selection_table(
        tabla_de_pares, reporte_de_ejemplo, output_dir=tmp_path, stem="p",
    )

    guardado = pd.read_csv(rutas["csv"])
    assert guardado.columns[0] == "rank"
    assert guardado["rank"].tolist() == [1, 2]


def test_export_respeta_ranking_existente(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """Si la tabla ya trae ``rank``, no se sobrescribe."""
    con_rank = tabla_de_pares.copy()
    con_rank.insert(0, "rank", [7, 8])

    rutas = export_pair_selection_table(
        con_rank, reporte_de_ejemplo, output_dir=tmp_path, stem="p",
    )

    assert pd.read_csv(rutas["csv"])["rank"].tolist() == [7, 8]


def test_export_tex_escapa_y_formatea(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """El LaTeX escapa guiones bajos y formatea ausentes, miles y booleanos."""
    rutas = export_pair_selection_table(
        tabla_de_pares,
        reporte_de_ejemplo,
        output_dir=tmp_path,
        stem="p",
        method="correlación de Pearson",
        caption="Tabla & prueba",
        label="tab:p",
        decimals=3,
    )
    tex = rutas["tex"].read_text(encoding="utf-8")

    # Cabecera escapada: ``_`` debe salir como ``\_``.
    assert r"ticker\_a" in tex
    assert r"n\_sesiones" in tex
    assert r"correlacion" in tex
    # El caption también va escapado.
    assert r"Tabla \& prueba" in tex
    # Flotante con 3 decimales y ausente como ``--``.
    assert "0.898" in tex
    assert "--" in tex
    # Entero con separador de miles y booleano en negrita.
    assert r"5\,047" in tex
    assert r"8\,307" in tex
    assert r"\textbf{Sí}" in tex
    assert "No" in tex
    # Estructura booktabs; las condiciones van en un ``minipage`` para no
    # depender de ``threeparttable``, que aborta con listas y párrafos.
    assert r"\toprule" in tex
    assert r"\midrule" in tex
    assert r"\bottomrule" in tex
    assert r"\begin{minipage}{0.98\linewidth}" in tex
    assert r"\footnotesize" in tex
    assert r"\textbullet\ " in tex
    assert r"\label{tab:p}" in tex
    assert r"\caption{Tabla \& prueba}" in tex
    assert "threeparttable" not in tex


def test_export_tex_cierra_cada_fila_con_doble_barra(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """Cada fila del ``tabular`` termina en ``\\\\``, incluida la cabecera.

    Un único ``\\`` deja la alineación rota y LaTeX aborta con
    ``Misplaced \\noalign`` en el ``\\midrule``; conviene fijarlo porque el
    fallo sólo aparece al compilar.
    """
    rutas = export_pair_selection_table(
        tabla_de_pares, reporte_de_ejemplo, output_dir=tmp_path, stem="p",
    )
    filas = rutas["tex"].read_text(encoding="utf-8").splitlines()

    inicio = next(
        i for i, linea in enumerate(filas) if linea.startswith(r"\begin{tabular}")
    )
    fin = filas.index(r"\bottomrule")
    cuerpo = filas[inicio + 1 : fin]

    # Cabecera + dos filas de datos, cada una cerrada con ``\\\\``.
    assert cuerpo[0] == r"\toprule"
    assert cuerpo[1].startswith("rank & ticker\\_a")
    assert cuerpo[1].endswith(r"\\")
    assert cuerpo[2] == r"\midrule"
    assert cuerpo[3].startswith("1 & AAPL & MSFT")
    assert cuerpo[3].endswith(r"\\")
    assert cuerpo[4].endswith(r"\\")
    # El spec declara una columna por campo de la tabla con ``rank``.
    n_columnas = len(tabla_de_pares.columns) + 1
    spec = filas[inicio].removeprefix(r"\begin{tabular}{").removesuffix("}")
    assert len(spec) == n_columnas


def test_export_tex_documenta_las_condiciones(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """La nota incluye ventana, muestra, universo, filtro y cobertura."""
    rutas = export_pair_selection_table(
        tabla_de_pares,
        reporte_de_ejemplo,
        output_dir=tmp_path,
        stem="p",
        method="correlación de Pearson",
        notes=["Test ADF al 5% de significancia."],
    )
    tex = rutas["tex"].read_text(encoding="utf-8")

    assert "Método de selección: correlación de Pearson." in tex
    assert "Ventana solicitada: 1993-01-01 a 2026-01-01." in tex
    assert "Ventana efectiva: 2005-12-06 a 2025-12-31." in tex
    assert "5047 sesiones" in tex
    assert "251.0 sesiones/año" in tex
    assert "388 activos de 497 solicitados" in tex
    assert "2 descartados por historia insuficiente" in tex
    assert "al menos 5040 sesiones" in tex
    # El tanto por ciento va escapado para LaTeX.
    assert r"60.8\% de las 8307 fechas disponibles" in tex
    assert r"58.6\% de la ventana solicitada" in tex
    assert "Pares seleccionados: 2." in tex
    assert r"Test ADF al 5\% de significancia." in tex


def test_export_json_lleva_las_condiciones(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """El JSON serializa las condiciones como datos utilizables."""
    rutas = export_pair_selection_table(
        tabla_de_pares,
        reporte_de_ejemplo,
        output_dir=tmp_path,
        stem="p",
        method="correlación de Pearson",
        extra_metadata={"significancia_adf": 0.05, "pares_estacionarios": 1},
        notes=["Nota del cuaderno."],
    )
    meta = json.loads(rutas["json"].read_text(encoding="utf-8"))

    assert meta["metodo"] == "correlación de Pearson"
    assert meta["n_pares"] == 2
    assert meta["sesiones"] == 5047
    assert meta["dias_naturales"] == 7300
    assert meta["sesiones_por_ano"] == 251.0
    assert meta["activos_solicitados"] == 497
    assert meta["activos_conservados"] == 388
    assert meta["activos_descartados"] == 2
    assert meta["umbral_historia_sesiones"] == 5040
    assert meta["fechas_disponibles_union"] == 8307
    assert meta["ventana_solicitada"]["inicio"] == "1993-01-01"
    assert meta["ventana_efectiva"]["fin"].startswith("2025-12-31")
    assert meta["significancia_adf"] == 0.05
    assert meta["pares_estacionarios"] == 1
    assert meta["columnas"][0] == "rank"
    assert "Nota del cuaderno." in meta["condiciones"]
    # ``json`` no admite NaN: la cobertura se serializa como float finita.
    assert isinstance(meta["cobertura_panel_pct"], float)


def test_export_rechaza_tabla_vacia(reporte_de_ejemplo, tmp_path) -> None:
    """Una tabla sin filas no se exporta."""
    with pytest.raises(ValueError, match="no tiene filas"):
        export_pair_selection_table(
            pd.DataFrame(columns=["ticker_a"]),
            reporte_de_ejemplo,
            output_dir=tmp_path,
            stem="p",
        )


# =============================================================================
# Selección de columnas
# =============================================================================


def test_select_columns_respeta_el_orden(tabla_de_pares) -> None:
    """Las columnas salen en el orden pedido, no en el de la tabla."""
    elegidas = ["correlacion", "ticker_a", "ticker_b"]

    resultado = select_table_columns(tabla_de_pares, elegidas)

    assert list(resultado.columns) == elegidas
    assert list(resultado.index) == list(tabla_de_pares.index)


def test_select_columns_none_devuelve_todo(tabla_de_pares) -> None:
    """``None`` conserva todas las columnas."""
    resultado = select_table_columns(tabla_de_pares, None)
    assert list(resultado.columns) == list(tabla_de_pares.columns)


def test_select_columns_rechaza_inexistente(tabla_de_pares) -> None:
    """Pedir una columna inexistente nombra la que falta y las disponibles."""
    with pytest.raises(ValueError, match="inexistentes") as exc:
        select_table_columns(tabla_de_pares, ["ticker_a", "beta"])

    mensaje = str(exc.value)
    assert "'beta'" in mensaje
    assert "ticker_a" in mensaje


def test_select_columns_rechaza_repetidas(tabla_de_pares) -> None:
    """No se admiten columnas repetidas."""
    with pytest.raises(ValueError, match="repetidas"):
        select_table_columns(tabla_de_pares, ["ticker_a", "ticker_a"])


def test_select_columns_rechaza_lista_vacia(tabla_de_pares) -> None:
    """Una lista vacía es un error, no una tabla sin columnas."""
    with pytest.raises(ValueError, match="vacía"):
        select_table_columns(tabla_de_pares, [])


def test_select_columns_rechaza_cadena_suelta(tabla_de_pares) -> None:
    """Pasar un nombre suelto evita que se interprete como iterable."""
    with pytest.raises(TypeError, match="secuencia"):
        select_table_columns(tabla_de_pares, "ticker_a")


def test_export_usa_solo_las_columnas_pedidas(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """CSV, LaTeX y JSON comparten exactamente las columnas exportadas."""
    elegidas = ["ticker_a", "ticker_b", "correlacion", "estacionario"]

    rutas = export_pair_selection_table(
        tabla_de_pares,
        reporte_de_ejemplo,
        output_dir=tmp_path,
        stem="p",
        columns=elegidas,
    )

    assert list(pd.read_csv(rutas["csv"]).columns) == elegidas
    meta = json.loads(rutas["json"].read_text(encoding="utf-8"))
    assert meta["columnas"] == elegidas

    cabecera = next(
        linea for linea in rutas["tex"].read_text(encoding="utf-8").splitlines()
        if linea.startswith("ticker\\_a")
    )
    assert cabecera.startswith("ticker\\_a & ticker\\_b")
    assert "sector" not in cabecera
    assert "n\\_sesiones" not in cabecera


def test_export_permite_omitir_el_ranking(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """Si ``rank`` no se pide, la tabla exportada no lo lleva."""
    rutas = export_pair_selection_table(
        tabla_de_pares,
        reporte_de_ejemplo,
        output_dir=tmp_path,
        stem="p",
        columns=["ticker_a", "ticker_b"],
    )

    assert "rank" not in pd.read_csv(rutas["csv"]).columns
    assert "rank" not in json.loads(
        rutas["json"].read_text(encoding="utf-8")
    )["columnas"]


def test_export_admite_ranking_inexistente_en_la_tabla(
    reporte_de_ejemplo, tmp_path
) -> None:
    """``rank`` puede pedirse aunque el DataFrame no lo traiga."""
    sin_rank = pd.DataFrame(
        {"ticker_a": ["AAPL"], "ticker_b": ["MSFT"], "mi": [0.5]}
    )

    rutas = export_pair_selection_table(
        sin_rank,
        reporte_de_ejemplo,
        output_dir=tmp_path,
        stem="p",
        columns=["rank", "ticker_a"],
    )

    assert pd.read_csv(rutas["csv"])["rank"].tolist() == [1]


def test_export_no_escribe_ficheros_si_falla_la_validacion(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """Una columna inexistente aborta antes de dejar ficheros a medias."""
    with pytest.raises(ValueError):
        export_pair_selection_table(
            tabla_de_pares,
            reporte_de_ejemplo,
            output_dir=tmp_path,
            stem="p",
            columns=["ticker_a", "no_existe"],
        )

    assert list(tmp_path.iterdir()) == []


def test_export_crea_el_directorio(
    tabla_de_pares, reporte_de_ejemplo, tmp_path
) -> None:
    """El directorio de destino se crea si no existe."""
    destino = tmp_path / "anidado" / "tablas"

    rutas = export_pair_selection_table(
        tabla_de_pares, reporte_de_ejemplo, output_dir=destino, stem="p",
    )

    assert destino.is_dir()
    assert rutas["tex"].exists()


# =============================================================================
# Traducción de sectores GICS
# =============================================================================


def test_traduce_los_once_sectores_gics() -> None:
    """Los once sectores GICS del S&P 500 están traducidos."""
    expected = {
        "Communication Services": "Servicios de Comunicación",
        "Consumer Discretionary": "Consumo Discrecional",
        "Consumer Staples": "Consumo Básico",
        "Energy": "Energía",
        "Financials": "Finanzas",
        "Health Care": "Salud",
        "Industrials": "Industriales",
        "Information Technology": "Tecnología de la Información",
        "Materials": "Materiales",
        "Real Estate": "Inmobiliario",
        "Utilities": "Servicios Públicos",
    }
    for ingles, espanol in expected.items():
        assert GICS_SECTOR_ES[ingles] == espanol


def test_traduce_ambos_nombres_del_sector_financiero() -> None:
    """GICS renombró ``Financials``; se aceptan ambas denominaciones."""
    assert "Financial Services" in GICS_SECTOR_ES

    series = pd.Series(["Financials", "Financial Services"])
    traducidos = translate_gics_sectors(series)
    assert traducidos.notna().all()
    assert not traducidos.str.startswith("Financial").any()


def test_los_alias_de_un_sector_deben_coincidir() -> None:
    """Las dos denominaciones de un sector deben dar la misma etiqueta.

    Si divergieran y un dataset contuviera ambas, la partición sectorial se
    fragmentaría en dos y purity, entropía y ARI quedarían sesgados.
    """
    assert (
        GICS_SECTOR_ES["Financials"] == GICS_SECTOR_ES["Financial Services"]
    )


def test_traduce_sectores_de_una_serie() -> None:
    """La serie conserva índice, nombre y estructura, sólo cambia el valor."""
    series = pd.Series(
        ["Information Technology", "Health Care", "Real Estate"],
        index=["AAPL", "JNJ", "PLD"],
        name="sector",
    )

    traduccion = translate_gics_sectors(series)

    assert traduccion.tolist() == [
        "Tecnología de la Información", "Salud", "Inmobiliario",
    ]
    assert traduccion.index.tolist() == ["AAPL", "JNJ", "PLD"]
    assert traduccion.name == "sector"


def test_conserva_sectores_desconocidos() -> None:
    """Un sector sin mapeo pasa intacto en lugar de romper el análisis."""
    series = pd.Series(["Utilities", "Sector Inventado"])

    traduccion = translate_gics_sectors(series)

    assert traduccion.tolist() == ["Servicios Públicos", "Sector Inventado"]


def test_traduce_ignorando_nan() -> None:
    """Los ``NaN`` no se convierten en la cadena ``"nan"``."""
    series = pd.Series(["Energy", np.nan])
    traduccion = translate_gics_sectors(series)
    assert traduccion.iloc[0] == "Energía"
    assert pd.isna(traduccion.iloc[1])


def test_detecta_sectores_sin_traducir() -> None:
    """``unmapped_gics_sectors`` avisa de lo que falta por traducir."""
    series = pd.Series(
        ["Energy", "Sector Inventado", "Utilities", "Otro", "Sector Inventado"]
    )
    assert unmapped_gics_sectors(series) == ["Otro", "Sector Inventado"]


def test_no_hay_sectores_sin_traducir_en_el_universo_real() -> None:
    """La tabla de Wikipedia de hoy está cubierta por el mapeo."""
    from thesis.pair_trading import filter_duplicate_issuers, get_sp500_constituents

    kept, _ = filter_duplicate_issuers(get_sp500_constituents())
    sectores = kept.set_index("Symbol")["GICS Sector"]
    assert unmapped_gics_sectors(sectores) == []


def test_traducir_no_altera_las_metricas_de_alineacion() -> None:
    """Renombrar sectores no cambia purity, entropía, HHI ni ARI."""
    # Sectores sin empate, para que el sector dominante sea inequívoco.
    clusters = pd.Series([0, 0, 0, 1], index=TICKERS[:4], name="cluster")
    sectores_en = pd.Series(
        ["Information Technology", "Information Technology",
         "Health Care", "Real Estate"],
        index=TICKERS[:4],
        name="sector",
    )

    original = evaluate_cluster_sector_alignment(clusters, sectores_en)
    traducido = evaluate_cluster_sector_alignment(
        clusters, translate_gics_sectors(sectores_en)
    )

    assert traducido.weighted_purity == pytest.approx(original.weighted_purity)
    assert traducido.normalized_entropy == pytest.approx(original.normalized_entropy)
    assert traducido.mean_hhi == pytest.approx(original.mean_hhi)
    assert traducido.ari == pytest.approx(original.ari)
    assert traducido.ami == pytest.approx(original.ami)
    assert traducido.v_measure == pytest.approx(original.v_measure)
    assert set(traducido.per_cluster["dominant_sector"]) == {
        "Tecnología de la Información", "Inmobiliario",
    }
    # Y las etiquetas inglesas ya no aparecen en la salida.
    assert not (
        traducido.contingency.index.astype(str)
        .isin(["Information Technology", "Health Care", "Real Estate"])
        .any()
    )


# =============================================================================
# Helpers
# =============================================================================


def _labels(clusters: list[int], sectors: list[str]):
    """Construye las series alineadas de cluster y sector para los tests."""
    index = pd.Index(TICKERS[: len(clusters)], name="ticker")
    return (
        pd.Series(clusters, index=index, name="cluster"),
        pd.Series(sectors, index=index, name="sector"),
    )

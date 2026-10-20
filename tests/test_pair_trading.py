"""Pruebas de los contratos públicos de :mod:`thesis.pair_trading`.

El foco está en los dataclasses de resultado: si un ``@dataclass`` declara
una documentación de atributos pero ninguna anotación, el constructor
generado acepta cero argumentos y la función que lo instancia falla en
tiempo de ejecución con ``TypeError``.  Estos tests fijan ese contrato.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from thesis.pair_trading import (
    ADFResult,
    ClusterSectorReport,
    HedgeResult,
    OUParameters,
    evaluate_cluster_sector_alignment,
)

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PFE", "KO", "WMT"]


def _labels(clusters: list[int], sectors: list[str]):
    """Construye las series alineadas de cluster y sector para los tests."""
    index = pd.Index(TICKERS[: len(clusters)], name="ticker")
    return (
        pd.Series(clusters, index=index, name="cluster"),
        pd.Series(sectors, index=index, name="sector"),
    )


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
# evaluate_cluster_sector_alignment
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

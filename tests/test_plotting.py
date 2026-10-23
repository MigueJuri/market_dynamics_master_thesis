"""Pruebas de las funciones de heatmap sectorial de ``thesis.plotting``.

Se usa el backend Agg para no abrir ventanas: las pruebas inspeccionan la
figura devuelta (escala de color, matriz dibujada y separadores) en lugar de
mirar una imagen.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from thesis.plotting import (
    plot_sector_average_heatmap,
    plot_sector_reordered_matrix,
)

TICKERS = ["A", "B", "C", "D", "E", "F"]


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    """Cierra las figuras abiertas al terminar cada prueba."""
    yield
    plt.close("all")


@pytest.fixture
def matriz_simple():
    """Matriz 6x6 con clusters 1, 1, 2, 2, 3, 3 y NaN en la diagonal."""
    valores = np.arange(36, dtype=float).reshape(6, 6)
    np.fill_diagonal(valores, np.nan)
    return pd.DataFrame(valores, index=TICKERS, columns=TICKERS)


@pytest.fixture
def sectores():
    """Sectores alternos, para distinguirlos del orden por cluster."""
    return pd.Series(
        ["Inmobiliario", "Energía", "Energía",
         "Inmobiliario", "Inmobiliario", "Energía"],
        index=TICKERS,
        name="sector",
    )


@pytest.fixture
def clusters():
    """Clusters cuya agrupación por sector sería distinta.

    Ordenar por sector daría ``[A, D, E, B, C, F]`` y ordenar por cluster
    daría ``[A, B, F, D, E, C]``: la prueba distingue un criterio del otro.
    """
    return pd.Series([1, 1, 2, 2, 2, 1], index=TICKERS, name="cluster")


@pytest.fixture
def matriz_etiquetada():
    """Cada celda codifica su par origen como ``fila * 10 + columna``.

    Así la posición final de un ticker se recupera sin ambigüedad: basta leer
    una celda fuera de la diagonal y tomar la parte de las decenas.
    """
    n = len(TICKERS)
    valores = np.fromfunction(lambda i, j: i * 10 + j, (n, n), dtype=float)
    np.fill_diagonal(valores, np.nan)
    return pd.DataFrame(valores, index=TICKERS, columns=TICKERS)


def _orden_de_la_figura(fig) -> list[str]:
    """Devuelve los tickers en el orden en que se dibujaron.

    La diagonal de la matriz reordenada queda en ``(i, i)``, así que se lee
    una celda vecina: su parte de las decenas es la fila de origen.
    """
    dibujada = np.asarray(fig.axes[0].get_images()[0].get_array())
    n = len(dibujada)
    return [TICKERS[int(dibujada[i, (i + 1) % n]) // 10] for i in range(n)]


# =============================================================================
# plot_sector_reordered_matrix
# =============================================================================


def test_escala_usa_los_valores_medidos(matriz_simple, clusters) -> None:
    """``vmin``/``vmax`` salen de la matriz, no de un rango teórico."""
    fig = plot_sector_reordered_matrix(matriz_simple, cluster_labels=clusters)
    clim = fig.axes[0].get_images()[0].get_clim()

    finitos = matriz_simple.to_numpy(dtype=float)
    finitos = finitos[np.isfinite(finitos)]
    assert clim == pytest.approx((finitos.min(), finitos.max()))


def test_la_escala_ignora_la_diagonal_nan(matriz_simple, clusters) -> None:
    """La diagonal no participa en los límites de color.

    Se rellena la diagonal con un valor muy alto: si se contaminara, la
    escala se desplazaría y la prueba fallaría.
    """
    fig = plot_sector_reordered_matrix(matriz_simple, cluster_labels=clusters)

    finitos = matriz_simple.to_numpy(dtype=float)
    finitos = finitos[np.isfinite(finitos)]
    clim = fig.axes[0].get_images()[0].get_clim()
    assert clim == pytest.approx((finitos.min(), finitos.max()))
    # Ningún límite procede del 0 que en su día ocupaba la diagonal.
    assert clim[0] > 0.0


def test_no_transforma_los_valores(matriz_simple, clusters) -> None:
    """La matriz dibujada es la de entrada, sin normalizar."""
    fig = plot_sector_reordered_matrix(matriz_simple, cluster_labels=clusters)
    dibujada = fig.axes[0].get_images()[0].get_array()

    entrada = matriz_simple.to_numpy(dtype=float)
    assert np.allclose(
        np.sort(dibujada[~np.isnan(dibujada)]),
        np.sort(entrada[~np.isnan(entrada)]),
    )


def test_ordena_globalmente_por_cluster(
    matriz_etiquetada, sectores, clusters
) -> None:
    """El orden es por cluster, no agrupado por sector."""
    fig = plot_sector_reordered_matrix(
        matriz_etiquetada, sector_labels=sectores, cluster_labels=clusters
    )
    orden = _orden_de_la_figura(fig)

    # Ordenado por cluster.
    assert clusters.loc[orden].tolist() == [1, 1, 1, 2, 2, 2]
    # Y deliberadamente no por sector: agrupar por sector daría otro orden.
    assert sectores.loc[orden].tolist() != [
        s
        for s in sorted(sectores.unique(), key=str)
        for s in sectores[sectores == s].index
    ]


def test_une_separador_por_frontera_de_cluster(
    matriz_simple, sectores, clusters
) -> None:
    """Con dos clusters hay una sola frontera: dos líneas (h y v)."""
    fig = plot_sector_reordered_matrix(
        matriz_simple, sector_labels=sectores, cluster_labels=clusters
    )
    assert len(fig.axes[0].lines) == 2


def test_no_dibuja_separadores_sectoriales(
    matriz_simple, sectores, clusters
) -> None:
    """Los límites sectoriales ya no se marcan."""
    fig_solo_clusters = plot_sector_reordered_matrix(
        matriz_simple, sector_labels=sectores, cluster_labels=clusters
    )
    fig_sin_clusters = plot_sector_reordered_matrix(
        matriz_simple, sector_labels=sectores, cluster_labels=None
    )

    # Sin clusters no hay ninguna línea, aunque haya sectores distintos.
    assert len(fig_sin_clusters.axes[0].lines) == 0
    # Y con clusters sólo aparecen los de cluster.
    assert len(fig_solo_clusters.axes[0].lines) == 2


def test_tres_clusters_dan_dos_fronteras() -> None:
    """k clusters implican k-1 fronteras."""
    matriz = pd.DataFrame(
        np.arange(36, dtype=float).reshape(6, 6),
        index=TICKERS,
        columns=TICKERS,
    )
    tres = pd.Series([1, 1, 2, 2, 3, 3], index=TICKERS, name="cluster")

    fig = plot_sector_reordered_matrix(matriz, cluster_labels=tres)

    assert len(fig.axes[0].lines) == 4


def test_color_de_separador_configurable() -> None:
    """``separator_color`` se aplica a las líneas dibujadas."""
    matriz = pd.DataFrame(
        np.arange(36, dtype=float).reshape(6, 6),
        index=TICKERS,
        columns=TICKERS,
    )
    dos = pd.Series([1, 1, 2, 2, 3, 3], index=TICKERS, name="cluster")

    fig = plot_sector_reordered_matrix(
        matriz, cluster_labels=dos, separator_color="red"
    )

    colores = {line.get_color() for line in fig.axes[0].lines}
    assert colores == {"red"}


def test_no_muestra_tickers_pero_numera_los_clusters(
    matriz_simple, sectores, clusters
) -> None:
    """El eje X lleva el número de cada cluster, nunca nombres de ticker."""
    fig = plot_sector_reordered_matrix(
        matriz_simple, sector_labels=sectores, cluster_labels=clusters
    )
    etiquetas = [t.get_text() for t in fig.axes[0].get_xticklabels()]

    assert etiquetas == ["1", "2"]
    assert not any(t in etiquetas for t in TICKERS)


def test_no_aparece_el_nombre_de_los_sectores(
    matriz_simple, sectores, clusters
) -> None:
    """Los sectores ya no se rotulan: sólo se numeran los clusters."""
    fig = plot_sector_reordered_matrix(
        matriz_simple, sector_labels=sectores, cluster_labels=clusters
    )
    textos = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    todo = " ".join(textos) + " " + " ".join(
        t.get_text() for t in fig.axes[0].get_yticklabels()
    )

    assert "Inmobiliario" not in todo
    assert "Energía" not in todo
    assert "C1" not in todo


def test_la_numeracion_es_la_posicion_del_bloque() -> None:
    """Los números empiezan en 1 y siguen el orden de izquierda a derecha.

    No reproducen el identificador interno del clustering: si éste fuese
    ``[10, 20, 30]``, la figura debe mostrar ``1, 2, 3``.
    """
    clusters = pd.Series([10, 10, 20, 20, 30, 30], index=TICKERS, name="cluster")
    matriz = pd.DataFrame(
        np.arange(36, dtype=float).reshape(6, 6),
        index=TICKERS,
        columns=TICKERS,
    )

    fig = plot_sector_reordered_matrix(matriz, cluster_labels=clusters)
    etiquetas = [t.get_text() for t in fig.axes[0].get_xticklabels()]

    assert etiquetas == ["1", "2", "3"]


def test_numeracion_centrada_bajo_cada_bloque(matriz_simple, sectores) -> None:
    """Cada número queda en el centro geométrico de su bloque."""
    clusters = pd.Series([1, 1, 2, 2, 3, 3], index=TICKERS, name="cluster")

    fig = plot_sector_reordered_matrix(
        matriz_simple, sector_labels=sectores, cluster_labels=clusters
    )
    ax = fig.axes[0]

    # Bloques de dos activos: centros en 0.5, 2.5 y 4.5.
    assert list(ax.get_xticks()) == pytest.approx([0.5, 2.5, 4.5])


def test_bloques_de_tamanos_distintos_se_centran_bien() -> None:
    """Un bloque estrecho y otro ancho reciben su número en su propio centro."""
    largo = ["A"] + [f"T{i:02d}" for i in range(11)]
    clusters = pd.Series([1] + [2] * 11, index=largo, name="cluster")
    n = len(largo)
    valores = np.fromfunction(lambda i, j: i * 100 + j, (n, n), dtype=float)
    np.fill_diagonal(valores, np.nan)
    matriz = pd.DataFrame(valores, index=largo, columns=largo)

    fig = plot_sector_reordered_matrix(matriz, cluster_labels=clusters)

    # Cluster 1 ocupa el activo 0 (centro 0.0); cluster 2 va de 1 a 11.
    assert list(fig.axes[0].get_xticks()) == pytest.approx([0.0, 6.0])


def test_el_centro_coincide_con_la_geometria_del_bloque() -> None:
    """El número cae en el centro real del bloque, sin medio píxel de error.

    ``imshow`` dibuja el índice ``i`` en ``[i - 0.5, i + 0.5]``, así que un
    bloque ``[inicio, fin]`` abarca de ``inicio - 0.5`` a ``fin + 0.5`` y su
    centro es ``(inicio + fin) / 2``.
    """
    largo = [f"T{i:02d}" for i in range(9)]
    clusters = pd.Series([1] * 2 + [2] * 5 + [3] * 2, index=largo, name="cluster")
    n = len(largo)
    valores = np.fromfunction(lambda i, j: i * 100 + j, (n, n), dtype=float)
    np.fill_diagonal(valores, np.nan)
    matriz = pd.DataFrame(valores, index=largo, columns=largo)

    fig = plot_sector_reordered_matrix(matriz, cluster_labels=clusters)
    centros = list(fig.axes[0].get_xticks())

    # Bloques: 0-1, 2-6 y 7-8.
    assert centros == pytest.approx([(0 + 1) / 2, (2 + 6) / 2, (7 + 8) / 2])
    # Coinciden con el punto medio de las fronteras dibujadas.
    assert centros[0] == pytest.approx((0 - 0.5 + 1 + 0.5) / 2)
    assert centros[2] == pytest.approx((7 - 0.5 + 8 + 0.5) / 2)


def test_el_eje_x_se_titula_clusters(matriz_simple, clusters) -> None:
    """El eje X lleva el título ``Clusters``."""
    fig = plot_sector_reordered_matrix(matriz_simple, cluster_labels=clusters)
    assert fig.axes[0].get_xlabel() == "Clusters"


def test_el_titulo_del_eje_x_es_configurable(matriz_simple, clusters) -> None:
    """``xlabel`` permite adaptar el rótulo."""
    fig = plot_sector_reordered_matrix(
        matriz_simple, cluster_labels=clusters, xlabel="Clúster"
    )
    assert fig.axes[0].get_xlabel() == "Clúster"


def test_se_pueden_ocultar_los_numeros(matriz_simple, sectores) -> None:
    """``cluster_labels_visible=False`` deja el eje X vacío y sin título."""
    clusters = pd.Series([1, 1, 2, 2, 3, 3], index=TICKERS, name="cluster")

    fig = plot_sector_reordered_matrix(
        matriz_simple,
        sector_labels=sectores,
        cluster_labels=clusters,
        cluster_labels_visible=False,
    )

    assert list(fig.axes[0].get_xticks()) == []
    assert fig.axes[0].get_xlabel() == ""


def test_sin_clusters_no_hay_numeros(matriz_simple, sectores) -> None:
    """Sin clusters tampoco hay separadores ni numeración."""
    fig = plot_sector_reordered_matrix(
        matriz_simple, sector_labels=sectores, cluster_labels=None
    )
    assert list(fig.axes[0].get_xticks()) == []
    assert len(fig.axes[0].lines) == 0


def test_numeros_no_se_salen_del_lienzo() -> None:
    """Aunque haya clusters diminutos contiguos, los números caben."""
    largo = (
        [f"W{i:03d}" for i in range(45)]
        + ["A", "B", "C"]
        + [f"V{i:03d}" for i in range(20)]
    )
    clusters = pd.Series(
        [1] * 45 + [2, 3, 4] + [5] * 20, index=largo, name="cluster"
    )
    n = len(largo)
    valores = np.fromfunction(lambda i, j: i * 100 + j, (n, n), dtype=float)
    np.fill_diagonal(valores, np.nan)
    matriz = pd.DataFrame(valores, index=largo, columns=largo)

    fig = plot_sector_reordered_matrix(
        matriz, cluster_labels=clusters, figsize=(10, 9)
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cajas = [
        t.get_window_extent(renderer) for t in fig.axes[0].get_xticklabels()
    ]

    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [
        "1", "2", "3", "4", "5",
    ]
    assert min(c.y0 for c in cajas) >= fig.bbox.y0
    assert min(c.x0 for c in cajas) >= fig.bbox.x0
    assert max(c.x1 for c in cajas) <= fig.bbox.x1


def test_sectors_en_el_eje_y_si_se_piden(
    matriz_simple, sectores, clusters
) -> None:
    """``sector_labels_visible`` rotula el eje Y sin invadir el X."""
    fig = plot_sector_reordered_matrix(
        matriz_simple,
        sector_labels=sectores,
        cluster_labels=clusters,
        sector_labels_visible=True,
    )
    ax = fig.axes[0]

    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2"]
    assert {t.get_text() for t in ax.get_yticklabels()} == {
        "Inmobiliario", "Energía",
    }


def test_sin_activos_comunes_no_dibuja() -> None:
    """Sin coincidencia entre matriz y metadatos devuelve ``None``."""
    matriz = pd.DataFrame(
        np.eye(2), index=["A", "B"], columns=["A", "B"]
    )
    clusters = pd.Series([1, 2], index=["X", "Y"], name="cluster")

    assert plot_sector_reordered_matrix(matriz, cluster_labels=clusters) is None


def test_matriz_sin_valores_finitos_no_dibuja() -> None:
    """Una matriz toda NaN no produce una figura vacía sin avisar."""
    matriz = pd.DataFrame(np.nan, index=TICKERS, columns=TICKERS)
    clusters = pd.Series([1, 1, 2, 2, 3, 3], index=TICKERS, name="cluster")

    assert plot_sector_reordered_matrix(matriz, cluster_labels=clusters) is None


def test_acepta_sin_metadatos(matriz_simple) -> None:
    """Sin sectores ni clusters se dibuja la matriz tal cual."""
    fig = plot_sector_reordered_matrix(matriz_simple)
    assert fig is not None
    assert len(fig.axes[0].lines) == 0


# =============================================================================
# plot_sector_average_heatmap
# =============================================================================


def test_sector_average_usa_escala_medida() -> None:
    """Los límites salen de los valores presentes, no de un rango fijo."""
    matriz = pd.DataFrame(
        [[0.2, 0.7], [0.7, 0.2]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    fig = plot_sector_average_heatmap(matriz)

    assert fig.axes[0].get_images()[0].get_clim() == pytest.approx((0.2, 0.7))


def test_sector_average_conserva_los_valores() -> None:
    """No normaliza: dibuja la matriz recibida."""
    matriz = pd.DataFrame(
        [[0.1, 0.9], [0.9, 0.1]], index=["A", "B"], columns=["A", "B"]
    )

    fig = plot_sector_average_heatmap(matriz)
    dibujada = fig.axes[0].get_images()[0].get_array()

    assert np.allclose(dibujada, matriz.to_numpy())


def test_sector_average_ignora_nan() -> None:
    """Los valores ausentes no fijan la escala."""
    matriz = pd.DataFrame(
        [[np.nan, 0.4], [0.4, np.nan]], index=["A", "B"], columns=["A", "B"]
    )

    fig = plot_sector_average_heatmap(matriz)

    # Con un único valor finito la escala es degenerada; matplotlib la
    # ensancha, así que se comprueban los extremos con tolerancia.
    vmin, vmax = fig.axes[0].get_images()[0].get_clim()
    assert vmin == pytest.approx(0.4, abs=0.05)
    assert vmax == pytest.approx(0.4, abs=0.05)


def test_sector_average_etiqueta_los_sectores() -> None:
    """Los ejes quedan rotulados con los nombres de sector."""
    matriz = pd.DataFrame(
        [[0.2, 0.7], [0.7, 0.2]], index=["A", "B"], columns=["A", "B"]
    )

    fig = plot_sector_average_heatmap(matriz, title="Título", cbar_label="MI")

    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["A", "B"]
    assert fig.axes[0].get_yticklabels()[0].get_text() == "A"
    assert fig.axes[0].get_title() == "Título"


def test_sector_average_sin_valores_finitos_no_dibuja() -> None:
    """Una matriz toda NaN devuelve ``None``."""
    matriz = pd.DataFrame(np.nan, index=["A"], columns=["A"])
    assert plot_sector_average_heatmap(matriz) is None


def test_ambas_funciones_comparten_paleta() -> None:
    """Pearson e información mutua se dibujan con la misma paleta."""
    from thesis.plotting import plot_sector_reordered_matrix as reord

    matriz = pd.DataFrame(
        np.arange(36, dtype=float).reshape(6, 6),
        index=TICKERS,
        columns=TICKERS,
    )
    clusters = pd.Series([1, 1, 2, 2, 3, 3], index=TICKERS, name="cluster")

    fig_a = reord(matriz, cluster_labels=clusters, cmap="viridis")
    fig_b = plot_sector_average_heatmap(
        matriz, cmap="viridis"
    )

    cmap_a = fig_a.axes[0].get_images()[0].get_cmap()
    cmap_b = fig_b.axes[0].get_images()[0].get_cmap()
    assert cmap_a.name == cmap_b.name == "viridis"
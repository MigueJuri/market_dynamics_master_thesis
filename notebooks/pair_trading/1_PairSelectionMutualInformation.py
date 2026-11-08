# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: tesis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Selección de Pares vía Información Mutua (KSG)
#
# ## ¿Por qué MI en lugar de correlación?
#
# La **correlación de Pearson** sólo captura dependencia *lineal*.  Dos
# series pueden tener correlación más fuerte (un valor bajo) y aun así
# estar acopladas de forma no lineal (por ejemplo, $Y = f(X)$ monótona).
# La **Información Mutua** (MI) detecta dependencias estadísticas
# generales:
#
# $$I(X;Y) = \sum_{y \in Y} \sum_{x \in X}
#   p(x,y)\, \log \frac{p(x,y)}{p(x)\,p(y)}$$
#
# $I(X;Y) = 0$ si y sólo si $X$ e $Y$ son independientes.
#
# ## El estimador KSG (k-NN)
#
# Para datos continuos no podemos estimar $p(x,y)$ mediante
# histogramas sin sesgo de discretización.  Usamos el estimador de
# **Kraskov-Stögbauer-Grassberger (KSG)**, basado en k-vecinos
# cercanos.  Las ideas clave son:
#
# - Para cada punto, se busca la distancia al k-ésimo vecino en el
#   espacio conjunto y en cada marginal.
# - La diferencia entre conteos de vecinos en el espacio conjunto y
#   en los marginales aproxima el logaritmo de la densidad conjunta
#   dividido por el producto de marginales.
# - Se promedia sobre todos los puntos y se simetriza estimando
#   $MI(X,Y)$ y $MI(Y,X)$ por separado.
#
# El parámetro ``k_neighbors=3`` controla el balance sesgo-varianza:
# valores más bajos reducen el sesgo pero aumentan la varianza.
#
# ## Relación con el cuaderno de Pearson
#
# Este cuaderno es el gemelo de
# [`2_PairSelectionPearson.py`](./2_PairSelectionPearson.py): comparte el
# universo, la ventana de análisis y el pipeline (tabla de pares, test
# ADF del spread, tabla de pares cointegrados, clustering sectorial y
# figuras), y sólo cambia la métrica de acoplamiento.  La MI captura
# dependencias no lineales que Pearson no ve, a costa de un estimador
# más caro y con más varianza.
#
# ## Flujo del notebook
#
# 1. Universo S&P 500 completo con ventana explícita y filtro de historia,
#    más el diagnóstico de cobertura del panel.
# 2. Matriz MI KSG sobre todo el universo (calculada y cacheada).
# 3. Top pares por MI, filtrando emisores del mismo grupo.
# 4. Test ADF del spread de cada par sobre la historia completa, con el
#    tamaño muestral de cada par.
# 5. Tabla de pares cointegrados (spread estacionario) dentro del conjunto
#    de candidatos.
# 6. Clustering jerárquico de Ward con tantos clusters como sectores.
# 7. Evaluación de la calidad sectorial del clustering.
# 8. Visualización de matrices y composición sectorial.

# %%
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

from thesis.config import (
    DATA_DIR,
    END,
    PAIR_SELECTION_MIN_HISTORY_OBS,
    SEED,
    START,
)
from thesis.pair_trading import (
    adf_test,
    cached_ksg_mi_matrix,
    compute_hedge_ratio,
    evaluate_cluster_sector_alignment,
    export_pair_selection_table,
    hierarchical_clustering,
    load_log_prices,
    mi_distance,
    prepare_pair_selection_universe,
    sector_average_matrix,
    top_pairs,
    translate_gics_sectors,
    unmapped_gics_sectors,
)
from thesis.plotting import (
    plot_cluster_sector_composition,
    plot_cluster_sector_heatmap,
    plot_cluster_sector_summary,
    plot_sector_average_heatmap,
    plot_sector_reordered_matrix,
)

# %%
# ## Universo, ventana de análisis y retornos
#
# ``prepare_pair_selection_universe`` ejecuta la fase compartida con
# ``2_PairSelectionPearson.py``:
#
# 1. ``exclude_undownloadable`` descarta los constituyentes cuyo ticker no
#    se puede descargar de forma consistente: ``FDXF`` (FedEx Freight),
#    ``HONA`` (Honeywell Aerospace) y ``VYLR`` (Vylor) devuelven "Data
#    doesn't exist" en ``yfinance`` para cualquier ventana.  No cambia
#    ningún resultado --ya quedaban fuera del panel al no aportar
#    observaciones--, pero evita reintentos fallidos en cada ejecución y
#    deja la exclusión documentada.
# 2. ``filter_duplicate_issuers`` normaliza el nombre del valor (elimina
#    sufijos como ``Class A``, ``Inc.``, ``Corp.``, etc.) y conserva **una
#    sola clase de acción por emisor** (ej. ``GOOGL``/``GOOG`` o
#    ``FOXA``/``FOX``), evitando evaluar pares inválidos y reduciendo el
#    coste del KSG.
# 3. ``load_returns_panel`` **filtra por historia antes de alinear las
#    fechas**.  Sin ese filtro, ``dropna`` conservaría sólo las fechas en las
#    que cotizan los ~500 activos a la vez y un único emisor recién salgado a
#    bolsa reduciría el rango completo a dos meses.
# 4. Se alinean sectores y emisores con los tickers que sobreviven y se
#    imprime el diagnóstico de cobertura.
#
# El umbral de historia vive en ``thesis.config`` como
# ``PAIR_SELECTION_MIN_HISTORY_OBS``, de modo que ambos cuadernos estiman su
# métrica sobre exactamente la misma muestra y las dos matrices son
# comparables.  Medido sobre la caché actual, con la ventana 1993-2026:
#
# ==========  ========  ==========  =============
# mínimo      activos   sesiones    desde
# ==========  ========  ==========  =============
# 0           497       43         2025-10-28
# 1260 (~5a)  484       1 267      2020-12-11
# 5040 (~20a) 388       5 047      2005-12-06
# 7560 (~30a) 301       7 557      1995-12-18
# ==========  ========  ==========  =============
#
# Se elige el mínimo de veinte años: mantiene 388 activos (suficientes para
# el clustering sectorial) y veinte años de observaciones.  Subir el umbral
# acorta la intersección; bajarlo obliga a usar un panel reciente o
# directamente inservible.
universe = prepare_pair_selection_universe(
    start=START,
    end=END,
    cache_dir=DATA_DIR,
    min_history_obs=PAIR_SELECTION_MIN_HISTORY_OBS,
)

returns_full = universe.returns
issuer_map = universe.issuers
n_clusters = universe.n_sectors

#: Constituyentes excluidos del universo porque su ticker no se puede
#: descargar de forma consistente.  Se documentan en los metadatos y en las
#: notas de las tablas exportadas para que el universo sea reproducible.
TICKERS_SIN_DESCARGA = sorted(universe.excluded_undownloadable["Symbol"])

# Los sectores GICS llegan en inglés desde Wikipedia y se traducen aquí para
# que toda la salida de la tesis (métricas, figuras y tablas) esté en
# castellano.  Traducir no altera los resultados: las métricas de alineación
# sólo dependen de la partición.
sectors_en = universe.sectors
available_sectors = translate_gics_sectors(sectors_en)

sin_traducir = unmapped_gics_sectors(sectors_en)
if sin_traducir:
    print(f"⚠ Sectores GICS sin traducción al castellano: {sin_traducir}")
print("Sectores GICS (traducidos):")
for sector in sorted(available_sectors.unique()):
    n_activos = int((available_sectors == sector).sum())
    print(f"  {sector}: {n_activos} activos")

# %%
# ## Información mutua KSG sobre el universo completo
#
# ``cached_ksg_mi_matrix`` persiste la matriz en
# ``data/pair_trading/mi/ksg/`` con un hash determinista de fechas,
# activos, número de observaciones y contenido de los retornos.
# Reejecuciones idénticas recuperan el resultado sin volver a
# calcular el estimador KSG (O(N²) en el número de activos).
#
# Cada par ``(i, j)`` se estima dos veces (``mi_xy`` y ``mi_yx``) y
# se simetriza como la media, mitigando asimetrías numéricas de la
# búsqueda k-NN.  Series con varianza cero se omiten y la diagonal
# queda vacía.
# El parámetro ``k_neighbors`` controla el balance sesgo-varianza del
# estimador.  Se fija una única constante y se reutiliza tanto en el cálculo
# de la matriz como en las condiciones de la tabla exportada, para que la
# tabla documente exactamente el estimador que la produjo.
K_NEIGHBORS = 3

mi_matrix = cached_ksg_mi_matrix(returns_full, k_neighbors=K_NEIGHBORS)

#: Número de candidatos evaluados por MI.  Es el conjunto sobre el que se
#: aplica después el filtro de cointegración (ADF), igual que en
#: ``2_PairSelectionPearson.py``; el tamaño está limitado por el coste de
#: un test ADF por par, que se estima sobre la historia completa del par y
#: no sobre el panel alineado.
N_CANDIDATOS = 200

top_20_pairs = top_pairs(mi_matrix, n=N_CANDIDATOS, issuer_map=issuer_map)

print(f"\nTop {len(top_20_pairs)} pares por Información Mutua KSG:")
for rank, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f} nats")

# %%
# ## Tabla de pares principales con sector y diagnóstico ADF del spread
#
# Cada par lleva el sector GICS de ambos miembros.  Permite distinguir
# combinaciones intra-sectoriales (Utilities, Energy, ...) de
# combinaciones inter-sectoriales.
#
# Para cada par se toman los log-precios de las observaciones comunes a
# ambos activos mediante ``load_log_prices`` y se estima el hedge ratio por
# OLS de
# $\log A_t = \alpha + \beta \log B_t + \varepsilon_t$.  El spread
# resultante $\varepsilon_t$ es la serie sobre la que se aplica el test
# Augmented Dickey-Fuller:
#
# - ``estadistico_adf``: valores muy negativos rechazan la raíz unitaria;
# - ``p_valor_adf``: por debajo de 0.05 el spread se considera
#   estacionario;
# - ``estacionario``: bandera booleana con ese mismo criterio.
#
# El ADF se estima sobre $\log P$ y no sobre retornos: acumular los
# log-retornos con ``cumsum`` perdería el primer nivel y todas las fechas en
# las que un activo no cotizó, dando series distintas de las que produce
# ``PairTrading.py`` para un mismo par.
#
# ## Dos ventanas distintas y por qué
#
# El panel de información mutua y el ADF no usan exactamente la misma
# muestra.  La MI se estima sobre el panel alineado (intersección de fechas
# de los activos con historia suficiente), mientras que el ADF usa
# ``load_log_prices`` sobre las fechas comunes de **ese par**, que recupera
# historia adicional de los activos que no cotizaban en el periodo en que
# empezó el último superviviente del universo.
#
# La diferencia no es un descuido: la selección de candidatos necesita un
# panel común para poder clusterizar, mientras que la cointegración se
# estima mejor sobre toda la historia del par.  Ambas consultas parten de
# ``START`` y terminan en ``END``, y la del ADF **coincide con el rango de
# ``PairTrading.py`**, así que el estadístico de un mismo par es
# directamente comparable entre ambos cuadernos.  La columna
# ``n_sesiones_par`` deja explícito cuánto soporte tiene cada resultado.
#
# El objetivo del estadístico es el spread, no la MI: dos activos con MI
# alta pueden tener un spread no estacionario.
SIGNIFICANCIO_ADF = 0.05

#: Un ADF sobre menos de un año de sesiones no es interpretable.
MIN_SESIONES_ADF = 252

top_pairs_df = pd.DataFrame(
    top_20_pairs,
    columns=["ticker_a", "ticker_b", "mi"],
)
top_pairs_df["sector_a"] = top_pairs_df["ticker_a"].map(available_sectors)
top_pairs_df["sector_b"] = top_pairs_df["ticker_b"].map(available_sectors)


def adf_para_par(ticker_a, ticker_b):
    """Calcula el estadístico, el p-value y el tamaño muestral del par.

    Devuelve ``(np.nan, np.nan, n)`` si faltan datos, no se puede calcular
    el spread o la muestra es demasiado corta, para no interrumpir la tabla
    completa.  ``n`` es el número de sesiones comunes del par, que acota
    cuánta evidencia respalda cada estadístico.
    """
    try:
        log_precios = load_log_prices([ticker_a, ticker_b], START, END, DATA_DIR)
    except ValueError:
        return np.nan, np.nan, 0

    n = len(log_precios)
    if n < MIN_SESIONES_ADF:
        return np.nan, np.nan, n

    try:
        hedge = compute_hedge_ratio(log_precios[ticker_a], log_precios[ticker_b])
        spread = log_precios[ticker_a] - (
            hedge.alpha + hedge.beta * log_precios[ticker_b]
        )
        adf = adf_test(spread, significance=SIGNIFICANCIO_ADF)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan, n
    return adf.statistic, adf.pvalue, n


adf_stats = [
    adf_para_par(row.ticker_a, row.ticker_b)
    for row in top_pairs_df.itertuples(index=False)
]
top_pairs_df["n_sesiones_par"] = [n for _, _, n in adf_stats]
top_pairs_df["estadistico_adf"] = [s for s, _, _ in adf_stats]
top_pairs_df["p_valor_adf"] = [p for _, p, _ in adf_stats]
top_pairs_df["estacionario"] = top_pairs_df["p_valor_adf"] < SIGNIFICANCIO_ADF

print(f"\nTop {len(top_pairs_df)} pares con sector y diagnóstico ADF del spread:")
print(top_pairs_df.to_string(index=False))
print(
    f"\nSpread estacionario al {SIGNIFICANCIO_ADF:.0%} en "
    f"{int(top_pairs_df['estacionario'].sum())} de {len(top_pairs_df)} pares."
)
print(
    f"Muestra del ADF: mediana {top_pairs_df['n_sesiones_par'].median():.0f} "
    f"sesiones, rango {top_pairs_df['n_sesiones_par'].min()}-"
    f"{top_pairs_df['n_sesiones_par'].max()}."
)

# %%
# ## Exportación de la tabla
#
# ``export_pair_selection_table`` escribe tres versiones de la tabla en
# ``outputs/tables``: el CSV con los valores sin formatear, el LaTeX listo
# para insertar en la tesis y un JSON con las condiciones de cálculo.
#
# El ``.tex`` incluye como nota las condiciones que producen la tabla:
# método, ventanas solicitada y efectiva, sesiones, ritmo anual, tamaño
# del universo, filtro de historia y cobertura.  Aquí es especialmente
# importante registrar el número de vecinos, porque la MI KSG es
# sesgo-varianza dependiente de ``k`` y dos ejecuciones con distinto ``k``
# no son comparables.
#
# ## Columnas de la tabla
#
# ``COLUMNAS_TABLA`` decide qué se exporta al CSV y al LaTeX, y en qué
# orden.  Edita esta lista para recortar la tabla de la tesis; el DataFrame
# completo sigue disponible arriba.  Para una tabla más estrecha, por
# ejemplo sin las columnas del ADF:
#
# ```python
# COLUMNAS_TABLA = [
#     "rank",
#     "ticker_a", "ticker_b", "sector_a", "sector_b",
#     "mi", "p_valor_adf",
# ]
# ```
COLUMNAS_TABLA = [
    "rank",
    "ticker_a",     
    "ticker_b",
    "sector_a",
    "sector_b",
    "mi",
    "n_sesiones_par",
    "estadistico_adf",
    "p_valor_adf",
    "estacionario",
]

export_paths = export_pair_selection_table(
    top_pairs_df,
    universe.report,
    stem="mutual_information_top20",
    columns=COLUMNAS_TABLA,
    method=(
        f"información mutua estimada con KSG (k = {K_NEIGHBORS} vecinos) "
        f"sobre log-retornos diarios, con diagnóstico ADF del spread"
    ),
    caption=(
        "Principales pares por información mutua KSG y diagnóstico ADF "
        "de su spread"
    ),
    label="tab:mi_top20",
    extra_metadata={
        "vecinos_ksg": K_NEIGHBORS,
        "semilla": SEED,
        "tickers_sin_descarga": TICKERS_SIN_DESCARGA,
        "significancia_adf": SIGNIFICANCIO_ADF,
        "sesiones_minimas_adf": MIN_SESIONES_ADF,
        "n_sesiones_adf_mediana": float(
            top_pairs_df["n_sesiones_par"].median()
        ),
        "pares_estacionarios": int(top_pairs_df["estacionario"].sum()),
    },
    notes=[
        (
            f"Estimador KSG con k = {K_NEIGHBORS} vecinos sobre "
            f"{universe.report.n_sessions} observaciones; cada par se estima "
            f"en ambos sentidos y se simetriza."
        ),
        "Métrica expresada en nats.",
        (
            f"Test ADF sobre el spread con significancia "
            f"{SIGNIFICANCIO_ADF:.0%} y al menos {MIN_SESIONES_ADF} sesiones "
            f"por par; mediana de {top_pairs_df['n_sesiones_par'].median():.0f} "
            f"sesiones (rango {top_pairs_df['n_sesiones_par'].min()}-"
            f"{top_pairs_df['n_sesiones_par'].max()})."
        ),
        (
            f"Spread estacionario en "
            f"{int(top_pairs_df['estacionario'].sum())} de "
            f"{len(top_pairs_df)} pares."
        ),
        (
            "Activos excluidos del universo por no tener descarga "
            f"consistente: {', '.join(TICKERS_SIN_DESCARGA)}."
        ),
    ],
)
print(f"LaTeX: {export_paths['tex']}")

# %%
# ## Tabla de pares cointegrados
#
# La tabla anterior ordena los candidatos por **información mutua** y
# adjunta el diagnóstico ADF de cada uno, pero mezcla pares estacionarios con
# pares que no lo son.  Esta sección aísla los pares **cointegrados**: de los
# candidatos ya evaluados se conservan sólo aquellos cuyo spread rechaza la
# raíz unitaria, y de ellos se toman los ``TOP_N_COINTEGRADOS`` de mayor MI.
#
# ## Criterio de cointegración
#
# El criterio es el **p-value del test ADF**, no el t-estadístico:
#
# $$\text{cointegrado} \iff p_{\text{ADF}} < \alpha.$$
#
# Se usa el p-value y no un umbral fijo sobre el t-stat porque el valor
# crítico del ADF depende del número de observaciones del par (a mayor
# muestra, el mismo nivel de significancia exige un t-stat más negativo) y de
# la especificación del test.  Fijar ``t < -3`` sería incomparable entre
# pares con muestras de tamaños distintos; el p-value ya tiene en cuenta
# ambos efectos.  El t-stat se conserva en la tabla como diagnóstico, porque
# es el estadístico que el test contrasta contra su valor crítico.
#
# ``SIGNIFICANCIO_ADF`` (5 % por defecto) es el único parámetro que controla
# el filtro: subirlo exige evidencia más fuerte para declarar un par
# cointegrado.
#
# ## Alcance del filtro
#
# El filtro se aplica **dentro del conjunto de candidatos ya evaluados** por
# el notebook (los pares con mayor MI), y no sobre el universo completo.  Es
# una decisión deliberada: el ADF requiere un test por par y estimarlo para
# los ~75.000 pares del universo completo multiplica por unas dos horas el
# tiempo de ejecución, frente a los minutos que tarda el panel ya calculado.
# La consecuencia metodológica es explícita: la tabla identifica los pares
# **más acoplados en MI y cointegrados a la vez**, no una lista exhaustiva de
# todos los pares cointegrados del S&P 500.  Un par cointegrado cuya MI lo
# situara fuera del conjunto de candidatos evaluados no aparece aquí.
#
# Dentro de ese conjunto, al ordenarse por MI descendente, los
# ``TOP_N_COINTEGRADOS`` primeros son exactamente los pares cointegrados con
# mayor MI.

#: Número de pares cointegrados que se exportan.
TOP_N_COINTEGRADOS = 20

cointegrated_df = (
    top_pairs_df.loc[top_pairs_df["p_valor_adf"] < SIGNIFICANCIO_ADF]
    .sort_values("mi", ascending=False)
    .head(TOP_N_COINTEGRADOS)
    .reset_index(drop=True)
)

n_cointegrados = int((top_pairs_df["p_valor_adf"] < SIGNIFICANCIO_ADF).sum())
if n_cointegrados == 0:
    raise ValueError(
        f"Ninguno de los {len(top_pairs_df)} candidatos supera el filtro ADF "
        f"con p-value < {SIGNIFICANCIO_ADF:.0%}.  Baja SIGNIFICANCIO_ADF o "
        "amplía el conjunto de candidatos evaluados."
    )

print(
    f"\nTop {len(cointegrated_df)} pares cointegrados "
    f"(ADF p-value < {SIGNIFICANCIO_ADF:.0%}) por información mutua KSG:"
)
print(cointegrated_df.to_string(index=False))
print(
    f"\nPares cointegrados en el conjunto de candidatos: "
    f"{n_cointegrados} de {len(top_pairs_df)}."
)

# %%
# ## Exportación de la tabla de pares cointegrados
#
# Se exporta en el mismo formato que la tabla anterior (CSV, LaTeX y JSON),
# con su propio ``stem`` para no sobrescribirla.  La nota del ``.tex``
# documenta el criterio de cointegración, el número de candidatos evaluados
# y el tamaño muestral, que es lo que hace falta para reproducir el filtro.
#
# ## Columnas de la tabla
#
# ``COLUMNAS_TABLA_COINTEGRADOS`` fija el subconjunto y el orden de las
# columnas exportadas.  Hereda de la tabla principal las columnas de
# identificación y de métrica, y añade el diagnóstico ADF completo (t-stat,
# p-value y bandera de estacionariedad) más el soporte muestral.
COLUMNAS_TABLA_COINTEGRADOS = [
    "rank",
    "ticker_a",
    "ticker_b",
    "sector_a",
    "sector_b",
    "mi",
    "n_sesiones_par",
    "estadistico_adf",
    "p_valor_adf",
    "estacionario",
]

export_paths_cointegrated = export_pair_selection_table(
    cointegrated_df,
    universe.report,
    stem="mutual_information_cointegrated_top20",
    columns=COLUMNAS_TABLA_COINTEGRADOS,
    method=(
        f"información mutua estimada con KSG (k = {K_NEIGHBORS} vecinos) "
        f"sobre log-retornos diarios, filtrando por cointegración (test ADF "
        f"con p-value < {SIGNIFICANCIO_ADF:.0%})"
    ),
    caption=(
        "Principales pares cointegrados por información mutua KSG "
        "(spread estacionario según ADF)"
    ),
    label="tab:mi_cointegrated_top20",
    extra_metadata={
        "vecinos_ksg": K_NEIGHBORS,
        "semilla": SEED,
        "tickers_sin_descarga": TICKERS_SIN_DESCARGA,
        "significancia_adf": SIGNIFICANCIO_ADF,
        "sesiones_minimas_adf": MIN_SESIONES_ADF,
        "candidatos_evaluados": len(top_pairs_df),
        "candidatos_cointegrados": n_cointegrados,
        "n_sesiones_adf_mediana": float(
            cointegrated_df["n_sesiones_par"].median()
        ),
    },
    notes=[
        (
            f"Estimador KSG con k = {K_NEIGHBORS} vecinos sobre "
            f"{universe.report.n_sessions} observaciones; métrica en nats."
        ),
        (
            f"Criterio de cointegración: p-value del test ADF "
            f"< {SIGNIFICANCIO_ADF:.0%}, sobre el spread de cada par y con "
            f"al menos {MIN_SESIONES_ADF} sesiones; mediana de "
            f"{cointegrated_df['n_sesiones_par'].median():.0f} sesiones "
            f"(rango {cointegrated_df['n_sesiones_par'].min()}-"
            f"{cointegrated_df['n_sesiones_par'].max()})."
        ),
        (
            f"Filtro aplicado a los {len(top_pairs_df)} candidatos con mayor "
            f"información mutua KSG: {n_cointegrados} resultaron "
            f"cointegrados."
        ),
        (
            "El filtro no recorre el universo completo de pares: la tabla "
            "lista los pares más acoplados en MI y cointegrados a la vez, no "
            "todos los pares cointegrados del índice."
        ),
        (
            "Activos excluidos del universo por no tener descarga "
            f"consistente: {', '.join(TICKERS_SIN_DESCARGA)}."
        ),
    ],
)
print(f"LaTeX: {export_paths_cointegrated['tex']}")

# %%
# ## Clustering jerárquico de Ward con k = número de sectores
#
# ``mi_distance`` convierte MI en distancia restando el máximo:
# $d_{ij} = \max(MI) - MI_{ij}$.  Activos con MI alta quedan
# cercanos; activos independientes quedan lejos.
#
# El método de **Ward** minimiza el incremento de varianza al fusionar
# clusters.  El número de clusters se fija igual al número de sectores
# GICS presentes para hacer la evaluación sectorial 1-a-1.
distance_matrix = mi_distance(mi_matrix)
cluster_labels, leaf_order = hierarchical_clustering(
    distance_matrix, n_clusters=n_clusters,
)
labels_ordered = mi_matrix.columns[leaf_order]
mi_ordered = mi_matrix.loc[labels_ordered, labels_ordered]

clusters = pd.Series(cluster_labels, index=mi_matrix.columns, name="cluster")
print(f"\nClustering jerárquico de Ward (k = {n_clusters}):")
for cid in sorted(clusters.unique()):
    members = clusters.index[clusters == cid].tolist()
    print(f"  Cluster {cid}: {members[:10]}{'...' if len(members) > 10 else ''}")

# %%
# ## Evaluación de la calidad sectorial del clustering
#
# Mide cuánto se parece el clustering a la partición oficial por
# sectores GICS.  Las métricas se imprimen primero en formato global
# y después desglosadas por cluster.  No son una medida absoluta de
# calidad financiera: evalúan **alineación con la taxonomía GICS**.
report = evaluate_cluster_sector_alignment(clusters, available_sectors)

print("\nMétricas globales de alineación sectorial:")
print(f"  Purity ponderada        : {report.weighted_purity:.4f}")
print(f"  Entropía normalizada    : {report.normalized_entropy:.4f}")
print(f"  HHI medio               : {report.mean_hhi:.4f}")
print(f"  Adjusted Rand Index     : {report.ari:.4f}")
print(f"  Adjusted Mutual Info    : {report.ami:.4f}")
print(f"  Homogeneity             : {report.homogeneity:.4f}")
print(f"  Completeness            : {report.completeness:.4f}")
print(f"  V-measure               : {report.v_measure:.4f}")

print("\nMétricas por cluster:")
print(report.per_cluster.to_string())

# %%
# ## Visualización sectorial
#
# Tres vistas complementarias:
# - **Heatmap**: cuántos activos de cada sector caen en cada cluster.
# - **Composición apilada**: porcentaje del cluster ocupado por cada
#   sector (útil para comparar clusters de distinto tamaño).
# - **Resumen**: purity, entropía normalizada y HHI por cluster.
plot_cluster_sector_heatmap(
    report.contingency,
    title=f"Sectores vs clusters (MI KSG, k = {n_clusters})",
)

plot_cluster_sector_composition(
    report.contingency,
    title=f"Composición sectorial por cluster (MI KSG, k = {n_clusters})",
)

plot_cluster_sector_summary(report.per_cluster)

# %%
# ## Matriz MI ordenada por cluster (sin etiquetas de ticker)
#
# La matriz se reordena **globalmente por cluster**, igual que en
# ``2_PairSelectionPearson.py``: los bloques del heatmap son los clusters
# que evalúa el análisis sectorial, y los únicos separadores dibujados son
# los cambios de cluster (línea discontinua).
#
# ## Escala de color comparable con la de Pearson
#
# Se usa la misma paleta (``viridis``) y la misma política de escala: los
# límites salen de los valores **medidos** en la matriz, ignorando la
# diagonal ``NaN``.  Ningún valor se transforma; la MI se dibuja en nats.
#
# Un mismo color equivale a la misma intensidad **dentro de cada métrica**,
# no al mismo valor numérico: Pearson e información mutua tienen unidades
# distintas y cada barra de color etiqueta las suyas.  Lo comparable entre
# ambos cuadernos es el patrón de los bloques, no la magnitud.
CMAP_COMPARABLE = "viridis"

plot_sector_reordered_matrix(
    mi_matrix,
    sector_labels=available_sectors,
    cluster_labels=clusters,
    cmap=CMAP_COMPARABLE,
    title="MI KSG ordenada por cluster",
    cbar_label="MI (nats)",
    cluster_labels_visible=False,
    separator_color=None,
)

# %%
# ## Matriz promedio sector × sector
#
# ``sector_average_matrix`` agrega la matriz de activos a una matriz
# sector × sector usando la media (omitiendo la diagonal dentro del
# bloque).  Valores altos en la diagonal confirman acoplamiento
# intra-sectorial; valores altos fuera de la diagonal indican
# dependencias inter-sectoriales.
#
# Comparte paleta y escala medida con el heatmap de la matriz de activos
# y con las figuras del cuaderno de Pearson.
sector_mi = sector_average_matrix(mi_matrix, available_sectors, operation="mean")

plot_sector_average_heatmap(
    sector_mi,
    cmap=CMAP_COMPARABLE,
    title="MI KSG promedio entre sectores GICS",
    cbar_label="MI media (nats)",
)

# %%
# ## Símbolos finales del top de pares por MI
#
# Lista única de tickers que aparecen en el conjunto de candidatos.  Estos
# son los candidatos a estudiar en `PairTrading.py` para el análisis del
# spread y el modelo OU.
final_symbols = sorted({t1 for t1, _, _ in top_20_pairs}
                       | {t2 for _, t2, _ in top_20_pairs})
print(final_symbols)

# %%
# ## Símbolos finales de los pares cointegrados
#
# Los tickers de los pares que superan el filtro ADF.  Es un subconjunto
# más restrictivo del anterior: los mismos candidatos, pero con el spread
# verificado como estacionario.
cointegrated_symbols = sorted(
    set(cointegrated_df["ticker_a"]) | set(cointegrated_df["ticker_b"])
)
print(cointegrated_symbols)
# %%

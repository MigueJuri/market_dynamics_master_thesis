# Pair Trading — Documentación

Este directorio contiene los cuadernos y simulaciones asociadas a la
estrategia de *pair trading* (trading por pares), tanto para encontrar
pares candidatos como para modelar el spread entre dos activos como un
proceso de Ornstein-Uhlenbeck.

## Estructura

| Archivo | Propósito |
|---|---|
| [`1_PairSelectionMutualInformation.py`](./1_PairSelectionMutualInformation.py) | Selección de pares mediante Información Mutua estimada con el método KSG (k-NN). |
| [`2_PairSelectionPearson.py`](./2_PairSelectionPearson.py) | Selección de pares mediante correlación de Pearson, con animación de la correlación *rolling*. |
| [`PairTrading.py`](./PairTrading.py) | Construcción del spread, ajuste del modelo OU y las tres gráficas del par: spread, precios y vista conjunta con subplots. |
| [`OUProcess.py`](./OUProcess.py) | Simulación de un proceso OU y de dos activos GBM cuyo spread coincide con la trayectoria simulada. |

Toda la lógica común (descarga de precios, log-retornos, scraping del
S&P 500, filtro de emisores duplicados, estimadores de MI, clustering,
cálculo de métricas sectoriales, simulaciones OU y GBM) vive en
`src/thesis/pair_trading.py` y se reutiliza desde los cuadernos.

## Orden recomendado de ejecución

1. `OUProcess.py` (sólo requiere `simulate_ou_process`, `simulate_gbm`).
2. `PairTrading.py` (depende de `extract_and_construct_spread`,
   `fit_ou_process` y de `thesis.plotting` para las tres figuras:
   `plot_spread`, `plot_pair_prices` y `plot_pair_overview`).
3. `1_PairSelectionMutualInformation.py` y/o
   `2_PairSelectionPearson.py`.  Ambos comparten la misma fase de
   preparación (universo, retornos, sectores) pero luego divergen en
   la métrica central.  Pueden ejecutarse en cualquier orden; incluso
   en paralelo, ya que la caché KSG es local a cada notebook.

## Configuración común

Todos los cuadernos leen las fechas de `thesis.config`:

```python
from thesis.config import START, END, DATA_DIR
```

`START` y `END` por defecto son `"1993-01-01"` y `"2026-01-01"`.  Los
precios descargados se cachean en CSV bajo `DATA_DIR`
(`thesis/data/` por defecto) gracias a `thesis.market_data.load_multiple_prices`.

## Universo y filtrado de emisores

`get_sp500_constituents()` descarga la tabla oficial del S&P 500 desde
Wikipedia.  `filter_duplicate_issuers()` la normaliza (elimina sufijos
de `Class A/B`, `Inc.`, `Corp.`, etc.) y conserva **una sola fila por
emisor**, descartando explícitamente pares como:

- `GOOGL` y `GOOG` (Alphabet)
- `FOXA` y `FOX` (Fox Corporation)
- `NWSA` y `NWS` (News Corp)

Esto evita evaluar pares inválidos *antes* de descargar datos y calcular
matrices.  Adicionalmente, `top_pairs(issuer_map=...)` descarta
cualquier par residual del mismo emisor al construir el ranking.

## Estimadores de dependencia

- **Correlación de Pearson** (`returns.corr(method="pearson")`):
  captura únicamente relaciones lineales.
- **Información Mutua gaussiana** (`gaussian_mi_matrix`):
  cerrada para distribuciones normales, equivalente a
  $-\tfrac{1}{2}\log(1-\rho^2)$.
- **Información Mutua KSG** (`ksg_mi_matrix` / `cached_ksg_mi_matrix`):
  estimador por k-vecinos que captura dependencias no lineales.

## Clustering jerárquico

En ambos cuadernos de selección se aplica
`hierarchical_clustering()` con el método de **Ward** sobre una matriz
de distancias:

- `correlation_distance(corr, use_sqrt=True)`:
  $d_{ij} = \sqrt{1 - \rho_{ij}}$ para Pearson.
- `mi_distance(mi)`:
  $d_{ij} = \max(MI) - MI_{ij}$ para Información Mutua.

El número de clusters se fija igual al número de sectores GICS
presentes en el universo (`available_sectors.nunique()`, 11 en el
S&P 500 actual).  Esto facilita una comparación 1-a-1 entre sectores y
clientes GICS.

## Evaluación sectorial de los clusters

`evaluate_cluster_sector_alignment()` devuelve un `ClusterSectorReport`
con:

- **`contingency`**: tabla sector × cluster (más dos filas extra,
  `cluster_size` y `cluster_purity`).
- **`per_cluster`**: tamaño, sector dominante, purity, entropía
  normalizada y HHI por cluster.
- Métricas globales:
  - `weighted_purity` (pureza ponderada por tamaño);
  - `normalized_entropy` (entropía sectorial media, normalizada por
    $\log S_c$);
  - `mean_hhi` (Herfindahl-Hirschman medio);
  - `ari` (Adjusted Rand Index);
  - `ami` (Adjusted Mutual Information);
  - `homogeneity`, `completeness` y `v_measure` (de `sklearn.metrics`).

Estas funciones miden **alineación con los sectores GICS**, no la calidad
financiera del clustering.  Los sectores GICS no son la verdad
absoluta sobre la dependencia estadística, sólo una partición de
referencia basada en actividad económica.

## Caché de matrices KSG

KSG es O(N²) en el número de activos (≈125.000 pares en el S&P 500
completo).  `cached_ksg_mi_matrix()` persiste el resultado en disco:

```text
data/pair_trading/mi/ksg/
└── ksg_k{k}_seed{s}_{assets}x{obs}_{hash}.pkl
└── ksg_k{k}_seed{s}_{assets}x{obs}_{hash}.json
```

La clave de caché incorpora:

- versión del esquema (`KSG_CACHE_VERSION`);
- `k_neighbors` y `seed`;
- fechas de inicio/fin y número de observaciones;
- lista ordenada de activos;
- hash SHA-256 del contenido de los retornos.

Cualquier cambio en estos campos provoca una entrada de caché distinta.
La escritura es **atómica** (`tempfile` + `os.replace`) para evitar
ficheros corruptos si el proceso se interrumpe.

Para forzar el recálculo de una entrada concreta basta con pasar
`force=True`:

```python
mi_matrix = cached_ksg_mi_matrix(returns, k_neighbors=3, force=True)
```

La carpeta `data/` está en `.gitignore`, por lo que los ficheros
generados nunca se suben al repositorio.

## Construcción del panel y control de cobertura

`prepare_pair_selection_universe()` ejecuta la fase compartida por
`1_PairSelectionMutualInformation.py` y `2_PairSelectionPearson.py` y
devuelve un `PairSelectionUniverse` con `returns`, `report`, `sectors`,
`issuers`, `n_sectors` y `removed_duplicate_issuers`.  Encadena:

1. `filter_duplicate_issuers()` — una clase de acción por emisor.
2. `load_returns_panel()` — filtrado por historia y alineación.
3. Alineación de sectores y emisores con las columnas del panel.

Centralizarlo garantiza que la matriz de Pearson y la de Información
Mutua se estimen sobre exactamente la misma muestra.  El umbral vive en
`thesis.config` como `PAIR_SELECTION_MIN_HISTORY_OBS`, de modo que
cambiarlo no exige editar los dos cuadernos.

## Nombres de los sectores

`get_sp500_constituents()` descarga la tabla de Wikipedia en inglés.  Los
cuadernos aplican `translate_gics_sectors()` sobre los sectores, de modo que
métricas, figuras y tablas exportadas salen en castellano:

| GICS (inglés) | Castellano |
|---|---|
| Communication Services | Servicios de Comunicación |
| Consumer Discretionary | Consumo Discrecional |
| Consumer Staples | Consumo Básico |
| Energy | Energía |
| Financials / Financial Services | Servicios Financieros |
| Health Care | Salud |
| Industrials | Industriales |
| Information Technology | Tecnología de la Información |
| Materials | Materiales |
| Real Estate | Inmobiliario |
| Utilities | Servicios Públicos |

Traducir **no altera ningún resultado**: purity, entropía, HHI, ARI y AMI
sólo dependen de la partición sectorial, no de sus nombres.  GICS renombró
`Financials` a `Financial Services`, así que se aceptan ambas
denominaciones.  Un sector sin mapeo se conserva tal cual y `unmapped_gics_sectors()` lo
lista en la consola, para que una etiqueta en inglés no llegue a la tesis
sin que nadie se entere.

El orden de las operaciones es lo esencial: **primero se filtra por
historia, después se alinean las fechas**. `dropna(how="any")` conserva
sólo las fechas en las que cotizan todos los activos a la vez, así que sin
el filtro previo un único emisor recién salgado a bolsa determina la
ventana de todo el análisis. Medido sobre la caché actual del universo
S&P 500 con la ventana 1993-2026:

| `min_history_obs` | activos | sesiones | desde | cobertura |
|---|---|---|---|---|
| 0 | 497 | 43 | 2025-10-28 | 0.5% |
| 1260 (~5 años) | 484 | 1 267 | 2020-12-11 | 14.7% |
| 5040 (~20 años) | 388 | 5 047 | 2005-12-06 | 60.8% |
| 7560 (~30 años) | 301 | 7 557 | 1995-12-18 | 87.8% |

Conviene recordar que la cobertura nunca será del 100% aunque el filtro
se elimine: la mayoría de los componentes actuales del índice no cotizaba
todavía en 1993.

El valor por defecto es `PAIR_SELECTION_MIN_HISTORY_OBS = 7560` (~30 años,
301 activos, 7 557 sesiones).  Para priorizar cobertura de activos sobre
longitud del historial, bajarlo a 5040 (~20 años, 388 activos).

`print_dataset_report()` imprime el diagnóstico, que expone las
mediciones que hacen visible cualquier recorte:

- `n_sessions` y `sessions_per_year` (~251 confirma que el índice es de
  sesiones bursátiles y no de días calendario).
- `coverage_pct` frente a `union_sessions`: cuántas de las fechas en las
  que cotizó *algún* activo sobreviviente acaba usando la intersección. Es
  la métrica que delata un activo sin historia truncando el panel.
- `window_pct` frente a la ventana solicitada, que es informativa.
- `history_per_asset` y `dropped_assets` con `n_obs`, fechas y `motivo`
  (`historia_insuficiente` o `sin_datos`).

Si `n_sessions` baja de `MIN_SESSIONS_WARN` (252) se emite un
`UserWarning`: es preferible fallar de forma visible a producir
correlaciones y matrices sobre dos meses de datos.

## Exportación de tablas

`export_pair_selection_table()` escribe tres ficheros con el mismo nombre
base en `outputs/tables` (`thesis.config.TABLES_DIR`):

```text
outputs/tables/
├── pearson_top20.csv        valores sin formatear
├── pearson_top20.tex        tabla lista para \input
├── pearson_top20.json       condiciones estructuradas
├── mutual_information_top20.{csv,tex,json}
```

El CSV conserva la precisión completa para reutilizarlo; el `.tex` lleva
las condiciones de cálculo como nota bajo la tabla; el JSON las guarda como
datos, útiles para documentar o comparar ejecuciones.

### Uso

```python
export_pair_selection_table(
    top_pairs_df,
    universe.report,
    stem="pearson_top20",
    columns=["rank", "ticker_a", "ticker_b", "sector_a", "sector_b",
             "correlacion_pearson", "p_valor_adf"],
    method="correlación de Pearson sobre log-retornos diarios",
    caption="Principales pares por correlación de Pearson",
    label="tab:pearson_top20",
    extra_metadata={"significancia_adf": 0.05},
    notes=["Test ADF con significancia 5%."],
)
```

- `columns` fija qué se exporta al CSV y al LaTeX, y en qué orden.  Si se
  omite, se exportan todas.  La lista se respeta al pie de la letra: si no
  incluye `rank`, la tabla no lo lleva.  Cada cuaderno declara su lista en
  una constante (`COLUMNAS_TABLA`) para poder recortarla sin tocar nada más.
- `extra_metadata` alimenta únicamente el JSON.
- `notes` añade condiciones ya redactadas a la nota del `.tex`.
- La validación ocurre **antes** de escribir: una columna inexistente, una
  repetida, una lista vacía o un nombre suelto en lugar de secuencia abortan
  la exportación sin dejar ficheros a medias.

### Condiciones documentadas

Una tabla sin las condiciones que la producen es irreproducible: los
mismos tickers sobre otra ventana dan ADF distintos. La nota incluye
método, ventana solicitada y efectiva, sesiones, días naturales, ritmo
anual, tamaño del universo, descartes por historia, filtro aplicado,
cobertura frente al calendario disponible y frente a la ventana pedida.

### En LaTeX

Sólo necesita el paquete `booktabs`:

```latex
\usepackage{booktabs}
...
\input{outputs/tables/pearson_top20}
```

El `.tex` se verificó compilando con `pdflatex`. Las condiciones van en un
`minipage` y no en `tablenotes` porque dentro de `threeparttable` tanto
`itemize` como los párrafos sueltos abortan la compilación.

`outputs/` está en `.gitignore`: las tablas son artefactos que se regeneran
al ejecutar los cuadernos, y las que acaben usándose en la tesis se copian
a su repositorio.

### Sobre `DataFrame.to_latex`

No se usa `DataFrame.to_latex`: en pandas 3 delega en `Styler`, que exige
`jinja2`, y el proyecto no declara esa dependencia. El formateador propio
además controla el escapado, los valores ausentes (`--`), los booleanos
(`Sí`/`No`, con `Sí` en negrita) y el separador de miles.

## Numeración de clusters bajo la matriz

Bajo cada bloque del heatmap se imprime su **número**, centrado, y el eje X
lleva el título `Clusters`:

```text
[matriz de correlación o MI]

      1     2     3     4
      ⋮     ⋮     ⋮     ⋮
```

La numeración es la posición del bloque de izquierda a derecha, no el
identificador interno del clustering: aunque las etiquetas sean `[10, 20,
30]`, la figura muestra `1, 2, 3`, que es lo que hace legible el gráfico. Para
conocer el sector mayoritario de cada cluster hay que mirar la tabla de
contingencia (`evaluate_cluster_sector_alignment`), no el heatmap.

El centro se calcula como `(inicio + fin) / 2`, porque `imshow` sitúa el
índice `i` en el píxel `[i - 0.5, i + 0.5]`. Los separadores se dibujan en
`i - 0.5`, que es justo la frontera entre bloques, así que número y frontera
caen geométricamente alineados.

Opciones: `cluster_labels_visible=False` oculta la numeración y
`xlabel` cambia el rótulo del eje. `sector_labels_visible=True` rotula el eje
**Y** con los sectores, ya que el eje X queda para los clusters.

## Heatmaps de dependencia y escala comparable

`plot_sector_reordered_matrix()` dibuja la matriz de activos reordenada
**globalmente por cluster**: los bloques del heatmap son exactamente los
clusters que evalúa el análisis sectorial. Los únicos separadores son las
líneas discontinuas entre clusters; los límites sectoriales ya no se marcan,
porque en una matriz de ~300 activos sólo añadían ruido. Con
`sector_labels_visible=True` se pueden rotular las bandas sectoriales en los
ejes si hiciera falta.

### Escala de color

Pearson e información mutua comparten paleta (`viridis`) y política de
escala:

- los límites `vmin`/`vmax` salen de los valores **medidos** en la matriz,
  ignorando la diagonal `NaN`;
- **no se transforma ningún valor**: Pearson se dibuja en su rango real y la
  MI en nats;
- cada barra de color etiqueta sus propias unidades.

Lo comparable es el **patrón de bloques** y la intensidad relativa, no la
magnitud: un mismo color significa "intensidad equivalente dentro de cada
métrica", no el mismo valor numérico, porque las dos métricas tienen
unidades distintas.

Una asimetría conviene conocerla: en la matriz de correlación la diagonal
vale 1, así que `vmax` queda fijado en 1 aunque las correlaciones entre
activos distintos rara vez se acerquen; en la de información mutua la
diagonal es `NaN` y `vmax` es el mayor valor real. Si se prefiere que ambas
escalas se refieran sólo a valores fuera de la diagonal, basta enmascararla
antes de llamar a la función.

`plot_sector_average_heatmap()` aplica la misma política a las matrices
sector × sector, y ambos cuadernos la usan para que las dos vistas y los
dos cuadernos sean comparables entre sí.

## Log-precios frente a log-retornos

La selección de pares usa **log-retornos** (`load_log_returns`), porque la
correlación y la información mutua se estiman sobre las variaciones
diarias.

El análisis de cointegración, en cambio, necesita los **log-precios**
(`load_log_prices`): el hedge ratio y el test ADF se estiman sobre
$\log P$, que es una serie I(1), no sobre sus diferencias.

No son intercambiables.  Reconstruir los log-precios acumulando los
log-retornos (`returns.cumsum()`) no devuelve los niveles reales:

- descarta el primer nivel de cada activo (una constante que el
  intercepto de la OLS absorbería, pero que desplaza el spread);
- opera sobre el frame de retornos ya alineado, por lo que pierde todas
  las fechas en las que el otro activo no cotizó y salta days completos;
- parte de una intersección de fechas mucho más restringida que la de un
  par concreto, al intersectar los ~500 activos del universo.

Por eso `PairTrading.py` y la tabla ADF de `2_PairSelectionPearson.py`
obtienen los log-precios con `load_log_prices` y sólo así sus estadísticos
son comparables para un mismo par.

## Advertencias metodológicas

- **KSG es costoso**: sobre los ≈500 activos del S&P 500 puede tardar
  varios minutos.  La caché KSG evita repetir el cálculo.
- **Sesgo de supervivencia**: `get_sp500_constituents()` lee la
  composición *actual* del S&P 500 desde Wikipedia.  Los retornos
  históricos se asignan al sector *presente hoy*; esto introduce un
  sesgo si empresas cambiaron de sector en el pasado.
- **Cobertura de los datos**: la caché CSV de precios (`market_data`)
  puede tener huecos para tickeros poco negociados; esos activos se
  descartan en `filter_by_history` si quedan por debajo de `min_valid_obs`
  o `min_history_obs` observaciones. La caché se indexa por
  `{ticker}_{start}_{end}.csv`, así que cada ventana temporal usada
  (1993-2026 en los cuadernos) tiene su propio juego de ficheros.
- **Costes de descarga**: `load_multiple_prices` itera ticker por ticker
  sin descarga en paralelo. Si la ventana cambia, aparecen ficheros de
  caché nuevos y la primera ejecución descarga el universo completo; las
  siguientes son locales. Conviene comprobar qué ventanas existen en
  `data/` antes de lanzar un cuaderno.
- **GICS ≠ dependencia**: las métricas de evaluación sectorial miden
  cuánto se parece el clustering a la partición oficial del S&P 500.
  Un clustering "bueno" en términos financieros (pares cointegrados)
  puede no coincidir con los sectores GICS.
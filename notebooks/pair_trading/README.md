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
| [`Spread.py`](./Spread.py) | Construcción del spread, ajuste del modelo OU y las tres gráficas del par: spread, precios y vista conjunta con subplots. |
| [`OUProcess.py`](./OUProcess.py) | Simulación de un proceso OU y de dos activos GBM cuyo spread coincide con la trayectoria simulada. |
| [`PracticalPairTrading.py`](./PracticalPairTrading.py) | Backtest básico de pair trading con hedge ratio estático y z-score móvil. |

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
├── pearson_cointegrated_top20.{csv,tex,json}
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

## Tabla de pares cointegrados (Pearson + ADF)

`2_PairSelectionPearson.py` exporta **dos** tablas:

| Tabla | `stem` | Contenido |
|---|---|---|
| Candidatos | `pearson_top20` | Los 200 pares con mayor correlación, con el diagnóstico ADF de cada uno (estacionarios y no estacionarios mezclados). |
| Cointegrados | `pearson_cointegrated_top20` | Los 20 pares con mayor correlación **entre los que superan el filtro ADF**. |

La segunda tabla es un subconjunto de la primera: no introduce una métrica
nueva, sino que aplica el diagnóstico ADF que la primera ya calcula como
criterio de selección.

### Criterio de cointegración

El filtro es el **p-value del test ADF**, con significancia
`SIGNIFICANCIO_ADF = 0.05`:

```python
cointegrated_df = (
    top_pairs_df.loc[top_pairs_df["p_valor_adf"] < SIGNIFICANCIO_ADF]
    .sort_values("correlacion_pearson", ascending=False)
    .head(TOP_N_COINTEGRADOS)
    .reset_index(drop=True)
)
```

Se usa el p-value y no un umbral fijo sobre el t-estadístico porque el valor
crítico del ADF depende del tamaño muestral del par: a mayor muestra, el
mismo nivel de significancia exige un t-stat más negativo. Un corte como
`t < -3` no sería comparable entre pares con muestras de tamaños distintos,
mientras que el p-value ya tiene en cuenta ese efecto. El t-stat se conserva
en la tabla exportada como diagnóstico, que es el estadístico que el test
contrasta contra su valor crítico.

### Alcance del filtro

El filtro se aplica **dentro del conjunto de candidatos ya evaluados** por el
cuaderno, no sobre el universo completo de pares. Estimar el ADF para los
~75.000 pares del universo completo multiplica por unas dos horas el tiempo de
ejecución, frente a los minutos que tarda el panel ya calculado.

La consecuencia metodológica es explícita y queda documentada en la nota del
`.tex`: la tabla identifica los pares **más correlacionados y cointegrados a
la vez**, no una lista exhaustiva de todos los pares cointegrados del S&P
500. Un par cointegrado cuya correlación lo situara fuera del conjunto de
candidatos evaluados no aparece en ella.

### Columnas exportadas

```text
rank, ticker_a, ticker_b, sector_a, sector_b,
correlacion_pearson, n_sesiones_par,
estadistico_adf, p_valor_adf, estacionario
```

`rank` se reasigna según la posición dentro de la tabla filtrada, no
hereda el de la tabla de candidatos: el par 1 de la tabla de cointegrados es
el de mayor correlación entre los que pasan el filtro, que no coincide con
el par 1 de la tabla de candidatos si ese no está cointegrado.

### En LaTeX

```latex
\input{outputs/tables/pearson_cointegrated_top20}
```

Requiere sólo `booktabs`, igual que el resto de tablas.

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

## Backtest básico (`PracticalPairTrading.py`)

`PracticalPairTrading.py` es un notebook sencillo para probar una
estrategia de pair trading sobre un único par.  Está dividido en dos
partes claramente separadas:

1. **Análisis descriptivo in-sample**: calcula el hedge ratio, el test
   ADF y el proceso OU sobre todo el periodo.  Sirve para diagnosticar
   si el par parece cointegrado, pero **no se usa para generar
   señales**.
2. **Backtest walk-forward**: simula la ejecución real de la
   estrategia estimando cada día el hedge ratio y el z-score sólo con
   datos anteriores.

### Flujo del backtest walk-forward

1. Descarga los precios ajustados de dos tickers mediante
   `thesis.market_data.load_multiple_prices` y la caché CSV local.
2. Para cada día ``t``:
   - Estima ``alpha`` y ``beta`` por OLS sobre los últimos
     `hedge_window` días (hasta ``t-1``).
   - Calcula el spread histórico en los últimos `z_window` días.
   - Normaliza el spread de ``t`` con la media y desviación de esa
     ventana histórica.
   - Genera señal: long spread si ``z < -entry_z``, short spread si
     ``z > entry_z``, cierre al cruzar ``exit_z``.
3. La posición decidida al cierre de ``t`` se ejecuta desde ``t`` hasta
   ``t+1`` (lag de un día sin look-ahead).
4. El beta se fija en el momento de la entrada
   (`rebalance_hedge="on_entry"`) para no introducir turnover artificial.
5. El retorno se normaliza por el notional bruto
   ``1 + abs(beta)``, de modo que representa el P&L sobre el capital
   total comprometido.
6. Se restan costes proporcionales al turnover sobre ese notional.

### Normalización por notional

La estrategia largo/corto compromete ``1`` unidad en A y ``|beta|``
unidades en B.  El retorno bruto ``ret_A - beta * ret_B`` es el P&L
relativo a una unidad de A.  Para expresarlo sobre el capital total
invertido se divide por ``1 + |beta|``:

```python
gross_notional = 1.0 + abs(beta)
strategy_ret = position * (ret_A - beta * ret_B) / gross_notional
```

Esto hace comparables los resultados entre pares con distintos betas.

### Parámetros principales

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `HEDGE_WINDOW` | `252` | Días de historia para estimar el hedge ratio. |
| `Z_WINDOW` | `60` | Días de historia para estimar media y std del spread. |
| `ENTRY_Z` | `2.0` | Umbral de entrada en valor absoluto. |
| `EXIT_Z` | `0.0` | Nivel de z-score al que se cierra la posición. |
| `COST` | `0.001` | Coste por unidad de turnover sobre notional bruto. |

## Comparativa con benchmarks

`PracticalPairTrading.py` contrasta la estrategia con tres alternativas sobre
el **mismo periodo**: `SPY` (buy & hold del ETF del S&P 500) y el buy & hold
de cada componente del par.

```python
from thesis.pair_trading import build_strategy_comparison

comparativa = build_strategy_comparison(
    bt.results["strategy_ret"],
    {
        "Buy & hold SPY": ret_spy,
        "Buy & hold A": ret_a,
        "Buy & hold B": ret_b,
    },
)
print(comparativa.to_string(index=False))
```

| Columna | Significado |
|---|---|
| `estrategia` | Alternativa evaluada; la estrategia es siempre la primera fila. |
| `retorno_total` | `exp(sum(retornos logarítmicos)) - 1`. |
| `sharpe` | `media / desviación * sqrt(252)`, sin tasa libre de riesgo. |
| `n_sesiones` | Sesiones del periodo común. |
| `inicio` / `fin` | Extremos del periodo común. |

### Mismo periodo para todos

La función recorta las fechas en las que falta alguna alternativa, de modo
que todas las filas acumulan el mismo número de sesiones y el mismo
calendario.  Sin ese recorte la comparación no sería homogénea: un buy &
hold con más historia acumularía más retorno por el solo hecho de haber
existido durante más tiempo.

El recorte parte del índice de resultados del backtest, así que **incluye el
warm-up**: durante las primeras `HEDGE_WINDOW` sesiones la estrategia aún no
puede operar y permanece en efectivo (retorno cero), mientras que los
benchmarks sí acumulan su retorno.  Es el tratamiento honesto de "habría
desplegado esta estrategia con estos datos en esta fecha", y hace la
comparación conservadora para la estrategia.

### Convención de las métricas

Las métricas usan la misma convención que `pair_trading_backtest`, para que
la fila de la estrategia de la tabla reproduzca `bt.total_return` y
`bt.sharpe`.  El cuaderno lo comprueba con un `assert`.

El Sharpe **no lleva tasa libre de riesgo**, por coherencia con `bt.sharpe`.
Es la convención habitual para comparar estrategias entre sí, pero un Sharpe
sin `rf` no es el Sharpe "de mercado": al omitir el término `rf` el valor
queda inflado en la misma proporción que el retorno sin riesgo del periodo.

### Los benchmarks se calculan desde el precio crudo

Los retornos de los benchmarks se obtienen como `np.log(P).diff()`, es decir,
aplicando el logaritmo **una sola vez** al precio.  Es un error fácil de
cometer: si la entrada ya es un log-precio, `np.log()` produce
`log(log(P))`, que no es un retorno.

El efecto es silencioso porque el resultado sigue pareciendo plausible, pero
subestima el rendimiento: para `FRT` entre 2006 y 2025, `log(log(P))` da
+34,7 % frente al +225,0 % real, y con ello la estrategia parecía superar al
buy & hold de sus propios componentes.

El cuaderno incluye una validación que blinda este error: para cada
benchmark, el retorno total de la tabla se contrasta con el ratio de
precios `P_fin / P_ini - 1`, que es la identidad exacta de un buy & hold.

```python
precios = serie.reindex(backtest_index)
retorno_por_precios = precios.iloc[-1] / precios.iloc[0] - 1.0
assert np.isclose(fila["retorno_total"], retorno_por_precios)
```

### Curva de capital

La curva de SPY se añade a `results` como la columna `bh_equity_SPY`, siguiendo
el patrón de `bh_equity_A` y `bh_equity_B`.  `plot_equity`,
`plot_equity_diagnostics` y `plot_backtest_dashboard` la dibujan
automáticamente cuando la columna está presente, y la omiten si no lo está.

## Estado actual del par: última ventana

`summarize_pair_window` resume el hedge ratio, el test ADF y el ajuste OU
(vida media) de las últimas `window` observaciones y los compara con los
de la muestra completa:

```python
from thesis.pair_trading import summarize_pair_window

summary = summarize_pair_window(
    log_prices, TICKER_A, TICKER_B, window=HEDGE_WINDOW
)

print(summary.comparison_table())
```

| Métrica | Muestra completa | Última ventana |
|---|---|---|
| Observaciones | ... | ... |
| Alpha | ... | ... |
| Beta | ... | ... |
| R² | ... | ... |
| ADF statistic | ... | ... |
| ADF p-value | ... | ... |
| ¿Estacionario? | ... | ... |
| Half-Life (sesiones) | ... | ... |
| Theta (OU) | ... | ... |
| Sigma eq (OU) | ... | ... |
| R² AR(1) | ... | ... |

El resumen añade además `beta_change`, `beta_change_pct`,
`adf_pvalue_change`, `half_life_change`, `half_life_change_pct` y
`stationarity_lost`, que es `True` cuando el spread era estacionario en la
muestra completa y deja de serlo en la ventana reciente.

La vida media se obtiene ajustando un proceso de Ornstein-Uhlenbeck al
spread de cada ventana (`fit_ou_process`).  Si el spread no admite un
AR(1) estacionario, `summary.ou` es `None` y `summary.ou_error` explica el
motivo, en lugar de propagar el error.

> Esta ventana **incluye** la última observación, por lo que describe el
> estado *actual* del par.  Es exclusivamente diagnóstica: no debe usarse
> para generar señales históricas, porque dentro del backtest la ventana
> disponible en cada fecha termina en `t-1`.

La gráfica `plot_pair_window_summary` superpone el spread completo y el de
la ventana reciente, muestra las bandas de equilibrio de la ventana
reciente y contrasta los tres hedge ratio: walk-forward, muestra completa
y ventana reciente.  El eje temporal incluye la vida media de ambas
muestras como información auxiliar.

## Diagnóstico del backtest

`src/thesis/backtest_diagnostics.py` agrega métricas de diagnóstico y
ejecuta comprobaciones automáticas para auditar que el backtest funciona
correctamente.  Se usa desde el notebook con:

```python
from thesis.backtest_diagnostics import compute_pair_backtest_diagnostics

diag = compute_pair_backtest_diagnostics(
    bt.results,
    bt.equity,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
    hedge_window=HEDGE_WINDOW,
    z_window=Z_WINDOW,
    log_prices=log_prices,
    ticker_a=TICKER_A,
    ticker_b=TICKER_B,
    max_abs_beta=5.0,
)
```

### Bloques de métricas

| Bloque | Contenido |
|---|---|
| `diag.hedge` | Momentos, percentiles y extremos de `beta` y `alpha`, número de cambios significativos, betas fuera de rango, número de betas efectivos distintos y si el beta es constante dentro de cada operación. |
| `diag.signal` | NaNs del z-score, z-score máximo, fracción de tiempo en cash, entradas/salidas, round trips, Breakdown por dirección, duración media/mediana/máxima y cumplimiento de los umbrales. |
| `diag.returns` | Turnover total y medio, turnover por operación, costes acumulados y como porcentaje del retorno bruto, notional bruto (media y percentiles), exposición media a cada activo y error máximo de cuadre de las contribuciones. |
| `diag.risk` | Retorno total, CAGR, volatilidad, Sharpe, Sortino, Calmar, drawdown máximo y su duración, VaR y CVaR al 5%, mejor y peor operación, profit factor, win rate y ganancias/pérdidas medias. |
| `diag.trades` | Una fila por operación con entrada, salida, z-scores, beta aplicado, duración, retorno bruto, costes, resultado neto y motivo de salida. |
| `diag.validation` | Resultado de cada comprobación automática (`ValidationReport`). |

### Comprobaciones automáticas

`validate_pair_backtest` cubre:

- integridad del índice temporal y rango de las posiciones;
- respeto del warm-up (ninguna posición antes de `hedge_window`);
- beta efectivo nulo en cash y constante dentro de cada operación;
- alineación de un día entre la posición señalada y la aplicada;
- cuadre de contribuciones, costes, retorno neto, equity y drawdown;
- coherencia del turnover con la variación de los pesos;
- cumplimiento de los umbrales de entrada y de salida.

`verify_hedge_is_causal` recalcula `alpha`, `beta` y el z-score en una
muestra de fechas usando exclusivamente datos anteriores y los compara con
lo almacenado.  Es la comprobación que detecta look-ahead.

### Gráficas de diagnóstico

Las funciones de `src/thesis/plotting.py` cubren cuatro bloques:

| Función | Paneles |
|---|---|
| `plot_hedge_diagnostics` | Beta walk-forward vs beta efectivo, alpha, cambios de beta. |
| `plot_signal_diagnostics` | Spread con entradas/salidas, z-score con posición, días de permanencia. |
| `plot_return_diagnostics` | Contribución de cada pata, retorno bruto/neto, notional y pesos, turnover y costes acumulados. |
| `plot_equity_diagnostics` | Equity neta y bruta frente a benchmarks (A, B y SPY), drawdown, Sharpe y volatilidad rolling. |
| `plot_backtest_dashboard` | Las seis anteriores condensadas en un único panel de seis subgráficos. |
| `plot_pair_window_summary` | Spread completo vs última ventana, hedge ratio walk-forward vs completo vs reciente, z-score de la ventana reciente. |

El panel del beta efectivo es la comprobación visual más importante: en
modo `on_entry` debe verse una función escalonada con un valor distinto
en cada operación.  Una línea plana con un único valor indica que se está
reutilizando el beta de la última entrada.

### Ejecución sin red

Si ya existen los CSV de caché en `data/`, el notebook no necesita
conexión.  Puedes ejecutarlo celda a celda con Jupytext o convertirlo
a `.ipynb`:

```bash
uv run jupytext --to notebook notebooks/pair_trading/PracticalPairTrading.py
uv run jupyter notebook notebooks/pair_trading/PracticalPairTrading.ipynb
```

### Limitaciones

- El backtest es académico: no incluye restricciones de apalancamiento,
  tamaño de lote, impacto de mercado ni costes de financiación de
  cortos.
- El test ADF y el OU in-sample no deben usarse para decidir entradas;
   sólo describen el par.
- La selección del par, los umbrales y las ventanas no deben
  optimizarse sobre el mismo periodo que se evalúa; se requiere una
  partición train/validation/test para una estimación honesta.
- Las comprobaciones automáticas validan la coherencia interna del
  código, pero no pueden detectar sesgos metodológicos como la
  selección del par o el ajuste de hiperparámetros sobre la muestra.

# SPY: caracterización, composición sectorial y libro de órdenes (LOB)

Este directorio reúne tres cuadernos sobre el **SPY**:

1. [`SPY_caracterizacion.py`](SPY_caracterizacion.py): **caracterización
   estadística** de los retornos diarios y de sus cuadrados (hechos
   estilizados: leptocurtosis, asimetría, autocorrelación, clustering de
   volatilidad);
2. [`SPY_sectores.py`](SPY_sectores.py): **composición sectorial** del
   mercado estadounidense en tres niveles (snapshot actual, reconstrucción
   histórica y entropía de Shannon);
3. [`OrderBook.py`](OrderBook.py): **estructura de liquidez** del libro de
   órdenes del SPY a partir de los ficheros LOBSTER (perfil de liquidez
   relativo y absoluto, snapshot del libro y precio medio promediado).

Todos los cuadernos son **descriptivos**, no predictivos: resumen el estado del
mercado, pero no anticipan su evolución.

---

## Composición sectorial del S&P 500 y del SPY

Este módulo estudia la **composición sectorial** del mercado estadounidense en
tres niveles:

1. **snapshot actual**: los pesos sectoriales que publica yfinance para el
   ETF `SPY`, en un gráfico de anillo (*donut*);
2. **reconstrucción histórica**: la composición sectorial del S&P 500 por
   capitalización de mercado, reconstruida a partir de la lista actual de
   constituyentes (Wikipedia), los precios (yfinance) y las acciones en
   circulación (yfinance);
3. **entropía de Shannon**: cómo se ha diversificado (o concentrado) la
   composición sectorial a lo largo del tiempo.

La pregunta central es: **¿cómo se ha repartido el peso entre sectores y cómo
ha evolucionado esa reparto?**

La respuesta es **descriptiva**, no predictiva: la entropía resume la
diversificación de la composición en cada fecha, pero no anticipa la
rotación sectorial.

---

## Estructura

```text
notebooks/spy/
├── README.md                  ← este archivo
├── SPY_caracterizacion.py     ← caracterización de retornos (jupytext py:percent)
├── SPY_sectores.py            ← orquestador por etapas (jupytext py:percent)
└── OrderBook.py               ← orquestador por etapas (jupytext py:percent)

src/thesis/
├── characterization.py        ← momentos, ACF, EWMA y bootstrap
├── sector_composition.py      ← datos, ponderación y entropía
└── orderbook.py               ← carga, validación y extracción del LOB
```

El cuaderno está escrito en formato jupytext `py:percent` y tiene **una celda
por sección del pipeline**.  Cada celda deja su resultado en una variable
(`spy_weights`, `snapshot`, `constituents`, `prices`, `shares`,
`sector_weights`, `entropy`), de modo que se puede cambiar un parámetro o una
figura y re-ejecutar **sólo** la celda afectada, sin volver a descargar los
datos.

El **código de dibujo de las figuras vive en el cuaderno**, escrito de forma
directa (sin funciones de apoyo): así se pueden retocar etiquetas, escalas y
colores sin tocar `src/thesis/sector_composition.py`.  **Todas las etiquetas
de las figuras están en español.**

El módulo `thesis.sector_composition` **no** se exporta en
`thesis/__init__.py` para no arrastrar `lxml`/`requests` en cada import del
paquete; se usa `from thesis.sector_composition import ...`.

---

## Pipeline

1. **Snapshot (donut)**: `fetch_spy_sector_weights` descarga los pesos
   sectoriales actuales de `SPY` (`yf.Ticker(...).funds_data`, en fracción) y
   `aggregate_small_sectors` agrupa en "Otros" los sectores por debajo del
   umbral (8%), traduce las etiquetas al español y las ordena de mayor a
   menor.  El anillo se dibuja en el cuaderno.
2. **Datos históricos** (única celda con red):
   `get_sp500_constituents` scrapea la lista de constituyentes del S&P 500
   de Wikipedia (columnas `Symbol`, `Security`, `Sector`; `BRK.B` → `BRK-B`);
   `download_price_history` baja el cierre ajustado de todos los tickers
   **por lotes** (100 por petición, 1 s de pausa entre lotes) y
   `get_shares_outstanding` consulta las acciones en circulación
   (best-effort: los tickers que fallan quedan con `NaN`).
   Los **precios** se cachean en un único CSV
   `data/sp500_prices_<start>_<end>.csv` y las **acciones** en
   `data/sp500_shares.csv`: re-ejecutar la celda no vuelve a bajar ~500
   tickers.  La caché de precios exige una `end_date` **explícita** (con
   `None` la ventana sería "hasta hoy" y no se puede congelar en el nombre
   del archivo); `refresh_prices=True` / `refresh_shares=True` fuerzan la
   re-descarga.  El remuestreo (`resample_freq`) se aplica **después** de
   cargar el frame, así que cambiarlo no obliga a volver a descargar.
3. **Pesos**: `build_sector_weights` calcula
   `capitalización(t) = precio(t) · acciones` por ticker, descarta los
   tickers sin datos, suma por sector y devuelve **porcentajes (0-100)** que
   suman 100 en cada fecha.  El frame se puede cachear en
   `data/sp500_sector_composition.csv`.
4. **Figura de área apilada al 100%**: los sectores se ordenan por peso medio
   y se anotan a la derecha los que superan el 2% en la última fecha.
5. **Entropía**: `sector_entropy` calcula `H = -Σ p·ln(p)` fecha a fecha
   (con el límite `p·ln(p) → 0` para `p = 0`, de modo que un sector con peso
   nulo no da `-inf`).  El cálculo es **puntual**: truncar la serie no altera
   las filas ya calculadas.

---

## API pública (`src/thesis/sector_composition.py`)

* ``GICS_SECTOR_ES``
  Traducción al español de los sectores GICS que usa yfinance.

* ``SP500_WIKI_URL``
  URL de la tabla de constituyentes del S&P 500 en Wikipedia.

* ``MINOR_SECTOR_THRESHOLD = 0.08``
  Umbral (fracción) por debajo del cual un sector se agrupa en "Otros".

* ``SECTOR_COMPOSITION_CACHE_FILENAME``
  Nombre del CSV de caché con los pesos sectoriales históricos.

* ``PRICE_CACHE_FILENAME_TMPL = "sp500_prices_{start}_{end}.csv"``
  Plantilla del nombre del CSV de caché con el frame de precios (la clave
  es la ventana ``start:end``; ``end`` debe ser explícito).

* ``SHARES_CACHE_FILENAME = "sp500_shares.csv"``
  Nombre del CSV de caché con las acciones en circulación.

* ``price_cache_path(cache_dir, start, end) -> Path``
  Ruta del CSV de caché de precios para la ventana ``start:end``.

* ``aggregate_small_sectors(weights, threshold=0.08, translation=GICS_SECTOR_ES) -> pd.Series``
  Agrupa en "Otros" los sectores por debajo del umbral, traduce las
  etiquetas y ordena el resultado de mayor a menor (conserva la suma).

* ``fetch_spy_sector_weights(ticker="SPY") -> pd.Series``
  Snapshot actual de los pesos sectoriales del ETF (en fracción).

* ``get_sp500_constituents(session=None) -> pd.DataFrame``
  Constituyentes del S&P 500 desde Wikipedia (`session` inyectable para
  tests).

* ``download_price_history(tickers, start, end, batch_size=100, sleep=1.0, cache_dir=None, refresh=False) -> pd.DataFrame``
  Cierre ajustado por lotes, frame ancho (fecha × ticker).  Con `cache_dir`
  **y** `end` explícito, el frame se cachea en un único CSV
  (`sp500_prices_<start>_<end>.csv`) y las siguientes llamadas no tocan la
  red; la caché se descarta si no cubre todos los tickers pedidos.

* ``get_shares_outstanding(tickers, cache_dir=None, refresh=False) -> pd.Series``
  Acciones en circulación actuales (best-effort; `NaN` si falla).  Con
  `cache_dir` el resultado se cachea en `sp500_shares.csv`.

* ``build_sector_weights(prices, shares, sector_map) -> pd.DataFrame``
  Pesos sectoriales en porcentaje (0-100) por fecha.  Falla rápido si
  `sector_map` no cubre los tickers con datos, si no hay datos utilizables o
  si los pesos salen negativos.

* ``sector_entropy(sector_weights) -> pd.Series``
  Entropía de Shannon (nats) por fecha.

Importar el módulo **no** descarga datos ni dibuja figuras: los efectos con
red, disco o pantalla viven dentro de funciones explícitas.

---

## Ejecución

El cuaderno se abre con Jupyter (el archivo `.py` es un cuaderno jupytext
`py:percent`) y se ejecuta **celda a celda**:

```bash
uv run jupyter notebook notebooks/spy/SPY_sectores.py
```

También se puede ejecutar como script, en cuyo caso las celdas corren en
orden:

```bash
uv run python notebooks/spy/SPY_sectores.py
```

### Celdas del pipeline

| Celda | Contenido | Resultado en el espacio de nombres |
| --- | --- | --- |
| Configuración | parámetros del experimento | `CONFIG` |
| Etapa 1: snapshot (donut) | `matplotlib` en línea | `spy_weights`, `snapshot`, `donut_fig` |
| Etapa 2: datos históricos | constituyentes, precios y acciones | `constituents`, `tickers`, `sector_map`, `prices`, `shares` |
| Etapa 3: pesos sectoriales | ponderación por capitalización | `sector_weights` |
| Etapa 4: área apilada | `matplotlib` en línea | `area_fig` |
| Etapa 5: entropía | `sector_entropy` + `matplotlib` en línea | `entropy`, `entropy_fig` |

Para iterar: cambia `CONFIG` (o la figura) y re-ejecuta **sólo** la celda
afectada.  Por ejemplo, cambiar `CONFIG["minor_threshold"]` y re-ejecutar la
celda 1 recalcula el anillo; cambiar `CONFIG["resample_freq"]` y re-ejecutar
las celdas 2 y 3 rehace los snapshots (sin volver a descargar: los precios
vienen de la caché); cambiar `CONFIG["use_cache"]` a `True` reutiliza
`data/sp500_sector_composition.csv` en lugar de recalcular los pesos.

### Cachés

| Caché | Dónde | Cuándo se usa |
|---|---|---|
| Precios | `data/sp500_prices_<start>_<end>.csv` | `CONFIG["use_price_cache"]` y `end_date` explícita |
| Acciones | `data/sp500_shares.csv` | `CONFIG["use_shares_cache"]` |
| Pesos sectoriales | `data/sp500_sector_composition.csv` | `CONFIG["use_cache"]` |

Para forzar la re-descarga: `CONFIG["refresh_prices"] = True` (precios) o
`CONFIG["refresh_shares"] = True` (acciones).  La caché de precios se
descarta automáticamente si el universo de constituyentes ha cambiado
(algún ticker pedido no está en el CSV).

Tests (sintéticos, sin red):

```bash
uv run pytest tests/test_sector_composition.py -v
```

Los tests marcados con `integration` (scraping de Wikipedia y pesos de `SPY`)
requieren red y se omiten por defecto:

```bash
uv run pytest -m integration -v
```

---

## Artefactos

| Artefacto | Descripción |
|---|---|
| `data/sp500_prices_<start>_<end>.csv` | Frame de precios histórico (caché, ignorado por Git) |
| `data/sp500_shares.csv` | Acciones en circulación (caché, ignorado por Git) |
| `data/sp500_sector_composition.csv` | Pesos sectoriales históricos (caché, ignorado por Git) |
| `outputs/spy_sector_weights.png` | Anillo con los pesos sectoriales actuales de SPY |
| `outputs/sp500_sector_composition.png` | Área apilada al 100% de la composición histórica |
| `outputs/sp500_sector_entropy.png` | Entropía de Shannon de la composición a lo largo del tiempo |

Las figuras se guardan en `outputs/` con una ruta **anclada a la raíz del
repositorio**, no al directorio de trabajo.

---

## Interpretación y precauciones

- **Sesgo de supervivencia**: la reconstrucción usa la lista de
  constituyentes **actual**, así que las empresas que salieron del índice
  (bancruptas, compradas o deslistadas) están ausentes de toda la
  historia.  No es un bug a corregir sino una limitación metodológica.
- **Clasificación sectorial retroactiva**: se aplican las etiquetas GICS
  **actuales** a todo el pasado.  Por ejemplo, "Servicios de Comunicación"
  no existía como sector GICS antes de septiembre de 2018 (absorbió nombres
  de Tecnología y Consumo Cíclico), y el sector "Real Estate" se separó de
  Servicios Financieros en 2016.
- **Acciones en circulación constantes**: se usan las acciones **actuales**
  para todo el histórico (no se modelan recompras, emisiones ni splits), de
  modo que las capitalizaciones de los años tempranos son aproximadas.
- **Precios ajustados**: el cierre es ajustado (`auto_adjust=True`), lo que
  implica que los pesos por sector incorporan losplits y dividendos de
  forma implícita en la serie de precios.
- Para una versión rigurosa (point-in-time) haría falta una base de datos
  de pago (Compustat, S&P Capital IQ, Bloomberg).
- La **entropía** es una medida de diversificación del weights, no una señal:
  una entropía alta significa una composición repartida entre sectores; una
  baja, una composición concentrada.
- Los sectores del ETF (`SPY`) y los del índice (S&P 500) **no son las mismas
  series**: el primero es la composición *publicada* por el fondo (pesos
  oficiales, con las posiciones reales del ETF) y el segundo es una
  **reconstrucción** por capitalización de mercado a partir de los
  constituyentes.  Las dos series no tienen por qué coincidir.

---

## OrderBook (LOB)

### Pregunta central

**¿cómo se distribuye la liquidez alrededor del precio medio del SPY y cómo
evoluciona el precio medio del libro?**

La respuesta es **descriptiva**: el perfil de liquidez (volumen pasivo
agregado en bins de distancia al precio medio) y el precio medio del libro
describen el estado del mercado en la ventana analizada; no anticipan nada
y no se usan como señales.

### Pipeline

El cuaderno `OrderBook.py` tiene **una celda por etapa** y toda la lógica
reutilizable vive en `src/thesis/orderbook.py` (carga, validación y
extracción; **sin** `matplotlib`):

1. **Configuración**: rutas de los ficheros LOBSTER (`THESIS_LOB_DIR` o
   `config.DATA_DIR / "OrderBook"`), ventana temporal, número de niveles y
   anchos de bin.
2. **Distribución relativa** (`extract_relative_lob_distribution`): perfil de
   volumen pasivo en bins de distancia al precio medio, $\Delta P = P -
   P_{mid}$, promediado por snapshot.
3. **Distribución absoluta** (`extract_absolute_lob_distribution`): el mismo
   perfil en bins de **precio**; los bins se re-agregan a un ancho mayor
   (`aggregate_volume_bins`) para la figura.
4. **Snapshot único** (`extract_single_snapshot`): estado del libro en el
   primer instante con `Time >= target_time`.
5. **Precio medio promediado** (`extract_averaged_mid_price`): precio medio
   del libro en buckets de `interval` segundos.

El **código de dibujo de las figuras vive en el cuaderno** (convención del
repo): cada etapa deja sus resultados en el espacio de nombres
(`bids_binned`, `asks_binned`, `max_range`, `absolute_bids`, `price_range`,
`snap_bids`, `snap_asks`, `actual_time`, `df_avg`) y su figura (`relative_fig`,
`absolute_fig`, `snapshot_fig`, `mid_fig`).

### API pública (`src/thesis/orderbook.py`)

* ``PRICE_SCALE = 10_000.0``
  Factor de escala de los precios del *orderbook* LOBSTER (ticks → USD).

* ``DUMMY_PRICE = 9_999_999_999`` / ``DUMMY_NEG_PRICE = -9_999_999_999``
  Centinelas de LOBSTER para "nivel vacío".  Se filtran **antes** de dividir
  por `PRICE_SCALE` (comparar tras dividir introduce errores de coma
  flotante).

* ``MESSAGE_COLUMNS = ("Time", "Type", "OrderID", "Size", "Price", "Direction")``
  Columnas del fichero de mensajes; ``Type`` es el ``Event`` de LOBSTER
  (1=nueva orden, 2=modificación, 3=cancelación, 4=ejecución, 5=ejecución
  oculta, 7=halt).

* ``lob_data_dir() -> Path``
  Directorio de los ficheros LOBSTER: variable de entorno
  ``THESIS_LOB_DIR`` o ``config.DATA_DIR / "OrderBook"``.

* ``lob_file_paths(symbol, date, start, end, levels, directory=None) -> (Path, Path)``
  Rutas de los ficheros de mensajes y de orderbook siguiendo la convención
  de nombres de LOBSTER.

* ``load_message_times(message_path) -> pd.Series``
  Columna ``Time`` (se lee sólo esa columna: `usecols=[0]`).

* ``extract_relative_lob_distribution(message_path, orderbook_path,
  start_time, end_time, levels, bin_width=0.01) -> (bids, asks, max_range)``
  Perfil de liquidez relativo al precio medio.  ``max_range`` es la
  distancia máxima (en USD) al precio medio alcanzada por el libro.

* ``extract_absolute_lob_distribution(...) -> (bids, asks, (p_min, p_max))``
  Perfil de liquidez en precio absoluto; el rango se redondea a múltiplos de
  ``bin_width``.

* ``extract_single_snapshot(message_path, orderbook_path, target_time,
  levels) -> (df_bids, df_asks, actual_time)``
  Snapshot en el primer instante con ``Time >= target_time``.

* ``extract_averaged_mid_price(message_path, orderbook_path, start_time=None,
  end_time=None, interval=1.0) -> pd.DataFrame``
  Precio medio del libro (USD) en buckets de ``interval`` segundos; columnas
  ``TimeBin`` y ``MidPrice``.

* ``aggregate_volume_bins(price, volume, bin_width) -> pd.DataFrame``
  Helper compartido: agrupa precios en bins de ancho ``bin_width`` y suma los
  volúmenes (``floor(round(precio / bin_width, 6))`` para evitar los errores
  de coma flotante al truncar).  Lo usan la re-agregación de la distribución
  absoluta y la agregación del snapshot.

Importar el módulo **no** lee ficheros ni dibuja figuras: los efectos con
disco o pantalla viven dentro de funciones explícitas.  El módulo no se
exporta en `thesis/__init__.py`; se usa
`from thesis.orderbook import ...`.

### Unidades

| Magnitud | Unidad | Nota |
|---|---|---|
| Precio del *orderbook* | ticks (entero) | USD = ticks / 10 000 |
| Precio del *mensaje* | USD (decimal) | no se usa en la extracción |
| `Time` | segundos desde medianoche | puede superar 86 400 en *after-hours* |
| Volumen | acciones | suma dentro del bin |
| `Volume` de los bins | acciones por snapshot | normalizado por el **nº de snapshots válidos**, no por la duración de la ventana |
| `BinMid` | USD | centro del bin |

### Ejecución

Los ficheros LOBSTER **no** están en el repositorio: hay que apuntar
`THESIS_LOB_DIR` al directorio que los contiene (por defecto
`data/OrderBook`, con la estructura `<raíz>/<TICKER>/<ficheros>.csv`):

```bash
export THESIS_LOB_DIR=/ruta/a/OrderBook
```

El cuaderno se abre con Jupyter (el archivo `.py` es un cuaderno jupytext
`py:percent`) y se ejecuta **celda a celda**:

```bash
uv run jupyter notebook notebooks/spy/OrderBook.py
```

También se puede ejecutar como script, en cuyo caso las celdas corren en
orden:

```bash
uv run python notebooks/spy/OrderBook.py
```

| Celda | Contenido | Resultado en el espacio de nombres |
| --- | --- | --- |
| Configuración | rutas y parámetros | `CONFIG`, `MESSAGE_PATH`, `ORDERBOOK_PATH` |
| Etapa 1: distribución relativa | extracción + `matplotlib` | `bids_binned`, `asks_binned`, `max_range`, `relative_fig` |
| Etapa 2: distribución absoluta | extracción + `matplotlib` | `absolute_bids`, `absolute_asks`, `price_range`, `absolute_fig` |
| Etapa 3: snapshot único | extracción + `matplotlib` | `snap_bids`, `snap_asks`, `actual_time`, `snapshot_fig` |
| Etapa 4: precio medio promediado | extracción + `matplotlib` | `df_avg`, `mid_fig` |

Tests (sintéticos, sin red):

```bash
uv run pytest tests/test_orderbook.py -v
```

El test marcado con `integration` hace un *spot-check* sobre los ficheros
LOBSTER reales (se omite por defecto si no están disponibles):

```bash
uv run pytest -m integration -v
```

### Artefactos

| Artefacto | Descripción |
|---|---|
| `outputs/spy_lob_relative_distribution.png` | Perfil de liquidez relativo al precio medio |
| `outputs/spy_lob_absolute_distribution.png` | Perfil de liquidez en precio absoluto |
| `outputs/spy_lob_snapshot.png` | Estado del libro en un instante |
| `outputs/spy_lob_mid_price.png` | Precio medio del libro promediado en el tiempo |

### Interpretación y precauciones

- **Descriptivo, no predictivo**: ninguna de las extracciones es una señal.
  No hay train/test que se pueda contaminar.
- **El snapshot es retrospectivo**: `extract_single_snapshot` elige la
  primera fila con `Time >= target_time`, es decir, un instante *posterior*
  al objetivo.  No debe usarse como variable de entrada de un modelo.
- **Promedio por snapshots, no por tiempo**: el volumen de cada bin se
  normaliza por el número de snapshots válidos de la ventana.  Una ventana
  con poco tráfico tiene menos snapshots y el mismo volumen se reparte
  entre menos bins.
- **Centinelas**: los niveles vacíos (`±9999999999`) se descartan antes de
  escalar los precios; un snapshot con el *top of book* vacío se descarta
  entero.  La rejilla de bins se calcula **después** de filtrarlos (si no,
  un centinela contaminaría el rango y la rejilla entera).
- **Alineación fila a fila**: los ficheros de mensajes y de orderbook deben
  compartir orden y número de filas; el módulo lo comprueba y falla rápido
  si no es así.
- **Atípicos en los niveles profundos**: el fichero real de SPY contiene
  algunos niveles muy profundos con precios lejanos (1,04 – 172,38 USD
  frente a un precio en torno a 135 USD).  Las figuras acotan el eje X por
  defecto (`CONFIG["relative_xlim"]` y `CONFIG["absolute_xlim"]`); con
  `None` se ve el rango completo, atípicos incluidos.
- **Memoria**: la extracción de distribuciones lee el fichero de orderbook
  completo (con 50 niveles son ~1,4 GB por sesión).  El snapshot usa
  `skiprows`/`nrows` y el precio medio sólo lee las columnas 0 y 2.

---

## Caracterización estadística del SPY (retornos y retornos²)

### Pregunta central

**¿qué propiedades estadísticas debe respetar cualquier modelo que se ajuste
después a esta serie?**

La respuesta es **descriptiva**: el cuaderno establece hechos estilizados
(crecimiento aproximadamente exponencial, leptocurtosis, asimetría negativa,
ausencia de autocorrelación lineal, persistencia de la autocorrelación de los
retornos al cuadrado y clustering de volatilidad), no señales ni predicciones.

### Pipeline

El cuaderno `SPY_caracterizacion.py` tiene **una celda por etapa** y todo el
cómputo reutilizable vive en `src/thesis/characterization.py` (**sin**
`matplotlib`):

| Celda | Contenido | Resultado en el espacio de nombres |
| --- | --- | --- |
| 0 | magics `%autoreload 2` (recarga en caliente de `src/`) | `_INTERACTIVE` |
| 1 | imports y `CONFIG` | `CONFIG`, `OUTPUT_DIR` |
| 2 | datos (cierre ajustado cacheado + retornos) | `price`, `log_ret`, `simple_ret`, `r2` |
| 3 | trayectoria del precio (lineal y log) | `price_smooth`, `fig` |
| 4 | densidad de retornos (normal y t de Student) | `params`, `moments` |
| 5 | momentos con *moving block bootstrap* | `k_hat`, `k_ci`, `ann_hat`, `ann_ci` |
| 6 | ACF de retornos vs. retornos² (bandas por *bootstrap* i.i.d.) | `acf_r`, `ci_r`, `acf_sq`, `ci_sq` |
| 7 | histograma de los retornos al cuadrado | `r_2`, `r_2_log` |
| 8 | regímenes de volatilidad EWMA + precio semilog | `volatility`, `high_vol_mask` |
| 9 | bandas incondicionales ±kσ | `excursions` |
| 10 | tabla resumen | `summary` |

### Fórmulas

* **EWMA (convención RiskMetrics)**: $\sigma_t^2 = \lambda\,\sigma_{t-1}^2 +
  (1-\lambda)\,x_{t-1}^2$ con $\lambda = 0.94$.  En la fecha $t$ se usa
  **sólo** $x_{t-1}^2$: la estimación es **causal** (corromper la cola no
  cambia el pasado).  `sigma_t` se annualiza por $\sqrt{252}$.
* **Bandas de la ACF por *bootstrap* i.i.d.**: la ACF muestral se recalcula
  sobre cada réplica del bootstrap (remuestreo con reemplazo de observaciones
  individuales) y la banda percentil $\pm h_k$ es la envolvente del intervalo
  percentil alrededor de cero.  Como el remuestreo **destruye la dependencia
  temporal**, la banda es la distribución *nula* del estimador (equivalente
  empírico de la banda de Bartlett $\pm z/\sqrt{T}$); el lag 0 no lleva banda.
* **Freedman–Diaconis**: $h = 2\,\mathrm{IQR}\,n^{-1/3}$; robusto frente a los
  outliers (a diferencia de Scott, que usa el desvío estándar, inflado por las
  propias colas).
* **Moving block bootstrap**: remuestrea bloques contiguos de longitud $b$
  (10 sesiones) para preservar la dependencia temporal; la semilla es
  **explícita** (no se toca el estado global de numpy).  Se usa **solo** en
  los momentos (celda 5), no en las bandas de la ACF.

### API pública (`src/thesis/characterization.py`)

* ``describe_returns(x, label) -> pd.Series``
  Momentos muestrales (media, desvío, mediana, extremos, asimetría, exceso de
  curtosis) y test de Jarque–Bera.
* ``freedman_diaconis_bins(x) -> int``
  Número de bines según Freedman–Diaconis (mínimo 1).
* ``fit_return_distribution(x) -> dict``
  Ajuste por máxima verosimilitud de la normal (`mu`, `sigma`) y de la t de
  Student (`df`, `loc`, `scale`).
* ``excess_kurtosis(x) -> np.ndarray`` y ``annualized_stats(x, trading_days=252) -> np.ndarray``
  Estadísticos usados como estimador en el bootstrap.
* ``block_bootstrap_ci(x, func, block_size=10, reps=3000, alpha=0.05, seed=42) -> (est, ci)``
  Estimación puntual e intervalo percentil por *moving block bootstrap*; el CI
  tiene forma `(2, k)` (fila 0 = límite inferior, fila 1 = superior).
* ``bootstrap_ci(x, func, reps=3000, alpha=0.05, seed=42) -> (est, ci)``
  Estimación puntual e intervalo percentil por bootstrap **i.i.d.**
  (remuestreo con reemplazo de observaciones individuales): no preserva la
  dependencia temporal.
* ``bootstrap_acf(x, max_lags=50, reps=3000, alpha=0.05, seed=42) -> (acf, half_width)``
  ACF muestral con banda simétrica ``± half_width`` estimada por bootstrap
  i.i.d.; el lag 0 no lleva banda.  La banda es la distribución nula del
  estimador (no se ensancha con la dependencia de la serie).
* ``heteroskedastic_robust_acf(returns, max_lags=50, alpha=0.05) -> (acf, ci_bound)``
  ACF muestral con bandas robustas a heterocedasticidad (Diebold, 1986).
* ``sample_acf(x, max_lags=50) -> np.ndarray``
  ACF muestral simple (lag 0 incluido).
* ``compute_ewma_volatility(log_ret, lambda_ewma=0.94, trading_days=252) -> pd.Series``
  Volatilidad EWMA **causal** y anualizada.
* ``high_volatility_mask(volatility, quantile=0.90, index=None) -> pd.Series``
  Marca los valores por encima de un cuantil **global** (descriptivo, no
  señal).
* ``unconditional_band_excursions(log_ret, ks=(1, 2, 3, 5)) -> pd.DataFrame``
  Fracción de excursiones fuera de ±kσ frente a la predicción gaussiana.
* ``build_summary_table(...) -> pd.DataFrame``
  Tabla resumen de la caracterización empírica.

Importar el módulo **no** descarga datos ni dibuja figuras: el módulo no
exporta `matplotlib` ni `yfinance`, y no se exporta en `thesis/__init__.py`;
se usa `from thesis.characterization import ...`.

### Cómo retocar una figura

El **código de dibujo vive en el cuaderno**, escrito de forma directa (sin
funciones de apoyo): así se pueden retocar colores, bines, spans, cuantiles,
`xlim` o tamaño sin tocar `src/`.

1. editar la **celda de la figura** y re-ejecutarla **sólo** esa celda (no
   hace falta reiniciar el kernel);
2. editar `CONFIG` (celda 1) y re-ejecutar la celda de la figura afectada;
3. editar `src/thesis/characterization.py` y re-ejecutar la celda: el
   `%autoreload 2` de la celda 0 recarga el módulo **en caliente**.

Las figuras se guardan en `outputs/` con una ruta **anclada a la raíz del
repositorio**, no al directorio de trabajo.

### Artefactos

| Artefacto | Descripción |
|---|---|
| `outputs/spy_price_trajectory.png` | Precio ajustado en escala lineal y logarítmica |
| `outputs/spy_return_density.png` | Densidad de retornos con ajustes normal y t de Student |
| `outputs/spy_acf_comparison.png` | ACF de retornos vs. retornos² (bandas por *bootstrap* i.i.d.) |
| `outputs/spy_sq_returns_hist.png` | Histograma de $x_t^2$ y de $\log(x_t^2)$ |
| `outputs/spy_vol_regimes.png` | Volatilidad EWMA y regímenes de alta volatilidad |
| `outputs/spy_uncond_bands.png` | Retornos diarios con bandas incondicionales ±kσ |

Tests (sintéticos, sin red):

```bash
uv run pytest tests/test_characterization.py -v
```

### Interpretación y precauciones

- **El percentil 90 de volatilidad es descriptivo**: se calcula sobre toda la
  muestra, luego **no** es una señal libre de look-ahead.
- **La EWMA es causal**: $\sigma_t$ usa $x_{t-1}$; usarla como variable de
  entrada de un modelo no introduce look-ahead.
- **El bootstrap por bloques preserva la dependencia** (celda 5, momentos): el
  i.i.d. destruiría el clustering de volatilidad y daría intervalos
  artificialmente angostos.
- **Las bandas de la ACF son bandas nulas** (celda 6, bootstrap i.i.d.): al
  destruir la dependencia temporal quedan equivalentes a la banda de Bartlett
  $\pm z/\sqrt{T}$, que **subestima** la varianza de $\hat\rho_k$ en series
  heterocedásticas (los retornos) y persistentes (los retornos al cuadrado).
  Los lags "significativos" de la celda 6 se leen como evidencia *contra la
  independencia*, no como una medida de la magnitud de la persistencia.
- **El ajuste t de Student** con pocos grados de libertad indica colas cuyos
  momentos de orden alto **no existen**: no se deben estimar momentos de
  orden 4 como si fueran estables.
- La caracterización es **descriptiva**: sirve para fijar las restricciones
  del modelo, no para anticipar el precio.

---

## Referencias

* Wikipedia, *List of S&P 500 companies*:
  <https://en.wikipedia.org/wiki/List_of_S%26P_500_companies>
* yfinance: pesos sectoriales del ETF (`funds_data`), precios históricos y
  acciones en circulación (`fast_info`).
* `pandas.read_html` (parser `lxml`) para el scraping de Wikipedia.
* LOBSTER (Universidad de Humboldt, Berlin): ficheros de mensajes y de
  *orderbook* en formato CSV, precios en ticks y centinelas `±9999999999`
  para los niveles vacíos.

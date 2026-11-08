# Análisis de BTC: power-law y ciclos de mercado

Este módulo estudia el precio del Bitcoin con dos herramientas
descriptivas: una **regresión cuantílica power-law** sobre el tiempo desde
la génesis y una **detección de ciclos** (picos y valles macro) sobre el
precio logarítmico.

La pregunta central es: **¿el precio de BTC sigue una ley de potencia
determinista, y qué tan regulares son sus ciclos macro?**

La respuesta es descriptiva, no predictiva: los cuantiles dan una banda de
precios "de manual" para cada edad del activo y `find_peaks` etiqueta los
máximos y mínimos una vez que ya han ocurrido.

---

## Estructura

```text
notebooks/btc/
├── README.md     ← este archivo
└── BTC.py        ← orquestador por etapas (jupytext py:percent)

src/thesis/
└── btc.py        ← descarga, validación, power-law y ciclos
```

El cuaderno está escrito en formato jupytext `py:percent` y tiene **una celda
por sección del pipeline** (datos, power-law, figura, ciclos y curva
suavizada).  Cada celda deja su resultado en una variable (`btc_df`,
`reg_df`, `fit`, `figures`, `peaks_df`, `bottoms_df`, `cycle_df`,
`peaks_cand`, `peak_sel`, `cycle_fig`), de modo que se puede cambiar un
parámetro o una figura y re-ejecutar **sólo** la celda afectada, sin volver a
descargar los datos.

El **código de dibujo de las figuras vive en el cuaderno**, escrito de forma
directa (sin funciones de apoyo): la celda "Etapa 3: figura" dibuja los
tres gráficos con `matplotlib` y la celda "Etapa 5: curva suavizada y picos"
dibuja el precio dettrendizado con su media móvil y los picos marcados; ambas
se pueden retocar sin tocar `src/thesis/btc.py`.  **Todas las etiquetas de
las figuras están en español.**

El módulo `thesis.btc` **no** se exporta en `thesis/__init__.py` para no
arrastrar `statsmodels` en cada import del paquete; se usa
`from thesis.btc import ...`.

---

## Pipeline

1. **Datos**: descarga del OHLCV diario de Bitstamp (`btcusd`) con paginación
   de 1 000 velas por petición y pausa de 0,1 s para respetar el límite de
   la API. La caché es `data/btc_daily_ohlcv.csv` (anclada a
   `thesis.config.DATA_DIR`, no al directorio de trabajo).
2. **Validación**: columnas canónicas, índice `DatetimeIndex` monótono,
   sin `NaN` y con precios estrictamente positivos.
3. **Power-law**: `days` desde la génesis (`2009-01-03`), `log_days` y
   `log_price`; se descartan los días no positivos y los precios no
   positivos.
4. **Regresión cuantílica**: `log_price ~ log_days` ajustada por
   `statsmodels` en los cuantiles 0,10 / 0,50 / 0,90.
5. **Figura**: tres gráficos independientes —evolución temporal en escala
   lineal, eje temporal logarítmico y power-law log-log (días desde la
   génesis)— con las rectas de los cuantiles y las anotaciones de halvings,
   picos y valles. El código de dibujo está **en el cuaderno**, no en
   `src/thesis/btc.py`.
6. **Ciclos**: `scipy.signal.find_peaks` sobre `log10(close)` con distancia
   mínima de 300 muestras y prominencia mínima de 0,3; el pico de Mt. Gox
   (2011-06-08), anterior a los datos de Bitstamp, se inyecta a mano.
7. **Curva suavizada y picos**: se resta la banda del cuantil 0,10 al precio
   (precio dettrendizado), se aplica un **promedio móvil centrado** de
   `CONFIG["peak_selection"]["smooth_window"]` días a esa serie, se listan
   los máximos locales candidatos con un `id` numerado (`P1`, `P2`, ...) y se
   marcan en la figura los ids indicados en
   `CONFIG["peak_selection"]["selected_peaks"]`, **anclados al valor real de
   la curva suavizada**.

La transformación de `days` es **causal**: depende sólo de la fecha de cada
fila, de modo que truncar la serie no altera las filas ya calculadas.

---

## Fórmula del power-law

El precio sigue una ley de potencia en el tiempo desde la génesis:

$$\log_{10}(\text{price}) = \alpha + \beta \cdot \log_{10}(\text{days})$$

- $\alpha$ es el intercepto (precio de referencia a un día de la génesis);
- $\beta$ es el exponente: la elasticity del precio al tiempo transcurrido.

Los cuantiles $\tau \in \{0{,}10, 0{,}50, 0{,}90\}$ ajustan rectas paralelas
en espacio log-log que delimitan la banda de precios "de manual" para cada
edad del activo: el cuantil 0,10 es el escenario pesimista y el 0,90 el
optimista.

---

## API pública (`src/thesis/btc.py`)

* ``BTC_OHLCV_COLUMNS``
  Columnas canónicas: ``("open", "close", "high", "low", "volume")``.

* ``fetch_btc_ohlcv(symbol="btcusd", cache_dir=DATA_DIR, refresh=False, session=None) -> pd.DataFrame``
  Descarga (o lee de caché) el OHLCV diario y lo valida. ``session`` permite
  inyectar una sesión HTTP (tests).

* ``validate_btc_ohlcv(df) -> pd.DataFrame``
  Falla rápido ante columnas ausentes, `NaN`, precios no positivos o índice
  no monótono.

* ``prepare_power_law_frame(df, genesis="2009-01-03") -> pd.DataFrame``
  Añade ``days``, ``log_days`` y ``log_price``.

* ``fit_power_law_quantiles(df, quantiles=(0.10, 0.50, 0.90)) -> dict``
  Devuelve ``{cuantil: {"params": ..., "prediction": ...}}``.

* ``plot_btc_quantreg(df, halvings=None, peaks=None, bottoms=None, ax=None, quantiles=BTC_QUANTILES) -> Figure``
  Figura de tres gráficos con anotaciones parametrizables. **El cuaderno no
  la usa**: la figura del cuaderno se dibuja con código propio, escrito en
  la celda "Etapa 3: figura".

* ``detect_market_cycles(df, distance=300, prominence=0.3) -> (DataFrame, DataFrame)``
  Picos y valles macro con las columnas ``Days_Since_Last_Peak`` y
  ``Days_Since_Last_Bottom``.

* ``detrend_power_law_quantile(df, fit=None, quantile=0.10) -> pd.DataFrame``
  Resta la banda del cuantil pedido al precio y devuelve ``band``,
  ``log_band``, ``resid_log`` (``log_price - log_band``) y ``ratio``
  (``close / band``).

* ``find_local_extrema(series, distance=200, prominence=0.05) -> (DataFrame, DataFrame)``
  Extremos locales candidatos (valles y picos) con las columnas ``id``
  (``B1``, ``P2``, ...), ``value`` y ``prominence``. Los ``NaN`` (por
  ejemplo, el comienzo de la serie suavizada) se descartan antes de la
  detección.

* ``smooth_cycle_residual(resid, window=30, min_periods=None, centered=False) -> pd.Series``
  Promedio móvil del residuo logarítmico. Por defecto es **trailing** (causal):
  la ventana debe estar completa, así que los primeros ``window - 1`` valores
  quedan en ``NaN``. Con ``centered=True`` la ventana se centra en cada fecha
  (usa datos posteriores) y los ``NaN`` se concentran en los bordes (~``window // 2``).
  Es la serie sobre la que el cuaderno detecta los máximos locales.

* ``select_extrema(candidates, selection) -> pd.DataFrame``
  Resuelve la selección del autor: acepta ids (``"B3"``), fechas ISO
  (``"2015-01-14"``) o el número de orden de la tabla impresa.

* ``idealized_cycle(resid, bottoms, peaks) -> pd.Series``
  Poligonal del ciclo idealizado: los valles quedan en 0 y los picos a la
  amplitud de su ciclo; ``NaN`` fuera del rango cubierto.  **El cuaderno ya
  no la usa**: la etapa 5 se quedó en la curva suavizada y sus picos.

---

## Ejecución

El cuaderno se abre con Jupyter (el archivo `.py` es un cuaderno jupytext
`py:percent`) y se ejecuta **celda a celda**:

```bash
uv run jupyter notebook notebooks/btc/BTC.py
```

También se puede ejecutar como script, en cuyo caso las celdas corren en
orden:

```bash
uv run python notebooks/btc/BTC.py
```

### Celdas del pipeline

| Celda | Función | Resultado en el espacio de nombres |
| --- | --- | --- |
| Configuración | — | `CONFIG` (símbolo, caché, cuantiles, anotaciones, ciclos, picos) |
| Etapa 1: datos | `load_ohlcv()` | `btc_df` |
| Etapa 2: power-law | `fit_power_law()` | `reg_df`, `fit` |
| Etapa 3: figura | código de `matplotlib` en línea | `figures` |
| Etapa 4: ciclos | `find_cycles()` | `peaks_df`, `bottoms_df` |
| Etapa 5: curva suavizada y picos | código de `matplotlib` en línea | `cycle_df`, `peaks_cand`, `peak_sel`, `cycle_fig` |

Para iterar: cambia `CONFIG` (o la figura) y re-ejecuta **sólo** la celda
afectada.  Por ejemplo, cambiar `CONFIG["quantiles"]` y re-ejecutar las
celdas 2 y 3 actualiza los exponentes y las rectas sin volver a descargar
los datos; cambiar `CONFIG["distance"]` y re-ejecutar la celda 4 recalcula
los ciclos macro; cambiar `CONFIG["peak_selection"]["smooth_window"]` y
re-ejecutar la celda 5 vuelve a suavizar el precio dettrendizado y recalcula
los máximos detectados; cambiar
`CONFIG["peak_selection"]["selected_peaks"]` y re-ejecutar la celda 5
cambia los picos marcados en la figura.

Tests (sintéticos, sin red):

```bash
uv run pytest tests/test_btc.py -v
```

---

## Artefactos

| Artefacto | Descripción |
|---|---|
| `data/btc_daily_ohlcv.csv` | OHLCV diario en caché (ignorado por Git) |
| Figura del power-law | Tres gráficos (lineal, temporal-log y log-log) con los cuantiles y las anotaciones |
| Figura de la curva suavizada | Precio log-log con la banda q10 y precio dettrendizado (con su media móvil) con los máximos detectados y los picos destacados marcados sobre la curva |

---

## Curva suavizada y picos

La etapa 5 responde a una pregunta concreta: **¿qué pasa si el precio se
expresa respecto a su propio "suelo" estadístico?**

1. Se ajusta la regresión cuantílica del 0,10 (la banda inferior del
   power-law) sobre `log10(precio)` frente a `log10(días)`.
2. Se resta esa banda: el **precio dettrendizado**
   (`resid_log = log10(precio) − log10(banda)`) deja los **mínimos locales
   cerca de cero** — los valles que tocan el suelo del power-law se
   materializan alrededor de 0, mientras que los "mínimos" que no lo tocan
   quedan claramente por encima.
3. Se aplica un **promedio móvil centrado** de
   `CONFIG["peak_selection"]["smooth_window"]` días (200 por defecto) al
   precio dettrendizado: `resid_ma = media_móvil(resid_log)`.  La ventana
   usa datos posteriores (es descriptiva, no causal) y con
   `min_periods = window // 2` la serie suavizada cubre todo el histórico.
4. Se listan los **máximos locales** detectados **sobre la serie
   suavizada** con `find_peaks` (distancia 200 días, prominencia 0,05 en
   log10) y un `id` numerado.  Suavizar evita que el ruido diario genere
   candidatos espurios.
5. Los ids a destacar se eligen en
   `CONFIG["peak_selection"]["selected_peaks"]` (por defecto `P1`, `P2`,
   `P4` y `P5`): es una decisión editorial, no un detector.
6. En la figura, cada pico se marca **en su valor real** de la curva
   suavizada (`resid_ma` en la fecha del pico), de modo que el marker cae
   exactamente sobre la media móvil.

La tabla de máximos detectados se imprime en cada ejecución, de modo que los
ids son estables mientras no se cambien `distance` o `prominence`.

---

## Interpretación y precauciones

- **Las fechas de picos, valles y halvings son anotaciones descriptivas del
  autor**, no señales. Están parametrizadas (``BTC_HALVINGS``, ``BTC_PEAKS``,
  ``BTC_BOTTOMS``) e incluyen fechas futuras que sólo sirven como líneas de
  referencia del gráfico.
- **`find_peaks` es retrospectivo**: detecta máximos y mínimos *después* de
  que ocurren. No es una estrategia de trading y no debe evaluarse como
  señal en tiempo real.
- La separación entre picos y valles depende de `distance` y `prominence`;
  cambiarlos cambia el número de ciclos detectados y sus distancias medias.
- **La media móvil es centrada y descriptiva**: la ventana usa datos
  posteriores, así que los máximos detectados coinciden con los giros reales
  del precio, pero **no es una señal causal**; con
  `min_periods = window // 2` los bordes de la serie usan ventanas
  parciales.
- Bitstamp empieza a publicar en agosto de 2011: los picos anteriores
  (incluido Mt. Gox) no provienen de la API y se inyectan a mano.
- El exponente $\beta$ de la regresión cuantílica es una descripción del
  histórico, no una estimación fuera de muestra.
- **La banda q10 se ajusta sobre toda la serie**: es un ajuste *in-sample*,
  así que el precio dettrendizado de las fechas tempranas usa información
  posterior. Es una herramienta descriptiva, no un indicador causal.
- **La elección de picos es manual y subjetiva**: los ids marcados
  (`selected_peaks`) son una decisión editorial del autor, no un ciclo
  detectado automáticamente. Un lector puede destacar otros ids de la tabla
  de máximos.

---

## Referencias

* Bitstamp API: <https://www.bitstamp.net/api/>
* `statsmodels.formula.api.quantreg` para regresión cuantílica.
* `scipy.signal.find_peaks` para la detección de extremos.

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
└── btc.py        ← descarga, validación, power-law, ciclos y figura
```

El cuaderno está escrito en formato jupytext `py:percent` y tiene **una celda
por sección del pipeline** (datos, power-law, figura, ciclos).  Cada celda
deja su resultado en una variable (`btc_df`, `reg_df`, `fit`, `fig`,
`peaks_df`, `bottoms_df`), de modo que se puede cambiar un parámetro o una
figura y re-ejecutar **sólo** la celda afectada, sin volver a descargar los
datos.

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
5. **Figura**: dos paneles log-log (fecha y días desde la génesis) con las
   rectas de los cuantiles y las anotaciones de halvings, picos y valles.
6. **Ciclos**: `scipy.signal.find_peaks` sobre `log10(close)` con distancia
   mínima de 300 muestras y prominencia mínima de 0,3; el pico de Mt. Gox
   (2011-06-08), anterior a los datos de Bitstamp, se inyecta a mano.

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
  Figura de dos paneles log-log con anotaciones parametrizables.

* ``detect_market_cycles(df, distance=300, prominence=0.3) -> (DataFrame, DataFrame)``
  Picos y valles macro con las columnas ``Days_Since_Last_Peak`` y
  ``Days_Since_Last_Bottom``.

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
| Configuración | — | `CONFIG` (símbolo, caché, cuantiles, anotaciones, ciclos) |
| Etapa 1: datos | `load_ohlcv()` | `btc_df` |
| Etapa 2: power-law | `fit_power_law()` | `reg_df`, `fit` |
| Etapa 3: figura | `draw_power_law_figure()` | `fig` |
| Etapa 4: ciclos | `find_cycles()` | `peaks_df`, `bottoms_df` |

Para iterar: cambia `CONFIG` (o la figura) y re-ejecuta **sólo** la celda
afectada.  Por ejemplo, cambiar `CONFIG["quantiles"]` y re-ejecutar las
celdas 2 y 3 actualiza los exponentes y las rectas sin volver a descargar
los datos; cambiar `CONFIG["distance"]` y re-ejecutar la celda 4 recalcula
los ciclos macro.

`run_experiment(config=None)` encadena las cuatro etapas y devuelve
`{"ohlcv", "reg_df", "fit", "figure", "peaks", "bottoms"}`; es un atajo para
tests y scripts, no la vía de ejecución del cuaderno.

Tests (sintéticos, sin red):

```bash
uv run pytest tests/test_btc.py -v
```

---

## Artefactos

| Artefacto | Descripción |
|---|---|
| `data/btc_daily_ohlcv.csv` | OHLCV diario en caché (ignorado por Git) |
| Figura del power-law | Dos paneles log-log con los cuantiles y las anotaciones |

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
- Bitstamp empieza a publicar en agosto de 2011: los picos anteriores
  (incluido Mt. Gox) no provienen de la API y se inyectan a mano.
- El exponente $\beta$ de la regresión cuantílica es una descripción del
  histórico, no una estimación fuera de muestra.

---

## Referencias

* Bitstamp API: <https://www.bitstamp.net/api/>
* `statsmodels.formula.api.quantreg` para regresión cuantílica.
* `scipy.signal.find_peaks` para la detección de extremos.

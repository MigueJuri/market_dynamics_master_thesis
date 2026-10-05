# Análisis de predictibilidad de características

Este módulo cuantifica la predictibilidad de distintas características de
mercado —momentum, volatilidad, distancia a la media móvil— sobre el
ETF `SPY` y un universo de ETFs sectoriales.

La pregunta central es: **¿la característica histórica
$f_t(\tau_p)$ ayuda a explicar el retorno futuro acumulado
$R_{t,\tau_f} = \log P_{t+\tau_f} - \log P_t$?**

Para responderla se estiman correlaciones (Pearson o Spearman) entre
ambas magnitudes y se visualizan en una rejilla 2D donde el eje
horizontal representa $\tau_p$ (lookback) y el vertical $\tau_f$
(lookahead).

---

## Estructura

```text
notebooks/feature_analysis/
├── README.md                          ← este archivo
├── functions.py                       ← utilidades generales
├── momentum_analysis.py               ← análisis de momentum
├── volatility_analysis.py             ← análisis de volatilidad
└── moving_average_distance_analysis.py ← análisis de distancia a la MA
```

`functions.py` no ejecuta análisis al importarse: sólo define
utilidades. Cada análisis es un cuaderno Jupytext (`# %%`) que puede
ejecutarse celda a celda.

---

## Datos

Los precios se descargan con `yfinance` y se persisten como CSV en la
carpeta `data/` del directorio principal del proyecto. La ruta se
resuelve a partir de la ubicación de `functions.py`, por lo que es
independiente del directorio de trabajo.

| Variable    | Valor por defecto | Significado                          |
| ----------- | ----------------- | ------------------------------------ |
| `TICKER`    | `"SPY"`           | Activo principal.                    |
| `START`     | `"1993-01-01"`    | Inicio de la serie.                  |
| `END`       | `"2026-01-01"`    | Fin de la serie.                     |
| `TRADING_DAYS` | `252`          | Factor de anualización.              |
| `DATA_DIR`  | `<proyecto>/data` | Carpeta de caché de precios.         |

Para apuntar a otra carpeta de datos se puede exportar la variable de
entorno `THESIS_DATA_DIR` antes de ejecutar el cuaderno.

---

## Funciones generales (`functions.py`)

### Carga y series

* `load_single_price(ticker, start, end, cache_dir, refresh) -> pd.Series`
  Precio de cierre ajustado con caché CSV local.
* `load_multiple_prices(tickers, ...) -> pd.DataFrame`
  Descarga varios tickers, persiste cada CSV y devuelve un frame
  combinado por fecha.
* `log_returns(price)` y `simple_returns(price)`
  Retornos diarios (logarítmico y simple) sin el primer valor.
* `rolling_forward_window(series, window)`
  Suma de los próximos `window` valores, anclada en `t`.

### Ventanas: `apply_window`

`apply_window(series, window, operation, window_type="fixed",
ewm_adjust=False, min_periods=None)` unifica el cálculo de reducciones
sobre ventanas fijas o exponenciales:

| `window_type` | Implementación                  | Operaciones admitidas                  | Callables |
| ------------- | ------------------------------- | -------------------------------------- | --------- |
| `"fixed"`     | `series.rolling(window).<op>()` | cualquier método de `rolling`          | sí        |
| `"exponential"` | `series.ewm(span=window, adjust=...).<op>()` | `mean`, `std`, `var` | no        |

`ewm_adjust=False` usa la forma recursiva
$y_t = (1-\alpha) y_{t-1} + \alpha x_t$, que es la única que admite
actualizaciones incrementales sin recalcular la historia completa.

### Estimadores

* `compute_correlation_matrix(feature_series, target_series, ...)`
  Matriz de correlación $\tau_p \times \tau_f$.

  * `feature_op` define la operación móvil sobre la característica.
  * `feature_window_type` elige entre ventana fija o exponencial.
  * `feature_transform` permite combinar la salida de la ventana con
    la serie base (p. ej. `lambda w, s: s - w` para
    `s - MA(s)`).
  * El **target siempre es el retorno futuro acumulado** sobre el
    lookahead (`log_returns.rolling(τ_f).sum().shift(-τ_f)`), nunca
    una media exponencial.

* `compute_cross_sectional_correlation(log_returns_df, windows,
  feature_df=None, ...)`
  Análogo al anterior, pero aplicado a un universo de activos.
  `feature_df` permite calcular la característica sobre precios
  mientras el target se construye sobre retornos.

### Visualización

* `plot_correlation_matrix(matrix, lookbacks, lookaheads, ...)`
  Heatmap con ticks en fracciones de año.
* `plot_scatter_grid(feature_by_window, target_by_window, ...)`
  Grilla de scatter feature × target para todas las combinaciones de
  ventanas.
* `plot_feature_boxplots(feature, target, n_quantiles=5, x_axis="quantiles", ...)`
  Boxplot del target agrupado por la característica.  `x_axis`
  selecciona la discretización del eje X: `"quantiles"` (cuantiles
  clásicos con etiquetas `Q1..Qn`, mismo número de observaciones por
  grupo, vía `pd.qcut`) o `"bins"` (intervalos de igual anchura sobre
  el rango del feature, vía `pd.cut`, con la etiqueta de cada grupo
  mostrando el valor medio del intervalo).  En ambos casos
  `n_quantiles` controla el número de grupos.  Devuelve la correlación
  de Pearson sobre la muestra conjunta si `return_correlation=True`.
* `plot_cross_sectional_curves(corr_matrix, avg_corr, ...)`
  Curvas por activo con promedio cross-sectional. Usa
  `marker="o"` para que el caso de una sola ventana no quede en
  blanco.

---

## Características implementadas

### Momentum

```text
momentum_t(τ_p) = Σ_{k=1..τ_p}  r_{t-k+1}
```

donde $r = \log(P_t / P_{t-1})$ es el retorno logarítmico diario.

* Característica: `feature_op="sum"`.
* Eje $\tau_p$: ventana de lookback del momentum.
* Análisis: `momentum_analysis.py`.

### Volatilidad

```text
volatility_t(τ_p) = std( r_{t-τ_p+1..t} )
```

* Característica: `feature_op="std"`.
* Eje $\tau_p$: ventana de lookback de la desviación estándar móvil.
* Análisis: `volatility_analysis.py`.

### Distancia a la media móvil

```text
distance_t(τ_p) = log P_t − mean_window(log P, τ_p)
```

`mean_window` puede ser una **media simple fija** (SMA) o una **media
exponencial** (EMA). El cambio se controla con la constante
`WINDOW_TYPE` al inicio del cuaderno:

```python
WINDOW_TYPE = "fixed"        # o "exponential"
```

* Característica: `feature_op="mean"`,
  `feature_transform=lambda w, s: s - w`.
* Eje $\tau_p$: ventana de la media móvil.
* Análisis: `moving_average_distance_analysis.py`.

---

## Matriz de correlación

La rejilla tiene:

* **Eje horizontal** ($\tau_p$): ventana con la que se calcula la
  característica histórica.
* **Eje vertical** ($\tau_f$): horizonte del retorno futuro
  acumulado que se intenta predecir.

El target es siempre:

```python
log_returns.rolling(τ_f).sum().shift(-τ_f)
```

no una media exponencial. Esta convención se mantiene para que todas
las características (momentum, volatilidad, distancia a la MA) sean
directamente comparables.

Por defecto se usa Spearman (rangos) porque es robusto a outliers y
monótono, lo que importa cuando la característica no se relaciona
linealmente con el retorno futuro.

---

## Análisis cross-sectional

`compute_cross_sectional_correlation` produce una matriz de tamaño
`(len(windows), n_assets)` con la correlación por activo y horizonte,
más el promedio cross-sectional.

Para la distancia a la MA, la característica se calcula sobre
`log_price_df` (precios) y el target sobre `log_ret_df` (retornos
logarítmicos):

```python
cross_corr, avg_corr = compute_cross_sectional_correlation(
    log_ret_df, lags,
    feature_df=log_price_df,
    feature_op="mean",
    feature_window_type=WINDOW_TYPE,
    feature_transform=lambda w, s: s - w,
    target_op="sum",
    correlation="pearson", min_valid_n=50,
    overlap=False, shift_lookahead_by_lookback=True,
)
```

`shift_lookahead_by_lookback=True` coloca el inicio del lookahead
inmediatamente después del final del lookback, de modo que el
intervalo total cubierto por la celda $(\tau_p, \tau_f)$ es
$\tau = \tau_p + \tau_f$.

---

## Ejemplos de uso

### Cambiar la característica distancia a SMA / EMA

Editar la primera celda de `moving_average_distance_analysis.py`:

```python
WINDOW_TYPE = "exponential"   # o "fixed"
```

No es necesario tocar el resto: el callable `subtract_from_log_price`
y los gráficos se actualizan automáticamente.

### Calcular la matriz manualmente

```python
from feature_analysis.functions import (
    compute_correlation_matrix, plot_correlation_matrix,
    load_single_price, log_returns,
)

price = load_single_price("SPY")
log_price = np.log(price)
log_ret = (log_price - np.log(price.shift(1))).iloc[1:]

m, lb, lf = compute_correlation_matrix(
    log_price, log_ret,
    max_lookback=504, max_lookahead=504, steps=150,
    feature_op="mean",
    feature_window_type="exponential",
    feature_transform=lambda w, s: s - w,
    target_op="sum", correlation="spearman",
)
plot_correlation_matrix(m, lb, lf)
```

### Actualización incremental (ventana exponencial)

Con `feature_window_type="exponential"` y `ewm_adjust=False`, la
característica se puede actualizar al recibir un nuevo precio sin
recalcular toda la historia:

```python
y_t = (1 - alpha) * y_{t-1} + alpha * x_t
alpha = 2 / (1 + span)
```

Esto es especialmente útil en pipelines online o en producción, donde
llega un dato a la vez.

---

## Interpretación y precauciones

* **Signo**: una correlación positiva en la celda $(\tau_p, \tau_f)$
  sugiere que la característica en `t` anticipa el signo del retorno
  futuro. La magnitud absoluta mide fuerza, no rentabilidad
  esperable: en la práctica hay spreads, comisiones y riesgos de
  cola.
* **Escalas distintas**: la distancia a la MA es adimensional
  (log-precio), el momentum también (log-retorno acumulado) y la
  volatilidad es desviación estándar de log-retornos. Las
  correlaciones siguen siendo comparables entre características
  porque están acotadas en $[-1, 1]$.
* **Solapamiento de ventanas**: en `compute_correlation_matrix` se
  ofrece `overlap=False` para thin-ear la muestra con paso
  `min(τ_p, τ_f)` y reducir la autocorrelación inducida por las
  ventanas móviles.
* **Valores faltantes**: las primeras $\tau_p$ observaciones de la
  característica y las últimas $\tau_f$ del target son NaN; las
  celdas con menos de `min_valid` (3) o `min_valid_n` (50) muestras
  se devuelven como `NaN` en la matriz.
* **Ventanas exponenciales y callables**: pandas no admite callables
  en `Series.ewm.agg`. Por eso `apply_window` rechaza esa
  combinación y la distancia a la MA se construye con
  `feature_op="mean"` + `feature_transform` en lugar de un callable
  que devuelva `s - mean(s)`.
* **Backtest vs predictibilidad**: una correlación distinta de cero
  en la muestra no implica una oportunidad rentable: los
  coeficientes pueden no ser estables, el spread puede superar el
  edge y la inferencia múltiple (cientos de celdas) requiere
  corrección por tests múltiples.

---

## Verificación rápida

```bash
# Compilar todos los módulos
python -m py_compile \
    notebooks/feature_analysis/functions.py \
    notebooks/feature_analysis/momentum_analysis.py \
    notebooks/feature_analysis/volatility_analysis.py \
    notebooks/feature_analysis/moving_average_distance_analysis.py

# Smoke test aislado (no descarga datos)
python - <<'PY'
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'notebooks')
from feature_analysis.functions import apply_window, compute_correlation_matrix

rng = np.random.default_rng(0)
price = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 2000))))
log_price = np.log(price)
log_ret_aligned = log_price - np.log(price.shift(1))

# Distancia con ventana fija
m, _, _ = compute_correlation_matrix(
    log_price, log_ret_aligned,
    max_lookback=200, max_lookahead=200, steps=20,
    feature_op="mean", feature_window_type="fixed",
    feature_transform=lambda w, s: s - w,
    target_op="sum", correlation="spearman",
)
print("fixed finite cells:", int(np.isfinite(m).sum()))

# Distancia con ventana exponencial
m_exp, _, _ = compute_correlation_matrix(
    log_price, log_ret_aligned,
    max_lookback=200, max_lookahead=200, steps=20,
    feature_op="mean", feature_window_type="exponential",
    feature_transform=lambda w, s: s - w,
    target_op="sum", correlation="spearman",
)
print("exp finite cells:", int(np.isfinite(m_exp).sum()))
PY
```
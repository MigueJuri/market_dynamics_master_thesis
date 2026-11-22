# Portfolio long-only: 1/N, 1/σ e 1/σ²

Cuaderno: `notebooks/portfolio/portfolio_HRP.py`

## Pregunta central

¿Cuánto aporta predecir la volatilidad del próximo período de rebalanceo
con un HAR-RV en log-volatilidad frente a usar una estimación histórica
EWMA para construir portfolios long-only inverse-volatility?

## Pipeline

1. **Datos**: precios de cierre ajustado descargados con
   `thesis.market_data.load_multiple_prices` y cacheados en `data/`.
2. **Retornos**: log-retornos diarios con `thesis.timeseries.log_returns`.
3. **Volatilidad**
   * **EWMA**: ``σ² = Σ w_t (r_t - μ_w)² / (1 - Σw²)`` con ``w_t``
     exponenciales (λ=0.94 por defecto). La corrección ``1/(1-Σw²)``
     reduce a la varianza muestral cuando los pesos son uniformes.
   * **HAR-RV**: se predice la log-volatilidad anualizada del próximo
     horizonte ``H = rebalance_freq`` usando tres features causales:
     log-vol realizada sobre ``H``, ``3H`` y ``9H`` días. Cada activo
     tiene su propia regresión OLS con errores HAC.
4. **Pesos**:
   * ``1/N``: igual ponderado.
   * ``1/σ``: inverso de la volatilidad.
   * ``1/σ²``: inverso de la varianza (= IVP sin usar correlaciones).
5. **Backtest walk-forward causal**:
   * Los pesos se calculan en ``t-1`` y se aplican al retorno de ``t``.
   * Entre rebalanceos los pesos derivan con buy & hold.
   * Costo de transacción: ``tx_cost · Σ|w_target - w_drift|``.
6. **Métricas**: CAGR, volatilidad anual, Sharpe, máximo drawdown y
   turnover anual.
7. **Figuras**: curva de capital, drawdown, evolución de pesos,
   composición promedio y topología de mercado (dendrograma).

## API del módulo (`thesis.portfolio`)

```python
from thesis.portfolio import run_backtest, calculate_metrics

port_rets, weights, turnover = run_backtest(
    log_rets,
    lookback=252,
    rebalance_freq=21,
    tx_cost=0.005,
    volatility_model="har_rv",  # o "ewma"
    har_lookback=252 * 5,
    har_min_obs=60,
)
```

## Fórmulas

### EWMA

```
w_i = λ^(n-i) / Σ_j λ^(n-j)          # i=1..n, el dato más reciente pesa más
σ²  = Σ_i w_i (r_i - μ_w)² / (1 - Σ_i w_i²)
```

### HAR-RV

Target (log-vol forward anualizada a ``H`` días):

```
y_t = log( sqrt( (A/H) · Σ_{i=1..H} r_{t+i}² ) )
```

Features (log-vol trailing anualizada):

```
x_{t,w} = 0.5 · log( A · mean(r²_{t-w+1..t}) ),  w ∈ {H, 3H, 9H}
```

Regresión por activo:

```
y_t = β₀ + β₁ x_{t,H} + β₂ x_{t,3H} + β₃ x_{t,9H} + ε_t
σ̂_t = exp( ŷ_t )
```

## Ejecución

```bash
jupytext --to notebook notebooks/portfolio/portfolio_HRP.py
jupyter lab notebooks/portfolio/portfolio_HRP.ipynb
```

O directamente como script:

```bash
python notebooks/portfolio/portfolio_HRP.py
```

## Artefactos

El cuaderno guarda figuras en `outputs/`:

* `portfolio_ewma_equity_drawdown.pdf`
* `portfolio_har_rv_equity_drawdown.pdf`
* `portfolio_ewma_weights_1var.pdf`
* `portfolio_ewma_avg_weights.pdf`
* `portfolio_har_rv_avg_weights.pdf`
* `portfolio_market_topology.pdf`

## Precauciones

* **Look-audit del ajuste HAR-RV**: el modelo se reajusta en cada
  rebalanceo usando únicamente targets cuyo forward ya concluyó en
  ``t-1``. En la implementación esto se garantiza calculando features y
  target sobre ``data.loc[:end_date]``; ``compute_log_vol_target`` deja
  ``NaN`` en las últimas ``H`` filas, que se descartan.
* **Warm-up HAR-RV**: se necesitan al menos ``9H`` días de historia para
  las features más ``min_obs`` observaciones de entrenamiento. El primer
  rebalanceo válido ocurre en ``t ≥ 10·H + min_obs - 1``.
* **Costos**: ``tx_cost=0.005`` implica 50 bps por unidad de turnover.
  Sensibilizar este parámetro es clave para comparar EWMA vs HAR-RV.
* **λ EWMA**: 0.94 es el estándar diario de RiskMetrics; ajustarlo cambia
  la memoria efectiva de la estimación.

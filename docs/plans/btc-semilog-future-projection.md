# Plan: gráfico semilog con proyección futura del power-law de BTC

## Alcance

Añadir a `notebooks/btc/BTC.py` un gráfico que muestre los últimos **6 años**
de BTC en escala **semilog** (eje Y logarítmico, eje X = fecha lineal) junto con
las rectas de la **regresión cuantílica power-law** (`log10(price) = α + β·log10(days)`),
extendiendo los ejes **~2 años hacia delante** para insinuar el precio futuro.

Es un gráfico **descriptivo/exploratorio**, no una señal de trading: la
proyección se etiqueta explícitamente como tal. No se toca `src/thesis/btc.py`
(el código de dibujo vive en línea en el cuaderno, como manda la Etapa 3).

## Archivos a tocar

- `notebooks/btc/BTC.py` (único archivo).
  - `import numpy as np` (nuevo; hoy el cuaderno no importa numpy).
  - Nueva clave de configuración en `CONFIG` (sección "Figura"/proyección).
  - Nueva celda markdown + celda de código "Etapa 3b: proyección semilog a futuro".

## Configuración nueva

```python
"prediction": {
    "lookback_years": 6,      # ventana hacia atrás (últimos 6 años)
    "horizon_days": 730,      # ~2 años hacia delante
},
```

## Código de la nueva celda (depende de `reg_df` y `fit` de la Etapa 2)

1. Fechas/días de corte:
   - `last_date = reg_df.index.max()`, `last_days = reg_df["days"].max()`.
   - `future_days = last_days + np.arange(1, horizon_days + 1)`.
   - `future_dates = last_date + pd.to_timedelta(np.arange(1, horizon_days + 1), unit="D")`.
2. Proyección por cuantil a partir de los **parámetros ajustados**:
   ```python
   params = fit[q]["params"]
   intercept, beta = params["Intercept"], params["log_days"]
   future_log_price = intercept + beta * np.log10(future_days)
   future_price = 10 ** future_log_price
   ```
   (el ajuste vive en espacio log-log; aquí se evalúa la misma recta en días futuros).
3. Filtro in-sample a la ventana lookback:
   - `cutoff = last_date - pd.DateOffset(years=lookback_years)`.
   - `hist = reg_df[reg_df.index >= cutoff]`.
4. Dibujo (figura nueva, p. ej. `figsize=(12, 6)`):
   - `ax.plot(hist.index, hist["close"], label="Cierre de BTC", color="black", ...)`.
   - Por cada `q` en `CONFIG["quantiles"]`: línea **sólida** en-sample
     (`hist.index` vs `fit[q]["prediction"].reindex(hist.index)`) y línea
     **discontinua** de proyección (`future_dates` vs `future_price`), con los
     colores/estilos ya usados en la Etapa 3 (`quantile_colors`/`quantile_styles`).
   - `ax.set_yscale("log")`.
   - `ax.axvline(last_date, color="gray", ls=":", ...)` separando dato/proyección.
   - `ax.set_xlim(cutoff, future_dates[-1])` para extender el eje hacia delante.
   - Título/etiquetas aclarando "proyección power-law (no señal)".
   - `ax.grid(True, which="both", ls="--", alpha=0.4)`; `fig.tight_layout()`.
5. Mostrar/cerrar según `CONFIG["show_figure"]`, igual que las etapas existentes.

## Criterios de aceptación

- `python notebooks/btc/BTC.py` (y el cuaderno) corren sin error hasta el final.
- La nueva figura tiene **eje Y logarítmico** y eje X que cubre los últimos 6
  años más **~2 años hacia delante**.
- Muestra el cierre de BTC más las rectas de los cuantiles (q10/q50/q90): tramo
  histórico en trazo continuo y proyección futura en trazo discontinuo, con un
  separador vertical en la última fecha con datos.
- La proyección se etiqueta como tal (no se presenta como señal).
- No se añade lógica a `src/`; todo el dibujo queda en línea en el cuaderno.

## Riesgos principales

- **Aplanamiento visual**: `log10(days)` crece muy lento (~+0.1 en 2 años), así
  que las rectas proyectadas son casi horizontales en escala semilog; es
  esperable y hay que dejar claro que es el comportamiento del power-law.
- **Extrapolación especulativa**: proyectar una regresión fuera de muestra no es
  un pronóstico riguroso; mitigar con el etiquetado "proyección" y la línea
  separadora.
- **Sin look-ahead**: la proyección usa sólo parámetros ajustados con datos ≤
  hoy; no introduce leakage (es descriptiva), pero debe evitarse presentarla
  como predictor causal.
- **Dependencia de import**: añadir `import numpy as np`; verificar que no rompe
  el orden/estilo de imports del cuaderno.
- **Consistencia de estilos**: reutilizar `quantile_colors`/`quantile_styles`
  definidos en la Etapa 3 para no duplicar literales; la celda nueva debe poder
  re-ejecutarse sola (depende de `reg_df`/`fit` ya en memoria).

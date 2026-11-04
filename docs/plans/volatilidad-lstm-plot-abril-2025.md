# Plan — Plot LSTM: volatilidad realizada vs predicción (abril 2025)

## Alcance
Añadir una celda de gráfica al notebook `notebooks/volatility/Volatilidad_LSTM.py`
que, para el **período de abril de 2025**, superponga la volatilidad realizada
forward (`y_true`) y la predicción de la LSTM (`y_lstm`), y marque con línea(s)
vertical(es) el anuncio de las tarifas de Trump. La celda sigue el patrón ya
existente en el bloque `## Gráficas`: depende solo del diccionario `results` y se
puede re-ejecutar sin reentrenar.

## No-goals
- No modificar `src/` (features, modelo, evaluación, backtest).
- No reentrenar ni cambiar hiperparámetros; es puramente visualización.
- No tocar las otras figuras ni el pipeline de `run_experiment`.
- No editar el texto de la tesis ni crear helpers reutilizables nuevos.

## Archivos a leer/tocar
**Leer (ya inspeccionado):** `notebooks/volatility/Volatilidad_LSTM.py`
(bloque de gráficas, líneas ~395-618).

**Tocar (única mutación):** `notebooks/volatility/Volatilidad_LSTM.py`.

**No tocar:** `src/thesis/volatility*.py`, `tests/`, `data/`, `outputs/`.

## Cambios concretos
1. En la celda `### Estilo` (líneas ~430-438), añadir una entrada a `PLOT_PATHS`,
   p. ej. `"april2025": "lstm_vol_april2025.png"`.
2. Añadir una nueva celda (markdown `### 7. Abril 2025: tarifas` + código) tras
   la figura 3 (`Evolución de la volatilidad`) o al final del bloque de gráficas.
   Reutilizar `dates`, `y_true`, `y_lstm` (opcional `y_har`) ya definidos en la
   celda `### Datos comunes`:
   - Máscara temporal: `mask = (dates >= "2025-04-01") & (dates <= "2025-04-30")`.
   - `ax.plot(dates[mask], y_true[mask], ...)` y `ax.plot(dates[mask], y_lstm[mask], ...)`.
   - Línea(s) vertical(es): `ax.axvline(pd.Timestamp("2025-04-02"), color=..., ls="--", lw=1, alpha=...)`
     con `ax.text`/`label` descriptivo.
3. Eventos a marcar (confirmar fechas exactas en implementación):
   - **2025-04-02** — "Liberation Day": anuncio de tarifas recíprocas (10% base + por país).
   - **2025-04-09** — pausa de 90 días de las tarifas recíprocas (escalada con China).
4. Ajustes visuales coherentes con el resto: etiquetas en español
   ("Volatilidad anualizada", "Fecha", "Volatilidad realizada forward"), título,
   `ax.legend()`, `mdates.DateFormatter("%Y-%m-%d")` (o `%d %b`) +
   `fig.autofmt_xdate()`, `fig.tight_layout()`, `fig.savefig(..., dpi=150,
   bbox_inches="tight")`, `plt.show()`. Mantener tamaño ~`(11, 4)`.

## Criterios de aceptación
- La celda genera un PNG `lstm_vol_april2025.png` con la vol realizada vs la
  predicción LSTM restringida a abril de 2025.
- Aparece(n) línea(s) vertical(es) en la(s) fecha(s) de anuncio de tarifas, con
  leyenda/etiqueta identificable.
- Se re-ejecuta sin reentrenar (solo depende de `results`).
- Etiquetas y estilo en español, consistentes con el bloque de estilo existente.
- El eje Y está en volatilidad anualizada (espacio σ), igual que las otras figuras.
- `ruff`/criterios de estilo del repo satisfechos (sin imports nuevos salvo si
  faltara `pd`/`mdates`, ya importados).

## Riesgos
- **Leakage / look-ahead**: ninguno esperado: el filtrado es solo por fecha sobre
  predicciones ya calculadas. No reajustar escaladores ni reentrenar. No filtrar
  por magnitud de volatilidad (eso introduciría un sesgo de selección); filtrar
  únicamente por fecha.
- **Unidades**: `y_true`/`y_lstm` están en **vol anualizada** (ya `np.exp` del
  log-vol y EWMA `span=15`). No volver a aplicar `exp` ni confundir con log-vol.
- **Alineación de fechas**: `dates = results["test_df"].index[LB:LB+len(y_true)]`
  (días hábiles); abril 2025 está dentro del set de test (>2022-01-01 y sin
  `end_date`, por lo que los datos llegan al presente). Verificar que las fechas
  de evento sean días hábiles (02/04 y 09/04/2025 lo son) y que `axvline` use
  `pd.Timestamp` naive, coherente con el índice de `dates`.
- **Celdas con `if __name__ == "__main__":`**: en el notebook esto se evalúa a
  `True`, así que la celda se dibuja; si se ejecuta como script (`python3`), solo
  se dibuja si `results` está definido (igual que las demás figuras).
- **Fechas de evento exactas**: confirmar/ajustar las fechas del anuncio (02/04 y
  09/04/2025) antes de fijar los `axvline`; no es crítica para el código pero sí
  para la corrección del mensaje del gráfico.

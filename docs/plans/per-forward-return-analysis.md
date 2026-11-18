# Plan: Análisis de predictibilidad del PER sobre retornos futuros

## Objetivo y alcance

Crear un cuaderno Jupytext (`# %%`) en `notebooks/feature_analysis/` que cuantifique
si el ratio PER del S&P 500 anticipa los retornos futuros del SPY, siguiendo las
convenciones de los análisis existentes (momentum, volatilidad, distancia a la MA).

- **Feature**: nivel del PER (serie mensual rellenada a diario con *forward fill*).
  A diferencia de los otros cuadernos, el PER **no** es una característica
  "windowed" (no tiene eje τ_p de lookback): es un nivel puntual en `t`.
- **Target**: retorno futuro acumulado `log P_{t+τ_f} − log P_{t+1}` para
  horizontes `τ_f` **> 1 mes** (> 21 sesiones), p. ej. 2, 3, 6, 12, 24, 36 meses.
- **Resultados**: curva de correlación `ρ(τ_f)` (Pearson y Spearman), grilla de
  scatter PER × retorno futuro, y boxplots del retorno futuro agrupado por
  cuantiles/bins del PER.

### No-objetivos
- No se modifica `src/thesis/` (se reutilizan helpers existentes).
- No hay análisis cross-sectional por sectores (el PER es solo del índice).
- No se construye estrategia/backtest.
- No se genera `.ipynb` (gitignored); solo el `.py` Jupytext.

## Archivos

- **Crear**: `notebooks/feature_analysis/per_analysis.py` (header Jupytext idéntico
  a los demás; bootstrap del path con `_HERE/_ROOT/_SRC`).
- **Leer**: `data/sp500_pe_mensual.csv` (columnas `fecha, pe`, mensual 1871–2026).
- **Editar (opcional)**: `notebooks/feature_analysis/README.md` — añadir el archivo
  a la lista de estructura y una subsección "PER" en "Características implementadas".
- **Reutilizar (sin editar)**:
  - `thesis.config` (`DATA_DIR`, `TICKER`, `START`, `END`, `TRADING_DAYS`).
  - `thesis.market_data.load_single_price`.
  - `thesis.timeseries.log_returns`.
  - `thesis.plotting.plot_scatter_grid`, `plot_feature_boxplots`.

## Pipeline de datos (sin look-ahead)

1. `price = load_single_price(TICKER, START, END, DATA_DIR)` (SPY 1993–2026).
2. `pe = pd.read_csv(DATA_DIR / "sp500_pe_mensual.csv", parse_dates=["fecha"])`
   → `set_index("fecha")["pe"]`, `sort_index()`.
3. Forward fill sobre el índice diario:
   `pe_daily = pe.reindex(price.index, method="ffill")`.
   Verificar que comparte índice con `price` y que no quedan NaN tras la primera
   fecha válida de PER (imprimir nº de días rellenados y % de días cubiertos).
4. **Variantes de feature (requeridas)**:
   - `pe_daily_shifted = pe_daily.shift(1)` — lag de 1 mes para corregir el fechado
     "as-of" (ver riesgo 1); el primer mes queda NaN y se descarta del análisis.
   - `log_pe_daily = np.log(pe_daily)` — log del nivel para comprimir outliers
     (PER≈124 en 2009); análogamente `log_pe_daily_shifted = np.log(pe_daily_shifted)`.
5. `log_ret = log_returns(price)`; `log_price = np.log(price)`.
6. Retorno futuro: `fwd(τ_f) = log_price.shift(-τ_f) - log_price.shift(-1)`
   (equivale a `rolling_forward_window(log_ret, τ_f)`), anclado en `t`.

## Secciones del cuaderno

1. **Carga + prints**: rango de precios, rango de PER, observaciones, % rellenado.
2. **Correlación ρ(τ_f)**: bucle sobre `τ_f ∈ {42, 63, 126, 252, 504, 756}`
   (2–36 meses); correlación Pearson y Spearman entre cada variante de feature y
   `fwd(τ_f)` usando solo posiciones válidas; gráfico de `ρ` vs `τ_f` en años.
   - Curva superpuesta para **4 variantes**: `pe_daily`, `pe_daily_shifted`,
     `log_pe_daily`, `log_pe_daily_shifted` (el shift mide la sensibilidad al
     fechado; el log mide la sensibilidad a outliers).
   - Variante adicional opcional: *earnings yield* `1 / pe_daily` (unidades 1/año,
     comparable al retorno).
3. **Grilla de scatter**: `log_pe_daily_shifted` (eje X, sin look-ahead) × retorno
   futuro (eje Y) para los horizontes elegidos (reutilizar `plot_scatter_grid` con
   `feature_by_window`/`target_by_window` cuyas claves son los horizontes; o scatter
   directo si las claves de texto complican los rótulos).
4. **Boxplots por cuantiles/bins**: `plot_feature_boxplots(log_pe_daily_shifted,
   fwd(τ_f), n_quantiles=10, x_axis="bins", show_cumulative=True)` en 1–2 horizontes
   representativos (p. ej. 126 y 504 sesiones); reportar Pearson de la muestra.

## Criterios de aceptación

- El cuaderno se ejecuta celda a celda (sin red más allá del caché de precios ya
  presente en `data/`); si no existe caché de SPY, se descarga una vez.
- `pe_daily` tiene el índice exacto de `price`, sin NaN tras el primer PER válido;
  se imprime el número de días forward-filled.
- Se computan explícitamente las variantes requeridas: `pe_daily_shifted` (shift de
  1 mes por fechado) y `log_pe_daily` (log del PER), y la curva `ρ(τ_f)` se muestra
  superponiendo las 4 variantes (nivel/shifted × raw/log).
- Todos los horizontes `τ_f` usados son **> 21 sesiones** (explícito en el código).
- Se producen: curva `ρ(τ_f)` (Pearson + Spearman), grilla de scatter y boxplots.
- `python3 -m py_compile notebooks/feature_analysis/per_analysis.py` y
  `uv run ruff check` limpios.
- README actualizado si se edita.

## Riesgos principales

1. **Look-ahead / punto temporal del PER**: las fechas del CSV son día 1 de mes,
   pero el valor puede ser de cierre de mes (la última fila `2026-11-17` sugiere
   fechado "as-of"). Con `ffill` el PER del mes actual se usa durante todo el mes,
   lo que filtraría información si el valor es de fin de mes. **Mitigación**:
   documentar el supuesto y ofrecer una variante conservadora `pe.shift(1)` (lag de
   un mes) para comparar sensibilidades.
2. **Degeneración numérica / empates de rango**: con forward fill el feature es
   constante ~21 días seguidos; en horizontes cortos el target queda dominado por
   ruido y Spearman se degrada por empates masivos. **Mitigación**: usar solo
   `τ_f > 1 mes` (requisito), y reportar Pearson además de Spearman.
3. **Outliers extremos del PER** (2009 ≈ 124, 2001 ≈ 46) distorsionan Pearson.
   **Mitigación**: reportar también `log(PER)` y earnings yield `1/PER`, y Spearman.
4. **Unidades**: PER es adimensional; retorno futuro es log-retorno acumulado
   (adimensional). No hay conflicto de unidades, pero el *earnings yield* (1/año)
   es la forma directamente comparable con el retorno y conviene mostrarla.
5. **Ventanas solapadas**: los retornos futuros solapados autocorrelacionan la
   muestra; los ρ puntuales son válidos pero p-valores/IC son optimistas. Reportar
   sin inferencia formal o con submuestreo (paso `τ_f`).
6. **Desajuste temporal**: PER cubre 1871–2026 y precios 1993–2026; solo se usa la
   intersección (~33 años, SPY desde 1993).

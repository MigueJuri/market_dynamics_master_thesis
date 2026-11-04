# Plan: Integrar `notebooks/BTC.py` al esquema del repo

## Objetivo

Convertir el script suelto `notebooks/BTC.py` (análisis de BTC: descarga
OHLCV de Bitstamp, regresión cuantílica power-law y detección de ciclos de
mercado) en una pieza integrada: lógica reutilizable en `src/thesis/`,
orquestador ligero en `notebooks/btc/`, tests y documentación.

## Alcance

**Fuera de alcance**: cambiar la semántica numérica del análisis, entrenar
modelos, o tocar la suite LSTM/volatilidad existente. Se corrigen sólo
defectos estructurales listados abajo, preservando resultados.

## Defectos detectados (auditoría)

1. **Efectos al importar**: descarga, guardado CSV y `plt.show()` a nivel de
   módulo (`btc_df = fetch_historical_btc()`, `to_csv`, `plot_btc_quantreg`,
   `analyze_market_cycles`).
2. **Ruta relativa al CWD**: `data_dir = Path.cwd().parent / "data"` (frágil);
   además `DATA_DIR = Path("data")` está definido y sin uso.
3. **Datos sin validar**: sin chequeo de columnas, precios positivos ni NaN.
4. **Dependencia no declarada**: `requests` ausente de `pyproject.toml`.
5. **Imports duplicados** entre celdas (`requests`, `pandas`, `matplotlib`,
   `time`, `datetime`, `Path`).
6. **Datos de anotación dudosos**: `'2022-11-21 '` (espacio final) y fechas
   futuras hardcodeadas en `peaks`/`bottoms`; se parametrizan y se limpia el
   espacio.

## Estructura propuesta

```
src/thesis/btc.py          ← nuevo módulo de dominio BTC (fetch + análisis + plot)
notebooks/btc/BTC.py       ← orquestador (mover desde notebooks/BTC.py)
notebooks/btc/README.md    ← documentación del experimento
tests/test_btc.py          ← tests unitarios (sintéticos, sin red)
pyproject.toml             ← + requests
README.md (raíz)           ← sección de estructura
```

## Archivos y funciones

### `src/thesis/btc.py` (nuevo)

- `BTC_OHLCV_COLUMNS: tuple = ("open", "close", "high", "low", "volume")`
- `fetch_btc_ohlcv(symbol="btcusd", cache_dir=DATA_DIR, refresh=False) -> DataFrame`
  Porta el bucle de paginación de Bitstamp; usa `requests.Session` (o
  parámetro inyectable para mock); escribe caché en `cache_dir /
  "btc_daily_ohlcv.csv"` (anclado a `config.DATA_DIR`, no al CWD) y valida.
- `validate_btc_ohlcv(df) -> DataFrame`: falla rápido ante columnas ausentes,
  precios `<= 0`, NaN y no-monotonicidad de índice (patrón `validate_ohlcv`).
- `prepare_power_law_frame(df, genesis="2009-01-03") -> DataFrame`:
  `days`, `log_days`, `log_price` con filtros `days>0` y `close>0`.
- `fit_power_law_quantiles(df, quantiles=(0.10, 0.50, 0.90)) -> dict`:
  `smf.quantreg('log_price ~ log_days')`; devuelve params/predicciones por q.
- `plot_btc_quantreg(df, halvings=None, peaks=None, bottoms=None, ax=None) -> Figure`
  Gráfico original (dos paneles log-log) con anotaciones parametrizadas.
- `detect_market_cycles(df, distance=300, prominence=0.3) -> (DataFrame, DataFrame)`
  `scipy.signal.find_peaks` sobre `log10(close)`; retorna peaks/bottoms con
  `Days_Since_Last_*`; mantiene inyección manual del pico Mt.Gox (2011-06-08).

**Export**: no añadir a `src/thesis/__init__.py` (arrastra `statsmodels`);
importar con `from thesis.btc import ...`.

### `notebooks/btc/BTC.py` (orquestador)

- `run_experiment()`: descarga (o lee caché), imprime resumen, ajusta
  cuantiles, dibuja, detecta ciclos.
- `if __name__ == "__main__": run_experiment()`.
- Guardar `btc_daily_ohlcv.csv` vía `fetch_btc_ohlcv`, no en el cuaderno.

### `notebooks/btc/README.md`

Objetivo/pregunta central, pipeline, API pública, fórmula del power-law
(`log10(price) = α + β·log10(days)`), ejecución, artefactos
(`btc_daily_ohlcv.csv`, figuras), interpretación y precauciones (las fechas de
picos/valles son anotaciones descriptivas, no señales; `find_peaks` es
retrospectivo, no trading).

### `pyproject.toml`

Añadir `"requests>=2.31.0"` a `[project].dependencies`.

## Criterios de aceptación

- `python3 -m py_compile src/thesis/btc.py notebooks/btc/BTC.py` sin errores.
- `uv run pytest` en verde, incluido `tests/test_btc.py`.
- Importar `thesis.btc` **no** descarga datos ni dibuja nada.
- `validate_btc_ohlcv` rechaza columnas ausentes y precios no positivos.
- `prepare_power_law_frame` es causal (test de truncamiento: cortar la serie
  no altera filas ya calculadas).
- `detect_market_cycles` sobre sintético (serie con picos/valles conocidos)
  recupera fechas esperadas.
- `fetch_btc_ohlcv` con `requests` mockeado devuelve DataFrame validado y
  escribe caché en `tmp_path`.
- `ruff check` y `ruff format --check` limpios en `src/` y `tests/`.

## Riesgos principales

- **Bitstamp API**: bucle de paginación depende de la red y del límite 1000
  por petición; los tests unitarios lo evitan mockeando `requests`. No
  ejecutar integración en la suite por defecto (opcional marcar con
  `@pytest.mark.integration`).
- **Conservar resultados numéricos**: `find_peaks` es determinista dado el
  input, pero la regresión cuantílica depende de `statsmodels`; los tests usan
  datos sintéticos y sólo comprueban invariantes, no valores exactos.
- **Fechas futuras hardcodeadas** (pico 2025, valle 2026): se mantienen como
  defaults parametrizables para no alterar la figura, pero se documenta que
  son anotaciones del autor y no deben tratarse como señales.
- **Trailing space** en `'2022-11-21 '`: se limpia (no cambia el parseo).

## Pasos de implementación (sesión `/exec`)

1. Crear `src/thesis/btc.py` con las funciones arriba.
2. Mover `notebooks/BTC.py` → `notebooks/btc/BTC.py` y convertirlo en
   orquestador con `run_experiment()`.
3. Añadir `requests` a `pyproject.toml`.
4. Escribir `tests/test_btc.py`.
5. Escribir `notebooks/btc/README.md` y actualizar `README.md` raíz.
6. Verificar (`py_compile`, `pytest`, `ruff`), mostrar diffs y preguntar por
   el commit.

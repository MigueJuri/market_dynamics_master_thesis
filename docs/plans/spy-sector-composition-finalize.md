# Plan: finalizar integración de `notebooks/spy/yfinance_chart.py`

> Nota de slug: la tarea no traía slug usable (era el texto completo). Se usa
> `spy-sector-composition-finalize` (continuación de
> `spy-sector-composition.md`, que ya planificó esta integración).

## Estado actual (verificado)

La integración **ya está implementada** en el working tree (sin commitear),
según el plan previo `docs/plans/spy-sector-composition.md`:

- `src/thesis/sector_composition.py` — lógica reutilizable (snapshot, scrape
  Wikipedia, descarga por lotes, ponderación, entropía). Traducción GICS al
  español en `GICS_SECTOR_ES`. ✓
- `notebooks/spy/SPY_sectores.py` — orquestador por etapas (py:percent). ✓
- `notebooks/spy/README.md` — documentación. ✓
- `tests/test_sector_composition.py` — unitarios sintéticos + integración. ✓
- `lxml` declarado en `pyproject.toml` y fijado en `uv.lock`. ✓
- `README.md` raíz (Estructura, líneas 52/62) ya referencia el módulo y
  `notebooks/spy/`. ✓

**Pendientes detectados** (lo que falta para cerrar):

1. `notebooks/spy/yfinance_chart.py` sigue presente (duplicado legacy; el plan
   previo decía que se elimina y `SPY_sectores.py` lo reemplaza). El
   `notebooks/spy/README.md` (línea 29) documenta un árbol sin este archivo.
2. `notebooks/spy/sp500_sector_composition.png` es un artefacto generado en
   sitio incorrecto (los artefactos van a `outputs/`, ignorado por Git).
3. **Reutilización de datos**: hoy `download_price_history` vuelve a descargar
   ~500 tickers en cada re-ejecución; sólo se cachea el *peso* final
   (`sp500_sector_composition.csv`). Falta cachear el frame de precios.

## Alcance

- Eliminar los dos archivos legacy (punto 1 y 2).
- Añadir caché CSV del frame de precios histórico (ancho, fecha×ticker) en
  `data/`, siguiendo la convención de `market_data.load_single_price` pero
  **por lotes** (un único archivo, no por-ticker: mantener la descarga en lotes
  de 100, no forzar `load_multiple_prices`).
- Conectar el caché en el cuaderno y documentarlo.
- Verificación final: pytest, ruff, py_compile, import sin efectos.

## No-goals

- No reconstruir datos point-in-time (limitación metodológica ya documentada:
  sesgo de supervivencia, clasificación retroactiva, shares constantes).
- No exportar `sector_composition` en `thesis/__init__.py`.
- No tocar `pair_trading.py`, `plotting.py`, `notebooks/pair_trading/*` ni sus
  tests (cambios preexistentes del usuario).
- No commitear (sólo preparar; el commit es decisión de `build`/usuario).

## Archivos a tocar

- **Eliminar**: `notebooks/spy/yfinance_chart.py`,
  `notebooks/spy/sp500_sector_composition.png`.
- **Editar** `src/thesis/sector_composition.py`:
  - Nueva constante `PRICE_CACHE_FILENAME_TMPL = "sp500_prices_{start}_{end}.csv"`.
  - `download_price_history(..., cache_dir: Path | None = None, refresh: bool = False)`:
    si `cache_dir` es dado y el CSV existe y `refresh=False`, leerlo (índice
    parseado como fecha) y devolver sin red; si no, descargar (lógica actual) y
    escribir el CSV. La clave de caché es `(start, end)` **explícitos**: si
    `end is None`, no cachear (no se puede congelar "hoy" en el nombre) o usar
    fecha de descarga en metadatos; decisión de `build`: preferir cachear sólo
    con `end` explícito y documentarlo.
  - Opcional (menor prioridad): caché de `get_shares_outstanding` en
    `sp500_shares.csv`.
- **Editar** `notebooks/spy/SPY_sectores.py`:
  - `CONFIG` gana `"use_price_cache": True` / `"refresh_prices": False` y
    `"prices_cache_filename"`.
  - Etapa 2: pasar `cache_dir=Path(DATA_DIR)` y `refresh` a
    `download_price_history`; el `resample("ME").last()` sigue aplicándose
    **después** de cargar (así cambiar `resample_freq` no re-descarga).
- **Editar** `notebooks/spy/README.md`: actualizar estructura (ya sin
  `yfinance_chart.py` ni PNG), tabla de artefactos (`data/sp500_prices_*.csv`)
  y pipeline (paso 2 lee caché).
- **Editar** `tests/test_sector_composition.py`: test de caché con
  `monkeypatch` — si el CSV existe, `yf.download` **no** se invoca y el frame
  leído coincide; si no existe, se descarga y escribe el CSV (en `tmp_path`).
- **Verificar** (sin cambios esperados): `README.md` raíz, `pyproject.toml`,
  `uv.lock`.

## Criterios de aceptación

- `notebooks/spy/` queda con `README.md` + `SPY_sectores.py` únicamente (sin
  `.py` legacy ni `.png` generado).
- `uv run pytest` en verde; `uv run pytest -m integration` en verde.
- `uv run ruff check src tests` y `uv run ruff format --check src tests` en
  verde.
- Importar `thesis.sector_composition` no descarga ni dibuja (test existente).
- Re-ejecutar Etapa 2 con caché presente no llama a `yf.download` (cubierto por
  el nuevo test).
- Cuaderno corre como script y en Jupyter; figuras a `outputs/` con etiquetas
  en español.

## Riesgos

- **Staleness del caché**: con `end=None` el CSV no puede congelar "hoy";
  restringir caché a `end` explícito y permitir `refresh=True`.
- **Unidades**: snapshot en fracción (0-1) vs pesos históricos en % (0-100).
  `aggregate_small_sectors` espera fracción; `build_sector_weights` devuelve %.
  No mezclar en el cuaderno ni en el README.
- **Red volátil** en integración (Wikipedia/yfinance): tests marcados y
  omitidos por defecto.
- **`auto_adjust=True`**: el cierre ajustado usa factores de ajuste actuales
  (ajuste retroactivo de splits/dividendos) — aceptable para análisis
  descriptivo; ya documentado como precaución.
- **No es leakage**: composición sectorial es descriptiva; `sector_entropy` es
  puntual (sin look-ahead). Sin splits train/test involucrados.

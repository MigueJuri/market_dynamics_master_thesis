# Plan: integrar `notebooks/spy/yfinance_chart.py`

## Alcance

Convertir el cuaderno suelto `notebooks/spy/yfinance_chart.py` (análisis de
composición sectorial de SPY/S&P 500) en una pieza integrada del repo:
lógica reutilizable en `src/thesis/sector_composition.py`, orquestador por
etapas en `notebooks/spy/`, README y tests. Siguiendo el patrón de
`thesis.btc` + `notebooks/btc/`.

El cuaderno contiene **tres análisis** que se conservan:

1. **Donut** de pesos sectoriales actuales de SPY (`yf.Ticker(...).funds_data`).
2. **Reconstrucción histórica** de la composición sectorial del S&P 500 por
   capitalización de mercado (Wikipedia → constituyentes; yfinance → precios
   y shares; ponderación por sector). Es una **aproximación** con sesgo de
   supervivencia y clasificación retroactiva (ya documentado en el propio
   archivo).
3. **Entropía de Shannon** de los pesos sectoriales a lo largo del tiempo.

## Archivos

Nuevos:
- `src/thesis/sector_composition.py` — datos + cómputo reutilizable.
- `tests/test_sector_composition.py` — tests unitarios (sintéticos, sin red)
  + integración marcada.
- `notebooks/spy/README.md` — documentación del experimento.
- `notebooks/spy/SPY_sectores.py` — orquestador py:percent (reemplaza
  `yfinance_chart.py`, que se elimina).

Modificados:
- `pyproject.toml` — declarar `lxml` (lo usa `pd.read_html`; hoy ausente).
- `README.md` — sección Estructura: añadir `sector_composition.py` y
  `notebooks/spy/`.

No tocar: cambios preexistentes del usuario en `pair_trading.py`,
`plotting.py`, `notebooks/pair_trading/*`, tests y `docs/plans/*`.

## API de `src/thesis/sector_composition.py`

Constantes: `GICS_SECTOR_ES` (traducción), `SP500_WIKI_URL`,
`MINOR_SECTOR_THRESHOLD = 0.08`, `SECTOR_COMPOSITION_CACHE_FILENAME`.

- `aggregate_small_sectors(weights, threshold=0.08, translation=GICS_SECTOR_ES) -> pd.Series`
  — agrupa sectores `< threshold` en "Otros", traduce etiquetas, ordena
  descendente. Corrige la inconsistencia comentario(9%) vs código(0.08): se
  fija 0.08 y el comentario se alinea.
- `fetch_spy_sector_weights(ticker="SPY") -> pd.Series` — snapshot actual.
- `get_sp500_constituents(session=None) -> pd.DataFrame` — scrape Wikipedia
  (columnas `Symbol`, `Security`, `Sector`; `BRK.B → BRK-B`). Sesión
  inyectable para tests.
- `download_price_history(tickers, start, end, batch_size=100, sleep=1.0) -> pd.DataFrame`
  — cierre ajustado en lotes, frame wide (fecha × ticker).
- `get_shares_outstanding(tickers) -> pd.Series` — best-effort vía
  `yf.Ticker(t).fast_info`.
- `build_sector_weights(prices, shares, sector_map) -> pd.DataFrame` —
  market-cap = price·shares; descarta tickers sin datos; suma por sector;
  devuelve % (0-100). Validación fail-fast: `sector_map` cubre las columnas,
  al menos un ticker con datos, pesos no negativos.
- `sector_entropy(sector_weights) -> pd.Series` — entropía de Shannon por
  fila (p = weights/100; `p·ln p → 0` para p=0). **Solo** cómputo; el dibujo
  vive en el cuaderno.

Los efectos con red/disco/pantalla viven **dentro** de funciones; importar el
módulo no descarga ni dibuja. No se exporta en `thesis/__init__.py` (evita
arrastrar `lxml`/`requests` en cada import del paquete).

## Cuaderno `notebooks/spy/SPY_sectores.py` (py:percent)

Una celda por etapa, con `_HERE/_ROOT/_SRC` + `sys.path` como en
`notebooks/btc/BTC.py`. Celdas:

1. **Configuración**: `START_DATE`, `END_DATE`, `RESAMPLE_FREQ="ME"`,
   `BATCH_SIZE`, umbral, ticker.
2. **Donut (snapshot)**: invoca `fetch_spy_sector_weights` +
   `aggregate_small_sectors`; el código `matplotlib` de la figura (anillo +
   leyenda) **en el cuaderno**, etiquetas en español.
3. **Datos históricos**: `get_sp500_constituents` →
   `download_price_history` (con `resample("ME").last()`) →
   `get_shares_outstanding`.
4. **Pesos**: `build_sector_weights` → `sector_weights`.
5. **Figura área apilada 100%**: `matplotlib` en línea; guarda en
   `outputs/` vía ruta anclada (no CWD) o solo `plt.show()`; ejes en español.
6. **Entropía**: `sector_entropy` + figura en línea.

Las celdas dejan resultados en variables (`sector_weights`, `entropy`,
...), re-ejecutables sin repetir descargas.

## Defectos corregidos (documentar en el commit)

- Efectos al importar: pipeline a nivel de módulo → funciones + guard
  `if __name__ == "__main__"`.
- Estado global (`sectores`, `constituents`, `sector_weights`) → argumentos.
- `savefig("sp500_sector_composition.png")` relativo al CWD → ruta anclada.
- Umbral ambiguo (comentario 9% vs 0.08) → unificarlo en 0.08.
- Variable muerta `colores` y fragmentos comentados → limpiar.
- Dependencia no declarada `lxml`.

## Tests (`tests/test_sector_composition.py`)

Unitarios sintéticos: `aggregate_small_sectors` (agrupa < umbral, traduce,
ordena, conserva suma=1); `build_sector_weights` (filas suman 100, descarta
tickers all-NaN, `ValueError` si `sector_map` no cubre columnas);
`sector_entropy` (uniforme → ln(n); p=0 no da -inf; coincide con cálculo
manual); `sector_entropy` es pointwise (truncar no altera filas previas).
Test de import: importar `thesis.sector_composition` no hace red ni dibuja
(patrón de `test_btc::test_importar_thesis_btc_no_descarga_ni_dibuja`).
Integración marcada `@pytest.mark.integration` (omitida por defecto):
`get_sp500_constituents`, `fetch_spy_sector_weights`.

## Criterios de aceptación

- `uv run pytest` y `uv run pytest -m integration` en verde.
- `uv run ruff check src tests` y `uv run ruff format --check src tests` en
  verde.
- `python3 -m py_compile` de los archivos nuevos sin errores.
- Importar `thesis.sector_composition` no descarga ni dibuja.
- El cuaderno se abre en Jupyter (py:percent) y corre como script.

## Riesgos

- **Sesgo de supervivencia / clasificación retroactiva** en la reconstrucción
  histórica: no es un bug a corregir sino una limitación metodológica ya
  declarada; debe quedar documentada en `README.md` (sección precauciones),
  no "arreglada".
- **Red volátil** en la integración (Wikipedia/yfinance): tests marcados y
  omitidos por defecto; no ejecutar en la suite por defecto.
- **`pd.read_html` sin parser** falla sin `lxml`: declarar la dependencia y
  fijar el lock con `uv lock`.
- **Colisión de dominios**: `plotting.py` ya tiene funciones "sectoriales"
  (clusters de pair-trading); no reutilizarlas ni renombrarlas: dominio
  distinto.
- Descarga de ~500 tickers es lenta: mantener el enfoque por lotes del
  original (no forzar `load_multiple_prices` per-ticker).

# Plan: Integrar `notebooks/spy/OrderBook.py` al esquema del repo

## Objetivo

Convertir el cuaderno suelto `notebooks/spy/OrderBook.py` (extracción y
perfiles de liquidez del LOB de SPY en formato LOBSTER) en una pieza integrada:
lógica reutilizable en `src/thesis/`, orquestador por etapas en `notebooks/`,
tests y documentación. Sin cambiar la semántica numérica.

## Alcance / no-objetivos

**Incluye**: extracción vectorizada de distribuciones relativas y absolutas del
LOB, snapshot único, precio medio promediado, y sus figuras.

**Fuera de alcance**: features/modelos sobre el LOB, trading, métricas de
microestructura nuevas (spread, profundidad, OIR), y leer/escribir tesis. Se
corrigen sólo defectos estructurales, preservando resultados numéricos.

## Estructura propuesta

```
src/thesis/orderbook.py        ← nuevo módulo de dominio (carga + extracción, sin matplotlib)
notebooks/spy/OrderBook.py     ← orquestador por etapas (reemplaza el contenido actual)
tests/test_orderbook.py        ← tests unitarios con LOBSTER sintético (sin red)
notebooks/spy/README.md        ← añadir sección "OrderBook (LOB)"
```

No se tocan `pyproject.toml` (numpy/pandas/matplotlib ya declarados) ni
`src/thesis/__init__.py` (el módulo no arrastra dependencias pesadas, pero se
deja fuera para no acoplar; import con `from thesis.orderbook import ...`).

## Defectos detectados (auditoría)

1. **Efectos al importar**: rutas absolutas hardcodeadas
   (`/home/miguel/GoogleDriveOffline/Data/OrderBook/SPY/...`) y llamadas de
   ejemplo a nivel de módulo (`extract_...`, `plot_...` con `plt.show()`).
2. **Rutas no portables**: anclar a `config.DATA_DIR` + env var
   (`THESIS_LOB_DIR`) o aceptar `message_path`/`orderbook_path` explícitos.
3. **`extract_and_plot_averaged_mid_price` mezcla cálculo y gráfico**: separar
   en `extract_averaged_mid_price` (src) + celda de figura (notebook).
4. **Lógica de agregación duplicada** (`reaggregate`, `aggregate_book_side`):
   extraer a un único helper compartido.
5. **`levels` sin validar**: puede producir `IndexError`/columnas fuera de
   rango; añadir fail-fast.
6. **`plot_absolute_lob` con `xlim(134, 136.5)` hardcodeado**: específico del
   dataset; parametrizar o eliminar.
7. **Aliasing sentinel**: `!= 9999999999` vs `!= 999999.9999` (tras /10000);
   centralizar constantes.
8. **Imports duplicados entre celdas** (`pandas`, `numpy`, `matplotlib`).

## Archivos y funciones

### `src/thesis/orderbook.py` (nuevo)

Constantes:
- `PRICE_SCALE = 10_000.0`
- `DUMMY_PRICE = 9_999_999_999`, `DUMMY_NEG_PRICE = -9_999_999_999`
- `MESSAGE_COLUMNS = ("Time", "Type", "OrderID", "Size", "Price", "Direction")`
  (conservar el nombre `Type` del original; documentar equivalencia con
  `Event` de LOBSTER).

Carga:
- `load_message_times(message_path) -> pd.Series` (`usecols=[0]`, eficiente).
- `_load_orderbook(orderbook_path) -> pd.DataFrame`.

Validación:
- `_validate_levels(levels, n_cols)`: `4*levels <= n_cols`, `levels >= 1`.

Extracción (portan el código actual, preservando bugs-fixes de linspace y
bin-primero-luego-sumar):
- `extract_relative_lob_distribution(message_path, orderbook_path, start_time,
  end_time, levels, bin_width=0.01) -> (bids_binned, asks_binned, max_range)`
- `extract_absolute_lob_distribution(...) -> (bids_binned, asks_binned,
  (p_min_rounded, p_max_rounded))`
- `extract_single_snapshot(message_path, orderbook_path, target_time, levels)
  -> (df_bids, df_asks, actual_time)`
- `extract_averaged_mid_price(message_path, orderbook_path, start_time=None,
  end_time=None, interval=1.0) -> DataFrame` (sin plot)
- `aggregate_volume_bins(price: Series, volume: Series, bin_width: float)
  -> DataFrame` (helper compartido; usa `np.floor(np.round(x/bw, 6))`).

### `notebooks/spy/OrderBook.py` (orquestador, `py:percent`)

Celdas por etapa; el código de figuras vive aquí (convención del repo):
1. Configuración: rutas (`DATA_DIR`/env), `start_time`, `end_time`, `levels`,
   `bin_width`.
2. Distribución relativa: llama `extract_relative_lob_distribution` + figura.
3. Distribución absoluta: `extract_absolute_lob_distribution` + figura
   (xlim parametrizado).
4. Snapshot único: `extract_single_snapshot` + figura.
5. Precio medio promediado: `extract_averaged_mid_price` + figura.

Cada celda deja su resultado en el namespace; guard
`if __name__ == "__main__":`. Figuras devuelven `fig`/`ax` (no `plt.show()`
a nivel de módulo).

### `notebooks/spy/README.md`

Añadir sección "OrderBook (LOB)": pregunta central, pipeline, API pública,
unidades (ticks/10000 → USD; tiempo en segundos desde medianoche; volumen en
acciones; normalización por nº de snapshots válidos), ejecución y precauciones.

### `tests/test_orderbook.py`

- Sintéticos: construir CSV de mensajes y orderbook en `tmp_path` con
  centinelas y precios conocidos.
- Invariantes: filtrado de centinelas, mid-price `/10000`, `pd.cut`
  `right=False`, time-average = suma/n_snapshots, snapshot devuelve primera
  fila `>= target_time`, `aggregate_volume_bins` coincide con reagregación.
- `validate` rechaza `levels` inválidos.
- Truncamiento: cortar ventana temporal no altera resultados de la misma
  sub-ventana (ausencia de look-ahead en la extracción).
- (Opcional) test `@pytest.mark.integration` con datos reales para spot-check.

## Criterios de aceptación

- `python3 -m py_compile src/thesis/orderbook.py notebooks/spy/OrderBook.py` OK.
- `uv run pytest` en verde, incl. `tests/test_orderbook.py`.
- `from thesis import orderbook` **no** lee archivos ni dibuja.
- Salidas numéricas idénticas al original sobre el mismo input (spot-check).
- `uv run ruff check src tests` y `ruff format --check` limpios.
- Rutas de datos parametrizables (sin `/home/miguel/...` hardcodeado).

## Riesgos principales

- **Leakage**: extracciones descriptivas (no predictivas); el snapshot "a o
  después de `target_time`" es retrospectivo por diseño y `extract_single_snapshot`
  elige la primera fila `>= target_time` (posterior a target). Documentar que
  no son señales y que no hay train/test que contaminar.
- **Numerico**: centinelas `±9999999999` deben filtrarse antes de dividir por
  `PRICE_SCALE`; mid-price = `(ask+bid)/20000.0` equivale a `/2` y luego
  `/10000`; conservar `np.linspace` (no `np.arange` con paso float) y el
  redondeo a 6 decimales antes de `np.floor`.
- **Unidades**: precios en ticks (÷10000 → USD), `Time` en segundos desde
  medianoche (puede >86400 en after-hours), volumen en acciones; time-average
  normalizado por nº de snapshots válidos, **no** por duración de ventana.
- **Alineación fila a fila**: `message` y `orderbook` deben compartir orden y
  nº de filas (`.loc[valid_indices]` y `skiprows`/`nrows` lo asumen); validar
  igual longitud.
- **Memoria**: lectura de archivos completos; `use_cols` para tiempo y
  `skiprows`/`nrows` para snapshot mitigan; mantener.

## Pasos de implementación (sesión `/exec`)

1. Crear `src/thesis/orderbook.py`.
2. Reescribir `notebooks/spy/OrderBook.py` como orquestador por etapas.
3. Escribir `tests/test_orderbook.py`.
4. Actualizar `notebooks/spy/README.md`.
5. Verificar (`py_compile`, `pytest`, `ruff`), mostrar diffs, preguntar commit.

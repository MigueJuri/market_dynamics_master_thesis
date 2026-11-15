# Reutilizar el orden MI (leaf order) para la matriz de Pearson

## Objetivo y alcance

Hacer que la matriz de Pearson (`2_PairSelectionPearson.py`) se dibuje con **el
mismo orden de activos** que la matriz de información mutua
(`1_PairSelectionMutualInformation.py`), para que ambos heatmaps sean
comparables fila a fila. El orden de referencia es el **leaf order** del
dendrograma de Ward sobre la distancia MI (`hierarchical_clustering` →
`leaf_order` → `labels_ordered`), que hoy se calcula en el notebook de MI pero
**no se persiste ni se usa en la figura** (la figura se ordena internamente por
`cluster+sector` en `plot_sector_reordered_matrix`).

**No-objetivos**: no cambiar la evaluación sectorial del clustering de Pearson
(sigue con sus propios clusters); no tocar la animación rolling ni las matrices
sector×sector (son orden-independientes); no cambiar la métrica MI ni Pearson.

## Diseño

1. **Persistir el leaf order de MI**: en el notebook de MI, tras calcular
   `labels_ordered = mi_matrix.columns[leaf_order]`, guardar `list(labels_ordered)`
   en un JSON determinista bajo `data/pair_trading/mi/asset_order.json`, con
   metadatos (`k_neighbors`, `n_clusters`, `assets`, `created_at`) para auditar.
2. **Renderizar MI con ese orden explícito**: `plot_sector_reordered_matrix`
   gana un parámetro `order: list[str] | None = None`; cuando se pasa, se usa
   ese orden (intersección con el índice de la matriz) en lugar del sort por
   cluster/sector. El notebook de MI pasa `order=labels_ordered`.
3. **Renderizar Pearson con el mismo orden**: el notebook de Pearson carga el
   JSON (`load_asset_order`), reordena `corr_matrix` con
   `.reindex(index=order, columns=order)` y llama a
   `plot_sector_reordered_matrix(..., order=order)` (sin `cluster_labels`).

## Archivos y funciones a tocar

- `src/thesis/pair_trading.py` (nuevas funciones, junto a `cached_ksg_mi_matrix`):
  - `save_asset_order(order, *, k_neighbors, n_clusters, cache_dir=None) -> Path`
    — escribe el JSON (usar `tempfile` + `os.replace`, como la caché KSG).
  - `load_asset_order(*, cache_dir=None) -> list[str]` — lee y devuelve
    `assets`; `FileNotFoundError` claro si aún no se ejecutó el notebook de MI.
- `src/thesis/plotting.py` → `plot_sector_reordered_matrix` (línea ~547):
  añadir `order: list[str] | None = None`; si `order is not None`, filtrar a
  tickers presentes en `matrix` y saltarse la rama cluster/sector.
- `notebooks/pair_trading/1_PairSelectionMutualInformation.py`:
  - En la celda de clustering (~588-593): tras `labels_ordered`, llamar
    `save_asset_order(list(labels_ordered), k_neighbors=K_NEIGHBORS, n_clusters=n_clusters)`.
  - En la celda de `plot_sector_reordered_matrix` (~663): pasar
    `order=list(labels_ordered)` y quitar `cluster_labels`/`sector_labels` de
    esa llamada.
- `notebooks/pair_trading/2_PairSelectionPearson.py`:
  - Importar `load_asset_order`.
  - Antes del heatmap de la matriz (~610): `mi_order = load_asset_order()`;
    `order = [t for t in mi_order if t in corr_matrix.columns]`; reordenar
    `corr_matrix` con ese `order`.
  - Llamar `plot_sector_reordered_matrix(..., order=order)` (quitar
    `cluster_labels` de esa llamada; `sector_labels` opcional).

## Criterios de aceptación

- Ejecutar primero el notebook de MI escribe `asset_order.json`; su heatmap
  queda ordenado por leaf order.
- Ejecutar después el notebook de Pearson carga el orden y su heatmap de
  Pearson usa **exactamente el mismo orden de tickers** que el heatmap MI
  (verificable: mismo `order` en filas/columnas).
- La evaluación sectorial de Pearson, la animación rolling y las matrices
  sector×sector siguen produciendo los mismos resultados que hoy.
- `plot_sector_reordered_matrix` sin `order` mantiene el comportamiento actual
  (retrocompatible).
- `uv run ruff check src tests` y `uv run pytest` (sin integración) sin errores.

## Riesgos clave

- **Orden entre sesiones**: los dos notebooks corren en sesiones distintas; la
  ruta del JSON debe ser determinista y Pearson debe ejecutarse después de MI.
  Documentarlo; `load_asset_order` debe fallar con mensaje claro si falta.
- **Dependencia de `K_NEIGHBORS`**: el leaf order depende de `k`; registrar
  `k_neighbors` en el JSON para auditar y evitar usar un orden de otro `k`.
- **Desajuste de tickers**: si el universo difiere, `reindex` dejaría filas
  NaN; intersectar `order` con `corr_matrix.columns` (mismo universo ⇒ iguales).
- **Cambio visual en MI**: la figura MI pasará de orden cluster+sector a leaf
  order (bloques contiguos pero ordenados por dendrograma, no por id de
  cluster). Es intencional para igualar ambos heatmaps.
- **No leakage**: el cambio es puramente de orden de visualización; no altera
  métricas, clustering ni señales.

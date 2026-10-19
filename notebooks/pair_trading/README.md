# Pair Trading — Documentación

Este directorio contiene los cuadernos y simulaciones asociadas a la
estrategia de *pair trading* (trading por pares), tanto para encontrar
pares candidatos como para modelar el spread entre dos activos como un
proceso de Ornstein-Uhlenbeck.

## Estructura

| Archivo | Propósito |
|---|---|
| [`1_PairSelectionMutualInformation.py`](./1_PairSelectionMutualInformation.py) | Selección de pares mediante Información Mutua estimada con el método KSG (k-NN). |
| [`2_PairSelectionPearson.py`](./2_PairSelectionPearson.py) | Selección de pares mediante correlación de Pearson, con animación de la correlación *rolling*. |
| [`PairTrading.py`](./PairTrading.py) | Construcción del spread y ajuste del modelo OU para un par concreto. |
| [`OUProcess.py`](./OUProcess.py) | Simulación de un proceso OU y de dos activos GBM cuyo spread coincide con la trayectoria simulada. |

Toda la lógica común (descarga de precios, log-retornos, scraping del
S&P 500, filtro de emisores duplicados, estimadores de MI, clustering,
cálculo de métricas sectoriales, simulaciones OU y GBM) vive en
`src/thesis/pair_trading.py` y se reutiliza desde los cuadernos.

## Orden recomendado de ejecución

1. `OUProcess.py` (sólo requiere `simulate_ou_process`, `simulate_gbm`).
2. `PairTrading.py` (depende de `extract_and_construct_spread`,
   `fit_ou_process` y de `plotting.plot_ou_process` /
   `plot_pair_prices`).
3. `1_PairSelectionMutualInformation.py` y/o
   `2_PairSelectionPearson.py`.  Ambos comparten la misma fase de
   preparación (universo, retornos, sectores) pero luego divergen en
   la métrica central.  Pueden ejecutarse en cualquier orden; incluso
   en paralelo, ya que la caché KSG es local a cada notebook.

## Configuración común

Todos los cuadernos leen las fechas de `thesis.config`:

```python
from thesis.config import START, END, DATA_DIR
```

`START` y `END` por defecto son `"1993-01-01"` y `"2026-01-01"`.  Los
precios descargados se cachean en CSV bajo `DATA_DIR`
(`thesis/data/` por defecto) gracias a `thesis.market_data.load_multiple_prices`.

## Universo y filtrado de emisores

`get_sp500_constituents()` descarga la tabla oficial del S&P 500 desde
Wikipedia.  `filter_duplicate_issuers()` la normaliza (elimina sufijos
de `Class A/B`, `Inc.`, `Corp.`, etc.) y conserva **una sola fila por
emisor**, descartando explícitamente pares como:

- `GOOGL` y `GOOG` (Alphabet)
- `FOXA` y `FOX` (Fox Corporation)
- `NWSA` y `NWS` (News Corp)

Esto evita evaluar pares inválidos *antes* de descargar datos y calcular
matrices.  Adicionalmente, `top_pairs(issuer_map=...)` descarta
cualquier par residual del mismo emisor al construir el ranking.

## Estimadores de dependencia

- **Correlación de Pearson** (`returns.corr(method="pearson")`):
  captura únicamente relaciones lineales.
- **Información Mutua gaussiana** (`gaussian_mi_matrix`):
  cerrada para distribuciones normales, equivalente a
  $-\tfrac{1}{2}\log(1-\rho^2)$.
- **Información Mutua KSG** (`ksg_mi_matrix` / `cached_ksg_mi_matrix`):
  estimador por k-vecinos que captura dependencias no lineales.

## Clustering jerárquico

En ambos cuadernos de selección se aplica
`hierarchical_clustering()` con el método de **Ward** sobre una matriz
de distancias:

- `correlation_distance(corr, use_sqrt=True)`:
  $d_{ij} = \sqrt{1 - \rho_{ij}}$ para Pearson.
- `mi_distance(mi)`:
  $d_{ij} = \max(MI) - MI_{ij}$ para Información Mutua.

El número de clusters se fija igual al número de sectores GICS
presentes en el universo (`available_sectors.nunique()`, 11 en el
S&P 500 actual).  Esto facilita una comparación 1-a-1 entre sectores y
clientes GICS.

## Evaluación sectorial de los clusters

`evaluate_cluster_sector_alignment()` devuelve un `ClusterSectorReport`
con:

- **`contingency`**: tabla sector × cluster (más dos filas extra,
  `cluster_size` y `cluster_purity`).
- **`per_cluster`**: tamaño, sector dominante, purity, entropía
  normalizada y HHI por cluster.
- Métricas globales:
  - `weighted_purity` (pureza ponderada por tamaño);
  - `normalized_entropy` (entropía sectorial media, normalizada por
    $\log S_c$);
  - `mean_hhi` (Herfindahl-Hirschman medio);
  - `ari` (Adjusted Rand Index);
  - `ami` (Adjusted Mutual Information);
  - `homogeneity`, `completeness` y `v_measure` (de `sklearn.metrics`).

Estas funciones miden **alineación con los sectores GICS**, no la calidad
financiera del clustering.  Los sectores GICS no son la verdad
absoluta sobre la dependencia estadística, sólo una partición de
referencia basada en actividad económica.

## Caché de matrices KSG

KSG es O(N²) en el número de activos (≈125.000 pares en el S&P 500
completo).  `cached_ksg_mi_matrix()` persiste el resultado en disco:

```text
data/pair_trading/mi/ksg/
└── ksg_k{k}_seed{s}_{assets}x{obs}_{hash}.pkl
└── ksg_k{k}_seed{s}_{assets}x{obs}_{hash}.json
```

La clave de caché incorpora:

- versión del esquema (`KSG_CACHE_VERSION`);
- `k_neighbors` y `seed`;
- fechas de inicio/fin y número de observaciones;
- lista ordenada de activos;
- hash SHA-256 del contenido de los retornos.

Cualquier cambio en estos campos provoca una entrada de caché distinta.
La escritura es **atómica** (`tempfile` + `os.replace`) para evitar
ficheros corruptos si el proceso se interrumpe.

Para forzar el recálculo de una entrada concreta basta con pasar
`force=True`:

```python
mi_matrix = cached_ksg_mi_matrix(returns, k_neighbors=3, force=True)
```

La carpeta `data/` está en `.gitignore`, por lo que los ficheros
generados nunca se suben al repositorio.

## Advertencias metodológicas

- **KSG es costoso**: sobre los ≈500 activos del S&P 500 puede tardar
  varios minutos.  La caché KSG evita repetir el cálculo.
- **Sesgo de supervivencia**: `get_sp500_constituents()` lee la
  composición *actual* del S&P 500 desde Wikipedia.  Los retornos
  históricos se asignan al sector *presente hoy*; esto introduce un
  sesgo si empresas cambiaron de sector en el pasado.
- **Cobertura de los datos**: la caché CSV de precios (`market_data`)
  puede tener huecos para tickeros poco negociados; esos activos se
  descartan en `load_log_returns` si quedan por debajo de
  `min_valid_obs` observaciones válidas.
- **GICS ≠ dependencia**: las métricas de evaluación sectorial miden
  cuánto se parece el clustering a la partición oficial del S&P 500.
  Un clustering "bueno" en términos financieros (pares cointegrados)
  puede no coincidir con los sectores GICS.
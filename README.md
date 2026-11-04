# Modelado de Dinámica del Mercado de Valores

Repositorio de la tesis sobre modelado de la dinámica del mercado de valores.
Incluye análisis de predictibilidad de características, estrategias de par trading
basadas en cointegración e Información Mutua, y simulaciones de procesos
estocásticos.

---

## Instalación

El proyecto usa Python 3.10+ y se gestiona con `uv`.

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd thesis

# Crear entorno virtual e instalar dependencias
uv sync

# Activar el entorno
source .venv/bin/activate
```

Si no usas `uv`, también puedes instalar con `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Estructura

```text
.
├── src/thesis/                  # Paquete Python compartido
│   ├── config.py                # Constantes globales y estilo
│   ├── market_data.py           # Descarga cacheada de precios
│   ├── timeseries.py            # Retornos y ventanas móviles
│   ├── correlation.py           # Estimadores de correlación
│   ├── features.py              # Características y targets
│   ├── event_labeling.py        # Triple barrera
│   ├── pair_trading.py          # Spread, OU, MI y clustering
│   ├── volatility.py            # Features y target de volatilidad
│   ├── volatility_models.py     # LSTM, HAR-RV y entrenamiento
│   ├── volatility_evaluation.py # Métricas y backtest de volatilidad
│   ├── btc.py                   # OHLCV de BTC, power-law y ciclos
│   └── plotting.py              # Visualizaciones reutilizables
│
├── tests/                       # Tests del paquete
│   └── test_pair_trading.py     # Tests de pair_trading
│
├── notebooks/
│   ├── btc/                     # Análisis de BTC (power-law y ciclos)
│   ├── feature_analysis/        # Análisis de predictibilidad
│   ├── pair_trading/            # Selección de pares, OU y trading
│   ├── trend_following_strategy/# Estrategia de momentum
│   └── volatility/              # Predicción de volatilidad con LSTM
│
├── data/                        # Caché local de precios (no en Git)
├── outputs/                     # Resultados y figuras (no en Git)
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Tests

Ejecutar la batería completa con `pytest`:

```bash
uv run pytest
```

Para ver la salida detallada:

```bash
uv run pytest -v
```

Para una sola función de test:

```bash
uv run pytest tests/test_pair_trading.py::test_dataclass_declara_campos -v
```

Los tests de integración requieren red y datos reales, por lo que están
marcados con `integration` y se omiten por defecto.  Para ejecutarlos:

```bash
uv run pytest -m integration -v
```

---

## Linting

El proyecto usa `ruff` para formateo y análisis estático:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

Para aplicar el formateo automáticamente:

```bash
uv run ruff format src tests
```

---

## Reproducibilidad

- Las fechas por defecto y el ticker principal están en
  `src/thesis/config.py`.
- Los precios descargados se cachean localmente en `data/` (ignorado por Git).
- Las matrices de Información Mutua KSG se cachean en
  `data/pair_trading/mi/ksg/`.
- La semilla aleatoria por defecto es `SEED=42`.
- Los cuadernos de Jupytext pueden ejecutarse celda a celda o convertirse a
  notebooks con `jupytext`.

---

## Notas

- Los tests por defecto no requieren conexión a red: usan datos sintéticos
  generados en memoria.  Los de integración sí la requieren y se ejecutan
  con `pytest -m integration`.
- `torch` se resuelve desde el índice de wheels CPU de PyTorch
  (`download.pytorch.org/whl/cpu`), evitando descargar las dependencias
  CUDA (~3 GB) innecesarias para este proyecto.
- El directorio `data/` y los artefactos generados no deben subirse al
  repositorio; están incluidos en `.gitignore`.

"""Parámetros globales y estilo por defecto para los notebooks de la tesis.

Las constantes definen el activo principal, el rango temporal, el directorio
local de caché de precios y la configuración visual por defecto.  Pueden
sobreescribirse mediante variables de entorno (``THESIS_DATA_DIR``) o
importándolas y pasándolas como argumentos explícitos.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

TICKER: str = "SPY"
START: str = "1993-01-01"
END: str = "2026-01-01"
TRADING_DAYS: int = 252

#: Tasa libre de riesgo anual con la que se calculan los Sharpe de la tesis
#: (bono del Tesoro de EE.UU., ver Cap. Estrategias).
RISK_FREE_RATE: float = 0.03
SEED: int = 42

#: Mínimo de sesiones que un ticker debe tener dentro de ``START:END`` para
#: entrar en el panel de los cuadernos de selección de pares.  El filtro se
#: aplica **antes** de alinear las fechas: sin él, la intersección de fechas
#: la determina el activo con menos historia y un rango de treinta años se
#: convierte en dos meses.
#:
#: El umbral fija el equilibrio entre historia y número de activos.  Medido
#: sobre el S&P 500 actual con la ventana 1993-2026:
#:
#: ========  ========  ==========  =============
#: Sessions activos   sesiones    desde
#: ========  ========  ==========  =============
#: 0        497       43         2025-10-28
#: 1260     484       1 267      2020-12-11
#: 5040     388       5 047      2005-12-06
#: 7560     301       7 557      1995-12-18
#: ========  ========  ==========  =============
#:
#: El valor por defecto son 5 años bursátiles: 484 activos y unas 1 267
#: sesiones.  Subirlo a 7560 (~30 años) acorta la intersección a cambio de
#: descartar 183 empresas más.
PAIR_SELECTION_MIN_HISTORY_OBS: int = 252 * 20

#: Raíz del repositorio (dos niveles por encima de ``src/thesis/config.py``).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Directorio de caché de precios; configurable con ``THESIS_DATA_DIR``.
DATA_DIR: Path = Path(os.environ.get("THESIS_DATA_DIR", _PROJECT_ROOT / "data"))

#: Directorio donde los cuadernos exportan tablas en CSV, LaTeX y metadatos.
TABLES_DIR: Path = Path(
    os.environ.get("THESIS_TABLES_DIR", _PROJECT_ROOT / "outputs" / "tables")
)

#: Ruta al repositorio de la tesis (donde vive ``.thesis/results-ledger.md``).
#: Se resuelve desde ``THESIS_REPO_DIR``; si no está definida se asume que la
#: tesis comparte raíz con este repo de código.
THESIS_REPO_DIR: Path | None = (
    Path(p)
    if (p := os.environ.get("THESIS_REPO_DIR")) and Path(p).exists()
    else None
)

#: Publicar automáticamente métricas registradas al terminar el proceso.
PUBLISH_RESULTS_TO_LEDGER: bool = (
    os.environ.get("THESIS_PUBLISH_RESULTS", "1").lower() in ("1", "true", "yes")
)

# Reproducibilidad para consumidores del estado global de numpy.
np.random.seed(SEED)


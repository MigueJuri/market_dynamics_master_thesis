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
SEED: int = 42

#: Raíz del repositorio (dos niveles por encima de ``src/thesis/config.py``).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Directorio de caché de precios; configurable con ``THESIS_DATA_DIR``.
DATA_DIR: Path = Path(os.environ.get("THESIS_DATA_DIR", _PROJECT_ROOT / "data"))

# Reproducibilidad para consumidores del estado global de numpy.
np.random.seed(SEED)

FIGSIZE = (7.0, 3.5)
PLOT_RC = {
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "legend.frameon": False,
}

"""Extracción de distribuciones del libro de órdenes (LOB) en formato LOBSTER.

El módulo concentra la lógica reutilizable del cuaderno
``notebooks/spy/OrderBook.py``:

1. **distribución relativa** del LOB: perfil de volumen pasivo agregado en
   bins de distancia al precio medio (``extract_relative_lob_distribution``);
2. **distribución absoluta** del LOB: perfil de volumen pasivo agregado en
   bins de precio (``extract_absolute_lob_distribution``);
3. **snapshot único** del libro en el primer instante posterior a un tiempo
   objetivo (``extract_single_snapshot``);
4. **precio medio promediado** en buckets de tiempo
   (``extract_averaged_mid_price``).

Unidades:

* **precios**: los ficheros LOBSTER los guardan en *ticks* (enteros); el
  precio en USD es ``tick / PRICE_SCALE`` (``PRICE_SCALE = 10_000``);
* **tiempo**: segundos desde medianoche (puede superar 86 400 en
  *after-hours*);
* **volumen**: número de acciones.

Importar el módulo **no** lee ficheros ni dibuja figuras: todo el acceso a
disco vive dentro de funciones explícitas.  El módulo no se exporta en
``thesis/__init__.py``; se usa ``from thesis.orderbook import ...``.

.. note::
   Las extracciones son **descriptivas**, no señales: no hay train/test que
   contaminar.  ``extract_single_snapshot`` es retrospectivo por diseño
   (elige la primera fila con ``Time >= target_time``), así que no debe usarse
   como variable de entrada de un modelo.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.config import DATA_DIR

#: Factor de escala de los precios del *orderbook* LOBSTER (ticks → USD).
PRICE_SCALE: float = 10_000.0

#: Centinela de LOBSTER para un nivel de precio vacío (lado *ask*).
DUMMY_PRICE: int = 9_999_999_999

#: Centinela de LOBSTER para un nivel de precio vacío (lado *bid*).
DUMMY_NEG_PRICE: int = -9_999_999_999

#: Columnas del fichero de mensajes de LOBSTER.  ``Type`` es el ``Event``
#: de LOBSTER (1=nueva orden, 2=modificación, 3=cancelación,
#: 4=ejecución, 5=ejecución oculta, 7=halt).
MESSAGE_COLUMNS: tuple[str, ...] = (
    "Time",
    "Type",
    "OrderID",
    "Size",
    "Price",
    "Direction",
)


def lob_data_dir() -> Path:
    """Directorio raíz de los ficheros LOBSTER.

    Se toma de la variable de entorno ``THESIS_LOB_DIR`` y, si no está
    definida, de ``config.DATA_DIR / "OrderBook"``.
    """
    return Path(os.environ.get("THESIS_LOB_DIR", Path(DATA_DIR) / "OrderBook"))


def lob_file_paths(
    symbol: str = "SPY",
    date: str = "2012-06-21",
    start: str = "34200000",
    end: str = "37800000",
    levels: int = 50,
    directory: Path | None = None,
) -> tuple[Path, Path]:
    """Rutas de los ficheros LOBSTER (mensajes y orderbook) de una sesión.

    La convención de nombres de LOBSTER es
    ``<TICKER>_<fecha>_<hora_inicio>_<hora_fin>_{message,orderbook}_<niveles>.csv``
    y los ficheros de una sesión cuelgan de ``<raíz>/<TICKER>/``.
    """
    base = Path(directory) if directory is not None else lob_data_dir()
    stem = f"{symbol}_{date}_{start}_{end}"
    return (
        base / symbol / f"{stem}_message_{levels}.csv",
        base / symbol / f"{stem}_orderbook_{levels}.csv",
    )


def load_message_times(message_path: Path | str) -> pd.Series:
    """Columna ``Time`` (segundos desde medianoche) del fichero de mensajes."""
    return pd.read_csv(message_path, header=None, usecols=[0], names=["Time"])["Time"]


def _load_orderbook(orderbook_path: Path | str) -> pd.DataFrame:
    """Fichero de orderbook completo (sin cabecera, una fila por mensaje)."""
    return pd.read_csv(orderbook_path, header=None)


def _is_dummy(prices: np.ndarray) -> np.ndarray:
    """Máscara de los niveles vacíos (centinelas de LOBSTER).

    Se filtran **ambos** centinelas (``±9999999999``) en los dos lados del
    libro: en un fichero bien formado el centinela del *ask* es positivo y el
    del *bid* negativo, así que el filtro simétrico coincide con el
    original, pero evita que un centinela con el signo "equivocado" llegue a
    la rejilla de bins (y la dispare).
    """
    return (prices == DUMMY_PRICE) | (prices == DUMMY_NEG_PRICE)


def _validate_levels(levels: int, n_cols: int) -> None:
    """Falla rápido si ``levels`` no cabe en el fichero de orderbook.

    Cada nivel ocupa 4 columnas (precio ask, tamaño ask, precio bid, tamaño
    bid), luego el fichero debe tener al menos ``4 * levels`` columnas.
    """
    if levels < 1:
        raise ValueError(f"levels debe ser >= 1 (recibido {levels}).")
    if 4 * levels > n_cols:
        raise ValueError(
            f"levels={levels} requiere {4 * levels} columnas en el orderbook "
            f"(el fichero tiene {n_cols})."
        )


def _load_window(
    message_path: Path | str,
    orderbook_path: Path | str,
    start_time: float,
    end_time: float,
) -> pd.DataFrame:
    """Filas del orderbook dentro de la ventana temporal.

    ``message`` y ``orderbook`` comparten orden y número de filas, así que el
    subconjunto del orderbook se indexa con los índices del mensaje.
    """
    df_msg = pd.read_csv(message_path, header=None, names=list(MESSAGE_COLUMNS))
    df_ob = _load_orderbook(orderbook_path)
    if len(df_ob) != len(df_msg):
        raise ValueError(
            "Los ficheros de mensajes y orderbook deben tener el mismo número "
            f"de filas (mensajes: {len(df_msg)}, orderbook: {len(df_ob)})."
        )

    mask = (df_msg["Time"] >= start_time) & (df_msg["Time"] <= end_time)
    valid_indices = df_msg[mask].index
    if len(valid_indices) == 0:
        raise ValueError("No data found in the specified time window.")

    return df_ob.loc[valid_indices]


def _level_columns(levels: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Índices de columna de precios/tamaños de cada lado del libro."""
    ask_p_cols = [4 * i for i in range(levels)]
    ask_v_cols = [4 * i + 1 for i in range(levels)]
    bid_p_cols = [4 * i + 2 for i in range(levels)]
    bid_v_cols = [4 * i + 3 for i in range(levels)]
    return ask_p_cols, ask_v_cols, bid_p_cols, bid_v_cols


def _bin_and_average(
    prices: np.ndarray,
    volumes: np.ndarray,
    bins: np.ndarray,
    n_snapshots: int,
    price_column: str,
) -> pd.DataFrame:
    """Agrupa por bins (suma) y divide por el número de snapshots.

    El orden importa: primero se **binnea** y después se **suma** el volumen
    dentro de cada bin; el resultado es el volumen pasivo *promedio por
    snapshot* (no una suma de volúmenes por mensaje).
    """
    df = pd.DataFrame({price_column: prices, "Volume": volumes})
    df["Bin"] = pd.cut(df[price_column], bins=bins, right=False)
    binned = df.groupby("Bin", observed=True)["Volume"].sum().reset_index()
    binned["Volume"] = binned["Volume"] / n_snapshots
    # ``.apply`` sobre una columna categórica devuelve otra categórica: se
    # convierte a float para que el centro del bin sea aritmético.
    binned["BinMid"] = binned["Bin"].apply(lambda interval: interval.mid).astype(float)
    return binned


def extract_relative_lob_distribution(
    message_path: Path | str,
    orderbook_path: Path | str,
    start_time: float,
    end_time: float,
    levels: int,
    bin_width: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Perfil de liquidez relativo al precio medio, promediado en el tiempo.

    Devuelve ``(bids_binned, asks_binned, max_range)``:

    * ``bids_binned`` / ``asks_binned``: columnas ``Bin`` (intervalo),
      ``Volume`` (volumen pasivo promedio por snapshot) y ``BinMid``
      (centro del bin, en USD);
    * ``max_range``: distancia máxima (en USD) al precio medio alcanzada por
      el libro en la ventana.

    Los niveles vacíos (centinelas de LOBSTER) se descartan antes de
    escalar los precios.
    """
    if bin_width <= 0:
        raise ValueError(f"bin_width debe ser > 0 (recibido {bin_width}).")

    ob_subset = _load_window(message_path, orderbook_path, start_time, end_time)
    _validate_levels(levels, ob_subset.shape[1])

    # --- Precio medio (top of book) ---
    best_asks_raw = ob_subset.iloc[:, 0].to_numpy()
    best_bids_raw = ob_subset.iloc[:, 2].to_numpy()

    # Descarta los snapshots cuyo top of book es un valor dummy (lado vacío):
    # si no, el precio medio (~±10^11/10000) envenena RelPrice, max_range y
    # la rejilla de bins de todo el resultado.
    valid_top = ~_is_dummy(best_asks_raw) & ~_is_dummy(best_bids_raw)
    if not valid_top.any():
        raise ValueError("No valid top-of-book quotes in the specified time window.")

    ob_subset = ob_subset.loc[valid_top]
    best_asks = best_asks_raw[valid_top] / PRICE_SCALE
    best_bids = best_bids_raw[valid_top] / PRICE_SCALE
    mid_prices = (best_asks + best_bids) / 2.0
    n_snapshots = len(ob_subset)  # denominador del promedio temporal

    ask_p_cols, ask_v_cols, bid_p_cols, bid_v_cols = _level_columns(levels)
    bid_p_raw = ob_subset.iloc[:, bid_p_cols].to_numpy()
    bid_v = ob_subset.iloc[:, bid_v_cols].to_numpy()
    ask_p_raw = ob_subset.iloc[:, ask_p_cols].to_numpy()
    ask_v = ob_subset.iloc[:, ask_v_cols].to_numpy()

    mid_prices_matrix = np.tile(mid_prices, (levels, 1)).T

    # --- Aplanado (sin bucles) quitando los niveles dummy ---
    valid_bids_mask = ~_is_dummy(bid_p_raw)
    flat_bid_p_raw = bid_p_raw[valid_bids_mask]
    flat_bid_v = bid_v[valid_bids_mask]
    flat_mid_bids = mid_prices_matrix[valid_bids_mask]
    rel_bids_p = (flat_bid_p_raw / PRICE_SCALE) - flat_mid_bids

    valid_asks_mask = ~_is_dummy(ask_p_raw)
    flat_ask_p_raw = ask_p_raw[valid_asks_mask]
    flat_ask_v = ask_v[valid_asks_mask]
    flat_mid_asks = mid_prices_matrix[valid_asks_mask]
    rel_asks_p = (flat_ask_p_raw / PRICE_SCALE) - flat_mid_asks

    df_bids = pd.DataFrame({"RelPrice": rel_bids_p, "Volume": flat_bid_v})
    df_asks = pd.DataFrame({"RelPrice": rel_asks_p, "Volume": flat_ask_v})

    # --- Rango de los bins a partir de los precios relativos (antes de binnear) ---
    b_min = df_bids["RelPrice"].min() if not df_bids.empty else 0.0
    a_max = df_asks["RelPrice"].max() if not df_asks.empty else 0.0
    max_range = max(abs(b_min), abs(a_max))

    # ``linspace`` con un número explícito de bins (en vez de un ``arange``
    # con paso flotante, que acumula error y puede perder el último borde).
    n_bins = int(np.ceil((2 * max_range + bin_width) / bin_width)) + 1
    bins = np.linspace(-max_range - bin_width, max_range + bin_width, n_bins)

    bids_binned = _bin_and_average(
        df_bids["RelPrice"].to_numpy(),
        df_bids["Volume"].to_numpy(),
        bins,
        n_snapshots,
        "RelPrice",
    )
    asks_binned = _bin_and_average(
        df_asks["RelPrice"].to_numpy(),
        df_asks["Volume"].to_numpy(),
        bins,
        n_snapshots,
        "RelPrice",
    )

    return bids_binned, asks_binned, max_range


def extract_absolute_lob_distribution(
    message_path: Path | str,
    orderbook_path: Path | str,
    start_time: float,
    end_time: float,
    levels: int,
    bin_width: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    """Perfil de liquidez absoluto (en precio), promediado en el tiempo.

    Devuelve ``(bids_binned, asks_binned, (p_min_rounded, p_max_rounded))``,
    donde el rango está redondeado a múltiplos de ``bin_width`` para que los
    bordes de los bins sean "bonitos".
    """
    if bin_width <= 0:
        raise ValueError(f"bin_width debe ser > 0 (recibido {bin_width}).")

    ob_subset = _load_window(message_path, orderbook_path, start_time, end_time)
    _validate_levels(levels, ob_subset.shape[1])

    best_asks_raw = ob_subset.iloc[:, 0].to_numpy()
    best_bids_raw = ob_subset.iloc[:, 2].to_numpy()
    valid_top = ~_is_dummy(best_asks_raw) & ~_is_dummy(best_bids_raw)
    if not valid_top.any():
        raise ValueError("No valid top-of-book quotes in the specified time window.")

    ob_subset = ob_subset.loc[valid_top]
    n_snapshots = len(ob_subset)  # denominador del promedio temporal

    ask_p_cols, ask_v_cols, bid_p_cols, bid_v_cols = _level_columns(levels)
    bid_p_raw = ob_subset.iloc[:, bid_p_cols].to_numpy()
    bid_v = ob_subset.iloc[:, bid_v_cols].to_numpy()
    ask_p_raw = ob_subset.iloc[:, ask_p_cols].to_numpy()
    ask_v = ob_subset.iloc[:, ask_v_cols].to_numpy()

    valid_bids_mask = ~_is_dummy(bid_p_raw)
    flat_bid_p_raw = bid_p_raw[valid_bids_mask]
    flat_bid_v = bid_v[valid_bids_mask]

    valid_asks_mask = ~_is_dummy(ask_p_raw)
    flat_ask_p_raw = ask_p_raw[valid_asks_mask]
    flat_ask_v = ask_v[valid_asks_mask]

    df_bids = pd.DataFrame(
        {"AbsPrice": flat_bid_p_raw / PRICE_SCALE, "Volume": flat_bid_v}
    )
    df_asks = pd.DataFrame(
        {"AbsPrice": flat_ask_p_raw / PRICE_SCALE, "Volume": flat_ask_v}
    )

    b_min = df_bids["AbsPrice"].min() if not df_bids.empty else np.inf
    b_max = df_bids["AbsPrice"].max() if not df_bids.empty else -np.inf
    a_min = df_asks["AbsPrice"].min() if not df_asks.empty else np.inf
    a_max = df_asks["AbsPrice"].max() if not df_asks.empty else -np.inf

    p_min = min(b_min, a_min)
    p_max = max(b_max, a_max)
    if p_min == np.inf:
        raise ValueError("No valid prices found to bin.")

    # Redondea el mínimo hacia abajo y el máximo hacia arriba a múltiplos de
    # ``bin_width`` para que los bordes de los bins caigan en la rejilla.
    p_min_rounded = np.floor(p_min / bin_width) * bin_width
    p_max_rounded = np.ceil(p_max / bin_width) * bin_width

    n_bins = int(np.round((p_max_rounded - p_min_rounded) / bin_width)) + 1
    bins = np.linspace(p_min_rounded, p_min_rounded + (n_bins - 1) * bin_width, n_bins)

    bids_binned = _bin_and_average(
        df_bids["AbsPrice"].to_numpy(),
        df_bids["Volume"].to_numpy(),
        bins,
        n_snapshots,
        "AbsPrice",
    )
    asks_binned = _bin_and_average(
        df_asks["AbsPrice"].to_numpy(),
        df_asks["Volume"].to_numpy(),
        bins,
        n_snapshots,
        "AbsPrice",
    )

    return bids_binned, asks_binned, (p_min_rounded, p_max_rounded)


def extract_single_snapshot(
    message_path: Path | str,
    orderbook_path: Path | str,
    target_time: float,
    levels: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Estado del libro en el primer instante con ``Time >= target_time``.

    Devuelve ``(df_bids, df_asks, actual_time)`` con columnas ``Price``
    (USD) y ``Volume`` (acciones).  Es **retrospectivo** por diseño: el
    snapshot corresponde a un instante *posterior* al objetivo.
    """
    df_msg_times = pd.read_csv(message_path, header=None, usecols=[0], names=["Time"])
    valid_indices = df_msg_times[df_msg_times["Time"] >= target_time].index
    if len(valid_indices) == 0:
        raise ValueError("Target time is beyond the available data.")

    target_idx = int(valid_indices[0])

    # Lee sólo la fila necesaria del orderbook (``skiprows`` salta las
    # anteriores y ``nrows`` limita la lectura a una).
    df_ob_row = pd.read_csv(orderbook_path, header=None, skiprows=target_idx, nrows=1)
    if df_ob_row.empty:
        raise ValueError("Target time is beyond the available data.")
    row_data = df_ob_row.iloc[0].to_numpy()
    _validate_levels(levels, len(row_data))

    ask_p_cols, ask_v_cols, bid_p_cols, bid_v_cols = _level_columns(levels)
    ask_p_raw = row_data[ask_p_cols]
    ask_v = row_data[ask_v_cols]
    bid_p_raw = row_data[bid_p_cols]
    bid_v = row_data[bid_v_cols]

    # Filtra los niveles vacíos (centinelas) **antes** de escalar los
    # precios: comparar tras dividir introduciría errores de coma flotante.
    valid_asks = ~_is_dummy(ask_p_raw)
    valid_bids = ~_is_dummy(bid_p_raw)

    df_asks = pd.DataFrame(
        {"Price": ask_p_raw[valid_asks] / PRICE_SCALE, "Volume": ask_v[valid_asks]}
    )
    df_bids = pd.DataFrame(
        {"Price": bid_p_raw[valid_bids] / PRICE_SCALE, "Volume": bid_v[valid_bids]}
    )

    return df_bids, df_asks, df_msg_times.iloc[target_idx]["Time"]


def extract_averaged_mid_price(
    message_path: Path | str,
    orderbook_path: Path | str,
    start_time: float | None = None,
    end_time: float | None = None,
    interval: float = 1.0,
) -> pd.DataFrame:
    """Precio medio del libro, promediado en buckets de ``interval`` segundos.

    Devuelve un frame con columnas ``TimeBin`` (segundos desde medianoche,
    múltiplo de ``interval``) y ``MidPrice`` (USD).  El promedio es por
    **snapshots** (no por duración de la ventana).
    """
    if interval <= 0:
        raise ValueError(f"interval debe ser > 0 (recibido {interval}).")

    df_time = pd.read_csv(message_path, header=None, usecols=[0], names=["Time"])
    df_ob = pd.read_csv(
        orderbook_path, header=None, usecols=[0, 2], names=["BestAsk", "BestBid"]
    )
    if len(df_ob) != len(df_time):
        raise ValueError(
            "Los ficheros de mensajes y orderbook deben tener el mismo número "
            f"de filas (mensajes: {len(df_time)}, orderbook: {len(df_ob)})."
        )

    df = pd.concat([df_time, df_ob], axis=1)

    if start_time is not None:
        df = df[df["Time"] >= start_time]
    if end_time is not None:
        df = df[df["Time"] <= end_time]
    if df.empty:
        raise ValueError("No data found in the specified time window.")

    valid_mask = ~_is_dummy(df["BestAsk"].to_numpy()) & ~_is_dummy(
        df["BestBid"].to_numpy()
    )
    df = df[valid_mask].copy()

    df["MidPrice"] = (df["BestAsk"] + df["BestBid"]) / (2.0 * PRICE_SCALE)

    # Redondea el tiempo al múltiplo de ``interval`` para formar los buckets.
    df["TimeBin"] = np.floor(df["Time"] / interval) * interval
    df_avg = df.groupby("TimeBin", observed=True)["MidPrice"].mean().reset_index()

    return df_avg


def aggregate_volume_bins(
    price: pd.Series | np.ndarray,
    volume: pd.Series | np.ndarray,
    bin_width: float,
) -> pd.DataFrame:
    """Agrega volúmenes en bins de precio de ancho ``bin_width``.

    Devuelve un frame con columnas ``BinLeftEdge`` (borde izquierdo del bin),
    ``Volume`` (volumen agregado) y ``BinMid`` (centro del bin).  Es el
    helper compartido por la re-agregación de la distribución absoluta y la
    agregación del snapshot: el índice del bin se calcula como
    ``floor(round(precio / bin_width, 6))`` para evitar los errores de coma
    flotante al truncar.
    """
    if not bin_width > 0:
        raise ValueError(f"bin_width debe ser > 0 (recibido {bin_width}).")

    prices = pd.Series(np.asarray(price, dtype=float)).reset_index(drop=True)
    volumes = pd.Series(np.asarray(volume, dtype=float)).reset_index(drop=True)
    if len(prices) != len(volumes):
        raise ValueError(
            "price y volume deben tener la misma longitud "
            f"({len(prices)} vs {len(volumes)})."
        )
    if prices.empty:
        return pd.DataFrame(columns=["BinLeftEdge", "Volume", "BinMid"])

    bin_idx = np.floor(np.round(prices / bin_width, 6))
    df = pd.DataFrame(
        {"BinLeftEdge": bin_idx * bin_width, "Volume": volumes.to_numpy()}
    )
    df_agg = df.groupby("BinLeftEdge", observed=True)["Volume"].sum().reset_index()
    df_agg["BinMid"] = df_agg["BinLeftEdge"] + bin_width / 2.0
    return df_agg

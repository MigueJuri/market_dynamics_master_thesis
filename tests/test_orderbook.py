"""Tests para ``thesis.orderbook``.

Los tests unitarios son **sintéticos**: construyen ficheros LOBSTER en
``tmp_path`` (mensajes + orderbook) con centinelas y precios conocidos, y no
requieren red ni datos reales.  El test marcado con ``integration`` hace un
*spot-check* sobre los ficheros LOBSTER reales (si están disponibles).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from thesis.orderbook import (
    DUMMY_NEG_PRICE,
    DUMMY_PRICE,
    MESSAGE_COLUMNS,
    PRICE_SCALE,
    aggregate_volume_bins,
    extract_absolute_lob_distribution,
    extract_averaged_mid_price,
    extract_relative_lob_distribution,
    extract_single_snapshot,
    load_message_times,
    lob_file_paths,
)

_SRC = Path(__file__).resolve().parents[1] / "src"

#: Número de niveles del libro sintético (cada nivel ocupa 4 columnas).
LEVELS = 3

#: Filas del libro sintético:
#: ``(tiempo, precios ask, precios bid, tamaños ask, tamaños bid)``.  Los
#: precios están en ticks (división por 10 000 para obtener USD).
ROWS: tuple[tuple, ...] = (
    # Libro completo.
    (
        34200.0,
        (1350100, 1350200, 1350300),
        (1349900, 1349800, 1349700),
        (100, 50, 25),
        (200, 300, 400),
    ),
    # Nivel 2 y 3 vacíos en el lado ask.
    (
        34201.0,
        (1350200, DUMMY_PRICE, DUMMY_PRICE),
        (1349800, 1349700, DUMMY_NEG_PRICE),
        (110, 0, 0),
        (210, 310, 0),
    ),
    # Top of book vacío: el snapshot entero se descarta.
    (
        34202.0,
        (DUMMY_PRICE, DUMMY_PRICE, DUMMY_PRICE),
        (1349900, 1349800, 1349700),
        (0, 0, 0),
        (220, 320, 420),
    ),
    # Nivel 3 vacío en el lado ask.
    (
        34203.0,
        (1350000, 1350100, DUMMY_PRICE),
        (1349900, 1349800, 1349600),
        (120, 60, 0),
        (230, 330, 440),
    ),
    # Libro completo.
    (
        34204.0,
        (1350300, 1350400, 1350500),
        (1350000, 1349900, 1349800),
        (130, 70, 35),
        (240, 340, 450),
    ),
    # Nivel 2 vacío en el lado ask y nivel 3 vacío en el lado bid.
    (
        34205.0,
        (1350400, DUMMY_PRICE, DUMMY_PRICE),
        (1350100, 1349900, DUMMY_NEG_PRICE),
        (140, 0, 0),
        (250, 350, 0),
    ),
)

#: Ventana temporal que cubre todas las filas.
START_TIME = 34200.0
END_TIME = 34205.0


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    """Cierra las figuras abiertas al terminar cada prueba."""
    yield
    plt.close("all")


def _write_lobster_files(
    directory: Path, rows: tuple[tuple, ...] = ROWS
) -> tuple[Path, Path]:
    """Escribe un par de ficheros LOBSTER sintéticos y devuelve sus rutas."""
    directory.mkdir(parents=True, exist_ok=True)
    message_path = directory / "synthetic_message.csv"
    orderbook_path = directory / "synthetic_orderbook.csv"

    with message_path.open("w") as fh:
        for i, row in enumerate(rows):
            time = row[0]
            fh.write(
                f"{time},{1 + i % 6},{1000 + i},100,"
                f"{row[1][0] / PRICE_SCALE:.4f},{-1 if i % 2 else 1}\n"
            )

    with orderbook_path.open("w") as fh:
        for row in rows:
            _, ask_prices, bid_prices, ask_sizes, bid_sizes = row
            values: list[float] = []
            for ask_price, ask_size, bid_price, bid_size in zip(
                ask_prices, ask_sizes, bid_prices, bid_sizes
            ):
                values.extend([ask_price, ask_size, bid_price, bid_size])
            fh.write(",".join(str(value) for value in values) + "\n")

    return message_path, orderbook_path


@pytest.fixture
def lobster_files(tmp_path: Path) -> tuple[Path, Path]:
    """Rutas de los ficheros LOBSTER sintéticos."""
    return _write_lobster_files(tmp_path / "lob")


# ----------------------------------------------------------------------
# Rutas y carga
# ----------------------------------------------------------------------


def test_lob_file_paths() -> None:
    """Las rutas siguen la convención de nombres de LOBSTER."""
    message_path, orderbook_path = lob_file_paths(
        symbol="SPY",
        date="2012-06-21",
        start="34200000",
        end="37800000",
        levels=50,
        directory=Path("/tmp/lobster"),
    )
    assert message_path == Path(
        "/tmp/lobster/SPY/SPY_2012-06-21_34200000_37800000_message_50.csv"
    )
    assert orderbook_path == Path(
        "/tmp/lobster/SPY/SPY_2012-06-21_34200000_37800000_orderbook_50.csv"
    )


def test_load_message_times(lobster_files: tuple[Path, Path]) -> None:
    """La columna ``Time`` se lee sin cargar el resto del fichero."""
    message_path, _ = lobster_files
    times = load_message_times(message_path)
    assert isinstance(times, pd.Series)
    assert times.tolist() == [row[0] for row in ROWS]


def test_message_columns_documentan_el_evento_de_lobster() -> None:
    """``Type`` es el ``Event`` de LOBSTER."""
    assert MESSAGE_COLUMNS == (
        "Time",
        "Type",
        "OrderID",
        "Size",
        "Price",
        "Direction",
    )


# ----------------------------------------------------------------------
# Distribución relativa
# ----------------------------------------------------------------------


def test_relative_lob_promedia_por_snapshot(
    lobster_files: tuple[Path, Path],
) -> None:
    """El volumen de cada bin es la suma del bin dividida por los snapshots."""
    message_path, orderbook_path = lobster_files
    bids, asks, max_range = extract_relative_lob_distribution(
        message_path, orderbook_path, START_TIME, END_TIME, LEVELS, 0.02
    )

    # 5 snapshots válidos (el de las 34202.0 tiene el top of book vacío).
    # Volumen pasivo de los niveles válidos: bids 4 050 y asks 840.
    assert bids["Volume"].sum() == pytest.approx(4050.0 / 5.0)
    assert asks["Volume"].sum() == pytest.approx(840.0 / 5.0)

    # Distancia máxima al precio medio: 0,035 USD (135,05 - 135,015).
    assert max_range == pytest.approx(0.035, abs=1e-9)

    # Los centinelas nunca llegan a la rejilla: ningún bin se sale del rango.
    assert bids["BinMid"].abs().max() <= max_range + 0.02
    assert asks["BinMid"].abs().max() <= max_range + 0.02


def test_relative_lob_descarta_centinelas(lobster_files: tuple[Path, Path]) -> None:
    """Los niveles vacíos (centinelas) no aportan volumen ni precio."""
    message_path, orderbook_path = lobster_files
    bids, _, _ = extract_relative_lob_distribution(
        message_path, orderbook_path, START_TIME, END_TIME, LEVELS, 0.02
    )

    # El snapshot de las 34202.0 se descarta entero (top of book vacío):
    # su volumen (220 + 320 + 420 = 960 en el bid) no aparece en el total.
    expected_bid_volume = (
        (200 + 300 + 400)
        + (210 + 310)
        + (230 + 330 + 440)
        + (240 + 340 + 450)
        + (250 + 350)
    )
    assert bids["Volume"].sum() * 5 == pytest.approx(expected_bid_volume)

    # El nivel 3 del bid de las 34201.0 y 34205.0 es centinela: no cuenta.
    assert bids["Volume"].sum() * 5 == pytest.approx(4050.0)


def test_bins_son_cerrados_por_la_izquierda(tmp_path: Path) -> None:
    """``pd.cut(..., right=False)``: un precio en el borde cae en su bin."""
    # Un snapshot con ask == bid (precio relativo exactamente 0.0) y niveles
    # 2 a ±0,0125: el borde 0.0 de la rejilla es exacto.
    rows = (
        (
            34200.0,
            (1350000, 1350125),
            (1350000, 1349875),
            (10, 5),
            (20, 30),
        ),
    )
    message_path, orderbook_path = _write_lobster_files(tmp_path / "bordes", rows)
    bids, asks, _ = extract_relative_lob_distribution(
        message_path, orderbook_path, START_TIME, END_TIME, 2, 0.01
    )

    bid_at_zero = bids[bids["Bin"].apply(lambda interval: interval.left == 0.0)]
    assert len(bid_at_zero) == 1
    # El nivel con precio relativo 0.0 (volumen 20) cae en el bin que
    # empieza en 0.0; con ``right=True`` caería en el bin anterior.
    assert bid_at_zero["Volume"].iloc[0] == pytest.approx(20.0)
    assert bid_at_zero["BinMid"].iloc[0] > 0.0

    ask_at_zero = asks[asks["Bin"].apply(lambda interval: interval.left == 0.0)]
    assert len(ask_at_zero) == 1
    assert ask_at_zero["Volume"].iloc[0] == pytest.approx(10.0)
    assert ask_at_zero["BinMid"].iloc[0] > 0.0


# ----------------------------------------------------------------------
# Distribución absoluta
# ----------------------------------------------------------------------


def test_absolute_lob_precios_en_usd(lobster_files: tuple[Path, Path]) -> None:
    """Los precios absolutos están en USD (ticks / 10 000)."""
    message_path, orderbook_path = lobster_files
    bids, asks, (p_min, p_max) = extract_absolute_lob_distribution(
        message_path, orderbook_path, START_TIME, END_TIME, LEVELS, 0.02
    )

    # El rango está redondeado a múltiplos de ``bin_width``.
    assert p_min == pytest.approx(np.floor(p_min / 0.02) * 0.02)
    assert p_max == pytest.approx(np.ceil(p_max / 0.02) * 0.02)
    assert p_min == pytest.approx(134.96)
    assert p_max == pytest.approx(135.06)

    # Los bins cubren el rango completo y el volumen se conserva.
    assert bids["Bin"].min().left == pytest.approx(p_min)
    assert asks["Bin"].max().right == pytest.approx(p_max)
    assert bids["Volume"].sum() == pytest.approx(4050.0 / 5.0)
    assert asks["Volume"].sum() == pytest.approx(840.0 / 5.0)


# ----------------------------------------------------------------------
# Snapshot único
# ----------------------------------------------------------------------


def test_snapshot_toma_la_primera_fila_posterior(
    lobster_files: tuple[Path, Path],
) -> None:
    """El snapshot corresponde a la primera fila con ``Time >= target``."""
    message_path, orderbook_path = lobster_files

    _, _, actual_time = extract_single_snapshot(
        message_path, orderbook_path, 34202.5, LEVELS
    )
    assert actual_time == 34203.0

    _, _, actual_time = extract_single_snapshot(
        message_path, orderbook_path, 34202.0, LEVELS
    )
    assert actual_time == 34202.0

    with pytest.raises(ValueError):
        extract_single_snapshot(message_path, orderbook_path, 99999.0, LEVELS)


def test_snapshot_filtra_centinelas_y_escala_precios(
    lobster_files: tuple[Path, Path],
) -> None:
    """Los niveles vacíos se descartan y los precios están en USD."""
    message_path, orderbook_path = lobster_files
    bids, asks, _ = extract_single_snapshot(
        message_path, orderbook_path, 34201.0, LEVELS
    )

    # Fila de las 34201.0: ask1 y bid1 válidos, ask2/ask3 vacíos, bid3 vacío.
    assert len(bids) == 2
    assert len(asks) == 1
    assert bids["Price"].tolist() == pytest.approx([134.98, 134.97])
    assert asks["Price"].tolist() == pytest.approx([135.02])
    assert bids["Volume"].tolist() == [210, 310]
    assert asks["Volume"].tolist() == [110]


# ----------------------------------------------------------------------
# Precio medio promediado
# ----------------------------------------------------------------------


def test_averaged_mid_price(lobster_files: tuple[Path, Path]) -> None:
    """El precio medio es (ask + bid) / 20 000 y se agrupa por buckets."""
    message_path, orderbook_path = lobster_files
    df_avg = extract_averaged_mid_price(
        message_path, orderbook_path, START_TIME, END_TIME, interval=5.0
    )

    assert list(df_avg.columns) == ["TimeBin", "MidPrice"]
    # Bucket 34200: 135.00, 135.00, 134.995 y 135.015 (el snapshot de las
    # 34202.0 se descarta por tener el top of book vacío).
    assert df_avg["TimeBin"].tolist() == [34200.0, 34205.0]
    assert df_avg["MidPrice"].iloc[0] == pytest.approx(135.0025)
    assert df_avg["MidPrice"].iloc[1] == pytest.approx(135.025)


def test_averaged_mid_price_usa_buckets_de_un_segundo(
    lobster_files: tuple[Path, Path],
) -> None:
    """Con ``interval=1`` hay un bucket por snapshot válido."""
    message_path, orderbook_path = lobster_files
    df_avg = extract_averaged_mid_price(
        message_path, orderbook_path, START_TIME, END_TIME, interval=1.0
    )
    assert df_avg["TimeBin"].tolist() == [34200.0, 34201.0, 34203.0, 34204.0, 34205.0]
    assert df_avg["MidPrice"].tolist() == pytest.approx(
        [135.0, 135.0, 134.995, 135.015, 135.025]
    )


# ----------------------------------------------------------------------
# Agregación en bins
# ----------------------------------------------------------------------


def test_aggregate_volume_bins_agrupa_y_suma() -> None:
    """Los precios se agrupan en bins de ancho ``bin_width``."""
    prices = pd.Series([135.00, 135.01, 135.04, 134.99])
    volumes = pd.Series([10, 20, 30, 40])

    agg = aggregate_volume_bins(prices, volumes, 0.02)
    assert list(agg.columns) == ["BinLeftEdge", "Volume", "BinMid"]
    assert agg["BinLeftEdge"].tolist() == pytest.approx([134.98, 135.00, 135.04])
    assert agg["Volume"].tolist() == [40.0, 30.0, 30.0]
    assert agg["BinMid"].tolist() == pytest.approx([134.99, 135.01, 135.05])


def test_aggregate_volume_bins_coincide_con_la_reagregacion(
    lobster_files: tuple[Path, Path],
) -> None:
    """Re-agregar un frame ya binned usa el mismo helper."""
    message_path, orderbook_path = lobster_files
    bids, _, _ = extract_relative_lob_distribution(
        message_path, orderbook_path, START_TIME, END_TIME, LEVELS, 0.02
    )
    target_bin_width = 0.05

    reagg = aggregate_volume_bins(bids["BinMid"], bids["Volume"], target_bin_width)

    # Cálculo manual equivalente (el del cuaderno original).
    bin_idx = np.floor(np.round(bids["BinMid"].astype(float) / target_bin_width, 6))
    bin_left_edge = bin_idx * target_bin_width
    expected = bids.groupby(bin_left_edge)["Volume"].sum().reset_index()
    expected.columns = ["BinLeftEdge", "Volume"]
    expected["BinMid"] = expected["BinLeftEdge"] + target_bin_width / 2.0

    pd.testing.assert_frame_equal(reagg, expected, check_dtype=False)


def test_aggregate_volume_bins_vacio() -> None:
    """Una entrada vacía devuelve un frame vacío con las columnas esperadas."""
    agg = aggregate_volume_bins(pd.Series(dtype=float), pd.Series(dtype=float), 0.02)
    assert agg.empty
    assert list(agg.columns) == ["BinLeftEdge", "Volume", "BinMid"]


# ----------------------------------------------------------------------
# Validación y robustez
# ----------------------------------------------------------------------


def test_validate_levels_rechaza_niveles_invalidos(
    lobster_files: tuple[Path, Path],
) -> None:
    """``levels`` debe ser >= 1 y caber en el fichero de orderbook."""
    message_path, orderbook_path = lobster_files

    with pytest.raises(ValueError, match="levels"):
        extract_relative_lob_distribution(
            message_path, orderbook_path, START_TIME, END_TIME, 0, 0.02
        )

    with pytest.raises(ValueError, match="levels"):
        extract_relative_lob_distribution(
            message_path, orderbook_path, START_TIME, END_TIME, LEVELS + 1, 0.02
        )

    with pytest.raises(ValueError, match="levels"):
        extract_single_snapshot(message_path, orderbook_path, START_TIME, LEVELS + 1)


def test_ventana_sin_datos_falla(tmp_path: Path) -> None:
    """Una ventana temporal vacía lanza ``ValueError``."""
    message_path, orderbook_path = _write_lobster_files(tmp_path / "vacio")
    with pytest.raises(ValueError):
        extract_relative_lob_distribution(
            message_path, orderbook_path, 0.0, 1.0, LEVELS, 0.02
        )


def test_truncar_la_ventana_no_altera_el_resultado(
    tmp_path: Path, lobster_files: tuple[Path, Path]
) -> None:
    """Extraer una sub-ventana no depende de las filas fuera de ella."""
    message_path, orderbook_path = lobster_files

    full_bids, full_asks, full_range = extract_relative_lob_distribution(
        message_path, orderbook_path, 34200.0, 34203.0, LEVELS, 0.02
    )

    # Mismo fichero, pero con las filas fuera de la ventana eliminadas.
    trunc_rows = tuple(row for row in ROWS if row[0] <= 34203.0)
    trunc_message, trunc_orderbook = _write_lobster_files(
        tmp_path / "truncado", trunc_rows
    )
    trunc_bids, trunc_asks, trunc_range = extract_relative_lob_distribution(
        trunc_message, trunc_orderbook, 34200.0, 34203.0, LEVELS, 0.02
    )

    assert trunc_range == full_range
    pd.testing.assert_frame_equal(trunc_bids, full_bids)
    pd.testing.assert_frame_equal(trunc_asks, full_asks)


# ----------------------------------------------------------------------
# Importación sin efectos
# ----------------------------------------------------------------------


def test_importar_thesis_orderbook_no_lee_ni_dibuja(tmp_path: Path) -> None:
    """Importar el módulo no lee ficheros ni dibuja figuras."""
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "from thesis import orderbook\n"
        "assert not plt.get_fignums(), 'import drew a figure'\n"
        "print('ok')\n"
    )
    env = {
        **os.environ,
        "THESIS_DATA_DIR": str(tmp_path),
        "PYTHONPATH": str(_SRC),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------
# Integración (datos LOBSTER reales; se omite por defecto)
# ----------------------------------------------------------------------


_LOBSTER_DIR = Path(
    os.environ.get(
        "THESIS_LOB_DIR",
        "/home/miguel/GoogleDriveOffline/Data/OrderBook",
    )
)
_REAL_MESSAGE = _LOBSTER_DIR / "SPY" / "SPY_2012-06-21_34200000_37800000_message_50.csv"
_REAL_ORDERBOOK = (
    _LOBSTER_DIR / "SPY" / "SPY_2012-06-21_34200000_37800000_orderbook_50.csv"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not (_REAL_MESSAGE.exists() and _REAL_ORDERBOOK.exists()),
    reason="ficheros LOBSTER reales no disponibles",
)
def test_spot_check_con_datos_reales() -> None:
    """El módulo reproduce el resultado del cuaderno original sobre datos reales."""
    bids, asks, max_range = extract_relative_lob_distribution(
        _REAL_MESSAGE, _REAL_ORDERBOOK, 34200.0, 34230.0, 50, 0.02
    )
    assert max_range == pytest.approx(134.59, abs=0.01)
    assert not bids.empty and not asks.empty
    assert bids["Volume"].sum() > 0.0

    bids_abs, asks_abs, (p_min, p_max) = extract_absolute_lob_distribution(
        _REAL_MESSAGE, _REAL_ORDERBOOK, 34200.0, 34230.0, 50, 0.02
    )
    assert p_min == pytest.approx(1.04)
    assert p_max == pytest.approx(172.38)
    assert bids_abs["Volume"].sum() > 0.0
    assert asks_abs["Volume"].sum() > 0.0

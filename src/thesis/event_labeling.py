"""Etiquetado de eventos y triple barrera para backtests.

Este módulo contiene utilidades para construir eventos de trading,
detectar señales técnicas (cruces de medias móviles exponenciales),
definir barreras verticales y aplicar etiquetas según qué barrera se
activa primero (profit-take, stop-loss o barrera vertical).

La API sigue la filosofía de López de Prado (*Advances in Financial
Machine Learning*): cada evento es una entrada independiente con un
instante inicial, una barrera vertical y barreras dinámicas
proporcionales a una estimación de volatilidad.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def get_ewma_crossover_events(
    close: pd.Series,
    fast_span: int,
    slow_span: int,
    burn_in: Optional[int] = None,
) -> pd.DataFrame:
    """Detecta eventos de cruce entre dos medias exponenciales.

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre.
    fast_span : int
        Span de la EWMA rápida.
    slow_span : int
        Span de la EWMA lenta.
    burn_in : int, opcional
        Número de barras iniciales a descartar para evitar artefactos de
        calentamiento de las EWMA.  Por defecto es ``slow_span``.

    Devuelve
    --------
    pd.DataFrame
        DataFrame indexado por la fecha del cruce con columnas:
        ``fast_ewma``, ``slow_ewma`` y ``side`` (+1 alcista, -1 bajista).
    """
    if burn_in is None:
        burn_in = slow_span

    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()

    fast = fast.iloc[burn_in:]
    slow = slow.iloc[burn_in:]

    diff = fast - slow
    diff_prev = diff.shift(1)

    bullish = (diff_prev <= 0) & (diff > 0)
    bearish = (diff_prev >= 0) & (diff < 0)

    df = pd.DataFrame({"fast_ewma": fast, "slow_ewma": slow})
    df["side"] = np.where(bullish, 1, np.where(bearish, -1, 0))
    return df[df["side"] != 0]


def ewma_volatility_target(
    returns: pd.Series,
    span: int = 20,
    min_target: Optional[float] = None,
) -> pd.Series:
    """Estimación EWM de volatilidad diaria para usar como target de barrera.

    Parámetros
    ----------
    returns : pd.Series
        Serie de retornos (normalmente simples: ``pct_change()``).
    span : int
        Span de la EWMA de volatilidad.
    min_target : float, opcional
        Si se indica, descarta observaciones con target menor o igual a
        este valor (útil para filtrar eventos con volatilidad insignificante).

    Devuelve
    --------
    pd.Series
        Volatilidad EWM con nombre ``'trgt'``.
    """
    trgt = returns.ewm(span=span, adjust=False).std()
    trgt.name = "trgt"
    if min_target is not None:
        trgt = trgt[trgt > min_target]
    return trgt


def add_vertical_barrier(
    t_events: Union[pd.Index, Sequence],
    close_index: pd.Index,
    num_bars: int,
) -> pd.Series:
    """Crea una barrera vertical ``num_bars`` barras después de cada evento.

    Parámetros
    ----------
    t_events : pd.Index o secuencia
        Instantes de entrada de los eventos.
    close_index : pd.Index
        Índice temporal de la serie de precios.
    num_bars : int
        Distancia en barras hasta la barrera vertical.

    Devuelve
    --------
    pd.Series
        Serie indexada por ``t_events`` con el timestamp de la barrera
        vertical.  Los eventos cuya posición no se encuentre en
        ``close_index`` se descartan.
    """
    t_events = pd.Index(t_events).sort_values()
    event_pos = close_index.get_indexer(t_events, method="backfill")
    valid = event_pos != -1
    last_pos = len(close_index) - 1
    barrier_pos = np.minimum(event_pos[valid] + num_bars, last_pos)
    return pd.Series(close_index[barrier_pos], index=t_events[valid])


def apply_barriers(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: Sequence[float],
    return_type: str = "simple",
) -> pd.DataFrame:
    """Calcula el primer toque de profit-take y stop-loss para cada evento.

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre o, si ``return_type='log'``, serie de
        log-precios.
    events : pd.DataFrame
        Tabla de eventos con columnas ``t1`` (barrera vertical),
        ``trgt`` (target de volatilidad) y ``side`` (+1 largo, -1 corto).
    pt_sl : secuencia de dos floats
        Multiplicadores simétricos ``[pt, sl]`` del target para definir
        las barreras.  Un valor ``<= 0`` desactiva la barrera
        correspondiente.
    return_type : {'simple', 'log'}
        Tipo de retorno usado para las barreras.  Con ``'simple'`` se
        interpreta ``close`` como precio y se calcula
        ``(path / entry - 1) * side``.  Con ``'log'`` se interpreta
        ``close`` como log-precio y se calcula
        ``(path - entry) * side``; ``trgt`` debe entonces ser la
        volatilidad de retornos logarítmicos, obteniendo barreras
        simétricas en el espacio logarítmico.

    Devuelve
    --------
    pd.DataFrame
        DataFrame con columnas ``pt`` y ``sl`` (timestamps de primer
        toque, ``NaT`` si no se toca).
    """
    if return_type not in {"simple", "log"}:
        raise ValueError(
            f"return_type debe ser 'simple' o 'log', se recibió '{return_type}'."
        )

    out = pd.DataFrame(
        index=events.index,
        columns=["pt", "sl"],
        dtype="datetime64[ns]",
    )

    for loc, ev in events.iterrows():
        try:
            loc_pos = close.index.get_loc(loc)
        except KeyError:
            continue

        end_time = ev["t1"] if pd.notna(ev["t1"]) else close.index[-1]
        end_pos = close.index.get_indexer([end_time], method="bfill")[0]
        start_pos = loc_pos + 1

        if start_pos > end_pos:
            continue

        path = close.iloc[start_pos : end_pos + 1]
        if len(path) < 2:
            continue

        entry = path.iloc[0]
        if return_type == "simple":
            rets = (path.iloc[1:] / entry - 1.0) * ev["side"]
        else:
            rets = (path.iloc[1:] - entry) * ev["side"]

        if pt_sl[0] > 0:
            hit = rets[rets >= pt_sl[0] * ev["trgt"]]
            if not hit.empty:
                out.loc[loc, "pt"] = hit.index[0]

        if pt_sl[1] > 0:
            hit = rets[rets <= -pt_sl[1] * ev["trgt"]]
            if not hit.empty:
                out.loc[loc, "sl"] = hit.index[0]

    out["pt"] = pd.to_datetime(out["pt"])
    out["sl"] = pd.to_datetime(out["sl"])
    return out


def get_events(
    close: pd.Series,
    t_events: Union[pd.Index, Sequence],
    pt_sl: Sequence[float],
    trgt: pd.Series,
    min_ret: float,
    t1: pd.Series,
    side: pd.Series,
    return_type: str = "simple",
) -> pd.DataFrame:
    """Construye la tabla de eventos con etiqueta triple barrera.

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre o log-precios (según ``return_type``).
    t_events : pd.Index o secuencia
        Instantes de entrada de los eventos.
    pt_sl : secuencia de dos floats
        Multiplicadores ``[pt, sl]`` para las barreras.
    trgt : pd.Series
        Target de volatilidad indexado por evento.
    min_ret : float
        Mínimo target aceptable para conservar un evento.
    t1 : pd.Series
        Barrera vertical indexada por evento.
    side : pd.Series
        Dirección de la señal (+1 largo, -1 corto) indexada por evento.
    return_type : {'simple', 'log'}
        Ver :func:`apply_barriers`.

    Devuelve
    --------
    pd.DataFrame
        Tabla indexada por evento con columnas ``t1``, ``trgt``,
        ``side``, ``t_hit`` (primera barrera tocada) y ``label``
        (+1 PT, -1 SL, 0 vertical).
    """
    t_events = pd.Index(t_events).sort_values()

    # Alineamos target y side; descartamos eventos sin target válido.
    trgt_ = trgt.reindex(t_events).dropna()
    trgt_ = trgt_[trgt_ > min_ret]

    ev = pd.DataFrame(index=trgt_.index)
    ev["t1"] = t1.reindex(ev.index)
    ev["trgt"] = trgt_
    ev["side"] = side.reindex(ev.index).fillna(1.0)

    hits = apply_barriers(close, ev, pt_sl, return_type=return_type)

    # t_hit = earliest of {t1, pt, sl}
    candidates = pd.concat([ev["t1"], hits["pt"], hits["sl"]], axis=1)
    candidates.columns = ["t1", "pt", "sl"]
    ev["t_hit"] = candidates.stack().groupby(level=0).min()

    def _label(row: pd.Series) -> int:
        if pd.isna(row["t_hit"]):
            return 0
        if pd.notna(hits.loc[row.name, "pt"]) and row["t_hit"] == hits.loc[row.name, "pt"]:
            return int(row["side"])
        if pd.notna(hits.loc[row.name, "sl"]) and row["t_hit"] == hits.loc[row.name, "sl"]:
            return -int(row["side"])
        return 0

    ev["label"] = ev.apply(_label, axis=1)
    return ev


def compute_overlap_weights(labeled: pd.DataFrame) -> pd.Series:
    """Peso de solapamiento de eventos según López de Prado (AFML §4.3).

    Cada evento recibe un peso en ``(0, 1]`` igual a la fracción de su
    ventana ``[t_i, t_hit_i]`` que no está cubierta por el siguiente
    evento.  Si no hay solapamiento, el peso es 1.
    """
    ev = labeled[["t_hit"]].copy().sort_index()
    ev["t_next"] = ev.index.to_series().shift(-1).values

    weights = []
    for ts, row in ev.iterrows():
        if pd.isna(row["t_hit"]):
            weights.append(1.0)
            continue
        window = (row["t_hit"] - ts).total_seconds()
        if window <= 0:
            weights.append(1.0)
            continue
        if pd.notna(row["t_next"]) and row["t_next"] < row["t_hit"]:
            overlap = (row["t_hit"] - row["t_next"]).total_seconds()
            w = max(0.0, 1.0 - overlap / window)
        else:
            w = 1.0
        weights.append(w)

    return pd.Series(weights, index=ev.index, name="weight")

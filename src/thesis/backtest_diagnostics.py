"""Diagnóstico y validación de backtests de pair trading.

Este módulo resume el resultado de
:func:`thesis.pair_trading.pair_trading_backtest` en métricas de hedge
ratio, ejecución, retorno y riesgo, y ejecuta una batería de
comprobaciones automáticas que detectan los errores más habituales de un
backtest:

- look-ahead en la estimación del hedge ratio;
- desalineación entre la posición señalada y la posición aplicada;
- beta efectivo incoherente con la posición (por ejemplo, reutilizar el
  beta de la última entrada para toda la serie);
- turnover que no refleja los cambios de peso;
- contribuciones que no suman el retorno neto;
- equity o drawdown incoherentes con la serie de retornos.

Las funciones son puras: reciben el DataFrame ``results`` producido por el
backtest y devuelven dataclasses inmutables con las métricas.  Las
comprobaciones de causalidad que necesitan los precios se resuelven en
:func:`verify_hedge_is_causal`, que recalcula el estimador walk-forward
sobre una muestra de fechas y lo compara con el almacenado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# =============================================================================
# Estructuras de resultado
# =============================================================================


@dataclass(frozen=True)
class HedgeDiagnostics:
    """Estadísticas descriptivas del hedge ratio walk-forward.

    Attributes
    ----------
    beta_mean, beta_median, beta_std : float
        Moments del hedge ratio recalculado cada día.
    beta_min, beta_max : float
        Extremos del hedge ratio.
    beta_p01, beta_p25, beta_p75, beta_p99 : float
        Percentiles del hedge ratio.
    alpha_mean, alpha_std : float
        Momentos del intercepto walk-forward.
    beta_changes : int
        Número de cambios del hedge ratio superiores al umbral indicado.
    beta_out_of_bounds : int
        Días con ``|beta|`` por encima de ``max_abs_beta`` (0 si no se fijó).
    effective_beta_unique : int
        Número de valores distintos no nulos del hedge ratio efectivo.
    effective_beta_constant : bool
        ``True`` si todas las operaciones usaron exactamente el mismo beta.
        Es la señal más directa del bug de "último beta reutilizado".
    effective_beta_block_consistent : bool
        ``True`` si el beta efectivo es constante dentro de cada bloque
        continuo con posición (invariante del modo ``on_entry``).
    """

    beta_mean: float
    beta_median: float
    beta_std: float
    beta_min: float
    beta_max: float
    beta_p01: float
    beta_p25: float
    beta_p75: float
    beta_p99: float
    alpha_mean: float
    alpha_std: float
    beta_changes: int
    beta_out_of_bounds: int
    effective_beta_unique: int
    effective_beta_constant: bool
    effective_beta_block_consistent: bool

    def as_dict(self) -> dict[str, Any]:
        """Devuelve las métricas como diccionario plano."""
        return {
            "beta_mean": self.beta_mean,
            "beta_median": self.beta_median,
            "beta_std": self.beta_std,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "beta_p01": self.beta_p01,
            "beta_p25": self.beta_p25,
            "beta_p75": self.beta_p75,
            "beta_p99": self.beta_p99,
            "alpha_mean": self.alpha_mean,
            "alpha_std": self.alpha_std,
            "beta_changes": self.beta_changes,
            "beta_out_of_bounds": self.beta_out_of_bounds,
            "effective_beta_unique": self.effective_beta_unique,
            "effective_beta_constant": self.effective_beta_constant,
            "effective_beta_block_consistent": self.effective_beta_block_consistent,
        }


@dataclass(frozen=True)
class SignalDiagnostics:
    """Métricas de generación de señales y ejecución.

    Attributes
    ----------
    z_nan_count : int
        Observaciones con z-score ausente.
    z_nan_after_warmup : int
        Observaciones con z-score ausente una vez superado el warm-up,
        indicativas de desviación estándar nula o de datos insuficientes.
    z_max_abs : float
        Máximo valor absoluto del z-score.
    flat_fraction : float
        Fracción de tiempo con la estrategia en cash.
    entries, exits : int
        Número de entradas desde cash y de cierres (incluidos los flips de
        signo, que cierran una operación y abren otra).
    round_trips : int
        Operaciones cerradas.
    open_operations : int
        Operaciones todavía abiertas al final del periodo.
    long_entries, short_entries : int
        Entradas en largo y en corto del spread.
    mean_holding_days, median_holding_days : float
        Duración media y mediana de las operaciones cerradas.
    max_holding_days : int
        Duración de la operación más larga.
    max_consecutive_holding : int
        Mayor número de sesiones consecutivas con posición.
    entry_threshold_ok : bool
        ``True`` si todas las entradas_superaron ``|z| > entry_z``.
    exit_threshold_ok : bool
        ``True`` si todas las salidas por z-score cruzaron ``exit_z``.
    """

    z_nan_count: int
    z_nan_after_warmup: int
    z_max_abs: float
    flat_fraction: float
    entries: int
    exits: int
    round_trips: int
    open_operations: int
    long_entries: int
    short_entries: int
    mean_holding_days: float
    median_holding_days: float
    max_holding_days: int
    max_consecutive_holding: int
    entry_threshold_ok: bool
    exit_threshold_ok: bool

    def as_dict(self) -> dict[str, Any]:
        """Devuelve las métricas como diccionario plano."""
        return {
            "z_nan_count": self.z_nan_count,
            "z_nan_after_warmup": self.z_nan_after_warmup,
            "z_max_abs": self.z_max_abs,
            "flat_fraction": self.flat_fraction,
            "entries": self.entries,
            "exits": self.exits,
            "round_trips": self.round_trips,
            "open_operations": self.open_operations,
            "long_entries": self.long_entries,
            "short_entries": self.short_entries,
            "mean_holding_days": self.mean_holding_days,
            "median_holding_days": self.median_holding_days,
            "max_holding_days": self.max_holding_days,
            "max_consecutive_holding": self.max_consecutive_holding,
            "entry_threshold_ok": self.entry_threshold_ok,
            "exit_threshold_ok": self.exit_threshold_ok,
        }


@dataclass(frozen=True)
class ReturnDiagnostics:
    """Métricas de retorno, notional y costes.

    Attributes
    ----------
    total_turnover : float
        Turnover nocional acumulado.
    mean_daily_turnover : float
        Turnover nocional medio por sesión.
    turnover_per_operation : float
        Turnover medio por operación (entrada más salida).
    total_costs : float
        Costes acumulados en unidades de retorno.
    costs_pct_of_gross : float
        Costes como porcentaje del retorno bruto absoluto del periodo.
    mean_gross_notional, max_gross_notional : float
        Notional bruto medio y máximo (``1 + |beta|``).
    gross_notional_p01, gross_notional_p50, gross_notional_p99 : float
        Percentiles del notional bruto.
    mean_exposure_a, mean_exposure_b : float
        Exposición media absoluta a cada activo.
    gross_total, net_total : float
        Suma de retornos brutos y netos del periodo.
    contribution_max_error : float
        Máximo error entre ``gross_return - costs`` y ``strategy_ret``.
    """

    total_turnover: float
    mean_daily_turnover: float
    turnover_per_operation: float
    total_costs: float
    costs_pct_of_gross: float
    mean_gross_notional: float
    max_gross_notional: float
    gross_notional_p01: float
    gross_notional_p50: float
    gross_notional_p99: float
    mean_exposure_a: float
    mean_exposure_b: float
    gross_total: float
    net_total: float
    contribution_max_error: float

    def as_dict(self) -> dict[str, Any]:
        """Devuelve las métricas como diccionario plano."""
        return {
            "total_turnover": self.total_turnover,
            "mean_daily_turnover": self.mean_daily_turnover,
            "turnover_per_operation": self.turnover_per_operation,
            "total_costs": self.total_costs,
            "costs_pct_of_gross": self.costs_pct_of_gross,
            "mean_gross_notional": self.mean_gross_notional,
            "max_gross_notional": self.max_gross_notional,
            "gross_notional_p01": self.gross_notional_p01,
            "gross_notional_p50": self.gross_notional_p50,
            "gross_notional_p99": self.gross_notional_p99,
            "mean_exposure_a": self.mean_exposure_a,
            "mean_exposure_b": self.mean_exposure_b,
            "gross_total": self.gross_total,
            "net_total": self.net_total,
            "contribution_max_error": self.contribution_max_error,
        }


@dataclass(frozen=True)
class RiskMetrics:
    """Métricas de rendimiento y riesgo del periodo evaluado.

    Attributes
    ----------
    total_return : float
        Retorno total acumulado.
    cagr : float
        Tasa de crecimiento anual compuesta.
    volatility : float
        Volatilidad anualizada.
    sharpe, sortino, calmar : float
        Ratios de ajustado por riesgo.
    max_drawdown : float
        Drawdown máximo (valor negativo).
    max_drawdown_duration : int
        Número de sesiones que duró el máximo drawdown.
    var_95, cvar_95 : float
        Valor en riesgo y pérdida esperada condicional al 5%.
    best_trade, worst_trade : float
        Mejor y peor operación cerrada.
    profit_factor : float
        Ganancia total dividida por pérdida total absoluta.
    win_rate : float
        Fracción de operaciones cerradas con beneficio.
    avg_gain, avg_loss : float
        Ganancia y pérdida medias por operación.
    n_trades : int
        Número de operaciones cerradas.
    """

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    best_trade: float
    worst_trade: float
    profit_factor: float
    win_rate: float
    avg_gain: float
    avg_loss: float
    n_trades: int

    def as_dict(self) -> dict[str, Any]:
        """Devuelve las métricas como diccionario plano."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "n_trades": self.n_trades,
        }


@dataclass(frozen=True)
class ValidationCheck:
    """Resultado de una comprobación automática.

    Attributes
    ----------
    name : str
        Identificador de la comprobación.
    passed : bool
        ``True`` si la comprobación pasa.
    detail : str
        Descripción del resultado.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    """Conjunto de comprobaciones automáticas del backtest.

    Attributes
    ----------
    checks : tuple[ValidationCheck, ...]
        Comprobaciones ejecutadas.
    """

    checks: tuple[ValidationCheck, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """``True`` si todas las comprobaciones pasan."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[ValidationCheck, ...]:
        """Comprobaciones que fallan."""
        return tuple(check for check in self.checks if not check.passed)

    def as_dataframe(self) -> pd.DataFrame:
        """Devuelve las comprobaciones como tabla legible."""
        return pd.DataFrame(
            [
                {"check": check.name, "passed": check.passed, "detail": check.detail}
                for check in self.checks
            ]
        )

    def summary(self) -> str:
        """Resumen de una línea con el número de fallos."""
        n_failed = len(self.failures)
        return f"{len(self.checks) - n_failed}/{len(self.checks)} comprobaciones OK" + (
            "" if n_failed == 0 else f" ({n_failed} fallos)"
        )


@dataclass(frozen=True)
class PairBacktestDiagnostics:
    """Diagnóstico completo de un backtest de pair trading.

    Attributes
    ----------
    trades : pd.DataFrame
        Una fila por operación con entrada, salida, duración y resultado.
    hedge : HedgeDiagnostics
        Estadísticas del hedge ratio.
    signal : SignalDiagnostics
        Estadísticas de señales y ejecución.
    returns : ReturnDiagnostics
        Estadísticas de retorno, notional y costes.
    risk : RiskMetrics
        Métricas de rendimiento y riesgo.
    validation : ValidationReport
        Resultado de las comprobaciones automáticas.
    """

    trades: pd.DataFrame
    hedge: HedgeDiagnostics
    signal: SignalDiagnostics
    returns: ReturnDiagnostics
    risk: RiskMetrics
    validation: ValidationReport


# =============================================================================
# Extracción de operaciones
# =============================================================================


TRADE_COLUMNS = [
    "entry_date",
    "exit_date",
    "direction",
    "entry_z",
    "exit_z",
    "entry_beta",
    "exit_beta",
    "days",
    "gross_return",
    "costs",
    "net_return",
    "exit_reason",
    "is_open",
]


def extract_pair_trades(results: pd.DataFrame) -> pd.DataFrame:
    """Extrae las operaciones individuales del backtest.

    Recorre los cambios de ``results["position"]`` y empareja cada entrada
    con su salida.  Una operación abierta al final del periodo se marca con
    ``is_open=True`` y ``exit_reason="end"``.

    Parameters
    ----------
    results : pd.DataFrame
        Salida de :func:`thesis.pair_trading.pair_trading_backtest`.

    Returns
    -------
    pd.DataFrame
        Una fila por operación con las columnas de :data:`TRADE_COLUMNS`.
    """
    empty = pd.DataFrame(columns=TRADE_COLUMNS)
    if results.empty or "position" not in results.columns:
        return empty

    position = results["position"].to_numpy()
    z_score = results["z_score"].to_numpy()
    effective_beta = results["effective_beta"].to_numpy()
    exit_reason = (
        results["exit_reason"].to_numpy()
        if "exit_reason" in results.columns
        else np.array([""] * len(results), dtype=object)
    )
    index = results.index

    trades: list[dict[str, Any]] = []
    start: int | None = None
    direction = 0.0

    for i in range(len(results)):
        pos = position[i]
        previous = position[i - 1] if i > 0 else 0.0

        if pos != 0 and pos != previous:
            # Nueva operación: entrada desde cash o flip de signo.
            if start is not None:
                trades.append(
                    _build_trade(
                        results,
                        start,
                        i,
                        direction,
                        z_score,
                        effective_beta,
                        exit_reason,
                        index,
                        is_open=False,
                        close_reason="flip",
                    )
                )
            start = i
            direction = float(pos)
        elif pos == 0 and start is not None:
            trades.append(
                _build_trade(
                    results,
                    start,
                    i,
                    direction,
                    z_score,
                    effective_beta,
                    exit_reason,
                    index,
                    is_open=False,
                )
            )
            start = None
            direction = 0.0

    if start is not None:
        trades.append(
            _build_trade(
                results,
                start,
                len(results) - 1,
                direction,
                z_score,
                effective_beta,
                exit_reason,
                index,
                is_open=True,
            )
        )

    if not trades:
        return empty

    return pd.DataFrame(trades, columns=TRADE_COLUMNS)


def _build_trade(
    results: pd.DataFrame,
    start: int,
    end: int,
    direction: float,
    z_score: np.ndarray,
    effective_beta: np.ndarray,
    exit_reason: np.ndarray,
    index: pd.Index,
    is_open: bool,
    close_reason: str | None = None,
) -> dict[str, Any]:
    """Construye el registro de una operación a partir de sus índices."""
    gross = results["gross_return"].to_numpy()
    costs = results["costs"].to_numpy()
    position = results["position"].to_numpy()

    # La exposición de la posición decidida en ``start`` aparece en los
    # retornos de las filas start+1 ... end (en ``end`` aún seguía abierta).
    exposure_slice = slice(start + 1, end + 1) if end > start else slice(0, 0)
    gross_total = float(np.sum(gross[exposure_slice]))
    cost_total = float(np.sum(costs[start : end + 1]))

    # Días señalados con esta dirección (coincide con ``holding_days``).
    days = int(np.sum(position[start : end + 1] == direction))

    reason = str(exit_reason[end]) if end < len(exit_reason) else ""
    if is_open:
        reason = "end"
    elif close_reason is not None:
        reason = close_reason

    return {
        "entry_date": index[start],
        "exit_date": index[end],
        "direction": direction,
        "entry_z": float(z_score[start]),
        "exit_z": float(z_score[end]),
        "entry_beta": float(effective_beta[start]),
        "exit_beta": float(effective_beta[end]),
        "days": days,
        "gross_return": gross_total,
        "costs": cost_total,
        "net_return": gross_total - cost_total,
        "exit_reason": reason,
        "is_open": is_open,
    }


# =============================================================================
# Métricas por bloque
# =============================================================================


def _operation_starts(position: pd.Series) -> list[int]:
    """Índices de inicio de cada operación.

    Una operación empieza cuando la estrategia pasa de cash a una posición o
    cuando la posición cambia de signo (un flip abre una operación nueva con
    su propio hedge ratio).
    """
    pos = position.to_numpy()
    starts: list[int] = []
    previous = 0.0
    for i, value in enumerate(pos):
        if value != 0 and value != previous:
            starts.append(i)
        previous = value
    return starts


def _operation_lengths(position: pd.Series) -> list[int]:
    """Duración en sesiones de cada operación (días con la posición abierta)."""
    pos = position.to_numpy()
    lengths: list[int] = []
    for start in _operation_starts(position):
        direction = pos[start]
        end = start
        while end + 1 < len(pos) and pos[end + 1] == direction:
            end += 1
        lengths.append(end - start + 1)
    return lengths


def _effective_beta_block_consistent(position: pd.Series, beta: pd.Series) -> bool:
    """Comprueba que el beta efectivo es constante dentro de cada operación."""
    pos = position.to_numpy()
    betas = beta.to_numpy()
    for start in _operation_starts(position):
        direction = pos[start]
        end = start
        while end + 1 < len(pos) and pos[end + 1] == direction:
            end += 1
        values = {
            round(float(b), 12) for b in betas[start : end + 1] if not np.isnan(b)
        }
        if len(values) > 1:
            return False
    return True


def compute_hedge_diagnostics(
    results: pd.DataFrame,
    max_abs_beta: float | None = None,
    beta_change_threshold: float = 0.01,
) -> HedgeDiagnostics:
    """Calcula estadísticas descriptivas del hedge ratio.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    max_abs_beta : float | None
        Si se indica, cuenta los días con ``|beta|`` por encima del límite.
    beta_change_threshold : float
        Umbral a partir del cual se considera que el beta ha cambiado.

    Returns
    -------
    HedgeDiagnostics
        Estadísticas del hedge ratio walk-forward y del beta efectivo.
    """
    beta = results["beta"].dropna()
    alpha = results["alpha"].dropna()
    effective = results.loc[results["position"] != 0, "effective_beta"].dropna()

    if beta.empty:
        beta_stats = {k: np.nan for k in ("mean", "median", "std", "min", "max")}
        quantiles = {q: np.nan for q in ("p01", "p25", "p75", "p99")}
    else:
        beta_stats = {
            "mean": float(beta.mean()),
            "median": float(beta.median()),
            "std": float(beta.std()),
            "min": float(beta.min()),
            "max": float(beta.max()),
        }
        quantiles = {
            "p01": float(beta.quantile(0.01)),
            "p25": float(beta.quantile(0.25)),
            "p75": float(beta.quantile(0.75)),
            "p99": float(beta.quantile(0.99)),
        }

    beta_diff = results["beta"].diff().abs()
    beta_changes = int((beta_diff > beta_change_threshold).sum())

    out_of_bounds = 0
    if max_abs_beta is not None:
        out_of_bounds = int((results["beta"].abs() > max_abs_beta).sum())

    unique_effective = int(effective.round(12).nunique())

    return HedgeDiagnostics(
        beta_mean=beta_stats["mean"],
        beta_median=beta_stats["median"],
        beta_std=beta_stats["std"],
        beta_min=beta_stats["min"],
        beta_max=beta_stats["max"],
        beta_p01=quantiles["p01"],
        beta_p25=quantiles["p25"],
        beta_p75=quantiles["p75"],
        beta_p99=quantiles["p99"],
        alpha_mean=float(alpha.mean()) if not alpha.empty else np.nan,
        alpha_std=float(alpha.std()) if not alpha.empty else np.nan,
        beta_changes=beta_changes,
        beta_out_of_bounds=out_of_bounds,
        effective_beta_unique=unique_effective,
        effective_beta_constant=unique_effective <= 1,
        effective_beta_block_consistent=_effective_beta_block_consistent(
            results["position"], results["effective_beta"]
        ),
    )


def compute_signal_diagnostics(
    results: pd.DataFrame,
    entry_z: float,
    exit_z: float,
    hedge_window: int,
) -> SignalDiagnostics:
    """Calcula métricas de señales, ejecución y duración de operaciones.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    entry_z, exit_z : float
        Umbrales usados en el backtest.
    hedge_window : int
        Ventana de estimación del hedge ratio, usada para identificar el
        warm-up.

    Returns
    -------
    SignalDiagnostics
        Métricas de señales y ejecución.
    """
    position = results["position"]
    z_score = results["z_score"]
    n = len(results)

    z_nan = int(z_score.isna().sum())
    z_nan_after_warmup = int(z_score.iloc[hedge_window:].isna().sum())
    z_valid = z_score.dropna()
    z_max_abs = float(z_valid.abs().max()) if not z_valid.empty else np.nan

    flat_fraction = float((position == 0).mean()) if n else np.nan

    previous = position.shift(1).fillna(0)
    entries_mask = (position != 0) & (previous == 0)
    flips_mask = (position != 0) & (previous != 0) & (position != previous)
    exits_mask = (position == 0) & (previous != 0)

    entries = int(entries_mask.sum())
    flips = int(flips_mask.sum())
    exits = int(exits_mask.sum()) + flips

    long_entries = int((position[entries_mask] > 0).sum())
    short_entries = int((position[entries_mask] < 0).sum())

    entry_z_values = z_score[entries_mask].dropna()
    entry_threshold_ok = bool(
        (entry_z_values.abs() > entry_z).all() if not entry_z_values.empty else True
    )

    if "exit_reason" in results.columns:
        zscore_exits = results["exit_reason"] == "zscore"
        exit_z_values = z_score[exits_mask & zscore_exits].dropna()
        directions = position.shift(1).fillna(0)[exits_mask & zscore_exits]
        if exit_z_values.empty:
            exit_threshold_ok = True
        else:
            crossed = np.where(
                directions.to_numpy() > 0,
                exit_z_values.to_numpy() >= exit_z,
                exit_z_values.to_numpy() <= exit_z,
            )
            exit_threshold_ok = bool(crossed.all())
    else:
        exit_threshold_ok = True

    # La duración se mide sobre las operaciones completas: los días con la
    # posición abierta.  No se usa ``holding_days`` porque vale 0 en la fila
    # de salida (el contador se reinicia al cerrar).
    lengths = _operation_lengths(position)
    open_operations = 1 if n and position.iloc[-1] != 0 else 0
    closed_lengths = lengths[:-1] if open_operations and lengths else lengths
    mean_holding = float(np.mean(closed_lengths)) if closed_lengths else 0.0
    median_holding = float(np.median(closed_lengths)) if closed_lengths else 0.0
    max_holding = int(max(lengths)) if lengths else 0

    max_consecutive = 0
    current = 0
    for pos in position.to_numpy():
        if pos != 0:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0

    round_trips = exits

    return SignalDiagnostics(
        z_nan_count=z_nan,
        z_nan_after_warmup=z_nan_after_warmup,
        z_max_abs=z_max_abs,
        flat_fraction=flat_fraction,
        entries=entries,
        exits=exits,
        round_trips=round_trips,
        open_operations=open_operations,
        long_entries=long_entries,
        short_entries=short_entries,
        mean_holding_days=mean_holding,
        median_holding_days=median_holding,
        max_holding_days=max_holding,
        max_consecutive_holding=max_consecutive,
        entry_threshold_ok=entry_threshold_ok,
        exit_threshold_ok=exit_threshold_ok,
    )


def compute_return_diagnostics(results: pd.DataFrame) -> ReturnDiagnostics:
    """Calcula métricas de retorno, notional y costes.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.

    Returns
    -------
    ReturnDiagnostics
        Métricas de retorno y turnover.
    """
    turnover = results["turnover"]
    costs = results["costs"]
    gross = results["gross_return"]
    net = results["strategy_ret"]
    notional = results["gross_notional"]

    position = results["position"]
    operations = len(_operation_starts(position))
    turnover_per_operation = float(turnover.sum()) / operations if operations else 0.0

    gross_total = float(gross.sum())
    net_total = float(net.sum())
    total_costs = float(costs.sum())
    costs_pct = total_costs / abs(gross_total) if abs(gross_total) > 1e-12 else 0.0

    recomputed = gross - costs
    contribution_error = float(np.max(np.abs(recomputed.to_numpy() - net.to_numpy())))

    return ReturnDiagnostics(
        total_turnover=float(turnover.sum()),
        mean_daily_turnover=float(turnover.mean()),
        turnover_per_operation=turnover_per_operation,
        total_costs=total_costs,
        costs_pct_of_gross=costs_pct,
        mean_gross_notional=float(notional.mean()),
        max_gross_notional=float(notional.max()),
        gross_notional_p01=float(notional.quantile(0.01)),
        gross_notional_p50=float(notional.quantile(0.50)),
        gross_notional_p99=float(notional.quantile(0.99)),
        mean_exposure_a=float(results["weight_a"].abs().mean()),
        mean_exposure_b=float(results["weight_b"].abs().mean()),
        gross_total=gross_total,
        net_total=net_total,
        contribution_max_error=contribution_error,
    )


def compute_risk_metrics(
    results: pd.DataFrame,
    equity: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
) -> RiskMetrics:
    """Calcula métricas de rendimiento y riesgo.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    equity : pd.Series
        Curva de capital acumulada.
    trades : pd.DataFrame
        Operaciones extraídas con :func:`extract_pair_trades`.
    periods_per_year : int
        Número de sesiones por año usado en las anualizaciones.

    Returns
    -------
    RiskMetrics
        Métricas de rendimiento y riesgo.
    """
    rets = results["strategy_ret"].astype(float)
    n = len(rets)
    mean = float(rets.mean()) if n else 0.0
    std = float(rets.std(ddof=1)) if n > 1 else 0.0
    volatility = std * np.sqrt(periods_per_year)

    total_return = float(equity.iloc[-1] - 1.0) if n else 0.0
    years = n / periods_per_year if periods_per_year else 0.0
    if years > 0 and equity.iloc[-1] > 0:
        cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    else:
        cagr = 0.0

    sharpe = mean / std * np.sqrt(periods_per_year) if std > 0 else 0.0

    downside = rets[rets < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        mean / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0.0
    )

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if n else 0.0

    duration = 0
    current = 0
    longest = 0
    for value in drawdown.to_numpy():
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    duration = longest

    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    var_95 = float(rets.quantile(0.05)) if n else 0.0
    tail = rets[rets <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else var_95

    closed = trades[~trades["is_open"]] if not trades.empty else trades
    if closed.empty:
        best_trade = worst_trade = 0.0
        profit_factor = 0.0
        win_rate = 0.0
        avg_gain = avg_loss = 0.0
        n_trades = 0
    else:
        net_trades = closed["net_return"].astype(float)
        gains = net_trades[net_trades > 0]
        losses = net_trades[net_trades < 0]
        best_trade = float(net_trades.max())
        worst_trade = float(net_trades.min())
        total_loss = float(-losses.sum())
        profit_factor = (
            float(gains.sum()) / total_loss if total_loss > 1e-12 else np.inf
        )
        win_rate = float((net_trades > 0).mean())
        avg_gain = float(gains.mean()) if not gains.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        n_trades = len(closed)

    return RiskMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_drawdown,
        max_drawdown_duration=duration,
        var_95=var_95,
        cvar_95=cvar_95,
        best_trade=best_trade,
        worst_trade=worst_trade,
        profit_factor=profit_factor,
        win_rate=win_rate,
        avg_gain=avg_gain,
        avg_loss=avg_loss,
        n_trades=n_trades,
    )


# =============================================================================
# Comprobaciones automáticas
# =============================================================================


def detect_rebalance_mode(results: pd.DataFrame, default: str = "on_entry") -> str:
    """Deduce el modo de rebalanceo del hedge ratio usado en el backtest.

    El backtest guarda el modo en la columna ``rebalance_hedge``, de modo
    que las comprobaciones automáticas pueden adaptarse sin que el
    llamador tenga que repetir el parámetro.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    default : str
        Valor devuelto si la columna no está presente.

    Returns
    -------
    str
        ``"on_entry"`` o ``"daily"``.
    """
    if "rebalance_hedge" in results.columns:
        valores = results["rebalance_hedge"].dropna().unique()
        if len(valores) == 1:
            return str(valores[0])
    return default


def validate_pair_backtest(
    results: pd.DataFrame,
    entry_z: float,
    exit_z: float,
    hedge_window: int,
    rebalance_hedge: str | None = None,
) -> ValidationReport:
    """Ejecuta la batería de comprobaciones automáticas del backtest.

    Las comprobaciones cubren:

    - integridad del índice y de los rangos de posición;
    - respeto del warm-up;
    - coherencia entre beta efectivo, posición y bloques de operación;
    - alineación de un día entre posición señalada y posición aplicada;
    - cuadre de contribuciones, costes, retorno neto y equity;
    - coherencia del turnover con los cambios de peso;
    - cumplimiento de los umbrales de entrada y salida.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    entry_z, exit_z : float
        Umbrales usados en el backtest.
    hedge_window : int
        Ventana de estimación del hedge ratio (warm-up mínimo).
    rebalance_hedge : {"on_entry", "daily"} | None
        Modo de rebalanceo del hedge ratio.  La constancia del beta dentro
        de una operación sólo se exige en modo ``on_entry``.  Si es
        ``None`` se deduce de la columna ``rebalance_hedge`` de ``results``.

    Returns
    -------
    ValidationReport
        Resultado de cada comprobación.
    """
    if rebalance_hedge is None:
        rebalance_hedge = detect_rebalance_mode(results)
    checks: list[ValidationCheck] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(ValidationCheck(name=name, passed=bool(passed), detail=detail))

    if results.empty:
        add("results_no_vacios", False, "El DataFrame de resultados está vacío.")
        return ValidationReport(checks=tuple(checks))

    add(
        "indice_monotono",
        bool(results.index.is_monotonic_increasing),
        "El índice temporal debe ser estrictamente creciente.",
    )
    add(
        "indice_unico",
        bool(results.index.is_unique),
        "El índice temporal no debe tener fechas repetidas.",
    )

    position = results["position"]
    valid_positions = bool(position.isin([-1.0, 0.0, 1.0]).all())
    add(
        "posiciones_validas",
        valid_positions,
        "Las posiciones deben pertenecer a {-1, 0, 1}.",
    )

    add(
        "turnover_no_negativo",
        bool((results["turnover"] >= 0).all()),
        "El turnover no puede ser negativo.",
    )

    add(
        "retornos_sin_nan",
        bool(results["strategy_ret"].notna().all()),
        "El retorno neto no debe contener NaN.",
    )

    warmup_ok = bool((position.iloc[:hedge_window] == 0).all())
    add(
        "warmup_respetado",
        warmup_ok,
        f"No se permiten posiciones en las primeras {hedge_window} sesiones.",
    )

    flat_beta_ok = bool((results.loc[position == 0, "effective_beta"] == 0).all())
    add(
        "beta_efectivo_cero_en_cash",
        flat_beta_ok,
        "El beta efectivo debe ser 0 cuando la estrategia está en cash.",
    )

    block_ok = _effective_beta_block_consistent(
        results["position"], results["effective_beta"]
    )
    if rebalance_hedge == "on_entry":
        add(
            "beta_efectivo_constante_en_bloque",
            block_ok,
            "El beta efectivo debe ser constante dentro de cada operación.",
        )
    else:
        in_position = results["position"] != 0
        comparable = in_position & results["beta"].notna()
        if comparable.any():
            daily_ok = bool(
                np.allclose(
                    results.loc[comparable, "effective_beta"],
                    results.loc[comparable, "beta"],
                    atol=1e-12,
                )
            )
        else:
            daily_ok = True
        add(
            "beta_efectivo_igual_al_diario",
            daily_ok,
            "En modo diario el beta efectivo debe coincidir con la estimación diaria.",
        )

    expected_held_position = position.shift(1).fillna(0.0)
    lag_ok = bool(np.allclose(expected_held_position, results["held_position"]))
    add(
        "posicion_aplicada_con_lag",
        lag_ok,
        "La posición aplicada en t debe ser la posición señalada en t-1.",
    )

    expected_held_beta = results["effective_beta"].shift(1).fillna(0.0)
    beta_lag_ok = bool(np.allclose(expected_held_beta, results["held_beta"]))
    add(
        "beta_aplicado_con_lag",
        beta_lag_ok,
        "El beta aplicado en t debe ser el beta efectivo de t-1.",
    )

    contributions_ok = bool(
        np.allclose(
            results["contribution_a"] + results["contribution_b"],
            results["gross_return"],
            atol=1e-12,
        )
    )
    add(
        "contribuciones_suman_bruto",
        contributions_ok,
        "La contribución de A y B debe sumar el retorno bruto.",
    )

    net_ok = bool(
        np.allclose(
            results["gross_return"] - results["costs"],
            results["strategy_ret"],
            atol=1e-12,
        )
    )
    add(
        "bruto_menos_coste_es_neto",
        net_ok,
        "Retorno bruto menos costes debe igualar el retorno neto.",
    )

    cost_rate = float(results["cost_rate"].iloc[0])
    costs_ok = bool(
        np.allclose(results["turnover"] * cost_rate, results["costs"], atol=1e-12)
    )
    add(
        "costes_coherentes_con_turnover",
        costs_ok,
        "Los costes deben ser el turnover multiplicado por el coste unitario.",
    )

    expected_turnover = results["target_weight_a"].diff().abs().fillna(0.0) + results[
        "target_weight_b"
    ].diff().abs().fillna(0.0)
    turnover_ok = bool(np.allclose(expected_turnover, results["turnover"], atol=1e-12))
    add(
        "turnover_coherente_con_pesos",
        turnover_ok,
        "El turnover debe igualar la variación absoluta de los pesos.",
    )

    weights_ok = bool(
        np.allclose(
            results["weight_a"].abs() + results["weight_b"].abs(),
            results["held_position"].abs(),
            atol=1e-9,
        )
    )
    add(
        "pesos_normalizados",
        weights_ok,
        "La exposición bruta de los pesos debe ser 1 en las fechas con posición.",
    )

    equity = np.exp(results["strategy_ret"].cumsum())
    running_max = equity.cummax()
    expected_dd = (equity - running_max) / running_max
    drawdown_ok = bool(np.allclose(expected_dd, results["drawdown"], atol=1e-12))
    add(
        "drawdown_coherente",
        drawdown_ok,
        "El drawdown debe reconstruirse a partir del equity y su máximo.",
    )

    if "equity" in results.columns:
        equity_ok = bool(np.allclose(equity, results["equity"], atol=1e-12))
    else:
        equity_ok = True
    add(
        "equity_coherente",
        equity_ok,
        "El equity debe ser el exponente del cumulativo de retornos.",
    )

    entry_mask = (position != 0) & (position != position.shift(1).fillna(0))
    entry_z_values = results["z_score"][entry_mask].dropna()
    entry_ok = bool(
        (entry_z_values.abs() > entry_z).all() if not entry_z_values.empty else True
    )
    add(
        "entradas_superan_umbral",
        entry_ok,
        f"Todas las entradas deben cumplir |z| > {entry_z}.",
    )

    if "exit_reason" in results.columns:
        exit_mask = (position == 0) & (position.shift(1).fillna(0) != 0)
        zscore_exits = exit_mask & (results["exit_reason"] == "zscore")
        exit_z_values = results["z_score"][zscore_exits].dropna()
        directions = position.shift(1).fillna(0)[zscore_exits]
        if exit_z_values.empty:
            exit_ok = True
        else:
            crossed = np.where(
                directions.to_numpy() > 0,
                exit_z_values.to_numpy() >= exit_z,
                exit_z_values.to_numpy() <= exit_z,
            )
            exit_ok = bool(crossed.all())
    else:
        exit_ok = True
    add(
        "salidas_respetan_umbral",
        exit_ok,
        f"Las salidas por z-score deben cruzar exit_z = {exit_z}.",
    )

    return ValidationReport(checks=tuple(checks))


def verify_hedge_is_causal(
    log_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    results: pd.DataFrame,
    hedge_window: int,
    z_window: int,
    sample_size: int = 10,
) -> ValidationReport:
    """Recalcula el hedge walk-forward y comprueba que es causal.

    Vuelve a estimar ``alpha``, ``beta`` y el z-score en una muestra de
    fechas usando exclusivamente datos anteriores y compara el resultado
    con el almacenado en ``results``.  Si el backtest usara información
    futura, la diferencia sería relevante.

    Parameters
    ----------
    log_prices : pd.DataFrame
        Log-precios con una columna por ticker.
    ticker_a, ticker_b : str
        Tickers del par.
    results : pd.DataFrame
        Salida del backtest.
    hedge_window, z_window : int
        Ventanas utilizadas en el backtest.
    sample_size : int
        Número de fechas a verificar.

    Returns
    -------
    ValidationReport
        Comprobaciones de causalidad.
    """
    from thesis.pair_trading import _rolling_hedge_zscore

    checks: list[ValidationCheck] = []

    log_a = log_prices[ticker_a]
    log_b = log_prices[ticker_b]
    common = log_a.index.intersection(log_b.index)
    log_a = log_a.loc[common]
    log_b = log_b.loc[common]

    n = len(common)
    if n == 0:
        checks.append(
            ValidationCheck(
                name="causalidad_hedge",
                passed=False,
                detail="No hay fechas comunes para verificar.",
            )
        )
        return ValidationReport(checks=tuple(checks))

    valid_positions = np.flatnonzero(results["beta"].notna().to_numpy())
    if valid_positions.size == 0:
        checks.append(
            ValidationCheck(
                name="causalidad_hedge",
                passed=False,
                detail="No hay estimaciones de beta que verificar.",
            )
        )
        return ValidationReport(checks=tuple(checks))

    step = max(len(valid_positions) // max(sample_size, 1), 1)
    sampled = valid_positions[::step][:sample_size]

    max_alpha_error = 0.0
    max_beta_error = 0.0
    max_z_error = 0.0
    for t in sampled:
        alpha, beta, z = _rolling_hedge_zscore(
            log_a, log_b, int(t), hedge_window, z_window
        )
        stored = results.iloc[int(t)]
        if not np.isnan(alpha) and not np.isnan(stored["alpha"]):
            max_alpha_error = max(max_alpha_error, abs(alpha - stored["alpha"]))
        if not np.isnan(beta) and not np.isnan(stored["beta"]):
            max_beta_error = max(max_beta_error, abs(beta - stored["beta"]))
        if not np.isnan(z) and not np.isnan(stored["z_score"]):
            max_z_error = max(max_z_error, abs(z - stored["z_score"]))

    checks.append(
        ValidationCheck(
            name="causalidad_alpha",
            passed=max_alpha_error < 1e-9,
            detail=(
                f"Error máximo de alpha al recalcular con datos pasados: "
                f"{max_alpha_error:.3e}"
            ),
        )
    )
    checks.append(
        ValidationCheck(
            name="causalidad_beta",
            passed=max_beta_error < 1e-9,
            detail=(
                f"Error máximo de beta al recalcular con datos pasados: "
                f"{max_beta_error:.3e}"
            ),
        )
    )
    checks.append(
        ValidationCheck(
            name="causalidad_z_score",
            passed=max_z_error < 1e-9,
            detail=(
                f"Error máximo de z-score al recalcular con datos pasados: "
                f"{max_z_error:.3e}"
            ),
        )
    )

    return ValidationReport(checks=tuple(checks))


# =============================================================================
# Agregación
# =============================================================================


def compute_pair_backtest_diagnostics(
    results: pd.DataFrame,
    equity: pd.Series,
    entry_z: float,
    exit_z: float,
    hedge_window: int,
    z_window: int,
    log_prices: pd.DataFrame | None = None,
    ticker_a: str | None = None,
    ticker_b: str | None = None,
    max_abs_beta: float | None = None,
    periods_per_year: int = TRADING_DAYS,
    verify_causality: bool = True,
    rebalance_hedge: str | None = None,
) -> PairBacktestDiagnostics:
    """Agrega todas las métricas y validaciones del backtest.

    Parameters
    ----------
    results : pd.DataFrame
        Salida de :func:`thesis.pair_trading.pair_trading_backtest`.
    equity : pd.Series
        Curva de capital acumulada.
    entry_z, exit_z : float
        Umbrales usados en el backtest.
    hedge_window, z_window : int
        Ventanas usadas en el backtest.
    log_prices : pd.DataFrame | None
        Log-precios originales; necesarios para verificar la causalidad.
    ticker_a, ticker_b : str | None
        Tickers del par; necesarios para verificar la causalidad.
    max_abs_beta : float | None
        Límite para contar betas extremos.
    periods_per_year : int
        Sesiones por año para las anualizaciones.
    verify_causality : bool
        Si ``True`` y se aportan los precios, ejecuta la comprobación de
        causalidad del hedge ratio.
    rebalance_hedge : {"on_entry", "daily"} | None
        Modo de rebalanceo usado en el backtest.  Si es ``None`` se deduce
        de la columna ``rebalance_hedge`` de ``results``.

    Returns
    -------
    PairBacktestDiagnostics
        Diagnóstico completo.
    """
    trades = extract_pair_trades(results)

    hedge = compute_hedge_diagnostics(results, max_abs_beta=max_abs_beta)
    signal = compute_signal_diagnostics(
        results, entry_z=entry_z, exit_z=exit_z, hedge_window=hedge_window
    )
    returns = compute_return_diagnostics(results)
    risk = compute_risk_metrics(
        results, equity, trades, periods_per_year=periods_per_year
    )

    report = validate_pair_backtest(
        results,
        entry_z=entry_z,
        exit_z=exit_z,
        hedge_window=hedge_window,
        rebalance_hedge=rebalance_hedge,
    )

    if verify_causality and log_prices is not None and ticker_a and ticker_b:
        causal = verify_hedge_is_causal(
            log_prices,
            ticker_a,
            ticker_b,
            results,
            hedge_window=hedge_window,
            z_window=z_window,
        )
        report = ValidationReport(checks=report.checks + causal.checks)

    return PairBacktestDiagnostics(
        trades=trades,
        hedge=hedge,
        signal=signal,
        returns=returns,
        risk=risk,
        validation=report,
    )

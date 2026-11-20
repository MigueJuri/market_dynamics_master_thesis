"""Adaptadores de payload de notebooks a :class:`thesis.results.ResultRecord`.

Cada notebook canónico declara un ``experiment_id`` al publicar.  Este módulo
mapea el payload estructurado de cada experimento a records con unidades y
protocolo correctos.  Los experimentos sin adapter cerrado se marcan como
``candidate`` con nota ``adapter pendiente``; nunca fallan la publicación.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from thesis.results import _UNKNOWN, ResultRecord, derive_result_id

#: Unidad por nombre de métrica (inferencia común).
_METRIC_UNITS: dict[str, str] = {
    "R2": "adimensional",
    "RMSE": "sigma anualizada",
    "MAE": "sigma anualizada",
    "Spearman_r": "adimensional",
    "Spearman_p": "adimensional",
    "Dir_Acc": "adimensional",
    "Dir_p": "adimensional",
    "QLIKE": "adimensional",
    "MZ_alpha": "adimensional",
    "MZ_beta": "adimensional",
    "MZ_F_pval": "adimensional",
    "total_return": "adimensional",
    "sharpe": "adimensional",
    "maxdd": "adimensional",
    "calmar": "adimensional",
    "sharpe_improvement": "adimensional",
    "sharpe_vs_bh": "adimensional",
}

#: Nombre amigable por métrica.
_METRIC_LABELS: dict[str, str] = {
    "R2": "R²",
    "Dir_Acc": "Directional Accuracy",
}


def _coerce_str(value: Any) -> str:
    if value is None:
        return _UNKNOWN
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    if isinstance(value, (float, int)):
        return str(value)
    return str(value)


def _extract_period(config: dict[str, Any]) -> tuple[str, str, str, str]:
    """Extrae periodo de datos y fechas de split desde la config."""
    start = config.get("start_date", config.get("start", _UNKNOWN))
    train_end = config.get("train_end", _UNKNOWN)
    val_end = config.get("val_end", _UNKNOWN)
    end = config.get("end_date", config.get("end", _UNKNOWN))

    data_period = f"{start} → {end}" if start != _UNKNOWN and end != _UNKNOWN else _UNKNOWN
    train_dates = f"{_UNKNOWN} → {train_end}" if train_end != _UNKNOWN else _UNKNOWN
    val_dates = f"{train_end} → {val_end}" if train_end != _UNKNOWN and val_end != _UNKNOWN else _UNKNOWN
    test_dates = f"{val_end} → {end}" if val_end != _UNKNOWN and end != _UNKNOWN else _UNKNOWN
    return data_period, train_dates, val_dates, test_dates


def _protocol_from_config(config: dict[str, Any]) -> str:
    parts: list[str] = []
    if "forecast_horizon" in config:
        parts.append(f"h={config['forecast_horizon']}d")
    if "cost_bps" in config:
        parts.append(f"cost={config['cost_bps']}bps")
    if "lookback" in config:
        parts.append(f"lookback={config['lookback']}")
    return "; ".join(parts) if parts else _UNKNOWN


def _sample_size_from_metrics(metrics: dict[str, Any]) -> int | str:
    if "N" in metrics:
        return int(metrics["N"]) if pd.notna(metrics["N"]) else _UNKNOWN
    return _UNKNOWN


def _records_from_model_metrics(
    chapter: str,
    experiment: str,
    config: dict[str, Any],
    metrics_by_model: dict[str, dict[str, Any]],
    code_path: str = _UNKNOWN,
    asset: str = _UNKNOWN,
    baseline: str = _UNKNOWN,
    target_definition: str = _UNKNOWN,
) -> list[ResultRecord]:
    """Crea records a partir de un dict ``{modelo: métricas}``."""
    data_period, train_dates, val_dates, test_dates = _extract_period(config)
    protocol = _protocol_from_config(config)
    records: list[ResultRecord] = []

    for model, metrics in metrics_by_model.items():
        sample_size = _sample_size_from_metrics(metrics)
        for metric_name, value in metrics.items():
            if metric_name in ("model", "N"):
                continue
            label = _METRIC_LABELS.get(metric_name, metric_name)
            fingerprint = {"model": model, "metric": metric_name, "config_keys": sorted(config.keys())}
            rec = ResultRecord(
                result_id=derive_result_id(
                    chapter=chapter,
                    experiment=experiment,
                    metric_name=label,
                    model=model,
                    asset=asset,
                    fingerprint=fingerprint,
                ),
                chapter=chapter,
                experiment=experiment,
                metric_name=label,
                metric_value=value,
                unit=_METRIC_UNITS.get(metric_name, _UNKNOWN),
                model=model,
                baseline=baseline,
                asset=asset,
                data_source=config.get("data_source", "yfinance"),
                frequency=config.get("frequency", "daily"),
                adjustments=config.get("adjustments", _UNKNOWN),
                data_period=data_period,
                train_dates=train_dates,
                val_dates=val_dates,
                test_dates=test_dates,
                target_definition=target_definition or config.get("target_definition", _UNKNOWN),
                horizon=_coerce_str(config.get("forecast_horizon", _UNKNOWN)),
                sample_size=sample_size,
                protocol=protocol,
                code_path=code_path,
                status="published",
            )
            records.append(rec)
    return records


def _records_from_summary(
    chapter: str,
    experiment: str,
    config: dict[str, Any],
    summary: pd.DataFrame,
    code_path: str = _UNKNOWN,
    asset: str = _UNKNOWN,
    baseline: str = _UNKNOWN,
) -> list[ResultRecord]:
    """Crea records a partir de un DataFrame de comparación de estrategias."""
    records: list[ResultRecord] = []
    data_period, train_dates, val_dates, test_dates = _extract_period(config)
    protocol = _protocol_from_config(config)

    for _, row in summary.iterrows():
        model = str(row.get("strategy", _UNKNOWN))
        for metric_name in ("total_return", "sharpe", "maxdd", "calmar"):
            if metric_name not in row:
                continue
            label = f"{model} {metric_name}"
            fingerprint = {"strategy": model, "metric": metric_name}
            rec = ResultRecord(
                result_id=derive_result_id(
                    chapter=chapter,
                    experiment=experiment,
                    metric_name=label,
                    model=model,
                    asset=asset,
                    fingerprint=fingerprint,
                ),
                chapter=chapter,
                experiment=experiment,
                metric_name=label,
                metric_value=row[metric_name],
                unit=_METRIC_UNITS.get(metric_name, _UNKNOWN),
                model=model,
                baseline=baseline,
                asset=asset,
                data_source=config.get("data_source", "yfinance"),
                frequency=config.get("frequency", "daily"),
                data_period=data_period,
                train_dates=train_dates,
                val_dates=val_dates,
                test_dates=test_dates,
                horizon=_coerce_str(config.get("forecast_horizon", _UNKNOWN)),
                sample_size=_UNKNOWN,
                protocol=protocol,
                code_path=code_path,
                status="published",
            )
            records.append(rec)
    return records


def _lstm_daily_adapter(experiment_id: str, payload: Any) -> list[ResultRecord]:
    """Adapter para ``volatility/Volatilidad_LSTM.py``."""
    if not isinstance(payload, dict):
        return _candidate_record(experiment_id, "payload no es dict")

    config = payload.get("config", {})
    metrics = payload.get("metrics", {})
    backtest = payload.get("backtest", {})
    summary = payload.get("summary")
    chapter = "Volatilidad"
    experiment = "LSTM diario"
    asset = config.get("ticker", "SPY")
    code_path = "notebooks/volatility/Volatilidad_LSTM.py"
    target = "log-vol realizada forward 5d anualizada"

    records: list[ResultRecord] = []
    if isinstance(metrics, dict):
        records.extend(
            _records_from_model_metrics(
                chapter=chapter,
                experiment=experiment,
                config=config,
                metrics_by_model=metrics,
                code_path=code_path,
                asset=asset,
                baseline="HAR-RV",
                target_definition=target,
            )
        )

    # Métricas de vol-timing del backtest.
    if isinstance(backtest, dict):
        data_period, train_dates, val_dates, test_dates = _extract_period(config)
        protocol = _protocol_from_config(config)
        bt_metrics = {
            k: v
            for k, v in backtest.items()
            if k in ("sharpe_improvement", "sharpe_vs_bh", "total_vol", "total_base", "total_bh")
        }
        for metric_name, value in bt_metrics.items():
            label = metric_name.replace("_", " ")
            rec = ResultRecord(
                result_id=derive_result_id(
                    chapter=chapter,
                    experiment=experiment,
                    metric_name=label,
                    model="Vol-Timed",
                    asset=asset,
                    fingerprint={"metric": metric_name},
                ),
                chapter=chapter,
                experiment=experiment,
                metric_name=label,
                metric_value=value,
                unit=_METRIC_UNITS.get(metric_name, _UNKNOWN),
                model="Vol-Timed",
                baseline="Buy & Hold",
                asset=asset,
                data_period=data_period,
                train_dates=train_dates,
                val_dates=val_dates,
                test_dates=test_dates,
                target_definition=target,
                protocol=protocol,
                code_path=code_path,
                status="published",
            )
            records.append(rec)

    if isinstance(summary, pd.DataFrame):
        records.extend(
            _records_from_summary(
                chapter=chapter,
                experiment=experiment,
                config=config,
                summary=summary,
                code_path=code_path,
                asset=asset,
                baseline="Buy & Hold",
            )
        )

    if not records:
        return _candidate_record(experiment_id, "sin métricas estructuradas")
    return records


def _generic_dataframe_adapter(chapter: str) -> Callable[[str, Any], list[ResultRecord]]:
    """Factory de adapters para notebooks que exportan un ``metrics_df``."""

    def adapter(experiment_id: str, payload: Any) -> list[ResultRecord]:
        if not isinstance(payload, dict):
            return _candidate_record(experiment_id, "payload no es dict")
        df = payload.get("metrics_df")
        config = payload.get("config", {})
        if not isinstance(df, pd.DataFrame) or df.empty:
            return _candidate_record(experiment_id, "metrics_df vacío")

        experiment = experiment_id.split("/")[-1]
        records: list[ResultRecord] = []
        data_period, train_dates, val_dates, test_dates = _extract_period(config)
        protocol = _protocol_from_config(config)
        code_path = f"notebooks/{experiment_id}.py"

        # Intentar detectar columna de modelo/estrategia.
        model_col = None
        for col in ("model", "strategy", "Method", "method"):
            if col in df.columns:
                model_col = col
                break

        for _, row in df.iterrows():
            model = str(row.get(model_col, _UNKNOWN)) if model_col else _UNKNOWN
            for metric_name in df.columns:
                if metric_name == model_col:
                    continue
                label = _METRIC_LABELS.get(metric_name, metric_name)
                rec = ResultRecord(
                    result_id=derive_result_id(
                        chapter=chapter,
                        experiment=experiment,
                        metric_name=label,
                        model=model,
                        fingerprint={"row": int(_), "metric": metric_name},
                    ),
                    chapter=chapter,
                    experiment=experiment,
                    metric_name=label,
                    metric_value=row[metric_name],
                    unit=_METRIC_UNITS.get(metric_name, _UNKNOWN),
                    model=model,
                    asset=config.get("ticker", _UNKNOWN),
                    data_period=data_period,
                    train_dates=train_dates,
                    val_dates=val_dates,
                    test_dates=test_dates,
                    protocol=protocol,
                    code_path=code_path,
                    status="candidate",
                    notes="adapter genérico; revisar unidades y protocolo",
                )
                records.append(rec)
        return records

    return adapter


def _candidate_record(experiment_id: str, reason: str) -> list[ResultRecord]:
    """Genera un record ``candidate`` cuando no hay adapter cerrado."""
    rec = ResultRecord(
        result_id=derive_result_id(
            chapter="unknown",
            experiment=experiment_id,
            metric_name="adapter_pending",
            fingerprint=experiment_id,
        ),
        experiment=experiment_id,
        metric_name="adapter_pending",
        metric_value="unknown",
        status="candidate",
        notes=f"adapter pendiente: {reason}",
    )
    return [rec]


#: Registro global de adaptadores.  Los ``experiment_id`` ausentes se marcan
#: automáticamente como ``candidate`` en :func:`thesis.results.publish`.
ADAPTERS: dict[str, Callable[[str, Any], list[ResultRecord]]] = {
    "volatility/lstm_daily": _lstm_daily_adapter,
    # Feature analysis
    "feature_analysis/momentum": _generic_dataframe_adapter("Momentum"),
    "feature_analysis/volatility": _generic_dataframe_adapter("Volatilidad"),
    "feature_analysis/momentum_volatility": _generic_dataframe_adapter("Momentum-Volatilidad"),
    "feature_analysis/moving_average_distance": _generic_dataframe_adapter("Moving Average Distance"),
    "feature_analysis/per": _generic_dataframe_adapter("PER"),
    # Trend following
    "trend_following/strategy_v2": _generic_dataframe_adapter("Trend Following"),
    # Pair trading
    "pair_trading/selection_mi": _generic_dataframe_adapter("Pair Trading"),
    "pair_trading/selection_pearson": _generic_dataframe_adapter("Pair Trading"),
    "pair_trading/practical": _generic_dataframe_adapter("Pair Trading"),
    "pair_trading/spread": _generic_dataframe_adapter("Pair Trading"),
    # BTC / SPY
    "btc/btc": _generic_dataframe_adapter("Bitcoin"),
    "spy/characterization": _generic_dataframe_adapter("SPY"),
    "spy/sectors": _generic_dataframe_adapter("SPY"),
    "spy/orderbook": _generic_dataframe_adapter("SPY"),
}

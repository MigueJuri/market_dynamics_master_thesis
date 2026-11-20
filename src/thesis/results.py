"""Registro, validación y publicación de métricas en el ledger de resultados.

El módulo expone un esquema estandarizado (:class:`ResultRecord`), una función
:func:`publish` para enviar métricas desde notebooks, y un
:class:`MetricRegistry` global con un *hook* de :mod:`atexit` opcional para
publicar automáticamente al cerrar el intérprete.

Ejemplo
-------
Desde el final de un notebook canónico::

    from thesis.results import publish

    publish(
        experiment_id="volatility/lstm_daily",
        payload={"metrics_df": metrics_df, "config": CONFIG, "backtest": backtest},
        chapter="Volatilidad",
    )
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from thesis import config as _config

#: Campos mínimos que deben aparecer en un record para considerarse ``published``.
REQUIRED_FIELDS: tuple[str, ...] = ("result_id", "chapter", "metric_name", "metric_value")

#: Valor canónico para metadatos desconocidos.
_UNKNOWN: str = "unknown"

#: Estados válidos de un record.
VALID_STATUSES: tuple[str, ...] = ("published", "candidate", "deprecated", "rejected")

#: Patrón de "token largo": 20+ caracteres alfanuméricos (posible secreto).
_LONG_TOKEN_PATTERN = re.compile(r"\b[A-Za-z0-9-_]{20,}\b")

#: Patrones que activan la redacción de seguridad.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    _LONG_TOKEN_PATTERN,
    re.compile(r"password|passwd|secret|token|api_key|apikey", re.IGNORECASE),
)

#: Campos con texto libre, donde un segredo puede aparecer de verdad.  El
#: patrón de token largo **no** se aplica al resto de campos: los
#: identificadores del repositorio (nombre del experimento, modelo, ruta del
#: notebook) son largos por legibilidad y no son secretos; redactarlos
#: rechazaba filas completas del ledger.
_FREE_TEXT_FIELDS: frozenset[str] = frozenset(
    ["metric_value", "notes", "data_source", "target_definition", "protocol"]
)

#: Extensiones de artefactos que nunca se publican.
_SENSITIVE_EXTENSIONS: frozenset[str] = frozenset(
    [".env", ".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".pickle"]
)

#: Rutas o prefijos sensibles que deben rechazarse.
_SENSITIVE_PREFIXES: tuple[str, ...] = (".env", ".env.", "data/", "outputs/")


@dataclass(frozen=True, slots=True)
class LedgerConfig:
    """Configuración de publicación del ledger."""

    ledger_path: Path
    sidecar_path: Path

    @classmethod
    def from_thesis_repo(cls, thesis_repo_dir: Path | str | None = None) -> LedgerConfig:
        """Construye la configuración a partir de la raíz del repo de la tesis."""
        root = _resolve_thesis_repo_dir(thesis_repo_dir)
        ledger_dir = root / ".thesis"
        return cls(
            ledger_path=ledger_dir / "results-ledger.md",
            sidecar_path=ledger_dir / "results-ledger.json",
        )


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """Fila normalizada del ledger de resultados.

    Todos los campos son ``str`` excepto ``metric_value`` (cualquier tipo
    serializable) y ``sample_size`` (``int | str``).  Los valores faltantes se
    normalizan a ``"unknown"`` en :func:`validate_record`.
    """

    result_id: str = _UNKNOWN
    chapter: str = _UNKNOWN
    experiment: str = _UNKNOWN
    metric_name: str = _UNKNOWN
    metric_value: Any = _UNKNOWN
    unit: str = _UNKNOWN
    model: str = _UNKNOWN
    baseline: str = _UNKNOWN
    asset: str = _UNKNOWN
    data_source: str = _UNKNOWN
    frequency: str = _UNKNOWN
    adjustments: str = _UNKNOWN
    data_period: str = _UNKNOWN
    train_dates: str = _UNKNOWN
    val_dates: str = _UNKNOWN
    test_dates: str = _UNKNOWN
    target_definition: str = _UNKNOWN
    horizon: str = _UNKNOWN
    sample_size: int | str = _UNKNOWN
    protocol: str = _UNKNOWN
    code_path: str = _UNKNOWN
    execution_date: str = _UNKNOWN
    git_rev: str = _UNKNOWN
    status: str = "candidate"
    artifacts: str = _UNKNOWN
    notes: str = _UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_row(self) -> dict[str, Any]:
        """Diccionario con ``metric_value`` forzado a ``str`` para la tabla."""
        d = self.to_dict()
        d["metric_value"] = _format_value(d["metric_value"])
        return d


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _resolve_thesis_repo_dir(thesis_repo_dir: Path | str | None = None) -> Path:
    """Resuelve la raíz del repo de la tesis sin rutas absolutas embebidas."""
    if thesis_repo_dir is not None:
        return Path(thesis_repo_dir).expanduser().resolve()
    if _config.THESIS_REPO_DIR is not None:
        return _config.THESIS_REPO_DIR
    project_root = Path(__file__).resolve().parents[2]
    fallback = project_root / ".thesis"
    if fallback.exists() or (project_root / "AGENTS.md").exists():
        return project_root
    raise RuntimeError(
        "No se pudo resolver el repositorio de la tesis. "
        "Define THESIS_REPO_DIR o pasa thesis_repo_dir."
    )


def _get_git_rev() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return _UNKNOWN


def _serialize_fingerprint(obj: Any) -> str:
    """Serializa un objeto a JSON canónico para hashing estable."""
    if isinstance(obj, pd.DataFrame):
        payload = obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        payload = obj.to_dict()
    elif isinstance(obj, (str, int, float, bool, type(None))):
        payload = obj
    elif isinstance(obj, (list, tuple)):
        payload = [_serialize_fingerprint(x) for x in obj]
    elif isinstance(obj, dict):
        payload = {k: _serialize_fingerprint(v) for k, v in sorted(obj.items())}
    else:
        payload = str(obj)
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def derive_result_id(
    chapter: str,
    experiment: str,
    metric_name: str,
    model: str = _UNKNOWN,
    asset: str = _UNKNOWN,
    fingerprint: Any = None,
) -> str:
    """Deriva un ``result_id`` estable y corto.

    El hash se calcula sobre una representación canónica JSON de
    ``chapter + experiment + metric_name + model + asset + fingerprint``.
    """
    payload = {
        "chapter": chapter,
        "experiment": experiment,
        "metric_name": metric_name,
        "model": model,
        "asset": asset,
        "fingerprint": _serialize_fingerprint(fingerprint) if fingerprint is not None else "",
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def redact(record: ResultRecord) -> tuple[ResultRecord, list[str]]:
    """Revisa un record en busca de secretos o artefactos sensibles.

    Devuelve el record (posiblemente con valores redactados) y una lista de
    problemas encontrados.  No lanza excepciones: los problemas se reflejan en
    ``status=rejected`` o en ``notes``.
    """
    issues: list[str] = []
    data = record.to_dict()

    for key, value in data.items():
        if not isinstance(value, str):
            continue
        lower = value.lower()
        for ext in _SENSITIVE_EXTENSIONS:
            if ext in lower or value.endswith(ext):
                issues.append(f"{key} contiene artefacto sensible: {ext}")
                data[key] = "[REDACTED]"
        for prefix in _SENSITIVE_PREFIXES:
            if lower.startswith(prefix) or ("/" + prefix) in lower:
                issues.append(f"{key} contiene ruta sensible: {prefix}")
                data[key] = "[REDACTED]"
        for pattern in _SECRET_PATTERNS:
            if pattern is _LONG_TOKEN_PATTERN and key not in _FREE_TEXT_FIELDS:
                # Identificadores del repositorio: pueden ser largos sin ser
                # secretos (p. ej. ``moving_average_distance_analysis.py``).
                continue
            if pattern.search(value):
                issues.append(f"{key} puede contener secreto ({pattern.pattern})")
                data[key] = "[REDACTED]"

    if issues:
        data["status"] = "rejected"
        data["notes"] = _concat_notes(data.get("notes"), "; ".join(issues))

    return ResultRecord(**data), issues


def _concat_notes(existing: Any, extra: str) -> str:
    existing_str = str(existing) if existing and existing != _UNKNOWN else ""
    if existing_str:
        return f"{existing_str} | {extra}"
    return extra


def validate_record(record: ResultRecord) -> tuple[ResultRecord, list[str]]:
    """Valida y normaliza un :class:`ResultRecord`.

    - Rellena campos faltantes con ``"unknown"``.
    - Asegura que ``status`` sea uno de los valores válidos.
    - Si faltan campos obligatorios, marca ``status=candidate`` y añade
      ``missing_metadata`` a ``notes``.
    - Aplica :func:`redact`.

    Returns
    -------
    tuple[ResultRecord, list[str]]
        Record validado y lista de advertencias.
    """
    data = record.to_dict()
    known_fields = {f.name for f in fields(ResultRecord)}
    issues: list[str] = []

    # Normalizar faltantes.
    for key in known_fields:
        if data.get(key) in (None, "", []):
            data[key] = _UNKNOWN

    # Status válido.
    if data.get("status") not in VALID_STATUSES:
        data["status"] = "candidate"
        issues.append(f"status invalid '{record.status}' -> candidate")

    # Campos obligatorios.
    missing = [k for k in REQUIRED_FIELDS if data.get(k) == _UNKNOWN]
    if missing:
        data["status"] = "candidate"
        issues.append(f"missing_metadata: {', '.join(missing)}")
        data["notes"] = _concat_notes(data.get("notes"), f"missing_metadata: {', '.join(missing)}")

    # Fechas.
    if data.get("execution_date") == _UNKNOWN:
        data["execution_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    validated = ResultRecord(**data)
    validated, redact_issues = redact(validated)
    return validated, issues + redact_issues


def _records_to_dataframe(records: list[ResultRecord]) -> pd.DataFrame:
    rows = [r.to_row() for r in records]
    return pd.DataFrame(rows)


class MetricRegistry:
    """Registro en memoria de métricas pendientes de publicar."""

    def __init__(self) -> None:
        self._records: list[ResultRecord] = []
        self._lock = threading.Lock()

    def record(self, rec: ResultRecord) -> MetricRegistry:
        with self._lock:
            self._records.append(rec)
        return self

    def record_many(self, records: list[ResultRecord]) -> MetricRegistry:
        with self._lock:
            self._records.extend(records)
        return self

    def records(self) -> list[ResultRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> MetricRegistry:
        with self._lock:
            self._records.clear()
        return self

    def publish(
        self,
        thesis_repo_dir: Path | str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Publica todos los records pendientes en el ledger."""
        with self._lock:
            to_publish = list(self._records)

        if not to_publish:
            return {"published": 0, "skipped": 0, "records": []}

        cfg = LedgerConfig.from_thesis_repo(thesis_repo_dir)
        cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Importación diferida para evitar dependencia circular.
        from thesis.results_ledger import merge_records_into_ledger

        summary = merge_records_into_ledger(to_publish, cfg, dry_run=dry_run)
        self.clear()
        return summary


#: Registro global usado por :func:`publish` y el hook de ``atexit``.
GLOBAL_REGISTRY = MetricRegistry()


def _run_adapters(experiment_id: str, payload: Any) -> list[ResultRecord]:
    """Ejecuta el adapter registrado para ``experiment_id`` si existe."""
    from thesis.results_adapters import ADAPTERS

    adapter: Callable[[str, Any], list[ResultRecord]] | None = ADAPTERS.get(experiment_id)
    if adapter is None:
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
            notes="adapter pendiente",
        )
        return [rec]
    return adapter(experiment_id, payload)


def publish(
    experiment_id: str,
    payload: Any,
    chapter: str | None = None,
    thesis_repo_dir: Path | str | None = None,
    dry_run: bool = False,
    auto_commit: bool = True,
) -> dict[str, Any]:
    """Publica un payload de métricas en el ledger.

    Parameters
    ----------
    experiment_id
        Identificador del experimento (p. ej. ``"volatility/lstm_daily"``).
    payload
        Dict/DataFrame/dataclass con las métricas.  El adapter registrado en
        :mod:`thesis.results_adapters` lo transforma en :class:`ResultRecord`.
    chapter
        Capítulo de la tesis.  Puede ser sobreescrito por el adapter.
    thesis_repo_dir
        Raíz del repo de la tesis.  Si es ``None`` se resuelve desde
        ``THESIS_REPO_DIR`` o la raíz del proyecto.
    dry_run
        Si es ``True``, no escribe en disco.
    auto_commit
        Si es ``True`` y ``dry_run=False``, publica inmediatamente en el ledger.
        Si es ``False``, deja los records en :data:`GLOBAL_REGISTRY` para la
        publicación al final del proceso.

    Returns
    -------
    dict[str, Any]
        Resumen con ``records`` y ``summary``.
    """
    records = _run_adapters(experiment_id, payload)

    # El chapter por defecto solo aplica si el adapter no lo definió.
    if chapter:
        updated: list[ResultRecord] = []
        for rec in records:
            if rec.chapter == _UNKNOWN:
                data = rec.to_dict()
                data["chapter"] = chapter
                updated.append(ResultRecord(**data))
            else:
                updated.append(rec)
        records = updated

    validated: list[ResultRecord] = []
    for rec in records:
        vrec, _ = validate_record(rec)
        validated.append(vrec)

    if auto_commit:
        cfg = LedgerConfig.from_thesis_repo(thesis_repo_dir)
        cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        from thesis.results_ledger import merge_records_into_ledger

        summary = merge_records_into_ledger(validated, cfg, dry_run=dry_run)
        return {"experiment_id": experiment_id, "records": [r.to_dict() for r in validated], "summary": summary}

    GLOBAL_REGISTRY.record_many(validated)
    return {"experiment_id": experiment_id, "records": [r.to_dict() for r in validated], "queued": len(validated)}


def _atexit_publish() -> None:
    """Hook de ``atexit``: publica automáticamente si está habilitado."""
    if not _config.PUBLISH_RESULTS_TO_LEDGER:
        return
    if os.environ.get("THESIS_PUBLISH_RESULTS", "1").lower() in ("0", "false", "no"):
        return
    if not GLOBAL_REGISTRY.records():
        return
    if _config.THESIS_REPO_DIR is None:
        # Si no hay ruta explícita, intentamos la raíz del proyecto.
        try:
            LedgerConfig.from_thesis_repo()
        except RuntimeError:
            return
    try:
        GLOBAL_REGISTRY.publish()
    except (RuntimeError, OSError, ValueError):
        # En el hook de salida no queremos fallar ruidosamente.
        pass


atexit.register(_atexit_publish)

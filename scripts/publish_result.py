#!/usr/bin/env python3
"""Publica métricas de un experimento en el ledger de resultados.

Uso
---
Publicar desde un payload estructurado::

    python scripts/publish_result.py \
        --experiment volatility/lstm_daily \
        notebooks/volatility/payload.json

Publicar un record ya formado::

    python scripts/publish_result.py --record record.json

La ruta al repositorio de la tesis se resuelve en este orden:
``--thesis-repo`` > ``THESIS_REPO_DIR`` > raíz del proyecto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Asegurar que ``src`` esté en el path cuando se ejecuta desde scripts/.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.results import LedgerConfig, ResultRecord, publish, validate_record
from thesis.results_ledger import merge_records_into_ledger


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_payload(args: argparse.Namespace) -> int:
    payload_path = Path(args.input)
    if not payload_path.exists():
        print(f"ERROR: no existe {payload_path}", file=sys.stderr)
        return 1

    payload = _load_json(payload_path)
    summary = publish(
        experiment_id=args.experiment,
        payload=payload,
        chapter=args.chapter,
        thesis_repo_dir=args.thesis_repo,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _publish_record(args: argparse.Namespace) -> int:
    record_path = Path(args.input)
    if not record_path.exists():
        print(f"ERROR: no existe {record_path}", file=sys.stderr)
        return 1

    data = _load_json(record_path)
    if isinstance(data, dict):
        data = [data]

    records: list[ResultRecord] = []
    for item in data:
        rec, issues = validate_record(ResultRecord(**item))
        if issues and args.verbose:
            print(f"WARN: {issues}", file=sys.stderr)
        records.append(rec)

    cfg = LedgerConfig.from_thesis_repo(args.thesis_repo)
    cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    summary = merge_records_into_ledger(records, cfg, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publica métricas de un experimento en el ledger de resultados."
    )
    parser.add_argument("--experiment", help="Identificador del experimento")
    parser.add_argument("--record", action="store_true", help="El input es un record JSON")
    parser.add_argument("--chapter", default=None, help="Capítulo de la tesis")
    parser.add_argument("--thesis-repo", default=None, help="Ruta al repo de la tesis")
    parser.add_argument("--dry-run", action="store_true", help="No escribir en disco")
    parser.add_argument("--verbose", action="store_true", help="Mostrar advertencias")
    parser.add_argument("input", nargs="?", help="Archivo JSON con payload o record")

    args = parser.parse_args(argv)

    if not args.input:
        parser.error("Se requiere un archivo JSON de entrada.")

    if args.record:
        return _publish_record(args)

    if not args.experiment:
        parser.error("Se requiere --experiment para publicar un payload.")

    return _publish_payload(args)


if __name__ == "__main__":
    raise SystemExit(main())

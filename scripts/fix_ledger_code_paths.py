#!/usr/bin/env python3
"""Corrige la columna de código de las filas heredadas del ledger.

Las primeras versiones del ledger derivaban la ruta del notebook a partir del
``experiment_id`` (``notebooks/feature_analysis/momentum.py``), que no coincide
con el nombre real del archivo (``momentum_analysis.py``).  Este script
resuelve la ruta real desde :data:`thesis.results_adapters.CODE_PATHS` y
reescribe tanto el sidecar JSON como el bloque gestionado del markdown.

No altera valores de resultados, sólo metadatos de procedencia.  Es
idempotente: ejecutarlo sobre un ledger ya corregido no cambia nada.

Uso
---
::

    python scripts/fix_ledger_code_paths.py [--thesis-repo RUTA] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.results import _UNKNOWN, LedgerConfig, ResultRecord
from thesis.results_adapters import resolve_code_path
from thesis.results_ledger import merge_records_into_ledger


def _load_records(path: Path) -> list[ResultRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", [])
    known = ResultRecord.__dataclass_fields__
    records = []
    for item in data:
        fields = {k: v for k, v in item.items() if k in known}
        records.append(ResultRecord(**fields))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Corrige la ruta de código de las filas del ledger."
    )
    parser.add_argument("--thesis-repo", default=None, help="Raíz del repo de la tesis")
    parser.add_argument("--dry-run", action="store_true", help="No escribir en disco")
    args = parser.parse_args(argv)

    cfg = LedgerConfig.from_thesis_repo(args.thesis_repo)
    if not Path(cfg.sidecar_path).exists():
        print(f"ERROR: no existe {cfg.sidecar_path}", file=sys.stderr)
        return 1

    records = _load_records(Path(cfg.sidecar_path))

    changed = 0
    fixed: list[ResultRecord] = []
    for rec in records:
        if rec.code_path and rec.code_path != _UNKNOWN:
            expected = resolve_code_path(rec.experiment)
            if expected != _UNKNOWN and rec.code_path != expected:
                data = rec.to_dict()
                data["code_path"] = expected
                rec = ResultRecord(**data)
                changed += 1
        fixed.append(rec)

    summary = merge_records_into_ledger(fixed, cfg, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "records": len(records),
                "code_paths_corregidos": changed,
                "written": summary["written"],
                "ledger": summary["ledger_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Importador idempotente del ledger de resultados.

Este módulo separa el contenido gestionado (tabla de resultados) de las
anotaciones manuales, fusiona filas por ``result_id`` y re-escribe solo el
bloque gestionado.  El sidecar JSON conserva los metadatos completos de cada
record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis.results import _UNKNOWN, LedgerConfig, ResultRecord, _format_value

#: Marcadores que delimitan el bloque gestionado en el markdown.
BEGIN_MARKER = "<!-- BEGIN_RESULTS_TABLE -->"
END_MARKER = "<!-- END_RESULTS_TABLE -->"

#: Columnas de la tabla markdown (en orden).
_LEDGER_COLUMNS: tuple[str, ...] = (
    "result_id",
    "Capítulo",
    "Experimento",
    "Modelo",
    "Baseline",
    "Métrica",
    "Valor",
    "Unidad",
    "N",
    "Estado",
    "Fuente/Artefacto",
    "Código+Git",
    "Protocolo",
)

#: Mapeo campo -> columna (el orden sigue el contrato con method-auditor).
_FIELD_TO_COLUMN: dict[str, str] = {
    "result_id": "result_id",
    "chapter": "Capítulo",
    "experiment": "Experimento",
    "model": "Modelo",
    "baseline": "Baseline",
    "metric_name": "Métrica",
    "metric_value": "Valor",
    "unit": "Unidad",
    "sample_size": "N",
    "status": "Estado",
    "artifacts": "Fuente/Artefacto",
    "code_path": "Código+Git",
    "protocol": "Protocolo",
}


def _parse_table_rows(text: str) -> list[dict[str, Any]]:
    """Parsea filas de una tabla markdown."""
    rows: list[dict[str, Any]] = []
    lines = [ln.rstrip() for ln in text.splitlines()]
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("|") and "result_id" in line:
            header_idx = i
            break
    if header_idx is None:
        return rows

    header_cells = [c.strip() for c in lines[header_idx].split("|")[1:-1]]
    # Saltar la línea de separación.
    for line in lines[header_idx + 2 :]:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < len(header_cells):
            continue
        row = {}
        for h, c in zip(header_cells, cells):
            row[h] = c
        rows.append(row)
    return rows


def _row_to_record(row: dict[str, Any]) -> ResultRecord:
    """Convierte una fila del ledger en :class:`ResultRecord`."""
    column_to_field = {v: k for k, v in _FIELD_TO_COLUMN.items()}
    data: dict[str, Any] = {f: _UNKNOWN for f in ResultRecord.__dataclass_fields__}
    for col, value in row.items():
        field = column_to_field.get(col)
        if field is None:
            continue
        data[field] = value
    return ResultRecord(**data)


def parse_ledger(path: Path | str) -> dict[str, Any]:
    """Lee el ledger y separa bloque gestionado vs anotaciones manuales.

    Returns
    -------
    dict
        ``preamble`` (texto antes del bloque gestionado),
        ``postamble`` (texto después),
        ``manual_annotations`` (preamble + postamble),
        ``managed_records`` (lista de :class:`ResultRecord`).
    """
    path = Path(path)
    if not path.exists():
        return {
            "preamble": "",
            "postamble": "",
            "manual_annotations": "",
            "managed_records": [],
        }

    text = path.read_text(encoding="utf-8")
    begin = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)

    if begin == -1 or end == -1:
        # No hay bloque gestionado: todo es manual.
        return {
            "preamble": text,
            "postamble": "",
            "manual_annotations": text,
            "managed_records": [],
        }

    preamble = text[:begin]
    managed_text = text[begin + len(BEGIN_MARKER) : end]
    postamble = text[end + len(END_MARKER) :]

    rows = _parse_table_rows(managed_text)
    records = [_row_to_record(r) for r in rows]

    return {
        "preamble": preamble,
        "postamble": postamble,
        "manual_annotations": preamble + postamble,
        "managed_records": records,
    }


def _render_table(records: list[ResultRecord]) -> str:
    """Renderiza la tabla markdown a partir de records."""
    if not records:
        rows: list[list[str]] = []
    else:
        rows = []
        for rec in records:
            row = rec.to_row()
            cells = []
            for col in _LEDGER_COLUMNS:
                field = {v: k for k, v in _FIELD_TO_COLUMN.items()}.get(col)
                if field:
                    value = row.get(field, _UNKNOWN)
                elif col == "result_id":
                    value = row.get("result_id", _UNKNOWN)
                else:
                    value = _UNKNOWN
                cells.append(_format_value(value).replace("|", "\\|"))
            rows.append(cells)

    # Calcular anchos.
    headers = list(_LEDGER_COLUMNS)
    widths = [len(h) for h in headers]
    for cells in rows:
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))

    def _line(cells: list[str]) -> str:
        parts = [f" {cell:<{widths[i]}} " for i, cell in enumerate(cells)]
        return "|" + "|".join(parts) + "|"

    sep = "|" + "|".join(["-" * (w + 2) for w in widths]) + "|"
    lines = [_line(headers), sep] + [_line(cells) for cells in rows]
    return "\n".join(lines)


def _dedupe_records(records: list[ResultRecord]) -> list[ResultRecord]:
    """Conserva la última aparición por ``result_id``."""
    seen: dict[str, ResultRecord] = {}
    for rec in records:
        seen[rec.result_id] = rec
    return list(seen.values())


def merge_records_into_ledger(
    records: list[ResultRecord],
    cfg: LedgerConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fusiona ``records`` con el ledger existente y escribe el resultado.

    La fusión es por ``result_id``: los nuevos records reemplazan a los
    existentes.  Las anotaciones manuales fuera del bloque gestionado se
    preservan textualmente.
    """
    ledger_path = Path(cfg.ledger_path)
    sidecar_path = Path(cfg.sidecar_path)

    parsed = parse_ledger(ledger_path)
    existing = parsed["managed_records"]

    merged = _dedupe_records(existing + records)
    merged = sorted(merged, key=lambda r: (r.chapter, r.experiment, r.metric_name))

    table_md = _render_table(merged)
    managed_block = f"{BEGIN_MARKER}\n\n{table_md}\n\n{END_MARKER}"

    output = parsed["preamble"] + managed_block + parsed["postamble"]
    # Normalizar dobles saltos de línea al final.
    output = output.rstrip() + "\n"

    if not dry_run:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(output, encoding="utf-8")
        sidecar_path.write_text(
            json.dumps(
                [r.to_dict() for r in merged],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    return {
        "written": not dry_run,
        "ledger_path": str(ledger_path),
        "sidecar_path": str(sidecar_path),
        "before": len(existing),
        "added": len(records),
        "after": len(merged),
        "result_ids": [r.result_id for r in merged],
    }

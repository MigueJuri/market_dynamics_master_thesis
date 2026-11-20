"""Tests para ``thesis.results_ledger``.

No requieren red ni datos reales.
"""

from __future__ import annotations

from thesis.results import LedgerConfig, ResultRecord, derive_result_id
from thesis.results_ledger import (
    BEGIN_MARKER,
    END_MARKER,
    merge_records_into_ledger,
    parse_ledger,
)


def _make_record(chapter: str, experiment: str, metric: str, value: float) -> ResultRecord:
    return ResultRecord(
        result_id=derive_result_id(
            chapter=chapter,
            experiment=experiment,
            metric_name=metric,
            fingerprint=value,
        ),
        chapter=chapter,
        experiment=experiment,
        metric_name=metric,
        metric_value=value,
        unit="adimensional",
        status="published",
    )


def test_merge_records_into_ledger_creates_managed_block(tmp_path):
    cfg = LedgerConfig(
        ledger_path=tmp_path / "results-ledger.md",
        sidecar_path=tmp_path / "results-ledger.json",
    )
    records = [_make_record("Cap1", "exp1", "R²", 0.42)]
    summary = merge_records_into_ledger(records, cfg)
    assert summary["after"] == 1

    text = cfg.ledger_path.read_text(encoding="utf-8")
    assert BEGIN_MARKER in text
    assert END_MARKER in text
    assert "result_id" in text
    assert "R²" in text


def test_second_publish_does_not_duplicate(tmp_path):
    cfg = LedgerConfig(
        ledger_path=tmp_path / "results-ledger.md",
        sidecar_path=tmp_path / "results-ledger.json",
    )
    records = [_make_record("Cap1", "exp1", "R²", 0.42)]
    merge_records_into_ledger(records, cfg)
    merge_records_into_ledger(records, cfg)
    text = cfg.ledger_path.read_text(encoding="utf-8")
    assert text.count("R²") == 1


def test_manual_annotations_preserved(tmp_path):
    cfg = LedgerConfig(
        ledger_path=tmp_path / "results-ledger.md",
        sidecar_path=tmp_path / "results-ledger.json",
    )
    original = (
        "# Ledger de resultados\n\n"
        "Anotación manual que no debe borrarse.\n\n"
        f"{BEGIN_MARKER}\n\n"
        "| result_id | Capítulo | Experimento | Modelo | Baseline | Métrica | Valor | Unidad | N | Estado | Fuente/Artefacto | Código+Git | Protocolo |\n"
        "| - | - | - | - | - | - | - | - | - | - | - | - | - |\n"
        "| abc123 | Cap1 | exp1 | M1 | B1 | R² | 0.5 | adimensional | 100 | published | none | code.py | proto |\n"
        f"\n{END_MARKER}\n\n"
        "## Otra sección manual\n\nTexto extra.\n"
    )
    cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ledger_path.write_text(original, encoding="utf-8")

    new_records = [_make_record("Cap1", "exp1", "MAE", 0.1)]
    merge_records_into_ledger(new_records, cfg)

    text = cfg.ledger_path.read_text(encoding="utf-8")
    assert "Anotación manual que no debe borrarse" in text
    assert "## Otra sección manual" in text
    assert "Texto extra" in text
    assert "MAE" in text
    assert "abc123" in text


def test_migrate_placeholder_to_managed_structure(tmp_path):
    cfg = LedgerConfig(
        ledger_path=tmp_path / "results-ledger.md",
        sidecar_path=tmp_path / "results-ledger.json",
    )
    placeholder = (
        "# Ledger de resultados\n\n"
        "| Capítulo | Métrica | Valor | Fuente |\n"
        "| - | - | - | - |\n"
        "| Cap1 | R² | 0.5 | notebook |\n"
    )
    cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ledger_path.write_text(placeholder, encoding="utf-8")

    new_records = [_make_record("Cap1", "exp1", "RMSE", 0.2)]
    merge_records_into_ledger(new_records, cfg)

    text = cfg.ledger_path.read_text(encoding="utf-8")
    assert "# Ledger de resultados" in text
    assert BEGIN_MARKER in text
    assert END_MARKER in text
    # El contenido antiguo se preserva fuera del bloque gestionado.
    assert "| Capítulo | Métrica | Valor | Fuente |" in text


def test_parse_ledger_returns_records_and_annotations(tmp_path):
    cfg = LedgerConfig(
        ledger_path=tmp_path / "results-ledger.md",
        sidecar_path=tmp_path / "results-ledger.json",
    )
    records = [
        _make_record("Cap1", "exp1", "R²", 0.42),
        _make_record("Cap1", "exp1", "MAE", 0.1),
    ]
    merge_records_into_ledger(records, cfg)

    parsed = parse_ledger(cfg.ledger_path)
    assert len(parsed["managed_records"]) == 2
    assert BEGIN_MARKER not in parsed["manual_annotations"]
    assert END_MARKER not in parsed["manual_annotations"]


def test_table_is_human_readable(tmp_path):
    cfg = LedgerConfig(
        ledger_path=tmp_path / "results-ledger.md",
        sidecar_path=tmp_path / "results-ledger.json",
    )
    records = [_make_record("Cap1", "exp1", "R²", 0.42)]
    merge_records_into_ledger(records, cfg)
    text = cfg.ledger_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    assert len(lines) >= 2
    # La primera línea debe tener al menos tantas columnas como el header.
    header_cells = len(lines[0].split("|"))
    for line in lines[1:]:
        assert len(line.split("|")) == header_cells

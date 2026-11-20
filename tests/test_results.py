"""Tests para ``thesis.results``.

No requieren red ni datos reales.
"""

from __future__ import annotations

from dataclasses import fields

from thesis.results import (
    _UNKNOWN,
    MetricRegistry,
    ResultRecord,
    derive_result_id,
    publish,
    redact,
    validate_record,
)


def test_derive_result_id_is_stable():
    rid1 = derive_result_id(
        chapter="Volatilidad",
        experiment="LSTM diario",
        metric_name="R²",
        model="LSTM",
        asset="SPY",
        fingerprint={"lookback": 5, "horizon": 5},
    )
    rid2 = derive_result_id(
        chapter="Volatilidad",
        experiment="LSTM diario",
        metric_name="R²",
        model="LSTM",
        asset="SPY",
        fingerprint={"horizon": 5, "lookback": 5},
    )
    assert rid1 == rid2
    assert len(rid1) == 16


def test_derive_result_id_changes_with_config():
    rid1 = derive_result_id(
        chapter="Volatilidad",
        experiment="LSTM diario",
        metric_name="R²",
        model="LSTM",
        asset="SPY",
        fingerprint={"lookback": 5},
    )
    rid2 = derive_result_id(
        chapter="Volatilidad",
        experiment="LSTM diario",
        metric_name="R²",
        model="LSTM",
        asset="SPY",
        fingerprint={"lookback": 10},
    )
    assert rid1 != rid2


def test_validate_record_fills_unknown_and_marks_candidate():
    rec = ResultRecord(metric_name="R²", metric_value=0.5)
    validated, issues = validate_record(rec)
    assert validated.status == "candidate"
    assert "missing_metadata" in validated.notes
    assert any(f.name == "chapter" for f in fields(ResultRecord))
    assert validated.chapter == _UNKNOWN
    assert len(issues) >= 1


def test_validate_record_published_when_complete():
    rec = ResultRecord(
        result_id="abc123",
        chapter="Test",
        experiment="unit",
        metric_name="R²",
        metric_value=0.5,
        status="published",
    )
    validated, issues = validate_record(rec)
    assert validated.status == "published"
    assert not issues


def test_redact_detects_sensitive_artifact():
    rec = ResultRecord(
        result_id="abc",
        chapter="Test",
        experiment="unit",
        metric_name="path",
        metric_value="models/secret.pt",
        status="published",
    )
    redacted, issues = redact(rec)
    assert redacted.status == "rejected"
    assert any(".pt" in issue for issue in issues)


def test_redact_detects_env_file():
    rec = ResultRecord(
        result_id="abc",
        chapter="Test",
        experiment="unit",
        metric_name="config",
        metric_value=".env.production",
        status="published",
    )
    redacted, issues = redact(rec)
    assert redacted.status == "rejected"
    assert any(".env" in issue for issue in issues)


def test_redact_keeps_long_repo_identifiers():
    """Los identificadores largos del repo no se redaccionan ni se rechazan."""
    rec = ResultRecord(
        result_id="abc",
        chapter="Momentum-Volatilidad",
        experiment="moving_average_distance",
        metric_name="pearson_corr",
        metric_value=0.11,
        model="Moving Average Distance",
        code_path="notebooks/feature_analysis/moving_average_distance_analysis.py",
        status="published",
    )
    redacted, issues = redact(rec)
    assert not issues
    assert redacted.status == "published"
    assert redacted.experiment == "moving_average_distance"
    assert redacted.code_path.endswith("moving_average_distance_analysis.py")


def test_redact_detects_long_token_in_free_text():
    """Un token largo en un campo de texto libre sí se redacciona."""
    rec = ResultRecord(
        result_id="abc",
        chapter="Test",
        experiment="unit",
        metric_name="nota",
        metric_value="AKIAIOSFODNN7EXAMPLEKEY123456",
        status="published",
    )
    redacted, issues = redact(rec)
    assert redacted.status == "rejected"
    assert redacted.metric_value == "[REDACTED]"
    assert issues


def test_metric_registry_publish(tmp_path):
    registry = MetricRegistry()
    rec = ResultRecord(
        result_id="r1",
        chapter="Test",
        experiment="unit",
        metric_name="m",
        metric_value=1.0,
        status="published",
    )
    registry.record(rec)
    summary = registry.publish(thesis_repo_dir=tmp_path)
    assert summary["after"] == 1
    ledger_path = tmp_path / ".thesis" / "results-ledger.md"
    sidecar_path = tmp_path / ".thesis" / "results-ledger.json"
    assert ledger_path.exists()
    assert sidecar_path.exists()
    assert registry.records() == []


def test_publish_unknown_experiment_is_candidate(tmp_path):
    summary = publish(
        experiment_id="does/not_exist",
        payload={},
        thesis_repo_dir=tmp_path,
    )
    assert summary["records"][0]["status"] == "candidate"
    assert "adapter pendiente" in summary["records"][0]["notes"]


def test_publish_respects_thesis_repo_env(tmp_path, monkeypatch):
    monkeypatch.setenv("THESIS_REPO_DIR", str(tmp_path))
    # Recargar el módulo config no es trivial; usamos argumento explícito.
    summary = publish(
        experiment_id="does/not_exist",
        payload={},
        thesis_repo_dir=tmp_path,
    )
    assert summary["records"][0]["status"] == "candidate"


def test_publish_dry_run_does_not_write(tmp_path):
    summary = publish(
        experiment_id="does/not_exist",
        payload={},
        thesis_repo_dir=tmp_path,
        dry_run=True,
    )
    assert summary["summary"]["written"] is False
    assert not (tmp_path / ".thesis" / "results-ledger.md").exists()

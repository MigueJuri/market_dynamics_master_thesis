# Patrones de test

Bloques listos para copiar y adaptar. Los tres niveles son acumulativos:
el unitario no necesita red, el de integración sí, y el end-to-end cubre el
orquestador completo.

## Configuración del marcador

En `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: tests que requieren red o datos reales (se omiten por defecto)",
]
addopts = "-m 'not integration'"
```

Para ejecutar solo los de integración:

```bash
uv run pytest -m integration -v
```

## Nivel 1 — Unitarios con datos sintéticos

```python
def _make_ohlcv(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.015, n))
    low = close / (1 + rng.uniform(0.001, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
```

Si el módulo depende de una librería opcional o pesada:

```python
pytest.importorskip("torch")
```

## Nivel 2 — Integración con datos reales

```python
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def real_ohlcv() -> pd.DataFrame:
    """Fixture de sesión: descarga una sola vez y la comparte."""
    return download_ohlcv("SPY", start="2015-01-01")
```

Usa `scope="session"` para no descargar los mismos datos en cada test.

Invariantes de mercado que conviene comprobar sobre datos reales:

```python
def test_real_ohlcv_invariantes(real_ohlcv):
    assert not real_ohlcv.isna().any().any()
    assert (real_ohlcv[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert (real_ohlcv["High"] >= real_ohlcv["Low"]).all()
    assert (real_ohlcv["High"] >= real_ohlcv[["Open", "Close"]].max(axis=1)).all()
    assert (real_ohlcv["Low"] <= real_ohlcv[["Open", "Close"]].min(axis=1)).all()
```

## Nivel 3 — Ausencia de look-ahead

La prueba más valiosa en series temporales: corrompe el futuro y comprueba
que el pasado no se mueve.

```python
def test_predicciones_invariantes_al_futuro():
    raw = download_ohlcv("SPY", start="2015-01-01")
    data_a = build_volatility_dataset(raw, horizon=5)
    train_a, _, test_a = temporal_split(data_a, TRAIN_END, VAL_END)

    cut = test_a.index[len(test_a) // 2]

    raw_b = raw.copy()
    rng = np.random.default_rng(0)
    future = raw_b.index > cut
    n_fut = int(future.sum())
    permuted = raw_b.loc[future, "Close"].to_numpy()[rng.permutation(n_fut)]
    jitter = rng.uniform(0.9, 1.1, n_fut)
    raw_b.loc[future, "Close"] = permuted * jitter
    raw_b.loc[future, "Open"] = permuted * jitter
    raw_b.loc[future, "High"] = permuted * jitter * 1.01
    raw_b.loc[future, "Low"] = permuted * jitter * 0.99

    data_b = build_volatility_dataset(raw_b, horizon=5)
    train_b, _, test_b = temporal_split(data_b, TRAIN_END, VAL_END)

    # 1. El train no cambia: la corrupción es posterior.
    pd.testing.assert_frame_equal(train_a, train_b)

    # 2. Las features hasta el corte son idénticas.
    cols = [c for c in data_a.columns if c != "target_log_vol"]
    np.testing.assert_allclose(
        data_a.loc[:cut, cols].to_numpy(),
        data_b.loc[:cut, cols].to_numpy(),
        rtol=1e-12, atol=1e-12,
    )

    # 3. El escalador se ajusta solo con el train.
    scaled = scale_splits(
        train_a[cols].values,
        train_a[cols].values,
        train_a[cols].values,
        train_a["target_log_vol"].values,
        train_a["target_log_vol"].values,
        train_a["target_log_vol"].values,
    )
    assert np.allclose(scaled["X_scaler"].mean_, train_a[cols].values.mean(axis=0))

    # 4. Las predicciones hasta el corte son idénticas.
    ...
```

Una variante más ligera, suficiente para la mayoría de dominios, comprueba
solo la causalidad de las features truncando la serie:

```python
def test_features_son_causales(real_ohlcv):
    t = len(real_ohlcv) // 2
    truncated = compute_volatility_features(real_ohlcv.iloc[: t + 1])
    full = compute_volatility_features(real_ohlcv)
    for col in full.columns:
        value = full[col].iloc[t]
        if pd.isna(value):
            continue
        assert truncated[col].iloc[-1] == pytest.approx(value, rel=1e-12)
```

## Nivel 4 — End-to-end del orquestador

```python
def test_orquestador_end_to_end(tmp_path):
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")

    module = _cargar_modulo("notebooks/volatility/Volatilidad_LSTM.py")
    config = {
        "max_epochs": 2,
        "min_epochs": 1,
        "patience": 1,
        "model_path": str(tmp_path / "model.pt"),
        "metrics_path": str(tmp_path / "metrics.csv"),
        "config_path": str(tmp_path / "config.json"),
        "plot_path": str(tmp_path / "report.png"),
    }

    results = module.run_experiment(config)

    assert results["metrics"]["lstm"]["N"] > 100
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "report.png").exists()
```

Cargar un cuaderno como módulo:

```python
import importlib.util
from pathlib import Path


def _cargar_modulo(ruta: str):
    notebook = Path(__file__).resolve().parents[1] / ruta
    spec = importlib.util.spec_from_file_location("notebook_bajo_test", notebook)
    module = importlib.util.module_from_spec(spec)
    sys.modules["notebook_bajo_test"] = module
    spec.loader.exec_module(module)
    return module
```

## Nivel 5 — Import sin efectos

```python
def test_import_no_ejecuta_pipeline():
    module = _cargar_modulo("notebooks/volatility/Volatilidad_LSTM.py")
    assert callable(module.run_experiment)
    assert "features" in module.CONFIG
```

## Buenas prácticas

- Compara con `pytest.approx` para flotantes; reserva `assert_allclose` con
  tolerancia explícita para comparaciones elemento a elemento.
- Usa `tmp_path` para cualquier artefacto: nada se escribe en el repo.
- Reduce el config del end-to-end (épocas, batch size) para que el test no
  tarde más de unos segundos.
- Un test debe fallar por una razón clara. Si el mensaje de aserción no
  dice qué se rompió, reescribe la aserción.

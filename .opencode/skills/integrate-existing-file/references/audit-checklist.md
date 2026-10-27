# Catálogo de defectos y cómo detectarlos

Cada entrada indica cómo reconocer el defecto, por qué importa en un
repositorio de investigación y cuál es la corrección mínima.

## 1. Efectos secundarios al importar

**Señal.** Llamadas a nivel de módulo: descargas de datos, entrenamiento,
escritura de ficheros, `plt.show()`.

**Por qué importa.** Un módulo con efectos al importar no se puede testear,
no se puede reutilizar y falla si se importa desde otro cuaderno.

**Detección.**

```bash
# llamadas suspectas fuera de funciones
grep -nE "^(download|fit|train|plt\.show|open\(|to_csv|savefig)" <archivo>
```

**Corrección.** Encapsula el pipeline en `run_experiment(config=None)` y
protege la ejecución:

```python
if __name__ == "__main__":
    results = run_experiment()
```

En un cuaderno Jupytext, `__name__` vale `"__main__"`, así que el guard se
dispara una sola vez al ejecutar el cuaderno y no al importarlo.

**Prueba de regresión.**

```python
def test_import_no_ejecuta_pipeline():
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.run_experiment)
```

## 2. Estado global

**Señal.** Funciones que leen `CONFIG`, `DEVICE`, `SEED` o cualquier
variable definida a nivel de módulo en vez de recibirla.

**Por qué importa.** La función deja de ser testeable de forma aislada y su
resultado depende del orden de ejecución.

**Corrección.** Pasa la configuración como `dict` y el dispositivo como
argumento:

```python
def train_lstm(model, train_loader, X_val, y_val, config, device, criterion=None):
    ...
```

## 3. Variable global no definida

**Señal.** Una función referencia un nombre que se define en otro módulo o
en una celda anterior. Es el bug más costoso de estas refactorizaciones: no
falla hasta que se reordena el archivo.

**Detección.**

```bash
grep -nE "sigma_bar|^[A-Z_]+ =" <archivo>
```

**Corrección.** Conviértela en parámetro con valor por defecto explícito.

## 4. Orden de columnas no determinista

**Señal.** `df[list(un_set)]` o `df[list(diccionario)]`.

**Por qué importa.** El orden de columnas cambia entre ejecuciones según el
hash seed de Python, lo que rompe comparaciones, checkpoints y tests.

**Corrección.** Declara una tupla a nivel de módulo:

```python
OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
```

## 5. Imports muertos

**Señal.** Librerías importadas y nunca usadas: `optuna`,
`TimeSeriesSplit`, `BaseEstimator`.

**Corrección.** Elimínalas. No declares en `pyproject.toml` una dependencia
que el código no usa: infla el entorno sin motivo.

## 6. Código incrustado en una docstring

**Señal.** Un fragmento de código aparece dentro del texto de una
docstring, normalmente resto de una edición anterior.

**Por qué importa.** Rompe la función: el bloque se ejecuta como parte de
la cadena, no como código, y las variables que define no existen.

**Detección.** `python3 -m py_compile <archivo>` y busca líneas sueltas de
código dentro de cadenas.

## 7. Dependencia no declarada

**Señal.** `import torch` sin `torch` en `pyproject.toml`.

**Corrección.** Declárala. Si el wheel por defecto arrastra cientos de MB
irrelevantes (el caso de `torch`, que en PyPI incluye las bibliotecas CUDA
por ~3 GB), fija un índice específico:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Después `uv lock` y `uv sync`, y comprueba que el lock solo contenga la
dependencia y sus transitivas.

## 8. Rutas relativas al directorio de trabajo

**Señal.** `open("model.pt", "w")`, `savefig("figura.png")`.

**Por qué importa.** Los artefactos acaban en un directorio arbitrario
dependiendo de desde dónde se lance el script.

**Corrección.** Ancla las rutas con `pathlib.Path` relativo a la raíz del
repo, y acepta rutas como parámetros para poder testear en `tmp_path`.

## 9. Datos sin validar

**Señal.** El script descarga y usa los datos sin comprobar NaN, precios no
positivos ni columnas obligatorias.

**Por qué importa.** Un único valor corrupto produce features inválidas y
resultados silenciosamente incorrectos.

**Corrección.** Un validador que falle rápido:

```python
def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLCV: {missing}")
    ...
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Precio no positivo detectado en los datos OHLCV.")
    return df
```

## 10. Inconsistencias de mercado en los datos

**Señal.** En datos reales, `Close > High` o `Low > Close`.

**Por qué importa.** `yfinance` puede publicar la barra de la sesión en
curso con el cierre ya actualizado pero el máximo/mínimo todavía no. Eso
sesga los estimadores de rango (Parkinson, Garman-Klass) y cualquier
cálculo de rango intradía.

**Corrección.** Reconstruye la consistencia tras validar:

```python
df["High"] = df[["Open", "High", "Close"]].max(axis=1)
df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
```

## 11. Posible look-ahead

**Señal.** Una feature que consulta el futuro, o un escalador ajustado
sobre el train y el test juntos.

**Por qué importa.** Es el error que invalida un resultado entero sin
producir ningún síntoma visible: el modelo parece demasiado bueno.

**Corrección.** Tres barreras, todas verificables con tests:

1. **Features causales**: el valor en `t` solo depende de `t` y el pasado.
2. **Escalado sin fuga**: el scaler se ajusta **solo** con el train.
3. **División temporal disjunta**: las particiones no se solapan.

**Prueba decisiva.** Corrompe el futuro y comprueba que el pasado no cambia:

```python
raw_b = raw.copy()
raw_b.loc[raw_b.index > cut, "Close"] = valores_corruptos
# features y predicciones hasta `cut` deben ser idénticas
```

## 12. Comentarios que describen lo que el código ya no hace

**Señal.** Comentarios como "rolling scaler" sobre un `StandardScaler` global
por partición, o referencias a secciones que ya no existen.

**Por qué importa.** El comentario es documentación; si miente, el lector
toma decisiones erróneas.

**Corrección.** Actualiza el comentario o corrige el código. Si el
comentario describe una intención que el código no cumple, decide
explícitamente cuál de los dos cambia.

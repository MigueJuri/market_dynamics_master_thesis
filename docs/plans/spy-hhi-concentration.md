# Plan: figura de HHI por empresa del S&P 500

## Alcance

Añadir al cuaderno `notebooks/spy/SPY_sectores.py` una **nueva etapa** que
grafique la evolución del **Índice de Herfindahl-Hirschman (HHI) a nivel de
empresa** (concentración de la capitalización en unas pocas compañías del
S&P 500). El HHI se define como

    HHI(t) = Σ_i w_i(t)^2,  con  w_i(t) = market_cap_i(t) / Σ_j market_cap_j(t)

Toma valores en `[1/N, 1]`; cuanto más alto, más concentrado.

Siguiendo la convención del proyecto ("la lógica reutilizable vive en
`src/thesis`, el dibujo vive en el cuaderno"), la métrica se implementa en
`sector_composition.py` y sólo la figura en el notebook.

## No-objetivos

- No reconstruir series point-in-time de constituyentes ni de acciones en
  circulación: se reutiliza la aproximación existente (lista actual + acciones
  constantes), con sus sesgos ya documentados.
- No tocar `pair_trading.hhi` (mide concentración sectorial *dentro de un
  cluster* sobre categorías, no pesos de capitalización): no es reutilizable
  aquí.
- No modificar las figuras existentes (donut, área apilada, entropía).

## Archivos y funciones

1. **`src/thesis/sector_composition.py`**
   - Añadir `company_hhi(prices: pd.DataFrame, shares: pd.Series) -> pd.Series`
     junto a `build_sector_weights`, con la misma lógica de validación:
     - `market_cap = prices.mul(shares.reindex(prices.columns), axis=1)`;
     - `dropna(axis=1, how="all")` (descartar tickers sin datos);
     - chequear precios/acciones negativos (`ValueError`, como en
       `build_sector_weights`);
     - descartar filas con capitalización total `<= 0`;
     - `weights = market_cap.div(market_cap.sum(axis=1), axis=0)`;
     - `hhi = (weights ** 2).sum(axis=1)`, devolver `pd.Series` con
       `name="hhi"` e índice idéntico.
   - Opcional (bajo riesgo): factorizar `_market_cap(prices, shares)` para no
     duplicar la validación negativa. Si se hace, ejecutar toda la suite.
   - Actualizar el docstring del módulo (lista 1–3) para mencionar el HHI.

2. **`notebooks/spy/SPY_sectores.py`**
   - Importar `company_hhi` en el bloque de imports (líneas ~76-89).
   - Añadir celda `# %%` nueva "Etapa 6: HHI por empresa" (markdown + código),
     guardada con `if __name__ == "__main__":` como el resto:
     - `hhi = company_hhi(prices, shares)` (`prices`/`shares` ya están en
       scope tras la Etapa 2);
     - `plt.plot(hhi.index, hhi.to_numpy(), ...)` con título/etiquetas en
       español, `grid`, `margins(x=0)`;
     - `print` de HHI medio, mínimo y máximo (con fecha), como en la entropía;
     - guardar en `OUTPUT_DIR / "sp500_company_hhi.png"` respetando
       `CONFIG["save_figures"]` / `CONFIG["show_figure"]`.

3. **`tests/test_sector_composition.py`**
   - Tests unitarios sintéticos para `company_hhi`:
     - capitalización uniforme sobre `N` empresas → `HHI = 1/N`;
     - una empresa con todo el peso → `HHI ≈ 1`;
     - coincide con cálculo manual `Σ w_i^2`;
     - `HHI ∈ [1/N, 1]` y finito; nombre `"hhi"` e índice igual a la entrada;
     - ticker con precios/acciones `NaN` se descarta (no rompe);
     - precios negativos → `ValueError`.

## Criterios de aceptación

- `company_hhi` devuelve una `pd.Series` por fecha, valores en `[1/N, 1]`,
  sin `NaN`/`inf`; pesos por fila suman 1 (dentro de tolerancia ~1e-9).
- El notebook ejecutado produce `outputs/sp500_company_hhi.png` con la serie
  temporal del HHI.
- `uv run pytest tests/test_sector_composition.py` en verde (sin integración).
- `uv run ruff check src tests` sin errores.

## Riesgos principales

- **Sesgos heredados (no nuevos):** supervivencia + clasificación retroactiva
  + acciones en circulación constantes (ver `.. warning::` del notebook). El
  HHI hereda la misma aproximación; documentarlo en la celda markdown, no es
  un bug de la nueva etapa.
- **Leakage:** no aplica (cálculo puntual fila a fila, sin ventanas ni
  `shift`); el HHI sólo depende de la fila actual, igual que `sector_entropy`.
- **Numérico:** no hay logaritmos (sin riesgo de `-inf`); suma de cuadrados
  estable con ~500 empresas. Cuidar los `NaN` de tickers sin datos y las filas
  con total 0 (descartar como en `build_sector_weights`) para no producir
  `0/0`.
- **Unidades:** HHI adimensional en `[0,1]` (fracción). Indicar en el eje Y;
  opcionalmente anotar la convención alternativa ×10 000 (no aplicar en el
  dato, sólo comentario).
- **Refactor `_market_cap`:** si se factoriza, es un cambio de bajo riesgo pero
  toca `build_sector_weights`; validar con la suite completa de
  `test_sector_composition.py`.

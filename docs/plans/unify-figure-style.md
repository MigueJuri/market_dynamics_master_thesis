# Plan: estilo de figuras unificado para la tesis (LaTeX, 155 mm, 12 pt)

## Contexto

Las figuras de los notebooks irán en la tesis LaTeX a ancho completo de texto
(**155 mm = 6.102 in**, cuerpo a **12 pt**). Hoy cada notebook hace su propio
ajuste: `PLOT_RC` en `src/thesis/config.py` (que `plotting.py` aplica *al
importarse*), `rcParams.update` locales (Volatilidad_LSTM*, Trend_Following a
6.3 in/11 pt), `seaborn-v0_8-darkgrid` en portfolio_HRP, `figsize` de (6,3)
a (16,22), `dpi=300` sueltos, títulos en negrita, `bbox_inches="tight"` y casi
todo en PNG. Objetivo: un único estilo sobrio tipo LaTeX, aplicado igual en
todas las figuras, que se pueda editar y re-aplicar sin reiniciar el kernel y
personalizar figura a figura.

### Evaluación de la estrategia actual ((6,3) + dpi 300)

- **La idea de fondo es correcta**: si la figura se crea con su tamaño físico
  final, se incluye a escala 1 y los pt de matplotlib son pt reales del PDF.
- **El dpi no influye en el tamaño de la letra**; solo en la resolución del
  raster. Subirlo no corrige nada del tamaño.
- Fallos que rompen la equivalencia: 6 in ≠ 155 mm; `bbox_inches="tight"`
  recorta y cambia el ancho final → LaTeX re-escala y la fuente ya no mide
  12 pt; tamaños distintos por notebook; PNG en vez de vectorial.
- **Método nuevo**: ancho exacto 6.102 in en el estilo, `constrained_layout`
  en lugar de `tight`, guardado en PDF vectorial, e inclusión en LaTeX con
  `\includegraphics{fig.pdf}` (o `width=\textwidth`, que es lo mismo, escala 1).

## Diseño

### 1. Archivo de estilo `styles/thesis.mplstyle` (nuevo, en la raíz)

No es código de `src/`: `plt.style.use(ruta)` relee el archivo cada vez que se
ejecuta, así que editarlo y re-correr la celda de setup basta (sin reiniciar).
Contenido principal:

- Tamaño: `figure.figsize: 6.102, 3.771` (proporción áurea), `figure.dpi: 150`
  (pantalla), `savefig.dpi: 300` (solo raster), `savefig.format: pdf`,
  `savefig.bbox: standard`, `figure.constrained_layout.use: True`,
  `pdf.fonttype: 42`.
- Tipografía: `text.usetex: True`, `font.family: serif`,
  `text.latex.preamble: \usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}`
  (Latin Modern, igual que el documento; dependencias ya instaladas:
  latex, dvipng, lmodern, type1cm).
- Tamaños: `font.size`, `axes.labelsize`, `axes.titlesize`,
  `legend.fontsize`, `x/ytick.labelsize` = 12; `axes.titleweight: normal`.
- Aspecto sobrio: fondo blanco, spines superior/derecho off, `axes.linewidth`
  0.6, ticks finos, grid gris muy tenue (`alpha` 0.3, línea 0.4), leyenda sin
  marco, `lines.linewidth` 1.2, ciclo de colores apto daltónicos (Tol
  "bright"), `image.cmap: viridis`.

### 2. Cómo lo usa cada notebook (celda de setup)

```python
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)          # re-ejecutar tras editar el .mplstyle
W = plt.rcParams["figure.figsize"][0]   # ancho de texto en pulgadas
```

Personalización in situ: `figsize=(W, 0.45 * W)`, `with plt.rc_context({...}):`
o `ax.set_*` sobre los `fig, ax` devueltos. No se crea ningún helper en `src/`.

### 3. Limpieza en `src/thesis/`

- `config.py`: eliminar `PLOT_RC` y `FIGSIZE`; actualizar sus usos
  (`plotting.py`, `notebooks/spy/GeometricBrownianMotion.py`,
  `notebooks/spy/SPY_caracterizacion.py`).
- `plotting.py` (y gráficos en `btc.py`, `portfolio.py`): quitar el
  `plt.rcParams.update` a nivel de módulo; `figsize` por defecto `None`
  (→ rcParams) o, en paneles multi-fila, `None` resuelto en la llamada como
  `(W, k*W)` leyendo `rcParams`; quitar `dpi=300`, `fontweight="bold"` y
  `fontsize=8/9` hardcodeados; asegurar que devuelvan `fig`/`axes` y no
  guarden ni hagan `plt.show()` por su cuenta. Corregir de paso
  `plt.set_title` (bug en ~línea 1420).

### 4. Barrido de notebooks (mismo patrón en todos)

Todos los `.py` de `notebooks/**` incluido `notebooks/figures/*.py`
(representativos: `spy/SPY_caracterizacion.py`,
`volatility/Volatilidad_LSTM.py`, `portfolio/portfolio_HRP.py`,
`trend_following_strategy/Trend_Following_Strategy_v2.py`):

- Añadir la celda de setup; borrar `rcParams.update` locales,
  `TEXTWIDTH_IN`, `style.use("seaborn…")`, y entradas `figsize` en `CONFIG`.
- `figsize` → por defecto o `(W, ratio*W)`; quitar `dpi=` en `subplots` y
  `savefig`; quitar `bbox_inches="tight"`; extensión `.png` → `.pdf`
  (los nombres se conservan; renombrar según la convención de CLAUDE.md
  queda para un paso posterior de unificación).
- **Compatibilidad usetex**: escapar en etiquetas/títulos/leyendas `%`, `_`,
  `&`, `#` (p. ej. `"Retorno (%)"` → `r"Retorno (\%)"`, nombres de columna
  usados como leyenda vía `label=`). Buscar con `rg` en etiquetas y en
  `df.plot()` que hereda nombres de columnas.
- Títulos: en tesis el título va en el `\caption`; se dejan los existentes
  sin negrita (el usuario los quita si quiere).

### 5. Test

`tests/test_plot_style.py`: el estilo carga, `figure.figsize[0] ≈ 155/25.4`,
`font.size == 12`, `text.usetex` activo; y que funciones de `plotting.py`
creen figuras con el ancho de rcParams (sin figsize hardcodeado), usando
`rc_context({"text.usetex": False})` para que la suite no dependa de LaTeX.

### 6. Regla permanente para figuras futuras

Pedido del usuario: toda figura nueva que me pida debe usar este formato.

- `CLAUDE.md` §5 (Salidas), añadir: «Toda figura nueva aplica
  `plt.style.use(_ROOT / "styles" / "thesis.mplstyle")` en la celda de setup;
  ancho por defecto (= `\textwidth`, 155 mm) o `(W, ratio*W)`; sin `dpi=`,
  sin `bbox_inches="tight"`, sin tamaños de fuente ni negritas hardcodeados;
  guardar en `.pdf`; texto compatible con usetex (escapar `% _ & #`).
  Personalizar solo in situ (`rc_context`, kwargs, `ax.set_*`).»
- Memoria `feedback` (`thesis-figure-format`) con la misma regla y el porqué
  (figuras a escala 1 en LaTeX, fuente igual al cuerpo de 12 pt), enlazada
  desde `MEMORY.md`.
- Va en el commit 1 junto al estilo.

## Orden de trabajo y commits

0. Prerrequisito: el usuario commitea sus cambios pendientes (árbol limpio).
1. Copiar este plan a `docs/plans/unify-figure-style.md`.
2. Commit 1 `feat(plotting): agrega estilo thesis.mplstyle para LaTeX` (estilo + test).
3. Commit 2 `refactor(plotting): elimina PLOT_RC y tamaños hardcodeados en src`.
4. Commits 3..n, uno por carpeta de notebooks (`spy`, `volatility`,
   `pair_trading`, `portfolio`, `btc`, `feature_analysis`,
   `trend_following_strategy`, `figures`). `git push` tras cada commit.

## Verificación

- Figura de prueba: `pdfinfo` debe dar ancho de página 439.4 pt (155 mm);
  compilar un `.tex` mínimo `12pt` con `\includegraphics{prueba.pdf}` y
  comparar visualmente la fuente del eje con el cuerpo.
- `uv run ruff check src tests` y `uv run pytest`.
- Ejecutar cada notebook con `MPLBACKEND=Agg uv run python <nb>.py` (los que
  tengan datos locales) y revisar que no haya errores de LaTeX ni texto
  cortado; abrir 2–3 PDFs generados para chequear el aspecto.

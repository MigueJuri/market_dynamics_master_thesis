# Ledger de resultados de la tesis

Este archivo cruza los resultados canónicos del repositorio de código con el
texto de la tesis.  El bloque entre `<!-- BEGIN_RESULTS_TABLE -->

| result_id        | Capítulo | Experimento    | Modelo  | Baseline | Métrica         | Valor   | Unidad  | N       | Estado    | Fuente/Artefacto | Código+Git | Protocolo |
|------------------|----------|----------------|---------|----------|-----------------|---------|---------|---------|-----------|------------------|------------|-----------|
| e44c63ca64e4cfb7 | unknown  | does/not_exist | unknown | unknown  | adapter_pending | unknown | unknown | unknown | candidate | unknown          | unknown    | unknown   |

<!-- END_RESULTS_TABLE -->` se regenera automáticamente desde los notebooks;
no editarlo a mano porque se sobrescribirá.  Las secciones fuera de ese bloque
se preservan tal cual.

## Anotaciones manuales

Tabla placeholder original (4 columnas) conservada como referencia:

| Capítulo | Métrica | Valor | Fuente |
|----------|---------|-------|--------|
| —        | —       | —     | —      |

<!-- BEGIN_RESULTS_TABLE -->

| result_id | Capítulo | Experimento | Modelo | Baseline | Métrica | Valor | Unidad | N | Estado | Fuente/Artefacto | Código+Git | Protocolo |
|-----------|----------|-------------|--------|----------|---------|-------|--------|---|--------|------------------|------------|-----------|

<!-- END_RESULTS_TABLE -->

## Notas

- Los registros con estado `candidate` requieren revisión de unidades, periodo
  o protocolo antes de ser citados en la tesis.
- `Código+Git` apunta al notebook canónico correspondiente en el repo de
  código; `Fuente/Artefacto` indica el archivo CSV/PNG/PDF generado.

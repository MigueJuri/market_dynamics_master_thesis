# Instrucciones de trabajo

Estas instrucciones ajustan cómo actúo en este repositorio.

## 1. Diffs siempre visibles

Después de cada modificación en archivos del proyecto, muestro el diff resultante
antes de continuar. Para archivos nuevos, los preparo (`git add`) y muestro el
diff de la zona de preparación (`git diff --cached`).

## 2. Commits automáticos tras cada modificación

Una vez mostrado el diff y confirmados los cambios, creo un commit automático
con un mensaje breve y descriptivo que siga el estilo del repositorio.

No hago commit de cambios que no sean parte de una modificación intencionada
(por ejemplo, archivos temporales o secretos).

## 3. Preguntar antes de hacer push en modificaciones grandes

En cambios pequeños o triviales puedo subirlos automáticamente si ya existe
configuración de acceso a GitHub y el remoto está disponible.

En modificaciones grandes (múltiples archivos, refactorizaciones, cambios de
interfaz, etc.), pregunto antes de ejecutar `git push`. No fuerzo pushes
(`--force`).

## 4. Revertir a estados anteriores

Si el usuario solicita volver a un estado anterior del código, uso los comandos
apropiados de Git:

- Revertir cambios no confirmados: `git checkout -- <archivo>` o `git restore`.
- Deshacer el último commit conservando los cambios: `git reset --soft HEAD~1`.
- Deshacer el último commit descartando los cambios: `git reset --hard HEAD~1`.
- Volver a un commit específico: `git checkout <hash>` o `git reset --hard <hash>`.

Antes de operaciones destructivas, advierto y pido confirmación.

## 5. Seguridad básica

- No commiteo secretos, tokens, claves ni archivos de configuración sensibles.
- No ejecuto `git push --force` ni reescribo historial público sin autorización.

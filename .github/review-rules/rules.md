# Reglas de review — plat-pr-agent-fork

Reglas propias de este repo. Se suman a las globales del hub (`plat-code-review-agent`).

Nota: el hub todavía no tiene base de reglas para Python (hoy existen `ruby`, `javascript` y
`typescript`), así que `languages: [python]` queda declarado para cuando exista, pero hoy el
review se apoya en las globales más estas reglas.

Contexto: este repo es el **fork propio de PR-Agent** (upstream Qodo) — el motor que corre el
code review de toda la empresa. Dos cosas lo hacen distinto de un repo normal: un bug acá sale
publicado en la imagen del ECR y afecta a los 21 repos consumidores a la vez, y cada línea que se
aparta de upstream sin necesidad es un conflicto en el próximo bump de versión.

### FORK-001 · Toda variable nueva de prompt se agrega a `self.vars` y se guarda con `{%- if %}`
- **Severidad:** bloqueante
- **Categoría:** robustez

**Patrón:** los prompts se renderizan con `jinja2.Environment(undefined=StrictUndefined)`. Una
variable referenciada en el TOML que no está en `self.vars` no queda vacía: revienta en runtime al
renderizar, y como esto pasa dentro del `docker run` del review, el consumidor ve un review que
falló sin explicación. Las dos mitades —extender `self.vars` en la tool y guardar el bloque en el
template— van en el mismo PR.

**Antes:**
```toml
{# el bloque asume que jira_ticket siempre existe #}
Ticket asociado: {{ jira_ticket }}
```
**Después:**
```toml
{%- if jira_ticket %}
Ticket asociado: {{ jira_ticket }}
{%- endif %}
```

### FORK-002 · Un TOML de prompt nuevo se registra en `settings_files` de `config_loader.py`
- **Severidad:** bloqueante
- **Categoría:** arquitectura

**Patrón:** Dynaconf solo carga los archivos listados en `settings_files=[...]`. Un prompt nuevo
que no se registra no se carga en `global_settings`, y el fallo no es un error de archivo faltante
sino una key ausente mucho más adelante — o peor, el tool cae al prompt anterior y el review sigue
saliendo con el comportamiento viejo, que parece "el cambio no hizo nada".

**Pregunta clave al revisar:** "¿Este `.toml` nuevo aparece en la lista de `settings_files` de
`pr_agent/config_loader.py`?"

### FORK-003 · Config nueva se declara en `configuration.toml` con su comentario
- **Severidad:** advertencia
- **Categoría:** arquitectura

**Patrón:** `configuration.toml` es el listado autoritativo de opciones y la base sobre la que
`apply_repo_settings` hace el merge por sección. Una opción que solo se lee con
`get_settings().get("...")` pero no está declarada ahí no tiene default, no es descubrible para
los repos consumidores y rompe el merge parcial de esa sección.

**Antes:**
```python
if get_settings().config.get("enable_falsification_pass", False):
    ...   # no existe en configuration.toml: nadie sabe que la opción existe
```
**Después:**
```toml
# configuration.toml
[config]
# Segundo pase que intenta refutar cada hallazgo antes de publicarlo.
enable_falsification_pass = false
```

### FORK-004 · No ramificar por `isinstance(provider, …)`; preguntar con `is_supported(...)`
- **Severidad:** bloqueante
- **Categoría:** arquitectura

**Patrón:** los providers comparten la interfaz `GitProvider` y declaran capacidades vía
`is_supported("feature")`. Un `isinstance` ata el comportamiento a una clase concreta: cualquier
provider que stubee u override esa feature queda del lado equivocado del `if`, y el bug aparece
solo en el provider que nadie probó.

**Antes:**
```python
if isinstance(self.git_provider, GithubProvider):
    self.git_provider.publish_inline_comments(comments)
```
**Después:**
```python
if self.git_provider.is_supported("publish_inline_comments"):
    self.git_provider.publish_inline_comments(comments)
```

### FORK-005 · Mantener el diff contra upstream lo más chico posible
- **Severidad:** advertencia
- **Categoría:** arquitectura

**Patrón:** esto es un fork que se sincroniza con upstream. Reordenar imports, reformatear un
bloque que no se toca, renombrar una variable "de paso" o reindentar un TOML de prompts no cambia
nada del comportamiento y se cobra entero en el próximo merge de versión, como conflictos en
archivos que nadie quiso cambiar. El cambio local debería poder leerse como un parche acotado y
justificado.

**Pregunta clave al revisar:** "¿Cuáles de estas líneas cambian comportamiento, y cuáles son ruido
que va a conflictuar en el próximo bump de upstream?"

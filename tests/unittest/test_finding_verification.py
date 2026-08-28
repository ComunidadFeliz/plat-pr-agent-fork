"""El pase de falsificación: resolución de evidencia y sesgo a conservar.

Los casos vienen del set de medición real (11 PRs de un monolito Rails): el orden de la evidencia
y el sesgo asimétrico son lo que llevó el pase de matar 1 de 4 falsos a matar 4 de 4 sin borrar
ningún hallazgo cierto.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pr_agent.algo import finding_verification as fv


def _file(filename, patch_text="@@ -1 +1 @@\n-a\n+b"):
    return SimpleNamespace(filename=filename, patch=patch_text, head_file="", base_file="")


class _Provider:
    """Provider mínimo: diffs del PR y contenido por path, sin red."""

    def __init__(self, diff_files=None, contents=None):
        self._diff_files = diff_files or []
        self._contents = contents or {}
        self.github_client = None
        self.repo = "org/repo"

    def get_diff_files(self):
        return self._diff_files

    def get_repo_file_content(self, path, from_default_branch=False):
        return self._contents.get(path, "")


# --------------------------------------------------------------------------- extracción

def test_symbols_tolera_sintaxis_de_llamada_y_agrega_la_accion():
    syms = fv._symbols("El botón iba a `upselling_info_path(package: 'attendance_control')`")
    assert "upselling_info_path" in syms
    assert "upselling_info" in syms          # la evidencia vive en la acción, no en el helper


def test_symbols_descarta_palabras_que_resuelven_a_cualquier_cosa():
    assert fv._symbols("el selector queda `.slots-container-undefined` con `undefined`") == []
    assert fv._symbols("`nil`, `true`, `params`") == []


def test_named_files_solo_extensiones_de_codigo():
    files = fv._named_files("Ver `app/models/payment.rb` y el logo.png y config/routes.rb")
    assert files == ["app/models/payment.rb", "config/routes.rb"]


# --------------------------------------------------------------------------- resolución

def test_el_diff_del_pr_va_primero_y_los_tests_al_final():
    """Medido: pasarle los specs y no el diff dejaba al verificador razonando a ciegas."""
    provider = _Provider(diff_files=[
        _file("spec/models/broker_api_request_spec.rb"),
        _file("db/migrate/20260826120000_fix_fks.rb"),
    ])
    finding = {"issue_content": "El `down` ejecuta un DELETE en db/migrate/20260826120000_fix_fks.rb",
               "failure_scenario": "Un rollback borra filas de consumo."}
    with patch.object(fv, "get_settings") as gs:
        gs.return_value.pr_reviewer.get.return_value = 26000
        labels = [label for label, _ in fv.resolve_evidence(provider, finding)]
    assert labels[0].startswith("diff of db/migrate/")        # el archivo que el hallazgo nombra
    assert any("spec/models" in l for l in labels)            # el spec igual va, pero después


def test_resuelve_el_relevant_file_aunque_el_texto_no_lo_nombre():
    provider = _Provider(diff_files=[_file("app/javascript/controllers/common_spaces_controller.js")])
    finding = {"issue_content": "El lookup pasó a un selector global", "failure_scenario": "no matchea",
               "relevant_file": "app/javascript/controllers/common_spaces_controller.js"}
    with patch.object(fv, "get_settings") as gs:
        gs.return_value.pr_reviewer.get.return_value = 26000
        labels = [label for label, _ in fv.resolve_evidence(provider, finding)]
    assert labels and "common_spaces_controller.js" in labels[0]


def test_respeta_el_presupuesto_de_caracteres():
    provider = _Provider(diff_files=[_file(f"app/f{i}.rb", "x" * 5000) for i in range(6)])
    with patch.object(fv, "get_settings") as gs:
        gs.return_value.pr_reviewer.get.return_value = 12000
        total = sum(len(body) for _, body in fv.resolve_evidence(provider, {"issue_content": "a"}))
    assert total <= 12000


# --------------------------------------------------------------------------- parseo

@pytest.mark.parametrize("raw", [
    '{"verdict": "refuted", "evidence": "spec.rb:22 — cita"}',
    '```json\n{"verdict": "refuted", "evidence": "spec.rb:22 — cita"}\n```',
    'Acá va mi análisis:\n{"verdict": "refuted", "evidence": "spec.rb:22 — cita"}\n',
])
def test_parse_verdict_acepta_las_formas_habituales(raw):
    assert fv._parse_verdict(raw)["verdict"] == "refuted"


@pytest.mark.parametrize("raw", ["", None, "no puedo responder", '{"verdict": "refu'])
def test_parse_verdict_devuelve_vacio_si_no_puede(raw):
    assert fv._parse_verdict(raw) == {}


# --------------------------------------------------------------------------- sesgo a conservar

def _run_verify(response, finding=None):
    provider = _Provider(diff_files=[_file("app/x.rb")])
    handler = SimpleNamespace(chat_completion=AsyncMock(return_value=(response, "stop")))
    f = finding or {"issue_header": "Adapter no restaurado", "issue_content": "c", "failure_scenario": "s"}
    with patch.object(fv, "get_settings") as gs:
        gs.return_value.pr_reviewer.get.return_value = 26000
        gs.return_value.pr_finding_verification_prompt.system = "sys {{ issue_header }}"
        gs.return_value.pr_finding_verification_prompt.user = "usr {{ evidence }}{{ resolved_any }}{{ issue_content }}{{ failure_scenario }}"
        return asyncio.run(fv.verify_findings(handler, provider, [f], "modelo"))


def test_descarta_solo_con_refutado_y_evidencia_citada():
    assert _run_verify('{"verdict": "refuted", "evidence": "spec.rb:22 — la línea sí corre"}') == ["Adapter no restaurado"]


def test_refutado_sin_evidencia_conserva():
    assert _run_verify('{"verdict": "refuted", "evidence": ""}') == []


@pytest.mark.parametrize("verdict", ["upheld", "undetermined", "cualquier cosa"])
def test_todo_lo_que_no_sea_refutado_conserva(verdict):
    assert _run_verify('{"verdict": "%s", "evidence": "algo"}' % verdict) == []


def test_json_ilegible_conserva():
    assert _run_verify("el modelo se puso a charlar") == []


def test_acepta_las_claves_en_espanol():
    assert _run_verify('{"veredicto": "refutado", "evidencia": "x.rb:1 — cita"}') == ["Adapter no restaurado"]


def test_una_excepcion_del_modelo_conserva_el_hallazgo():
    provider = _Provider(diff_files=[_file("app/x.rb")])
    handler = SimpleNamespace(chat_completion=AsyncMock(side_effect=RuntimeError("timeout")))
    with patch.object(fv, "get_settings") as gs:
        gs.return_value.pr_reviewer.get.return_value = 26000
        gs.return_value.pr_finding_verification_prompt.system = "s"
        gs.return_value.pr_finding_verification_prompt.user = "u"
        out = asyncio.run(fv.verify_findings(handler, provider,
                                             [{"issue_header": "h", "issue_content": "c", "failure_scenario": "s"}],
                                             "modelo"))
    assert out == []


def test_sin_hallazgos_no_llama_al_modelo():
    handler = SimpleNamespace(chat_completion=AsyncMock())
    assert asyncio.run(fv.verify_findings(handler, _Provider(), [], "modelo")) == []
    handler.chat_completion.assert_not_called()


# --------------------------------------------------------------------------- búsqueda de símbolos

class _PaginatedVacio:
    """Imita a PyGithub: slicear un PaginatedList vacío levanta IndexError."""
    def __getitem__(self, item):
        raise IndexError("list index out of range")
    def __iter__(self):
        return iter([])


def test_busqueda_de_simbolos_tolera_resultado_vacio():
    """Regresión del e2e: `results[:2]` sobre un PaginatedList vacío mataba el canal de evidencia."""
    provider = SimpleNamespace(github_client=SimpleNamespace(search_code=lambda q: _PaginatedVacio()),
                               repo="org/repo")
    assert fv._search_symbol(provider, "notify_upcoming_to_residents") == []


def test_busqueda_de_simbolos_devuelve_hasta_dos_paths():
    hits = [SimpleNamespace(path=f"app/f{i}.rb") for i in range(5)]
    provider = SimpleNamespace(github_client=SimpleNamespace(search_code=lambda q: hits), repo="org/repo")
    assert fv._search_symbol(provider, "algo") == ["app/f0.rb", "app/f1.rb"]


def test_sin_cliente_de_github_no_busca():
    assert fv._search_symbol(SimpleNamespace(github_client=None, repo="org/repo"), "algo") == []

"""El filtro de hallazgos sin falla concreta (ver _drop_unfalsifiable_findings)."""
from unittest.mock import patch

from pr_agent.tools.pr_reviewer import PRReviewer


def _run(findings, flag=True):
    review = {"key_issues_to_review": findings}
    with patch("pr_agent.tools.pr_reviewer.get_settings") as gs:
        gs.return_value.pr_reviewer.get.side_effect = lambda k, d=None: flag if k == "drop_unfalsifiable_findings" else d
        PRReviewer._drop_unfalsifiable_findings(object.__new__(PRReviewer), review)
    return review["key_issues_to_review"]


def test_conserva_el_hallazgo_con_falla_concreta():
    kept = _run([{"issue_header": "Test no valida el efecto real",
                  "issue_content": "Los specs stubean el consumidor.",
                  "failure_scenario": "Con una comunidad en America/Hermosillo el spec pasa aunque la fecha del CFDI se arme en Mexico/General."}])
    assert len(kept) == 1
    assert "**Cómo falla:**" in kept[0]["issue_content"]
    assert "failure_scenario" not in kept[0]          # la clave no llega al renderer


def test_descarta_sin_failure_scenario():
    for scenario in ("", "   ", "...", "N/A", "no aplica"):
        assert _run([{"issue_header": "x", "issue_content": "algo", "failure_scenario": scenario}]) == []
    assert _run([{"issue_header": "x", "issue_content": "algo"}]) == []


def test_descarta_el_que_pide_verificar():
    assert _run([{"issue_header": "Consumidor no visible", "issue_content": "Se agrega la clave region.",
                  "failure_scenario": "Conviene confirmar que generate_uniq_complement soporta la clave."}]) == []
    assert _run([{"issue_header": "Tags", "issue_content": "Se reasignaron tags.",
                  "failure_scenario": "No es posible confirmar que las 21 operaciones tengan un tag valido."}]) == []


def test_conserva_el_hallazgo_que_describe_comportamiento_esperado_del_codigo():
    """Regresión: 'comportamiento esperado' puede ser lo que el hallazgo DENUNCIA.

    Falso positivo real, medido en saas-web-monolith#15928: un spec que fija el bug como
    contrato ("documenta como comportamiento esperado que el controller responda con redirect
    exitoso mientras descarta el día") se descartaba por esa frase en el cuerpo.
    """
    kept = _run([{"issue_header": "Test que fija el bug como contrato",
                  "issue_content": "El ejemplo documenta como comportamiento esperado que el controller responda con redirect exitoso mientras descarta el dia.",
                  "failure_scenario": "Cuando se agregue la validacion server-side, este ejemplo fallara por esperar redirect_to(common_space)."}])
    assert len(kept) == 1


def test_descarta_el_que_se_auto_refuta():
    assert _run([{"issue_header": "GEN-007", "issue_content": "Ambos campos estan presentes, por lo que no hay regresion real.",
                  "failure_scenario": "El orden de los parametros cambia."}]) == []
    assert _run([{"issue_header": "Password", "issue_content": "Esto es inofensivo en la practica.",
                  "failure_scenario": "La contrasena generada se descarta."}]) == []


def test_no_filtra_por_comportamiento_esperado_ni_en_el_escenario():
    """El filtro de keywords NO intenta juzgar 'comportamiento esperado/correcto'.

    Da falsos positivos en las dos direcciones (el hallazgo puede estar describiendo el codigo
    que denuncia, o el comportamiento correcto que todavia no existe). Esa familia la resuelve
    el pase de falsificacion contra el codigo, no una regex.
    """
    kept = _run([{"issue_header": "x", "issue_content": "algo",
                  "failure_scenario": "Es el comportamiento esperado segun el comentario."}])
    assert len(kept) == 1


def test_respeta_el_flag_apagado():
    findings = [{"issue_header": "x", "issue_content": "algo", "failure_scenario": "..."}]
    assert len(_run(findings, flag=False)) == 1


def test_no_toca_lo_que_no_es_lista_de_dicts():
    assert _run([]) == []
    assert _run(["texto suelto"]) == ["texto suelto"]

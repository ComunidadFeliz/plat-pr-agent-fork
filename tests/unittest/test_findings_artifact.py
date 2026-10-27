"""The JSON findings artifact (see PRReviewer._emit_findings_artifact).

This is a CONTRACT with the human-validation gate in the central repo, which runs in a later
step of the same job. The comment's prose is a rendering and changes with the template; this
does not. If a change here breaks these tests, it breaks the consumer too: bump
`schema_version` and say so, do not reshape the fixture.
"""
import json
from unittest.mock import patch

import pytest

from pr_agent.tools.pr_reviewer import PRReviewer


def _emit(review, path, model="anthropic/claude-opus-5"):
    with patch("pr_agent.tools.pr_reviewer.get_settings") as gs:
        gs.return_value.config.model = model
        PRReviewer._emit_findings_artifact(object.__new__(PRReviewer), review)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@pytest.fixture
def findings_file(tmp_path, monkeypatch):
    path = tmp_path / "findings.json"
    monkeypatch.setenv("FINDINGS_FILE", str(path))
    return path


# --- The exact payload shape ------------------------------------------------------

def test_payload_shape(findings_file):
    """Golden fixture: these keys ARE the contract. Changing them bumps schema_version."""
    review = {"key_issues_to_review": [{
        "issue_header": "Nombres de variables no descriptivos",
        "issue_content": "Las variables `d` y `x` no expresan su intención.",
        "relevant_file": "app/services/bulk/resumen_unidad.rb",
        "relevant_line": "d = community.name.upcase",
        "rule_ids": ["SBX-007"],
        "start_line": 8,
        "end_line": 10,
        "failure_scenario": "irrelevante para el artefacto",
    }]}
    payload = _emit(review, findings_file)
    assert payload == {
        "event": "review_findings",
        "schema_version": 2,
        "model": "anthropic/claude-opus-5",
        "findings": [{
            "issue_header": "Nombres de variables no descriptivos",
            "issue_content": "Las variables `d` y `x` no expresan su intención.",
            "relevant_file": "app/services/bulk/resumen_unidad.rb",
            "relevant_line": "d = community.name.upcase",
            "rule_ids": ["SBX-007"],
            "start_line": 8,
            "end_line": 10,
        }],
    }


def test_a_vanished_key_arrives_as_null_not_missing(findings_file):
    """The projection is explicit: if a field is renamed upstream the consumer sees null (and
    can complain), not a missing key its parser would interpret however it likes."""
    payload = _emit({"key_issues_to_review": [{"issue_header": "solo el header"}]},
                    findings_file)
    assert payload["findings"] == [{"issue_header": "solo el header", "issue_content": None,
                                    "relevant_file": None, "relevant_line": None,
                                    "rule_ids": [], "start_line": None, "end_line": None}]


# --- The three states the consumer must tell apart --------------------------

def test_zero_findings_still_writes_the_file(findings_file):
    """"Nothing found" and "could not read" MUST NOT look alike on the other side: that is
    why the file is always written, with an empty findings list."""
    payload = _emit({"key_issues_to_review": []}, findings_file)
    assert payload["findings"] == []
    assert payload["schema_version"] == 2


def test_without_the_env_var_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv("FINDINGS_FILE", raising=False)
    path = tmp_path / "findings.json"
    with patch("pr_agent.tools.pr_reviewer.get_settings") as gs:
        gs.return_value.config.model = "m"
        PRReviewer._emit_findings_artifact(object.__new__(PRReviewer), {"key_issues_to_review": []})
    assert not path.exists()


def test_a_write_failure_does_not_break_the_review(tmp_path, monkeypatch):
    """Best-effort, like llm_usage: the review is worth more than its telemetry."""
    monkeypatch.setenv("FINDINGS_FILE", str(tmp_path / "no" / "existe" / "f.json"))
    with patch("pr_agent.tools.pr_reviewer.get_settings") as gs:
        gs.return_value.config.model = "m"
        PRReviewer._emit_findings_artifact(object.__new__(PRReviewer), {"key_issues_to_review": []})


# --- Robustness against odd input -----------------------------------------------------

def test_ignores_non_dict_entries(findings_file):
    payload = _emit({"key_issues_to_review": ["texto suelto", None,
                                              {"issue_header": "válido"}]}, findings_file)
    assert [f["issue_header"] for f in payload["findings"]] == ["válido"]


def test_tolerates_missing_or_non_list_key_issues(findings_file):
    assert _emit({}, findings_file)["findings"] == []
    assert _emit({"key_issues_to_review": "no soy lista"}, findings_file)["findings"] == []


# --- normalization of the new fields (schema 2) ------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (["GEN-012", "RUBY-010"], ["GEN-012", "RUBY-010"]),
    ("GEN-012", ["GEN-012"]),          # the model sometimes returns a bare string
    ([" GEN-012 ", ""], ["GEN-012"]),  # whitespace and empties
    (None, []), ([], []), (123, []),
])
def test_rule_ids_normalized_to_string_list(findings_file, raw, expected):
    payload = _emit({"key_issues_to_review": [{"rule_ids": raw}]}, findings_file)
    assert payload["findings"][0]["rule_ids"] == expected


@pytest.mark.parametrize("raw,expected", [(12, 12), ("12", 12), ("...", None),
                                            (None, None), ("", None)])
def test_lines_are_ints_or_none(findings_file, raw, expected):
    """The model's YAML sometimes yields '...' where a number belongs. None is honest; the
    literal string '...' would make the consumer anchor anywhere."""
    payload = _emit({"key_issues_to_review": [{"start_line": raw, "end_line": raw}]},
                    findings_file)
    assert payload["findings"][0]["start_line"] == expected
    assert payload["findings"][0]["end_line"] == expected

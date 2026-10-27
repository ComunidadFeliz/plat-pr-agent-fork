import copy
import datetime
import json
import os
import re
from functools import partial
from typing import List, Optional, Tuple

from jinja2 import Environment, StrictUndefined

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.inline_comment_dedup import (
    InlineCommentStore,
    can_verify_inline_comment_publication,
    get_inline_comment_store,
    key_issue_body_with_markers,
    key_issue_fingerprint,
    key_issue_location_fingerprint,
)
from pr_agent.algo.pr_processing import add_ai_metadata_to_diff_files, get_pr_diff, retry_with_fallback_models
from pr_agent.algo.finding_verification import verify_findings
from pr_agent.algo.repo_context import build_repo_context
from pr_agent.algo.run_details import init_run_details
from pr_agent.algo.skills_loader import get_skills_context
from pr_agent.algo.token_handler import TokenHandler
from pr_agent.algo.utils import (
    ModelType,
    PRReviewHeader,
    convert_to_markdown_v2,
    github_action_output,
    load_yaml,
    show_relevant_configurations,
    show_run_details,
)
from pr_agent.config_loader import get_settings
from pr_agent.git_providers import get_git_provider_with_context
from pr_agent.git_providers.git_provider import IncrementalPR, get_main_pr_language
from pr_agent.log import get_logger
from pr_agent.servers.help import HelpMessage
from pr_agent.tools.ticket_pr_compliance_check import extract_and_cache_pr_tickets

MAX_REVIEW_COVERAGE_FILES = 50

# A finding that does not assert a defect is not a finding. Two shapes show up in practice, and
# every 👎 we have collected is one of them: the model concludes there is no real problem and
# reports it anyway, or it hands the reader a verification task instead of a failure. Measured on
# 11 real PRs: 10 of 11 published findings did not survive a check against the code, and every one
# of those was one of these two shapes. Patterns cover es-CL and en-US -- config.response_language
# decides which one the model writes in.
# Auto-refutación inequívoca: el hallazgo dice que no hay defecto. Se busca en todo el hallazgo.
_SELF_REFUTING_RE = re.compile(
    r"no hay (?:regresi[oó]n|problema|bug|riesgo|defecto|contradicci[oó]n|impacto)"
    r"|no (?:representa|implica) (?:un|ning[uú]n) (?:problema|riesgo|defecto)"
    r"|es inofensiv|resulta inofensiv"
    r"|no (?:es|constituye) un (?:problema|defecto|bug)"
    r"|(?:is|are) harmless|no (?:real|actual) (?:regression|bug|issue|defect|problem)"
    r"|not (?:an|a real) issue",
    re.IGNORECASE)
# NO se filtra por frases del tipo "comportamiento esperado/correcto" ni "ya está manejado".
# Se probó y da falsos positivos en las dos direcciones: un hallazgo legítimo las usa para
# DESCRIBIR el código que denuncia ("el spec documenta como comportamiento esperado que el
# controller responda con redirect exitoso" es el hallazgo) o para describir el comportamiento
# correcto que todavía no existe. Un filtro que borra hallazgos verdaderos en silencio es peor
# que el ruido que evita: esa familia la resuelve el pase de falsificación, no una keyword.
_ASKS_TO_VERIFY_RE = re.compile(
    r"conviene (?:confirmar|verificar|revisar|validar|chequear)"
    r"|vale (?:la pena )?(?:confirmar|verificar|revisar|validar)"
    r"|habr[ií]a que (?:confirmar|verificar|revisar|validar)"
    r"|ser[ií]a (?:bueno|prudente) (?:confirmar|verificar|revisar)"
    r"|(?:confirmar|verificar|validar|chequear) (?:que|si|el|la|los|las)"
    r"|no (?:es posible|se puede|puedo|podemos) (?:confirmar|verificar|determinar)"
    r"|no (?:se ve|est[aá]|aparece) en el diff"
    r"|(?:worth|should|please) (?:confirming|verifying|checking|confirm|verify|check)"
    r"|(?:cannot|can not|can\'t|unable to) (?:confirm|verify|determine)"
    r"|not (?:visible|included) in the diff|make sure (?:that|the)",
    re.IGNORECASE)
_PLACEHOLDER_SCENARIO = {"", "...", "n/a", "na", "none", "no", "no aplica", "-", "tbd"}
_SUGGESTION_FENCE_RE = re.compile(r"```[ \t]*suggestion\b", re.IGNORECASE)


def _as_id_list(value):
    """`rule_ids` normalized to a list of strings. The model may return a list, a bare
    string, or nothing; the consumer should not have to handle all three."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _as_int(value):
    """An int, or None. The model's YAML sometimes yields '12' or '...' instead of 12."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class PRReviewer:
    """
    The PRReviewer class is responsible for reviewing a pull request and generating feedback using an AI model.
    """

    def __init__(self, pr_url: str, is_answer: bool = False, is_auto: bool = False, args: list = None,
                 ai_handler: partial[BaseAiHandler,] = LiteLLMAIHandler):
        """
        Initialize the PRReviewer object with the necessary attributes and objects to review a pull request.

        Args:
            pr_url (str): The URL of the pull request to be reviewed.
            is_answer (bool, optional): Indicates whether the review is being done in answer mode. Defaults to False.
            is_auto (bool, optional): Indicates whether the review is being done in automatic mode. Defaults to False.
            ai_handler (BaseAiHandler): The AI handler to be used for the review. Defaults to None.
            args (list, optional): List of arguments passed to the PRReviewer class. Defaults to None.
        """
        self.git_provider = get_git_provider_with_context(pr_url)
        self.args = args
        self.incremental = self.parse_incremental(args)  # -i command
        if self.incremental and self.incremental.is_incremental:
            self.git_provider.get_incremental_commits(self.incremental)

        self.main_language = get_main_pr_language(
            self.git_provider.get_languages(), self.git_provider.get_files()
        )
        self.pr_url = pr_url
        self.is_answer = is_answer
        self.is_auto = is_auto

        if self.is_answer and not self.git_provider.is_supported("get_issue_comments"):
            raise Exception(f"Answer mode is not supported for {get_settings().config.git_provider} for now")
        self.ai_handler = ai_handler()
        self.ai_handler.main_pr_language = self.main_language
        self.patches_diff = None
        self.remaining_files_list = []
        self._refuted_headers = []
        self.prediction = None
        question_str, answer_str = self._get_user_answers()
        self.pr_description, self.pr_description_files = (
            self.git_provider.get_pr_description(split_changes_walkthrough=True))
        if (self.pr_description_files and get_settings().get("config.is_auto_command", False) and
                get_settings().get("config.enable_ai_metadata", False)):
            add_ai_metadata_to_diff_files(self.git_provider, self.pr_description_files)
            get_logger().debug(f"AI metadata added to the this command")
        else:
            get_settings().set("config.enable_ai_metadata", False)
            get_logger().debug(f"AI metadata is disabled for this command")

        self.vars = {
            "title": self.git_provider.pr.title,
            "branch": self.git_provider.get_pr_branch(),
            "description": self.pr_description,
            "language": self.main_language,
            "diff": "",  # empty diff for initial calculation
            "num_pr_files": self.git_provider.get_num_of_files(),
            "num_max_findings": get_settings().pr_reviewer.num_max_findings,
            "require_score": get_settings().pr_reviewer.require_score_review,
            "require_tests": get_settings().pr_reviewer.require_tests_review,
            "require_estimate_effort_to_review": get_settings().pr_reviewer.require_estimate_effort_to_review,
            "require_estimate_contribution_time_cost": get_settings().pr_reviewer.require_estimate_contribution_time_cost,
            'require_can_be_split_review': get_settings().pr_reviewer.require_can_be_split_review,
            'require_security_review': get_settings().pr_reviewer.require_security_review,
            'require_todo_scan': get_settings().pr_reviewer.get("require_todo_scan", False),
            'question_str': question_str,
            'answer_str': answer_str,
            "extra_instructions": get_settings().pr_reviewer.extra_instructions,
            "skills_context": get_skills_context(),
            "repo_context": build_repo_context(self.git_provider),
            "commit_messages_str": self.git_provider.get_commit_messages(),
            "custom_labels": "",
            "enable_custom_labels": get_settings().config.enable_custom_labels,
            "is_ai_metadata":  get_settings().get("config.enable_ai_metadata", False),
            "related_tickets": get_settings().get('related_tickets', []),
            'duplicate_prompt_examples': get_settings().config.get('duplicate_prompt_examples', False),
            "date": datetime.datetime.now().strftime('%Y-%m-%d'),
        }

        self.token_handler = TokenHandler(
            self.git_provider.pr,
            self.vars,
            get_settings().pr_review_prompt.system,
            get_settings().pr_review_prompt.user
        )

    def parse_incremental(self, args: List[str]):
        is_incremental = False
        if args and len(args) >= 1:
            arg = args[0]
            if arg == "-i":
                is_incremental = True
        incremental = IncrementalPR(is_incremental)
        return incremental

    async def run(self) -> None:
        init_run_details()
        try:
            if not self.git_provider.get_files():
                get_logger().info(f"PR has no files: {self.pr_url}, skipping review")
                return None

            if self.incremental.is_incremental:
                can_run = self._can_run_incremental_review()
                # If the gate disabled incremental (e.g., commits_range is None), fall through to full review.
                if not can_run and self.incremental.is_incremental:
                    return None

            # if isinstance(self.args, list) and self.args and self.args[0] == 'auto_approve':
            #     get_logger().info(f'Auto approve flow PR: {self.pr_url} ...')
            #     self.auto_approve_logic()
            #     return None

            get_logger().info(f'Reviewing PR: {self.pr_url} ...')
            relevant_configs = {'pr_reviewer': dict(get_settings().pr_reviewer),
                                'config': dict(get_settings().config)}
            get_logger().debug("Relevant configs", artifacts=relevant_configs)

            # ticket extraction if exists
            await extract_and_cache_pr_tickets(self.git_provider, self.vars)

            if (
                self.incremental.is_incremental
                and hasattr(self.git_provider, "unreviewed_files_map")
                and not self.git_provider.unreviewed_files_map
            ):
                get_logger().info(f"Incremental review is enabled for {self.pr_url} but there are no new files")
                previous_review_url = ""
                if hasattr(self.git_provider, "previous_review") and self.git_provider.previous_review is not None:
                    previous_review_url = getattr(self.git_provider.previous_review, "html_url", "") or ""
                if get_settings().config.publish_output:
                    self.git_provider.publish_comment(f"Incremental Review Skipped\n"
                                    f"No files were changed since the [previous PR Review]({previous_review_url})")
                return None

            if get_settings().config.publish_output and not get_settings().config.get('is_auto_command', False):
                self.git_provider.publish_comment("Preparing review...", is_temporary=True)

            await retry_with_fallback_models(self._prepare_prediction, model_type=ModelType.REGULAR)
            if not self.prediction:
                self.git_provider.remove_initial_comment()
                return None

            await self._verify_findings()

            pr_review = self._prepare_pr_review()
            get_logger().debug(f"PR output", artifact=pr_review)

            should_publish = get_settings().config.publish_output and self._should_publish_review_no_suggestions(pr_review)
            if not should_publish:
                reason = "Review output is not published"
                if get_settings().config.publish_output:
                    reason += ": no major issues detected."
                get_logger().info(reason)
                get_settings().data = {"artifact": pr_review}
                return

            # publish the review
            # Providers that support it (GitLab) can post the review's final comment as a resolvable thread.
            # This intent applies to the review only - never to status comments or the output of other tools.
            review_thread_kwargs = {"as_thread": True} if self.git_provider.should_publish_review_as_thread() else {}
            if get_settings().pr_reviewer.persistent_comment and not self.incremental.is_incremental:
                final_update_message = get_settings().pr_reviewer.final_update_message
                self.git_provider.publish_persistent_comment(pr_review,
                                                            initial_header=f"{PRReviewHeader.REGULAR.value} 🔍",
                                                            update_header=True,
                                                            final_update_message=final_update_message,
                                                            **review_thread_kwargs)
            else:
                self.git_provider.publish_comment(pr_review, **review_thread_kwargs)

            self.git_provider.remove_initial_comment()
        except Exception as e:
            get_logger().error(f"Failed to review PR: {e}")
            if get_settings().config.get("propagate_tool_errors", False):
                raise

    def _should_publish_review_no_suggestions(self, pr_review: str) -> bool:
        return get_settings().pr_reviewer.get('publish_output_no_suggestions', True) or "No major issues detected" not in pr_review

    async def _prepare_prediction(self, model: str) -> None:
        output = get_pr_diff(self.git_provider,
                             self.token_handler,
                             model,
                             add_line_numbers_to_hunks=True,
                             disable_extra_lines=False,
                             return_remaining_files=True,)
        if isinstance(output, tuple):
            self.patches_diff, self.remaining_files_list = output
        else:
            self.patches_diff = output
            self.remaining_files_list = []

        if self.patches_diff:
            get_logger().debug(f"PR diff", diff=self.patches_diff)
            self.prediction = await self._get_prediction(model)
        else:
            get_logger().warning(f"Empty diff for PR: {self.pr_url}")
            self.prediction = None

    async def _get_prediction(self, model: str) -> str:
        """
        Generate an AI prediction for the pull request review.

        Args:
            model: A string representing the AI model to be used for the prediction.

        Returns:
            A string representing the AI prediction for the pull request review.
        """
        variables = copy.deepcopy(self.vars)
        variables["diff"] = self.patches_diff  # update diff

        environment = Environment(undefined=StrictUndefined)
        system_prompt = environment.from_string(get_settings().pr_review_prompt.system).render(variables)
        user_prompt = environment.from_string(get_settings().pr_review_prompt.user).render(variables)

        response, finish_reason = await self.ai_handler.chat_completion(
            model=model,
            temperature=get_settings().config.temperature,
            system=system_prompt,
            user=user_prompt
        )

        return response



    async def _verify_findings(self) -> None:
        """Second pass: try to refute each finding against the code it names, before publishing.

        Opt-in (`pr_reviewer.verify_findings`) because it costs one extra model call per PR. Never
        raises and never removes a finding on its own: it only records which headers came back
        refuted WITH cited evidence, and `_prepare_pr_review` drops those.
        """
        self._refuted_headers = []
        if not get_settings().pr_reviewer.get("verify_findings", False):
            return
        try:
            data = load_yaml(self.prediction.strip()) or {}
            findings = [f for f in ((data.get('review') or {}).get('key_issues_to_review') or [])
                        if isinstance(f, dict)]
            if not findings:
                return
            model = get_settings().config.model
            self._refuted_headers = await verify_findings(self.ai_handler, self.git_provider, findings, model)
        except Exception as e:
            get_logger().warning(f"finding verification skipped: {e}")
            self._refuted_headers = []

    def _drop_unfalsifiable_findings(self, review: dict) -> None:
        """Keep only findings that state a concrete failure, and fold that statement into the body.

        Drops three shapes: no failure_scenario at all, a scenario that is really a request for the
        reader to go and verify something, and a finding whose own text concludes there is no
        defect. What gets dropped is logged -- a silent filter would read as "nothing was found".
        """
        if not get_settings().pr_reviewer.get("drop_unfalsifiable_findings", True):
            return
        findings = review.get('key_issues_to_review')
        if not isinstance(findings, list) or not findings:
            return

        kept, dropped = [], []
        for finding in findings:
            if not isinstance(finding, dict):
                kept.append(finding)
                continue
            scenario = str(finding.get('failure_scenario') or '').strip()
            body = str(finding.get('issue_content') or '')
            header = str(finding.get('issue_header') or '').strip()
            reason = None
            if scenario.strip('.').lower() in _PLACEHOLDER_SCENARIO:
                reason = "sin failure_scenario"
            elif _ASKS_TO_VERIFY_RE.search(scenario):
                reason = "el failure_scenario pide verificar, no afirma una falla"
            elif _SELF_REFUTING_RE.search(scenario) or _SELF_REFUTING_RE.search(body):
                reason = "el propio texto concluye que no hay defecto"
            if reason:
                dropped.append({"issue_header": header, "reason": reason})
                continue
            # El escenario es la parte util para el reviewer humano: va al cuerpo, y la clave se
            # saca para no cambiar el contrato del renderer.
            if scenario and scenario not in body:
                finding['issue_content'] = f"{body.rstrip()}\n\n**Cómo falla:** {scenario}".lstrip()
            finding.pop('failure_scenario', None)
            kept.append(finding)

        if dropped:
            get_logger().info(
                f"Dropped {len(dropped)} of {len(findings)} findings: no falsifiable failure stated",
                artifact={"dropped": dropped})
        review['key_issues_to_review'] = kept

    # Findings artifact contract. BUMP the version for any change a consumer cannot absorb
    # (renaming or removing a key, changing its meaning); adding a key does not bump it. The
    # consumer MUST reject a version it does not know rather than carry on parsing:
    # see tests/unittest/test_findings_artifact.py.
    FINDINGS_SCHEMA_VERSION = 2

    def _emit_findings_artifact(self, review: dict) -> None:
        """Write the final findings as JSON to FINDINGS_FILE, when it is set.

        This exists so a later step in the same job (the human-validation gate) does not
        have to scrape the comment's markdown: prose is a rendering, not a contract, and it
        changes whenever the template does.

        Best-effort, like llm_usage: a failure here NEVER breaks the review. But the consumer
        must be able to tell THREE states apart, not two -- there are findings, there are zero
        findings, or the file could not be read. That is why the file is always written in
        full (with `findings: []` when there are none) and carries `schema_version`: a missing
        or unreadable file means "unknown", never "nothing found".

        Keys are projected one by one on purpose. With `**finding`, an upstream rename would
        change the payload silently; this way a key that disappears arrives as null and shows.
        """
        path = os.getenv("FINDINGS_FILE")
        if not path:
            return
        try:
            raw = review.get('key_issues_to_review')
            findings = [
                {
                    "issue_header": f.get('issue_header'),
                    "issue_content": f.get('issue_content'),
                    "relevant_file": f.get('relevant_file'),
                    "relevant_line": f.get('relevant_line'),
                    # The violated rules as DATA. The consumer used to dig them out of the
                    # prose with a regex, which failed whenever the model described the
                    # problem without naming the id (measured: 14% of findings). With the
                    # field in the schema, stating it stops being optional.
                    "rule_ids": _as_id_list(f.get('rule_ids')),
                    # The line NUMBER, not its text: without it the consumer cannot anchor a
                    # comment where it belongs and falls back to the top of the file.
                    "start_line": _as_int(f.get('start_line')),
                    "end_line": _as_int(f.get('end_line')),
                }
                for f in (raw if isinstance(raw, list) else [])
                if isinstance(f, dict)
            ]
            payload = {
                "event": "review_findings",
                "schema_version": self.FINDINGS_SCHEMA_VERSION,
                "model": get_settings().config.model,
                "findings": findings,
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            get_logger().info(f"review_findings: {len(findings)} finding(s) -> {path}")
        except Exception as e:
            get_logger().warning(f"Could not write review findings to {path}: {e}")

    def _prepare_pr_review(self) -> str:
        """
        Prepare the PR review by processing the AI prediction and generating a markdown-formatted text that summarizes
        the feedback.
        """
        first_key = 'review'
        last_key = 'security_concerns'
        data = load_yaml(self.prediction.strip(),
                         keys_fix_yaml=["ticket_compliance_check", "estimated_effort_to_review_[1-5]:", "security_concerns:", "key_issues_to_review:",
                                        "relevant_file:", "relevant_line:", "suggestion:"],
                         first_key=first_key, last_key=last_key)
        github_action_output(data, 'review')

        if 'review' not in data:
            get_logger().exception("Failed to parse review data", artifact={"data": data})
            return ""

        self._drop_unfalsifiable_findings(data['review'])

        # getattr: hay tests (y llamadores) que construyen el objeto sin pasar por __init__.
        refuted = getattr(self, "_refuted_headers", None)
        if refuted:
            findings = data['review'].get('key_issues_to_review')
            if isinstance(findings, list):
                data['review']['key_issues_to_review'] = [
                    f for f in findings
                    if not (isinstance(f, dict)
                            and str(f.get('issue_header') or '').strip() in refuted)]

        # The final findings as JSON, so another step of the SAME job can consume them
        # without scraping the markdown. Same pattern as LLM_USAGE_FILE.
        self._emit_findings_artifact(data['review'])

        # move data['review'] 'key_issues_to_review' key to the end of the dictionary
        if 'key_issues_to_review' in data['review']:
            key_issues_to_review = data['review'].pop('key_issues_to_review')
            data['review']['key_issues_to_review'] = key_issues_to_review

        if get_settings().config.publish_output and get_settings().pr_reviewer.get('inline_key_issues', False):
            data = self._publish_key_issues_as_inline_comments(data)

        incremental_review_markdown_text = None
        # Add incremental review section
        if self.incremental.is_incremental:
            last_commit_url = f"{self.git_provider.get_pr_url()}/commits/" \
                              f"{self.git_provider.incremental.first_new_commit_sha}"
            incremental_review_markdown_text = f"Starting from commit {last_commit_url}"

        markdown_text = convert_to_markdown_v2(data, self.git_provider.is_supported("gfm_markdown"),
                                            incremental_review_markdown_text,
                                               git_provider=self.git_provider,
                                               files=self.git_provider.get_diff_files())

        if self.remaining_files_list and get_settings().pr_reviewer.enable_review_coverage_footer:
            displayed_files = self.remaining_files_list[:MAX_REVIEW_COVERAGE_FILES]
            markdown_text += (
                "\n\n<hr>\n\n"
                "⚠️ **Review coverage:** The following files were not included in this review "
                "because of the token budget:\n"
                + "\n".join(f"- `{file}`" for file in displayed_files)
            )
            remaining_count = len(self.remaining_files_list) - len(displayed_files)
            if remaining_count:
                markdown_text += f"\n... and {remaining_count} more"

        # Add help text if gfm_markdown is supported
        if self.git_provider.is_supported("gfm_markdown") and get_settings().pr_reviewer.enable_help_text:
            markdown_text += "<hr>\n\n<details> <summary><strong>💡 Tool usage guide:</strong></summary><hr> \n\n"
            markdown_text += HelpMessage.get_review_usage_guide()
            markdown_text += "\n</details>\n"

        # Output the relevant configurations if enabled
        if get_settings().get('config', {}).get('output_relevant_configurations', False):
            markdown_text += show_relevant_configurations(relevant_section='pr_reviewer')

        # Output the agent run details (model, tokens, time cost) if enabled
        if get_settings().get('config', {}).get('output_run_details', False):
            markdown_text += show_run_details(self.git_provider.is_supported("gfm_markdown"))

        # Add custom labels from the review prediction (effort, security)
        self.set_review_labels(data)

        if markdown_text == None or len(markdown_text) == 0:
            markdown_text = ""

        return markdown_text

    def _build_key_issue_comment(self, issue, diff_files: dict) -> Optional[dict]:
        if not isinstance(issue, dict):
            return None
        relevant_file = (issue.get("relevant_file") or "").strip()
        issue_content = _SUGGESTION_FENCE_RE.sub("```text", (issue.get("issue_content") or "").strip())
        issue_header = (issue.get("issue_header") or "").strip()
        if issue_header.lower() == "possible bug":
            issue_header = "Possible Issue"
        try:
            start_line = int(str(issue.get("start_line", 0)).strip())
            end_line = int(str(issue.get("end_line", 0)).strip())
        except ValueError:
            start_line, end_line = 0, 0

        if not relevant_file or not issue_content or start_line < 1 or end_line < start_line:
            get_logger().warning("Review finding has no usable location, keeping it in the summary",
                                 artifact={"relevant_file": relevant_file, "start_line": start_line,
                                           "end_line": end_line})
            return None

        file = diff_files.get(relevant_file) or diff_files.get(relevant_file.lstrip("/"))
        if file is None:
            get_logger().warning("Review finding points at a file that is not in the diff, "
                                 "keeping it in the summary", artifact={"relevant_file": relevant_file})
            return None
        if not file.head_file or end_line > len(file.head_file.splitlines()):
            get_logger().warning("Review finding points past the end of the file, keeping it in the summary",
                                 artifact={"relevant_file": relevant_file, "start_line": start_line,
                                           "end_line": end_line})
            return None

        relevant_file = file.filename.strip()
        body = f"**{issue_header}**\n\n{issue_content}" if issue_header else issue_content
        return {"body": body,
                "relevant_file": relevant_file,
                "relevant_lines_start": start_line,
                "relevant_lines_end": end_line,
                "fallback_to_pr_comment": False}

    def _can_verify_inline_key_issue_publication(self) -> bool:
        return can_verify_inline_comment_publication(self.git_provider)

    def _published_inline_key_issue_fingerprints(self, store: InlineCommentStore,
                                                 fingerprints: set[str]) -> set[str]:
        try:
            for body in self.git_provider.get_recent_inline_comment_bodies():
                store.add_body(body)
        except Exception as e:
            get_logger().warning(
                f"Inline key-issue publishing cannot verify new Azure DevOps threads, error: {e}; "
                "keeping findings in the review summary")
            return set()
        return {fingerprint for fingerprint in fingerprints if store.seen(fingerprint)}

    def _publish_key_issues_as_inline_comments(self, data: dict) -> dict:
        issues = (data.get("review") or {}).get("key_issues_to_review")
        if not isinstance(issues, list) or not issues:
            return data
        if not self._can_verify_inline_key_issue_publication():
            get_logger().info("Inline key-issue publishing is not verifiable for this provider; "
                              "keeping findings in the review summary")
            return data

        diff_files = {}
        for file in self.git_provider.get_diff_files() or []:
            if not file.filename:
                continue
            path = file.filename.strip()
            diff_files[path] = file
            diff_files.setdefault(path.lstrip("/"), file)
        store = get_inline_comment_store(self.git_provider)
        store.load()
        if store.load_failed:
            get_logger().warning("Inline key-issue publishing cannot verify existing Azure DevOps threads; "
                                 "keeping findings in the review summary")
            return data
        remaining_issues = []
        candidate_comments = {}
        candidate_issues = {}
        candidate_fingerprints = {}
        published = 0
        for issue in issues:
            try:
                comment = self._build_key_issue_comment(issue, diff_files)
                if comment is None:
                    remaining_issues.append(issue)
                    continue
                fingerprint = key_issue_fingerprint(comment["relevant_file"], comment["body"])
                if store.seen(fingerprint):
                    published += 1
                    continue
                location_fingerprint = key_issue_location_fingerprint(
                    fingerprint, comment["relevant_lines_start"], comment["relevant_lines_end"])
                if location_fingerprint in candidate_comments:
                    candidate_issues[location_fingerprint].append(issue)
                    continue
                comment["body"] = key_issue_body_with_markers(
                    comment["body"], fingerprint, location_fingerprint,
                    getattr(self.git_provider, "max_comment_chars", None))
                candidate_comments[location_fingerprint] = comment
                candidate_issues[location_fingerprint] = [issue]
                candidate_fingerprints[location_fingerprint] = fingerprint
            except Exception as e:
                get_logger().warning(f"Failed to prepare a review finding for inline publication, error: {e}",
                                     artifact={"issue": issue})
                remaining_issues.append(issue)

        if candidate_comments:
            try:
                self.git_provider.publish_code_suggestions(list(candidate_comments.values()))
            except Exception as e:
                locations = [{"relevant_file": comment["relevant_file"],
                              "start_line": comment["relevant_lines_start"],
                              "end_line": comment["relevant_lines_end"]}
                             for comment in candidate_comments.values()]
                get_logger().warning(
                    f"Failed to publish review findings as Azure DevOps threads, error: {e}",
                    artifact={"locations": locations})
            verified_locations = self._published_inline_key_issue_fingerprints(store, set(candidate_comments))
            for location_fingerprint, comment in candidate_comments.items():
                issues_for_location = candidate_issues[location_fingerprint]
                if location_fingerprint in verified_locations:
                    store.add(candidate_fingerprints[location_fingerprint])
                    store.add(location_fingerprint)
                    published += len(issues_for_location)
                    continue
                get_logger().warning("Failed to publish a review finding as an Azure DevOps inline comment, "
                                     "keeping it in the summary",
                                     artifact={"relevant_file": comment["relevant_file"],
                                               "start_line": comment["relevant_lines_start"],
                                               "end_line": comment["relevant_lines_end"]})
                remaining_issues.extend(issues_for_location)

        if not published:
            return data
        get_logger().info(f"Published {published} review finding(s) as inline comments")

        data = copy.deepcopy(data)
        if remaining_issues:
            data["review"]["key_issues_to_review"] = remaining_issues
        else:
            data["review"].pop("key_issues_to_review", None)
        return data

    def _get_user_answers(self) -> Tuple[str, str]:
        """
        Retrieves the question and answer strings from the discussion messages related to a pull request.

        Returns:
            A tuple containing the question and answer strings.
        """
        question_str = ""
        answer_str = ""

        if self.is_answer:
            discussion_messages = self.git_provider.get_issue_comments()

            # providers return the comments oldest-first. PyGithub's PaginatedList reverses lazily,
            # so prefer it and only materialise the plain lists other providers return.
            newest_first = getattr(discussion_messages, "reversed", None)
            if newest_first is None:
                newest_first = reversed(list(discussion_messages))

            for message in newest_first:
                if "Questions to better understand the PR:" in message.body:
                    question_str = message.body
                elif '/answer' in message.body:
                    answer_str = message.body

                if answer_str and question_str:
                    break

        return question_str, answer_str

    def _get_previous_review_comment(self):
        """
        Get the previous review comment if it exists.
        """
        try:
            if hasattr(self.git_provider, "get_previous_review"):
                return self.git_provider.get_previous_review(
                    full=not self.incremental.is_incremental,
                    incremental=self.incremental.is_incremental,
                )
        except Exception as e:
            get_logger().exception(f"Failed to get previous review comment, error: {e}")

    def _remove_previous_review_comment(self, comment):
        """
        Remove the previous review comment if it exists.
        """
        try:
            if comment:
                self.git_provider.remove_comment(comment)
        except Exception as e:
            get_logger().exception(f"Failed to remove previous review comment, error: {e}")

    def _can_run_incremental_review(self) -> bool:
        """
        Checks if we can run incremental review according the various configurations and previous review.
        """
        # checking if running is auto mode but there are no new commits
        if self.is_auto and not self.incremental.first_new_commit_sha:
            get_logger().info(f"Incremental review is enabled for {self.pr_url} but there are no new commits")
            return False

        if not hasattr(self.git_provider, "get_incremental_commits"):
            get_logger().info(f"Incremental review is not supported for {get_settings().config.git_provider}")
            return False
        if self.incremental.commits_range is None:
            get_logger().info(
                f"Incremental review not initialized for {get_settings().config.git_provider}; "
                f"falling back to full review."
            )
            self.incremental.is_incremental = False
            return False
        # checking if there are enough commits to start the review
        num_new_commits = len(self.incremental.commits_range)
        num_commits_threshold = get_settings().pr_reviewer.minimal_commits_for_incremental_review
        not_enough_commits = num_new_commits < num_commits_threshold
        # checking if the commits are not too recent to start the review
        recent_commits_threshold = datetime.datetime.now() - datetime.timedelta(
            minutes=get_settings().pr_reviewer.minimal_minutes_for_incremental_review
        )
        last_seen_commit_date = (
            self.incremental.last_seen_commit.commit.author.date if self.incremental.last_seen_commit else None
        )
        all_commits_too_recent = (
            last_seen_commit_date > recent_commits_threshold if self.incremental.last_seen_commit else False
        )
        # check all the thresholds or just one to start the review
        condition = any if get_settings().pr_reviewer.require_all_thresholds_for_incremental_review else all
        if condition((not_enough_commits, all_commits_too_recent)):
            get_logger().info(
                f"Incremental review is enabled for {self.pr_url} but didn't pass the threshold check to run:"
                f"\n* Number of new commits = {num_new_commits} (threshold is {num_commits_threshold})"
                f"\n* Last seen commit date = {last_seen_commit_date} (threshold is {recent_commits_threshold})"
            )
            return False
        return True

    def set_review_labels(self, data):
        if not get_settings().config.publish_output:
            return

        if not get_settings().pr_reviewer.require_estimate_effort_to_review:
            get_settings().pr_reviewer.enable_review_labels_effort = False # we did not generate this output
        if not get_settings().pr_reviewer.require_security_review:
            get_settings().pr_reviewer.enable_review_labels_security = False # we did not generate this output

        if (get_settings().pr_reviewer.enable_review_labels_security or
                get_settings().pr_reviewer.enable_review_labels_effort):
            try:
                review_labels = []
                if get_settings().pr_reviewer.enable_review_labels_effort:
                    estimated_effort = data['review']['estimated_effort_to_review_[1-5]']
                    estimated_effort_number = 0
                    if isinstance(estimated_effort, str):
                        try:
                            estimated_effort_number = int(estimated_effort.split(',')[0])
                        except ValueError:
                            get_logger().warning(f"Invalid estimated_effort value: {estimated_effort}")
                    elif isinstance(estimated_effort, int):
                        estimated_effort_number = estimated_effort
                    else:
                        get_logger().warning(f"Unexpected type for estimated_effort: {type(estimated_effort)}")
                    if 1 <= estimated_effort_number <= 5:  # 1, because ...
                        review_labels.append(f'Review effort {estimated_effort_number}/5')
                if get_settings().pr_reviewer.enable_review_labels_security and get_settings().pr_reviewer.require_security_review:
                    security_concerns = data['review']['security_concerns']  # yes, because ...
                    security_concerns_bool = 'yes' in security_concerns.lower() or 'true' in security_concerns.lower()
                    if security_concerns_bool:
                        review_labels.append('Possible security concern')

                current_labels = self.git_provider.get_pr_labels(update=True)
                if not current_labels:
                    current_labels = []
                get_logger().debug(f"Current labels:\n{current_labels}")
                if current_labels:
                    current_labels_filtered = [label for label in current_labels if
                                               not label.lower().startswith('review effort') and not label.lower().startswith(
                                                   'possible security concern')]
                else:
                    current_labels_filtered = []
                new_labels = review_labels + current_labels_filtered
                if (current_labels or review_labels) and sorted(new_labels) != sorted(current_labels):
                    get_logger().info(f"Setting review labels:\n{review_labels + current_labels_filtered}")
                    self.git_provider.publish_labels(new_labels)
                else:
                    get_logger().info(f"Review labels are already set:\n{review_labels + current_labels_filtered}")
            except Exception as e:
                get_logger().error(f"Failed to set review labels, error: {e}")

    def auto_approve_logic(self):
        """
        Auto-approve a pull request if it meets the conditions for auto-approval.
        """
        if get_settings().config.enable_auto_approval:
            is_auto_approved = self.git_provider.auto_approve()
            if is_auto_approved:
                get_logger().info("Auto-approved PR")
                self.git_provider.publish_comment("Auto-approved PR")
        else:
            get_logger().info("Auto-approval option is disabled")
            self.git_provider.publish_comment("Auto-approval option for PR-Agent is disabled. "
                                              "You can enable it via a [configuration file](https://github.com/Codium-ai/pr-agent/blob/main/docs/REVIEW.md#auto-approval-1)")

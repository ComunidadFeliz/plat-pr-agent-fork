"""Second pass over the review's findings: try to REFUTE each one before publishing it.

Why this exists. Measured on 11 real PRs of a large Rails monolith with claude-opus-5: of the 19
findings the engine published, only 10 survived a hand check against the code. Every one that did
not survive was refutable by opening a file or running a line -- in one case by reading a spec that
was part of the very same PR. The review pass cannot do that: it only ever sees the diff. So this
module resolves the evidence the finding refers to and asks the model to refute its own claim.

The resolution step is mechanical and matters more than the prompt: an earlier version handed the
verifier only the PR's spec files and killed 1 of 4 false findings; handing it the PR's diffs first
took the same prompt and the same model to 4 of 4.

Bias: a finding is dropped ONLY on a "refuted" verdict that cites evidence. Every other outcome --
upheld, undetermined, unparseable JSON, an exception -- keeps the finding. Deleting a true finding
in silence is worse than the noise it avoids, and a keyword filter already taught us that lesson by
almost dropping the best finding of the batch. With this bias the measured pass dropped 4 of 4 false
findings and 0 true ones across 30 verifications.
"""
import asyncio
import copy
import json
import re
from itertools import islice

from jinja2 import Environment, StrictUndefined

from pr_agent.algo.utils import ModelType
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

# Paths the finding text names. Kept to source-ish extensions on purpose: a finding that names a
# .png or a .lock file is not making a claim we can resolve.
FILE_RE = re.compile(r"[\w./-]+\.(?:rb|rake|erb|haml|js|jsx|ts|tsx|py|yml|yaml|scss|css|sh|sql|json)\b")
BACKTICK_RE = re.compile(r"`([^`]{2,90})`")
IDENT_RE = re.compile(r"^[A-Za-z_][\w:#.?!]*$")
# Words that appear in backticks constantly and resolve to thousands of unrelated files.
SYMBOL_STOPWORDS = {"undefined", "nil", "null", "none", "true", "false", "self", "params",
                    "default", "string", "integer", "boolean", "array", "hash", "object"}
MAX_SYMBOLS = 6
MAX_FILES = 8
WINDOW_BEFORE = 20
WINDOW_AFTER = 40


def _named_files(text: str) -> list:
    return list(dict.fromkeys(m.group(0) for m in FILE_RE.finditer(text)))


def _symbols(text: str) -> list:
    """Identifiers the finding names, tolerating call and method syntax.

    `upselling_info_path(package: 'x')` -> upselling_info_path, and also upselling_info: the finding
    names the route helper but the evidence lives in the action.
    """
    out = []
    for raw in BACKTICK_RE.findall(text):
        candidate = raw.strip().split("(")[0].strip()
        if " " in candidate or not IDENT_RE.match(candidate):
            continue
        base = candidate.split("#")[-1].split("::")[-1].strip("?!")
        if len(base) < 4 or base.lower() in SYMBOL_STOPWORDS:
            continue
        out.append(base)
        if base.endswith("_path"):
            out.append(base[:-len("_path")])
    return list(dict.fromkeys(out))[:MAX_SYMBOLS]


def _window(content: str, needle: str) -> str:
    lines = content.split("\n")
    hits = [i for i, line in enumerate(lines) if needle in line]
    if not hits:
        return ""
    chunks = []
    for i in hits[:2]:
        lo, hi = max(0, i - WINDOW_BEFORE), min(len(lines), i + WINDOW_AFTER)
        chunks.append("\n".join(f"{n + 1}: {lines[n]}" for n in range(lo, hi)))
    return "\n...\n".join(chunks)


def _is_test(path: str) -> bool:
    return "spec/" in path or "test/" in path or path.endswith(("_spec.rb", "_test.rb", ".test.ts", ".spec.ts"))


def _search_symbol(git_provider, symbol: str) -> list:
    """Best-effort code search for where a symbol lives. GitHub only; silent no-op elsewhere."""
    client = getattr(git_provider, "github_client", None)
    repo_name = getattr(git_provider, "repo", None)
    if client is None or not repo_name:
        return []
    try:
        # islice, no `results[:2]`: PyGithub levanta IndexError al slicear un PaginatedList que
        # vuelve vacío, y la búsqueda de código vuelve vacía todo el tiempo (símbolo que no está
        # indexado, repo grande, rate limit). Visto en el e2e del sandbox: cada símbolo dejaba un
        # "list index out of range" y este canal de evidencia quedaba muerto en silencio.
        return [item.path for item in islice(client.search_code(f"{symbol} repo:{repo_name}"), 2)]
    except Exception as e:
        get_logger().debug(f"finding verification: code search for '{symbol}' unavailable: {e}")
        return []


def _file_content(git_provider, path: str) -> str:
    """Content from the base branch: a finding is verified against what the PR builds on, and the
    head is author-controlled."""
    try:
        return git_provider.get_repo_file_content(path) or ""
    except Exception:
        return ""


def resolve_evidence(git_provider, finding: dict, diff_files=None) -> list:
    """Collect the code a finding talks about, most decisive first. Mechanical, no model calls.

    Order is load-bearing (measured): the PR's own diffs, then whole files the finding names, then
    windows around the symbols it names, then the PR's tests -- which is where a contract that
    refutes the finding tends to live.
    """
    text = f"{finding.get('issue_content', '')} {finding.get('failure_scenario', '')}"
    named = _named_files(text)
    relevant = str(finding.get("relevant_file") or "").strip()
    if relevant:
        named.insert(0, relevant)
    max_chars = get_settings().pr_reviewer.get("verify_findings_max_evidence_chars", 26000)

    def mentioned(path: str) -> bool:
        tail = path.split("/")[-1]
        return any(path.endswith(n) or n.endswith(tail) for n in named)

    refs, seen = [], set()

    if diff_files is None:
        try:
            diff_files = git_provider.get_diff_files() or []
        except Exception:
            diff_files = []

    # 1) the change itself, files the finding names first and tests last
    for f in sorted(diff_files, key=lambda f: (not mentioned(f.filename), _is_test(f.filename)))[:MAX_FILES]:
        if not getattr(f, "patch", None):
            continue
        seen.add(f.filename)
        refs.append((f"diff of {f.filename}", f.patch[:7000]))

    # 2) whole files the finding names
    for name in named:
        in_diff = [f.filename for f in diff_files
                   if f.filename.endswith(name) or name.endswith(f.filename.split("/")[-1])]
        for path in (in_diff[:1] or [name]):
            if path in seen:
                continue
            content = _file_content(git_provider, path)
            if content:
                seen.add(path)
                refs.append((f"{path} (full file)", content[:7000]))

    # 3) where the symbols live
    for symbol in _symbols(text):
        for path in _search_symbol(git_provider, symbol):
            if path in seen:
                continue
            excerpt = _window(_file_content(git_provider, path), symbol)
            if excerpt:
                seen.add(path)
                refs.append((f"{path} (around `{symbol}`)", excerpt[:6000]))

    # 4) the PR's tests
    for f in diff_files:
        if _is_test(f.filename) and f.filename not in seen:
            content = _file_content(git_provider, f.filename) or getattr(f, "head_file", "") or ""
            if content:
                seen.add(f.filename)
                refs.append((f"{f.filename} (test from this PR)", content[:6000]))

    budget, out = 0, []
    for label, body in refs:
        if budget + len(body) > max_chars:
            continue
        budget += len(body)
        out.append((label, body))
    return out


def _parse_verdict(response: str) -> dict:
    """Parse the verifier's JSON. Anything unparseable is treated as 'keep the finding'."""
    if not response:
        return {}
    text = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    for candidate in (text, (re.search(r"\{.*\}", text, re.S) or type("", (), {"group": lambda *_: ""})).group(0)):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


async def _verify_one(ai_handler, git_provider, finding: dict, model: str, diff_files) -> dict:
    evidence = resolve_evidence(git_provider, finding, diff_files)
    variables = {
        "issue_header": str(finding.get("issue_header") or "").strip(),
        "issue_content": str(finding.get("issue_content") or "").strip(),
        "failure_scenario": str(finding.get("failure_scenario") or "").strip(),
        "evidence": "\n\n".join(f"===== {label} =====\n{body}" for label, body in evidence),
        "resolved_any": bool(evidence),
    }
    env = Environment(undefined=StrictUndefined)
    system = env.from_string(get_settings().pr_finding_verification_prompt.system).render(variables)
    user = env.from_string(get_settings().pr_finding_verification_prompt.user).render(variables)
    response, _ = await ai_handler.chat_completion(model=model, system=system, user=user)
    verdict = _parse_verdict(response)
    verdict["_sources"] = [label for label, _ in evidence]
    return verdict


async def verify_findings(ai_handler, git_provider, findings: list, model: str) -> list:
    """Return the issue_headers of the findings that were refuted with cited evidence.

    Never raises: a verification that fails for any reason leaves its finding published.
    """
    if not findings:
        return []
    try:
        diff_files = git_provider.get_diff_files() or []
    except Exception:
        diff_files = []

    tasks = [_verify_one(ai_handler, git_provider, copy.deepcopy(f), model, diff_files) for f in findings]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    refuted, report = [], []
    for finding, result in zip(findings, results):
        header = str(finding.get("issue_header") or "").strip()
        if isinstance(result, Exception) or not isinstance(result, dict):
            get_logger().info(f"finding verification: keeping '{header}' (verification failed: {result})")
            continue
        verdict = str(result.get("verdict") or result.get("veredicto") or "").strip().lower()
        evidence = str(result.get("evidence") or result.get("evidencia") or "").strip()
        if verdict in ("refuted", "refutado") and evidence:
            refuted.append(header)
            report.append({"issue_header": header, "evidence": evidence[:400],
                           "sources": result.get("_sources", [])})
        else:
            get_logger().debug(f"finding verification: keeping '{header}' (verdict: {verdict or 'none'})")
    if report:
        get_logger().info(f"finding verification: refuted {len(report)} of {len(findings)} findings",
                          artifact={"refuted": report})
    return refuted

"""수정안(unified diff) 적용 → 재검사 — LLM 이 낸 수정안이 정말 문제를 고치는지 코드로 확인한다.

설계 문서 docs/AI_전용_업무_설계.md 8장 2단계: "수정안을 적용해 재검사 통과 여부를 코드로 확인한다".

  parse_patches : 답변의 ```diff 블록 → 파일별 hunk (줄 수가 틀린 머리줄 @@ 도 받아 준다 — 내용으로 위치를 찾는다)
  apply_hunks   : 원문 사본에 hunk 적용. 문맥·삭제 줄이 원문에 없으면 실패 (공백 차이만 한 번 더 너그럽게 본다)
  verify        : 파일별로 사본에 적용 → 내장 규칙으로 전체 재검사 → 발견마다 고침 / 못 고침 / 시도 안 함
                  '고침'은 세 가지를 모두 만족할 때만: ① 그 발견이 재검사에서 사라짐 (리소스를 지워 없앤 것은 제외)
                  ② 그 파일에 같거나 더 높은 심각도의 새 문제가 없음 ③ 파일이 여전히 구문상 읽힘
  하네스 검증기 iac_fix(core/harness/validators.py)와 화면 표·자율 실행 승인 요청이 모두 이 결과를 쓴다.

원본 파일은 건드리지 않는다 — 전부 메모리 안 사본에서 한다.
"""

import difflib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from core import iac
from core.harness.base import extract_code_blocks

STATUS_LABEL = {"fixed": "✅ 고침", "not_fixed": "❌ 못 고침", "not_attempted": "· 시도 안 함"}


# ─── diff 읽기 · 적용 ───────────────────────────────────────────────────


@dataclass
class Hunk:
    old_start: int  # 머리줄의 원래 시작 줄 (1부터, 0 = 맨 앞) — 위치 찾기의 힌트로만 쓴다
    old: list[str]  # 문맥 + 지울 줄
    new: list[str]  # 문맥 + 넣을 줄


@dataclass
class FilePatch:
    path: str
    hunks: list[Hunk] = field(default_factory=list)
    new_file: bool = False
    delete: bool = False


_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def _clean_path(raw: str) -> str:
    path = raw.split("\t")[0].strip().strip('"')
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path.removeprefix("./")


def parse_patches(answer: str) -> list[FilePatch]:
    """```diff / ```patch 블록의 unified diff → 파일별 patch (한 블록에 여러 파일도 된다)."""
    out: list[FilePatch] = []
    for block in extract_code_blocks(answer, "diff", "patch"):
        lines = block.split("\n")
        cur: FilePatch | None = None
        hunk: Hunk | None = None
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                old, new = _clean_path(ln[4:]), _clean_path(lines[i + 1][4:])
                cur = FilePatch(path=old if new == "/dev/null" else new, new_file=old == "/dev/null", delete=new == "/dev/null")
                out.append(cur)
                hunk = None
                i += 2
                continue
            m = _HUNK.match(ln)
            if m and cur is not None:
                hunk = Hunk(int(m.group(1)), [], [])
                cur.hunks.append(hunk)
            elif hunk is not None and not ln.startswith(("diff --git", "index ", "\\")):
                if ln.startswith("-"):
                    hunk.old.append(ln[1:])
                elif ln.startswith("+"):
                    hunk.new.append(ln[1:])
                else:  # 문맥 줄 — 모델이 빈 문맥 줄의 앞 공백을 빼먹는 일이 잦아 빈 줄도 문맥으로 본다
                    text = ln[1:] if ln.startswith(" ") else ln
                    hunk.old.append(text)
                    hunk.new.append(text)
            i += 1
    for p in out:  # 블록 끝의 빈 줄이 문맥으로 붙은 것 걷어내기
        for h in p.hunks:
            while h.old and h.new and h.old[-1] == "" and h.new[-1] == "":
                h.old.pop()
                h.new.pop()
    return out


class PatchError(ValueError):
    pass


def apply_hunks(text: str, hunks: list[Hunk]) -> str:
    """원문에 hunk 들을 차례로 적용한 새 글. 위치는 '문맥+지울 줄' 내용으로 찾고, 머리줄 줄 번호에 가장 가까운 곳을 쓴다."""
    lines = text.split("\n")
    trailing = text.endswith("\n")
    if trailing:
        lines = lines[:-1]
    cursor, delta = 0, 0
    for n, h in enumerate(hunks, 1):
        if not h.old:  # 문맥 없는 순수 추가
            at = min(max(h.old_start + delta, 0), len(lines))
            pos = max(at, cursor)
        else:
            pos = _find(lines, h.old, cursor, h.old_start - 1 + delta, strict=True)
            if pos is None:
                pos = _find(lines, h.old, cursor, h.old_start - 1 + delta, strict=False)
            if pos is None:
                have = {" ".join(x.split()) for x in lines}
                odd = next((x for x in h.old if x.strip() and " ".join(x.split()) not in have), None)
                where = f"원본에 없는 줄 '{odd.strip()[:80]}'" if odd else "줄들은 있지만 원본과 순서·연속이 다름"
                raise PatchError(f"{n}번째 hunk 의 문맥·지울 줄이 원본과 맞지 않음 — {where}. "
                                 "diff 의 ' '(문맥)·'-'(지울) 줄은 원본을 글자 그대로 복사해야 한다")
        lines[pos: pos + len(h.old)] = h.new
        cursor = pos + len(h.new)
        delta += len(h.new) - len(h.old)
    return "\n".join(lines) + ("\n" if trailing else "")


def _find(lines: list[str], old: list[str], start: int, hint: int, strict: bool) -> int | None:
    norm = (lambda s: s.rstrip()) if strict else (lambda s: " ".join(s.split()))
    want = [norm(x) for x in old]
    hits = [p for p in range(start, len(lines) - len(old) + 1) if [norm(x) for x in lines[p: p + len(old)]] == want]
    return min(hits, key=lambda p: abs(p - hint)) if hits else None


def _resolve(path: str, files: dict[str, str]) -> str | None:
    if path in files:
        return path
    ends = [f for f in files if f.endswith("/" + path) or path.endswith("/" + f)]
    return ends[0] if len(ends) == 1 else None


# ─── 답변 JSON ──────────────────────────────────────────────────────────


def answer_json(answer: str) -> dict:
    for src in reversed(extract_code_blocks(answer, "json")):
        try:
            data = json.loads(src)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def norm_id(raw) -> str:
    m = re.fullmatch(r"\s*[Ff]0*(\d+)\s*", str(raw))
    return f"F{int(m.group(1)):02d}" if m else str(raw).strip()


def _ids(items, key: str = "id") -> list[str]:
    out = []
    for x in items if isinstance(items, list) else []:
        if isinstance(x, dict) and x.get(key) is not None:
            out += [norm_id(v) for v in (x[key] if isinstance(x[key], list) else [x[key]])]
        elif isinstance(x, str):
            out.append(norm_id(x))
    return out


# ─── 검증 ───────────────────────────────────────────────────────────────


@dataclass
class FileOutcome:
    file: str
    applied: bool = False
    problems: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)  # 이 파일 수정안이 고치려는 발견 id
    new: list[iac.Finding] = field(default_factory=list)  # 재검사에서 새로 생긴 문제

    @property
    def ok(self) -> bool:
        return self.applied and not self.problems


@dataclass
class Verification:
    before: iac.ScanResult
    after: iac.ScanResult  # 적용된 수정안을 모두 넣은 사본의 재검사
    patched: dict[str, str]  # 그 사본
    outcomes: dict[str, FileOutcome]
    status: dict[str, str]  # 발견 id → fixed / not_fixed / not_attempted
    reasons: dict[str, str]
    problems: list[str] = field(default_factory=list)  # 파일에 묶이지 않는 문제 (없는 id, 삭제, 모르는 파일 …)
    diffs: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems and all(o.ok for o in self.outcomes.values())

    def count(self, status: str) -> int:
        return sum(1 for s in self.status.values() if s == status)

    @property
    def verified_files(self) -> list[str]:
        return [f for f, o in self.outcomes.items() if o.ok and o.targets]


def verify(files: dict[str, str], answer: str, require_fix: bool = False, cover_all: bool = False) -> Verification:
    """답변의 수정안을 사본에 적용해 재검사한다. 원본 files 는 바꾸지 않는다."""
    before = iac.scan_files(files)
    by_id = before.by_id()
    data = answer_json(answer)
    problems: list[str] = []

    listed = _ids(data.get("findings")) + _ids(data.get("fixes"), "findings") + _ids(data.get("not_attempted"))
    unknown = sorted(set(i for i in listed if i not in by_id))
    if unknown:
        problems.append(f"점검 결과에 없는 발견 번호 {', '.join(unknown[:10])} — 근거 자료 표의 ID(F01…)만 쓴다")
    for item in data.get("findings") or []:
        if isinstance(item, dict) and item.get("rule") and norm_id(item.get("id")) in by_id:
            fid = norm_id(item["id"])
            if str(item["rule"]).strip() != by_id[fid].rule:
                problems.append(f"{fid} 의 규칙은 {by_id[fid].rule} 인데 {item['rule']} 로 적음")
    if cover_all:
        missing = [f.id for f in before.findings if f.id not in set(listed)]
        if missing:
            problems.append(f"점검 결과 {len(before.findings)}건 중 {len(missing)}건을 빠뜨림: {', '.join(missing)} — 모든 발견을 findings 에 넣는다")

    listed_targets: dict[str, list[str]] = {}  # 파일 → 고치겠다고 적은 id
    for fid in _ids(data.get("fixes"), "findings"):
        if fid in by_id:
            listed_targets.setdefault(by_id[fid].file, []).append(fid)

    patches = parse_patches(answer)
    if require_fix and not patches:
        problems.append("수정안 diff 가 없음 — 고칠 수 있는 발견마다 ```diff 블록(unified diff)을 붙인다")
    patched = dict(files)
    outcomes: dict[str, FileOutcome] = {}
    for p in patches:
        if p.delete:
            problems.append(f"{p.path}: 파일 삭제는 수정으로 치지 않는다 — 설정을 고치는 diff 를 쓴다")
            continue
        target = p.path if p.new_file else _resolve(p.path, files)
        if target is None:
            problems.append(f"diff 의 파일 '{p.path}' 이 점검 대상에 없음 (있는 파일: {', '.join(before.files)})")
            continue
        out = outcomes.setdefault(target, FileOutcome(target))
        if p.new_file and target in files:
            out.problems.append("이미 있는 파일을 /dev/null 에서 새로 만드는 diff — 기존 파일을 고치는 diff 로 쓴다")
            continue
        try:
            patched[target] = apply_hunks("" if p.new_file else patched[target], p.hunks)
            out.applied = True
        except PatchError as e:
            out.problems.append(f"패치가 적용되지 않음: {e}")
    for file, ids in listed_targets.items():
        if file not in outcomes:
            outcomes[file] = FileOutcome(file, problems=["fixes 에 적었지만 이 파일의 diff 가 없음"], targets=ids)

    after = iac.scan_files(patched)
    after_keys = {f.key for f in after.findings}
    before_keys = {f.key for f in before.findings}
    for file, out in outcomes.items():
        if not out.applied:
            out.targets = out.targets or listed_targets.get(file, [])
            continue
        if file in after.errors and file not in before.errors:
            out.problems.append(f"수정 뒤 파일을 읽을 수 없음 (구문 오류) — {after.errors[file]}")
        in_file = [f for f in before.findings if f.file == file]
        out.targets = listed_targets.get(file) or [f.id for f in in_file if f.key not in after_keys]
        if not out.targets:
            out.problems.append("이 diff 로 사라진 발견이 없음 — 어느 발견(F…)을 고치려는지 fixes 에 적고, 규칙이 보는 값을 바꾼다")
        for fid in out.targets:
            f = by_id[fid]
            if f.key in after_keys:
                out.problems.append(f"{fid}({f.rule}) 이 재검사에 그대로 남음 — {f.detail}")
            elif f.anchor not in after.inventory and file not in after.errors:
                out.problems.append(f"{fid}: {f.anchor} 를 지워 없앴다 — 리소스·컨테이너를 지우는 것은 고침으로 치지 않는다")
        floor = min((iac.severity_rank(by_id[t].severity) for t in out.targets), default=0)
        out.new = [f for f in after.findings if f.file == file and f.key not in before_keys]
        worse = [f for f in out.new if iac.severity_rank(f.severity) >= floor]
        if worse:
            out.problems.append("수정 뒤 새 문제: " + " / ".join(f"{f.rule}({f.severity}) {f.resource} {f.item} — {f.detail}".strip()
                                                            for f in worse[:4]))
    stray = [f for f in after.findings if f.key not in before_keys and f.file not in outcomes]
    if stray:
        problems.append("수정 뒤 다른 파일에 새 문제: " + " / ".join(f"{f.file} {f.rule} {f.resource}" for f in stray[:4]))

    status: dict[str, str] = {}
    reasons: dict[str, str] = {}
    targeted = {t: out for out in outcomes.values() for t in out.targets}
    for f in before.findings:
        out = targeted.get(f.id)
        if out is None:
            status[f.id] = "not_attempted"
            reasons[f.id] = next((str(x.get("reason") or "") for x in data.get("not_attempted") or []
                                  if isinstance(x, dict) and norm_id(x.get("id")) == f.id), "")
        elif out.ok:
            status[f.id], reasons[f.id] = "fixed", "재검사에서 사라짐 · 새 문제 없음 · 구문 정상"
        else:
            status[f.id], reasons[f.id] = "not_fixed", out.problems[0] if out.problems else ""
    return Verification(before, after, patched, outcomes, status, reasons, problems, diffs=len(patches))


def feedback(v: Verification) -> list[str]:
    """하네스 피드백 문장 (한국어) — 실패 이유를 파일별로."""
    lines = list(v.problems)
    for file, out in v.outcomes.items():
        lines += [f"{file}: {p}" for p in out.problems]
    return lines


# ─── 보고서 · 승인 요청 ─────────────────────────────────────────────────


def file_diff(old: str, new: str, path: str, is_new: bool = False) -> str:
    """표준 unified diff 한 파일 (a/ b/ 접두, LF) — 마지막 줄바꿈이 없으면 git 과 같은 '\\ No newline' 표시를 붙인다."""
    out = []
    for line in difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                     "/dev/null" if is_new else f"a/{path}", f"b/{path}"):
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


def verified_patch(v: Verification, files: dict[str, str], prefix: str = "") -> tuple[str, dict[str, str]]:
    """검증을 통과한 파일만 넣은 표준 unified diff (코드가 다시 만든 것 — git apply 로 적용 가능) 와 그 사본.

    prefix: 경로 앞에 붙일 폴더 (예: 저장소 루트 기준 'LLM/Agent_office/teams/…/sample_corp/').
    """
    chosen = {f: v.patched[f] for f in v.verified_files}
    patch = "".join(file_diff(files.get(f, ""), new, prefix + f, f not in files) for f, new in chosen.items())
    return patch, {**files, **chosen}


def markdown_report(v: Verification) -> str:
    """화면·자율 실행 보고서에 붙일 표 — 발견마다 고침 / 못 고침 / 시도 안 함 (코드가 판정)."""
    before, after = v.before.counts(), v.after.counts()
    head = (f"발견 {before['total']}건 중 **고침 {v.count('fixed')}** · 못 고침 {v.count('not_fixed')} · 시도 안 함 {v.count('not_attempted')}"
            f" — 수정안 diff {v.diffs}개, 적용한 사본 재검사 {before['total']}건 → {after['total']}건")
    rows = ["| ID | 심각도 | 규칙 | 파일 | 결과 | 이유 |", "|---|---|---|---|---|---|"]
    order = {"fixed": 0, "not_fixed": 1, "not_attempted": 2}
    for f in sorted(v.before.findings, key=lambda x: (order[v.status[x.id]], -iac.severity_rank(x.severity), x.id)):
        reason = v.reasons.get(f.id, "").replace("|", "/")
        rows.append(f"| {f.id} | {f.severity} | {f.rule} | {f.file}:{f.line} | {STATUS_LABEL[v.status[f.id]]} | {reason[:160]} |")
    files = [f"- `{file}` " + ("✅ 적용 · 재검사 통과" if o.ok else "❌ " + " / ".join(o.problems)[:300]) for file, o in v.outcomes.items()]
    extra = [f"- ⚠️ {p}" for p in v.problems]
    return "\n".join([head, "", *rows, "", *files, *extra]).strip()


def verification_report(answer: str, options: dict) -> str:
    """화면용 — 옵션에 IaC 원본(iac_files)이 없거나 답변에 diff 가 없으면 빈 글."""
    files = options.get("iac_files") or {}
    if not files or not parse_patches(answer):
        return ""
    return markdown_report(verify(files, answer))


REPO_ROOT = iac.BASE_DIR.parents[1]  # LLM/Agent_office → 저장소 루트


def repo_prefix(source_dir: str) -> str | None:
    """IaC 폴더 → 저장소 루트 기준 '/' 경로 접두 (예: 'LLM/Agent_office/teams/infra_security/samples/sample_corp/').
    저장소 밖 폴더·붙여 넣은 글이면 None (브랜치를 만들 수 없다)."""
    if not source_dir or source_dir == "붙여 넣은 글":
        return None
    path = Path(source_dir)
    path = (path if path.is_absolute() else iac.BASE_DIR / path).resolve()
    try:
        rel = path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None
    return f"{rel}/" if rel not in ("", ".") else ""


def base_commit() -> str:
    """패치를 만든 기준 커밋 (git 이 없거나 저장소가 아니면 빈 글 = 승인 쪽이 HEAD 를 쓴다)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=iac.BASE_DIR, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def proposals(v: Verification, files: dict[str, str], source_dir: str) -> list[dict]:
    """재검사를 통과한 파일마다 제안 하나 — 승인 대기열 Proposal 모양의 평범한 dict
    (core/approvals.py Proposal · RescanResult, 승인 처리 세션이 정함: 파일 하나 · 저장소 루트 기준 경로 · LF diff).

    저장소 밖 폴더·붙여 넣은 글이면 target_file 은 IaC 폴더 기준 경로 (브랜치는 못 만들고 제안 저장만 할 수 있다).
    """
    prefix = repo_prefix(source_dir)
    commit = base_commit() if prefix is not None else ""
    _, applied = verified_patch(v, files)
    rescan = iac.scan_files(applied)  # 검증을 통과한 수정안만 넣은 사본
    out = []
    for file in v.verified_files:
        fixed = [v.before.by_id()[t] for t in v.outcomes[file].targets]
        before = [f.rule for f in v.before.findings if f.file == file]
        after = [f.rule for f in rescan.findings if f.file == file]
        rules = sorted({f.rule for f in fixed})
        target = (prefix or "") + file
        out.append({
            "schema_version": 1,
            "title": f"{file}: IaC 수정 {len(fixed)}건 ({', '.join(rules)})"[:120],
            "target_file": target,
            "patch": file_diff(files.get(file, ""), v.patched[file], target, file not in files),
            "rationale": " / ".join(f"{f.id} {f.title} — {iac.rules().get(f.rule, {}).get('why', '')}" for f in fixed)[:2000],
            "findings": [f"{f.id} {f.rule}: {f.resource} {f.item} — {f.detail}".replace("  ", " ") for f in fixed],
            "rescan": {
                "scanner": f"builtin ({len(iac.rules())} rules, config/iac_rules.yaml)",
                "passed": True,  # verified_files = 고치려던 발견이 모두 사라지고 새 문제·구문 오류가 없는 파일만
                "before": before,
                "after": after,
                "log": (f"고침 {', '.join(f.id for f in fixed)} · 파일 안 발견 {len(before)}→{len(after)} · "
                        f"전체 {len(v.before.findings)}→{len(rescan.findings)}")[:2000],
            },
            "base_commit": commit,
        })
    return out


def autonomy_outputs(answer: str, options: dict, job) -> dict:
    """자율 실행 한 번의 IaC 결과 → {audit: run_end 에 덧붙일 값, report: 보고서 뒤에 붙일 글, actions: 승인 요청}.

    재검사를 통과한 파일마다 제안(Proposal 모양) 하나를 승인 요청으로 낸다 — 저장소 안 폴더면 create_branch_with_patch,
    저장소 밖·붙여 넣은 글이면 save_proposal. 둘 다 gate 가 어느 수준에서든 승인 대기로 돌린다 (스스로 하지 않는다).
    통과한 diff 는 보고서(허용된 report 행동)에도 그대로 남긴다.
    """
    files = options.get("iac_files") or {}
    if not files:
        return {"audit": {}, "report": "", "actions": []}
    v = verify(files, answer)
    audit = {"iac_findings": len(v.before.findings), "iac_attempted": v.count("fixed") + v.count("not_fixed"),
             "iac_fixed": v.count("fixed"), "iac_not_fixed": v.count("not_fixed")}
    report = "## 🔁 수정안 재검사 (코드)\n\n" + markdown_report(v)
    source = str(options.get("iac_source") or "")
    props = proposals(v, files, source)
    if not props:
        return {"audit": audit, "report": report, "actions": []}
    action = "create_branch_with_patch" if repo_prefix(source) is not None else "save_proposal"
    patch = "".join(p["patch"] for p in props)
    report += f"\n\n### 재검사를 통과한 수정안 ({len(props)}개 파일 · {action} 승인 대기)\n\n```diff\n{patch}```"
    tools = iac.available_scanners()
    if tools:  # 외부 스캐너가 있으면 수정 전후를 한 번 더 (참고 — 판정은 내장 규칙)
        before_ext, notes = iac.run_external(files, tools)
        after_ext, notes2 = iac.run_external(verified_patch(v, files)[1], tools)
        audit["iac_external"] = {"tools": tools, "before": len(before_ext), "after": len(after_ext), "notes": (notes + notes2)[:4]}
    actions = [{"action": action, "summary": f"{p['title']} · 재검사 통과 · {p['rescan']['log']}"[:200], "payload": p} for p in props]
    audit["iac_proposals"] = len(actions)
    return {"audit": audit, "report": report, "actions": actions}

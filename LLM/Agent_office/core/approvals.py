"""사람 승인 처리 — 자율 실행이 남긴 승인 요청을 사람이 승인·기각하고, 안전한 행동만 실제로 실행한다.

설계 문서: docs/AI_전용_업무_설계.md 5.5장

흐름 (모두 같은 해시 체인 감사 로그 audit.jsonl 에 새 줄로 이어 쓴다 — 앞 줄은 고치거나 지우지 않는다)
  approval_requested  자율 실행이 허용 밖 행동을 실행하지 않고 남긴 요청 (core/autonomy.request_approval)
  approval_granted    사람이 승인 (request_id, decider, comment, via)
    → action_executed      실행기가 있는 행동: 실행 결과 (만든 파일·브랜치·커밋)
    → action_failed        실행기가 있는 행동: 실패 (error)
    → action_not_executed  실행기가 없는 행동: 승인 기록만 남기고 실행하지 않음 + 사람이 직접 할 명령 (manual_steps)
  approval_rejected   사람이 기각 (실행 없음)
  approval_expired    expiry_days 동안 결정이 없어 만료 (실행 없음)

대기열 = 요청 - (승인·기각·만료된 요청). 같은 요청을 두 번 승인해도 처음 결정을 돌려줄 뿐 다시 실행하지 않는다.
감사 로그가 어긋나 있으면(verify 실패) 어떤 결정도 받지 않는다 — 무엇을 승인하는지 믿을 수 없으므로.

실행기 (행동 이름 → 함수, 기본 거부: 여기 없는 행동은 승인해도 실행하지 않는다)
  save_proposal             제안(패치·새 파일 내용)을 data/autonomy/proposals/<id>/ 에 저장 + 손으로 적용하는 법 (로컬, 지우면 끝)
  create_branch_with_patch  위 저장 + 새 git 작업 폴더(worktree)·새 로컬 브랜치 autonomy/<id> 에 패치를 적용해 로컬 커밋.
                            push 하지 않는다. 사용자가 쓰는 checkout(작업 폴더)에는 손대지 않는다 (브랜치를 지우면 끝)
  push·merge·삭제·외부 전송·설정 적용·배포 — 실행기를 만들지 않는다. 승인은 '사람이 직접 하겠다'는 기록이고,
  화면은 그대로 따라 할 명령만 보여 준다. 되돌릴 수 없거나 밖으로 나가는 행동이라 마지막 손은 사람이 쓴다.

payload 스키마 (다른 팀이 만드는 쪽) — Proposal (아래). 🏗️ 인프라 보안 팀의 IaC 수정 패치가 이 모양으로 온다.
"""

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from core.autonomy import IRREVERSIBLE_ACTIONS, ApprovalsConfig, AuditLog, payload_path

BASE_DIR = Path(__file__).resolve().parent.parent  # LLM/Agent_office
DECIDER = "user"  # 1인용 앱이라 로그인 없음 — 결정한 사람은 늘 '사용자'
DECISION_EVENTS = {"approval_granted": "granted", "approval_rejected": "rejected", "approval_expired": "expired"}
OUTCOME_EVENTS = ("action_executed", "action_failed", "action_not_executed")


# ─── payload 스키마 (save_proposal · create_branch_with_patch) ─────────────────


class RescanResult(BaseModel):
    """패치를 적용한 뒤 같은 스캐너로 다시 검사한 결과 — 코드로 확인한 것만 적는다."""

    scanner: str  # 예: "checkov 3.2.1", "kube-linter 0.6"
    passed: bool  # 고치려던 탐지가 패치 뒤 모두 사라졌는가
    before: list[str] = Field(default_factory=list)  # 패치 전 탐지 규칙 id (예: CKV_AWS_24)
    after: list[str] = Field(default_factory=list)  # 패치 뒤에도 남은 탐지 규칙 id
    log: str = ""  # 짧은 요약 (원문은 2,000자 안으로 잘라서)


class Proposal(BaseModel):
    """바꿀 내용 제안 — 파일 하나에 대한 패치(또는 새 내용)와 근거.

    target_file: 저장소 루트 기준 '/' 경로 (예: LLM/Agent_office/teams/infra_security/samples/main.tf).
                 절대 경로·'..'·.git 아래는 받지 않는다.
    patch      : unified diff (git diff 모양, a/ b/ 접두). target_file 한 파일만 바꿔야 한다.
                 create_branch_with_patch 는 patch 가 꼭 있어야 한다.
    new_content: 패치 대신 새 파일 전체 내용 (save_proposal 만 — 사람이 보고 직접 옮긴다)
    """

    schema_version: Literal[1] = 1
    title: str  # 한 줄 제목 — 커밋 메시지 첫 줄이 된다
    target_file: str
    patch: str = ""
    new_content: str = ""
    rationale: str = ""  # 왜 바꾸는지 (탐지 설명, 근거 통제 등)
    findings: list[str] = Field(default_factory=list)  # 고치려는 탐지 (예: "CKV_AWS_24: SSH 0.0.0.0/0 허용")
    rescan: RescanResult | None = None
    base_commit: str = ""  # 패치를 만든 기준 커밋 (비우면 저장소 HEAD)

    @model_validator(mode="after")
    def _check(self):
        safe_relpath(self.target_file)
        if not self.patch.strip() and not self.new_content:
            raise ValueError("patch 나 new_content 중 하나는 있어야 합니다")
        if self.patch.strip():
            touched = patch_paths(self.patch)
            if not touched:
                raise ValueError("patch 에서 바꾸는 파일을 찾지 못했습니다 (unified diff 의 ---/+++ 줄)")
            if touched != {self.target_file}:
                raise ValueError(f"patch 는 target_file({self.target_file}) 한 파일만 바꿔야 합니다: {sorted(touched)}")
        return self


def safe_relpath(path: str) -> str:
    """저장소 안의 상대 경로만 통과 — 절대 경로·'..'·드라이브 문자·역슬래시·.git 아래면 ValueError."""
    if not path or "\\" in path or ":" in path or path.startswith("/"):
        raise ValueError(f"저장소 루트 기준 '/' 상대 경로여야 합니다: {path!r}")
    parts = PurePosixPath(path).parts
    if any(p in ("..", ".") for p in parts) or parts[0] == ".git":
        raise ValueError(f"허용하지 않는 경로입니다: {path!r}")
    return path


def patch_paths(patch: str) -> set[str]:
    """unified diff 가 건드리는 파일 경로들 (a/ b/ 접두를 떼고, /dev/null 은 뺀다). 경로가 안전하지 않으면 ValueError.

    hunk(@@) 안의 줄은 건너뛴다 — 지운 줄 '-- 주석' 이 '--- ' 머리줄처럼 보이는 것을 파일로 잘못 세지 않게.
    심볼릭 링크(mode 120000)를 만드는 패치는 받지 않는다.
    """
    lines, out, i = patch.splitlines(), set(), 0
    while i < len(lines):
        line = lines[i]
        hunk = re.match(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
        if hunk:
            old, new = int(hunk.group(1) or 1), int(hunk.group(2) or 1)
            i += 1
            while i < len(lines) and (old > 0 or new > 0):
                tag = lines[i][:1]
                if tag != "\\":
                    old -= tag != "+"
                    new -= tag != "-"
                i += 1
            continue
        if re.match(r"^(?:new file |deleted file |old |new )?mode 120000", line):
            raise ValueError("심볼릭 링크를 만드는 패치는 받지 않습니다")
        head = re.match(r"^(?:---|\+\+\+) (\S+)", line)
        if line.startswith("diff --git "):
            both = re.match(r"^diff --git a/(\S+) b/(\S+)$", line)
            if not both:
                raise ValueError(f"읽을 수 없는 diff 머리줄입니다 (따옴표·공백 경로는 받지 않음): {line[:120]}")
            paths = list(both.groups())
        else:
            paths = [head.group(1)] if head else []
        for raw in paths:
            if raw == "/dev/null":
                continue
            out.add(safe_relpath(raw[2:] if head and raw[:2] in ("a/", "b/") else raw))
        i += 1
    return out


# ─── 대기열·이력 읽기 ─────────────────────────────────────────────────────


def request_id(rec: dict) -> str:
    """요청 id — 새 요청은 request_id, 예전(id 없이 남은) 요청은 그 줄 해시 앞 12자리."""
    return rec.get("request_id") or rec["hash"][:12]


def _index(entries: list[dict]) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """(요청 id → 요청 줄 + '_line'(0부터), id → 결정 줄, id → 실행 결과 줄)."""
    reqs, decisions, outcomes = {}, {}, {}
    for i, rec in enumerate(entries):
        ev = rec["event"]
        if ev == "approval_requested":
            reqs.setdefault(request_id(rec), {**rec, "id": request_id(rec), "_line": i})
        elif ev in DECISION_EVENTS:
            decisions.setdefault(rec["request_id"], rec)  # 처음 결정만 유효 (뒤에 또 있어도 무시)
        elif ev in OUTCOME_EVENTS:
            outcomes.setdefault(rec["request_id"], rec)
    return reqs, decisions, outcomes


def pending(audit: AuditLog, entries: list[dict] | None = None) -> list[dict]:
    """아직 결정이 없는 요청 (최근 것이 앞). 줄마다 "id" 가 붙는다."""
    reqs, decisions, _ = _index(audit.entries() if entries is None else entries)
    return [r for r in reversed(reqs.values()) if r["id"] not in decisions]


def history(audit: AuditLog, entries: list[dict] | None = None) -> list[dict]:
    """결정된 요청 (최근 결정이 앞) — {id, action, job, summary, requested, decision, decided, comment, via, outcome}.

    decision: granted·rejected·expired / outcome: action_executed·action_failed·action_not_executed 줄 (승인만, 없으면 None)
    """
    reqs, decisions, outcomes = _index(audit.entries() if entries is None else entries)
    out = []
    for rid, dec in decisions.items():
        req = reqs.get(rid, {})
        out.append({"id": rid, "action": req.get("action", dec.get("action", "")), "job": req.get("job", ""),
                    "summary": req.get("summary", ""), "requested": req.get("ts"), "decision": DECISION_EVENTS[dec["event"]],
                    "decided": dec["ts"], "comment": dec.get("comment", ""), "via": dec.get("via", ""),
                    "outcome": outcomes.get(rid)})
    return sorted(out, key=lambda h: h["decided"], reverse=True)


def load_payload(audit: AuditLog, req: dict) -> tuple[dict | None, str]:
    """(payload, 문제) — 파일이 없거나 요청 때 적은 sha256 과 다르면 payload 대신 문제를 돌려준다."""
    if req.get("payload_error"):
        return None, req["payload_error"]
    if not req.get("payload_sha256"):
        return None, ""  # 내용 없이 요약만 있는 요청
    path = payload_path(audit, request_id(req))
    if not path.exists():
        return None, f"내용 파일이 없습니다: {path}"
    body = path.read_text(encoding="utf-8")
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != req["payload_sha256"]:
        return None, "내용 파일이 요청 뒤에 바뀌었습니다 (sha256 불일치) — 실행하지 않습니다"
    return json.loads(body), ""


def costs(entries: list[dict], req: dict) -> tuple[float | None, float]:
    """(요청을 낸 실행 1회의 추정 비용 — 그 실행이 아직 안 끝났으면 None, 그 작업의 지금까지 자율 실행 비용 합계)."""
    job = req.get("job")
    run_cost = None
    for rec in entries[req["_line"] + 1:]:
        if rec.get("job") == job and rec["event"] in ("run_end", "run_error"):
            run_cost = float(rec.get("cost_usd") or 0)
            break
    total = sum(float(rec.get("cost_usd") or 0) for rec in entries
                if rec.get("job") == job and rec["event"] in ("run_end", "run_error"))
    return run_cost, total


def why(req: dict) -> str:
    """왜 사람 승인이 필요한지 — 기본 거부 게이트의 이유를 사람 말로."""
    action, level = req.get("action", ""), req.get("level")
    if action in IRREVERSIBLE_ACTIONS:
        return f"'{action}' 은 되돌릴 수 없거나 밖으로 나가는 행동이라 어느 자율 수준에서도 스스로 하지 않습니다."
    if action in EXECUTORS:
        return (f"자율 수준 {level or '(기록 없음)'} 에서는 읽기·보고서(·제안 저장)만 스스로 합니다. "
                f"'{action}' 은 저장소나 파일을 바꾸므로 사람 승인이 필요합니다.")
    return f"'{action}' 은 허용 목록에 없는 행동입니다 (기본 거부) — 승인해도 실행기가 없어 기록만 남습니다."


def on_approve(action: str) -> str:
    """승인하면 무슨 일이 일어나는지 한 줄."""
    ex = EXECUTORS.get(action)
    return ex.describe if ex else "실행하지 않습니다 — 승인 기록만 남기고, 사람이 직접 할 명령을 보여 줍니다."


# ─── 사람이 직접 할 명령 (실행기가 없는 행동 · 실행 뒤 다음 단계) ─────────────────

_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,120}$")  # 브랜치·원격 이름으로 명령에 넣어도 되는 모양만


def _ref(value, default: str) -> str:
    value = str(value or "").strip()
    return value if _REF.match(value) else default


def manual_steps(req: dict, payload: dict | None, result: dict | None = None) -> list[str]:
    """사람이 직접 따라 할 명령·단계. 모델이 낸 문자열을 명령에 그대로 넣지 않는다 — 이름 모양을 검사한 값만, 아니면 <자리 표시>."""
    action = req.get("action", "")
    p = payload or {}
    if result and result.get("branch"):  # create_branch_with_patch 가 만든 브랜치 — 검토 뒤 올리는 것은 사람
        repo, branch = result["repo"], result["branch"]
        return [f'git -C "{repo}" show --stat {branch}', f'git -C "{repo}" push origin {branch}',
                "GitHub 에서 PR 을 열고 리뷰한 뒤 merge (자동으로 하지 않음)",
                f'다 쓴 작업 폴더 정리: git -C "{repo}" worktree remove "{result["worktree"]}"']
    if result and result.get("export_dir"):
        d = result["export_dir"]
        steps = [f"저장한 제안 보기: {d}"]
        if result.get("patch_file"):
            steps += [f'git apply --check "{result["patch_file"]}"   (적용할 저장소 루트에서)',
                      f'git apply "{result["patch_file"]}"']
        return steps
    branch = _ref(p.get("branch"), "<브랜치>")
    if action == "git_push":
        return [f"git push {_ref(p.get('remote'), 'origin')} {branch}"]
    if action == "merge":
        return [f"git switch {_ref(p.get('base'), 'main')}", f"git merge --no-ff {branch}", "테스트를 돌린 뒤 git push"]
    if action == "delete":
        return [f"지울 대상을 직접 확인한 뒤 지우세요: {p.get('path') or req.get('summary', '')}"]
    if action == "send_external":
        return ["보낼 내용과 받는 곳을 직접 확인한 뒤, 평소 쓰는 도구로 직접 보내세요 (앱이 대신 보내지 않음)"]
    if action == "apply_config":
        return ["제안 내용을 운영 설정에 직접 반영하세요 — 먼저 같은 내용을 save_proposal 로 저장해 diff 를 검토하는 것을 권합니다",
                f"대상: {p.get('target_file') or req.get('summary', '')}"]
    if action == "deploy":
        return ["평소 배포 절차(리뷰·테스트·배포 도구)를 사람이 직접 진행하세요"]
    return [f"'{action}' 은 실행기가 없는 행동입니다 — 요청 내용을 보고 필요하면 직접 하세요"]


# ─── 실행기 ───────────────────────────────────────────────────────────────


@dataclass
class ExecContext:
    audit: AuditLog
    cfg: ApprovalsConfig
    request: dict
    payload: dict | None

    @property
    def rid(self) -> str:
        return request_id(self.request)

    @property
    def autonomy_dir(self) -> Path:
        return self.audit.path.parent  # data/autonomy


@dataclass
class Executor:
    describe: str  # 승인하면 일어나는 일 (화면에 그대로)
    run: Callable[[ExecContext], dict]  # → action_executed 줄에 넣을 결과. 실패는 예외로


def _proposal(ctx: ExecContext) -> Proposal:
    if ctx.payload is None:
        raise ValueError("실행할 내용(payload)이 없습니다")
    try:
        return Proposal(**ctx.payload)
    except ValidationError as e:
        raise ValueError(f"payload 가 Proposal 모양이 아닙니다: {e.errors()[0]['msg']}") from None


def save_proposal(ctx: ExecContext) -> dict:
    """data/autonomy/proposals/<id>/ 에 제안을 저장 — 이미 같은 내용이 있으면 다시 쓰지 않는다 (두 번 실행해도 같음)."""
    prop = _proposal(ctx)
    out = ctx.autonomy_dir / "proposals" / ctx.rid
    files: dict[str, str] = {"payload.json": json.dumps(ctx.payload, ensure_ascii=False, sort_keys=True, indent=1)}
    if prop.patch.strip():
        files["proposal.patch"] = prop.patch if prop.patch.endswith("\n") else prop.patch + "\n"
    if prop.new_content:
        files["proposed/" + prop.target_file] = prop.new_content
    rescan = (f"{prop.rescan.scanner} · {'통과' if prop.rescan.passed else '실패'} · 전 {prop.rescan.before} → 후 {prop.rescan.after}"
              if prop.rescan else "없음")
    files["README.md"] = "\n".join([
        f"# {prop.title}", "", f"- 요청 id: {ctx.rid} · 작업: {ctx.request.get('job', '')} · 행동: {ctx.request.get('action', '')}",
        f"- 대상 파일: `{prop.target_file}`", f"- 재검사: {rescan}", "", "## 근거", "", prop.rationale or "-", "",
        *(["## 고치려는 탐지", "", *[f"- {f}" for f in prop.findings], ""] if prop.findings else []),
        "## 손으로 적용하기 (자동으로 적용하지 않음)", "",
        "저장소 루트에서 `git apply --check proposal.patch` 로 먼저 확인한 뒤 `git apply proposal.patch`."
        if "proposal.patch" in files else f"`proposed/{prop.target_file}` 를 보고 대상 파일에 직접 옮기세요.", "",
    ])
    reused = out.exists() and all((out / name).exists() and (out / name).read_text(encoding="utf-8") == body
                                  for name, body in files.items())
    if not reused:
        for name, body in files.items():
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8", newline="\n")
    result = {"export_dir": str(out), "files": sorted(files), "reused": reused}
    if "proposal.patch" in files:
        result["patch_file"] = str(out / "proposal.patch")
    return result


def _git(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    # core.autocrlf=false — 패치(LF)를 바꾸지 않은 채 적용·커밋하려고. 사용자 git 설정은 건드리지 않는다 (-c 는 이번 명령만)
    proc = subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} 실패: {(proc.stderr or proc.stdout).strip()[:400]}")
    return proc


def _repo_root(cfg: ApprovalsConfig) -> Path:
    start = Path(cfg.patch_repo).expanduser() if cfg.patch_repo else BASE_DIR
    return Path(_git(["rev-parse", "--show-toplevel"], cwd=start).stdout.strip()).resolve()


def worktree_parent(cfg: ApprovalsConfig, repo: Path) -> Path:
    """브랜치를 꺼낼 새 작업 폴더의 부모 — 기본은 저장소 폴더 '옆' (<저장소 이름>-autonomy-worktrees).

    저장소 안(예: data/)에 두면 pytest 가 사본의 테스트까지 모으고, 사용자가 쓰는 checkout 안에 파일이 생긴다.
    """
    return Path(cfg.worktree_dir).expanduser().resolve() if cfg.worktree_dir else repo.parent / f"{repo.name}-autonomy-worktrees"


def create_branch_with_patch(ctx: ExecContext) -> dict:
    """제안 저장 + 새 작업 폴더(git worktree)·새 브랜치 autonomy/<id> 에 패치를 적용해 로컬 커밋. push 하지 않는다.

    사용자가 쓰는 checkout 의 파일·브랜치는 바꾸지 않는다 — 새 폴더에 새 브랜치를 꺼내 거기서만 적용한다.
    이미 그 브랜치가 있으면(두 번째 승인·재실행) 다시 만들지 않고 있던 커밋을 돌려준다.
    """
    prop = _proposal(ctx)
    if not prop.patch.strip():
        raise ValueError("create_branch_with_patch 는 patch 가 있어야 합니다 (새 내용만 있으면 save_proposal)")
    export = save_proposal(ctx)
    repo = _repo_root(ctx.cfg)
    branch = f"autonomy/{ctx.rid}"
    wt = worktree_parent(ctx.cfg, repo) / ctx.rid
    base = {"repo": str(repo), "branch": branch, "worktree": str(wt), "export_dir": export["export_dir"],
            "patch_file": export["patch_file"], "target_file": prop.target_file}
    existing = _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=repo, check=False)
    if existing.returncode == 0:
        return {**base, "commit": existing.stdout.strip(), "reused": True}
    if wt == repo or repo in wt.parents:
        raise RuntimeError(f"작업 폴더는 저장소 checkout 밖이어야 합니다 (worktree_dir 확인): {wt}")
    if wt.exists():
        raise RuntimeError(f"작업 폴더가 이미 있습니다 (다른 것일 수 있어 건드리지 않음): {wt}")
    start = prop.base_commit or "HEAD"
    _git(["rev-parse", "--verify", "--quiet", f"{start}^{{commit}}"], cwd=repo)
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git(["worktree", "add", "-b", branch, str(wt), start], cwd=repo)
    try:
        _git(["apply", "--check", export["patch_file"]], cwd=wt)
        _git(["apply", "--index", export["patch_file"]], cwd=wt)
        msg = (f"autonomy: {prop.title}\n\n승인 요청 {ctx.rid} · 작업 {ctx.request.get('job', '')}\n"
               f"승인: {DECIDER} (Agent Office 승인 대기열) · push 는 사람이 직접\n")
        _git(["-c", "user.name=Agent Office (자율 실행)", "-c", "user.email=autonomy@agent-office.invalid",
              "commit", "-q", "-m", msg], cwd=wt)
    except Exception:
        # 이 실행기가 방금 만든 것만 되돌린다 (빈 브랜치·작업 폴더가 남아 '이미 있음'으로 보이지 않게)
        _git(["worktree", "remove", "--force", str(wt)], cwd=repo, check=False)
        _git(["branch", "-D", branch], cwd=repo, check=False)
        raise
    commit = _git(["rev-parse", "HEAD"], cwd=wt).stdout.strip()
    return {**base, "commit": commit, "reused": False}


EXECUTORS: dict[str, Executor] = {
    "save_proposal": Executor("제안(패치·새 내용)을 data/autonomy/proposals/<id>/ 에 저장합니다 — 로컬 파일만, 적용·push 없음.",
                              save_proposal),
    "create_branch_with_patch": Executor(
        "제안을 저장하고, 새 작업 폴더에 새 로컬 브랜치 autonomy/<id> 를 만들어 패치를 적용·커밋합니다. "
        "push·merge 는 하지 않고, 지금 쓰는 작업 폴더는 건드리지 않습니다.", create_branch_with_patch),
}


# ─── 결정 ─────────────────────────────────────────────────────────────────


class ApprovalError(Exception):
    """결정을 받을 수 없음 (요청 없음 · 감사 로그 어긋남)."""


def expire_stale(audit: AuditLog, cfg: ApprovalsConfig, now: float | None = None) -> list[str]:
    """expiry_days 가 지난 대기 요청을 approval_expired 로 닫는다 (실행 없음). 닫은 id 목록. 로그가 어긋나 있으면 아무것도 안 한다."""
    now = time.time() if now is None else now
    limit = cfg.expiry_days * 86400
    with audit.locked():
        if not audit.verify()[0]:
            return []
        stale = [r for r in pending(audit) if now - r["ts"] >= limit]
        for r in reversed(stale):  # 오래된 것부터
            audit.append("approval_expired", ts=now, request_id=r["id"], job=r.get("job", ""), action=r.get("action", ""),
                         decider="system", age_days=round((now - r["ts"]) / 86400, 1))
    return [r["id"] for r in stale]


def decide(audit: AuditLog, cfg: ApprovalsConfig, rid: str, decision: Literal["grant", "reject"], comment: str = "",
           via: str = "ui", now: float | None = None) -> dict:
    """사람의 결정을 감사 로그에 남기고, 승인이면 실행기가 있는 행동만 실행한다.

    돌려주는 값: {"status": granted·rejected·already·expired, "decision": 결정 줄, "outcome": 실행 결과 줄 (승인만)}
    이미 결정된 요청이면 새 줄 없이 처음 결정을 돌려준다 (두 번 눌러도 두 번 실행되지 않음).
    """
    now = time.time() if now is None else now
    comment = str(comment or "").strip()[:500]
    with audit.locked():  # 결정 줄을 쓰는 동안만 잠근다 — 결정 줄이 '이 요청은 내가 처리 중' 표시가 된다
        ok, bad = audit.verify()
        if not ok:
            raise ApprovalError(f"감사 로그가 {bad}번째 줄부터 어긋나 결정을 받지 않습니다 — 로그를 먼저 확인하세요")
        entries = audit.entries()
        reqs, decisions, outcomes = _index(entries)
        req = reqs.get(rid)
        if req is None:
            raise ApprovalError(f"승인 요청 {rid} 이(가) 없습니다")
        if rid in decisions:
            return {"status": "already", "decision": decisions[rid], "outcome": outcomes.get(rid)}
        if now - req["ts"] >= cfg.expiry_days * 86400:
            rec = audit.append("approval_expired", ts=now, request_id=rid, job=req.get("job", ""), action=req.get("action", ""),
                               decider="system", age_days=round((now - req["ts"]) / 86400, 1))
            return {"status": "expired", "decision": rec, "outcome": None}
        event = "approval_granted" if decision == "grant" else "approval_rejected"
        dec = audit.append(event, ts=now, request_id=rid, job=req.get("job", ""), action=req.get("action", ""),
                           decider=DECIDER, via=via, comment=comment)
    if decision != "grant":
        return {"status": "rejected", "decision": dec, "outcome": None}

    payload, problem = load_payload(audit, req)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", rid):  # id 가 폴더·브랜치 이름에 들어가므로
        problem = problem or f"요청 id 모양이 이상합니다: {rid!r}"
    executor = EXECUTORS.get(req.get("action", ""))
    fields = {"request_id": rid, "job": req.get("job", ""), "action": req.get("action", "")}
    if executor is None:
        out = audit.append("action_not_executed", **fields, reason="실행기가 없는 행동 — 승인 기록만 남김 (사람이 직접)",
                           manual_steps=manual_steps(req, payload))
    elif problem:
        out = audit.append("action_failed", **fields, error=problem)
    else:
        try:
            result = executor.run(ExecContext(audit, cfg, req, payload))
            out = audit.append("action_executed", **fields, result=result, manual_steps=manual_steps(req, payload, result))
        except Exception as e:  # 실패도 결정의 결과로 남긴다 (다시 승인해도 재실행하지 않음 — 새 요청으로)
            out = audit.append("action_failed", **fields, error=f"{type(e).__name__}: {e}"[:600])
    return {"status": "granted", "decision": dec, "outcome": out}

"""🛎 인포메이션 센터 — 질문을 받아 '어느 부서 일인지' 부서장 회의로 정하고, 없으면 새 부서를 제안한다 (화면 그리기 없음).

흐름
  1) meeting()        : 모든 부서장의 판단을 모은다 → {dept_id(또는 ""), verdicts[부서마다 할 수 있음/없음·이유], how, model}
     - 비용: 부서장마다 1회씩(부서 6개 → 6회) 부르지 않고, **보조(부장) 모델 1회 구조화 출력**으로 모든 부서를 한꺼번에 판단한다.
       프롬프트에 부서마다 부장 이름·관점(persona)·소속 팀·다루는/다루지 않는 질문을 넣어 '부서장 회의'를 흉내 낸다.
       → 호출 1회·대기 1번이고, 부서끼리 비교해 가장 알맞은 곳을 고를 수 있다 (따로 부르면 서로를 모른다).
     - 모델을 못 쓰면 낱말 비교: 팀 scope.keywords 가 가장 많이 맞는 부서. 아무 낱말도 안 맞으면 범용 기본 팀(accept_all)의 부서.
  2) 부서가 정해지면 → 그 부서장이 팀을 고른다 (core/building.route_request — 2번 흐름, building_ui 부서 데스크)
  3) 어느 부서도 못 하면 → propose() : 새 부서·팀 초안 (보조 모델 1회, 실패하면 질문으로 만든 기본 초안)
     → 사람이 [만들기]를 누르면 core/dept_builder.py 가 파일 초안을 만들고, 미리 보기를 확인해야 저장
     → [나중에]는 LaterStore (data/new_dept_later.json) 에 남긴다
"""

import json
import re
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from core.schemas import Team

LATER_FILE = "new_dept_later.json"
MIN_FIT = 5  # 이 점수(0~10) 미만이면 '할 수 있음'이라도 맡기지 않는다 — 억지로 끼워 맞추지 않게


# ─── 구조화 출력 스키마 ───────────────────────────────────────────────


class DeptVerdict(BaseModel):
    """부서장 한 명의 판단."""

    dept_id: str = Field(description="부서 id (목록에 있는 id 그대로)")
    can_handle: bool = Field(description="우리 부서 팀들이 이 일을 맡을 수 있는가")
    fit: int = Field(description="얼마나 잘 맞는지 0~10 (10 = 바로 우리 부서 전문 분야)")
    reason: str = Field(description="부서장이 하는 한 문장 판단")


class MeetingResult(BaseModel):
    """🛎 부서장 회의 결과 — 모든 부서를 한꺼번에."""

    analysis: str = Field(description="먼저 요청의 목적과 필요한 전문 분야·자료·도구를 1~2문장으로 분석")
    verdicts: list[DeptVerdict] = Field(description="부서 목록의 모든 부서마다 하나씩")
    best_dept_id: str = Field(description="맡길 부서 id. 어느 부서도 맡을 수 없으면 빈 문자열")


class ProposedMember(BaseModel):
    id: str = Field(description="영문 소문자 id (예: lead, analyst)")
    rank: str = Field(description="직급 id — team_lead / gm / deputy_gm / assistant_manager / staff 중 하나")
    name: str = Field(description="화면 이름 (예: 차장·세법 리서치)")
    specialty: str = Field(description="전문 분야 한 줄")


class NewDeptProposal(BaseModel):
    """어느 부서도 못 하는 일 → 새 부서·팀 제안."""

    analysis: str = Field(description="이 일에 필요한 전문 분야·자료·도구를 먼저 1~2문장으로 분석")
    dept_id: str = Field(description="새 부서 영문 id (소문자·밑줄, 예: legal)")
    dept_name: str = Field(description="새 부서 이름 (예: 법무부)")
    emoji: str = Field(description="부서 이모지 하나")
    dept_description: str = Field(description="부서가 하는 일 한 문장")
    head_persona: str = Field(description="부서장이 결과를 비평할 때의 관점 한 문장")
    team_id: str = Field(description="첫 팀 영문 id (소문자·밑줄, 예: contract_review)")
    team_name: str = Field(description="첫 팀 이름 (예: ⚖️ 계약 검토팀)")
    team_description: str = Field(description="팀이 하는 일 한 문장")
    scope_summary: str = Field(description="팀이 다루는 질문 한 줄")
    accepts: list[str] = Field(description="다루는 질문 2~4개")
    rejects: list[str] = Field(description="다루지 않는 질문 1~3개")
    keywords: list[str] = Field(description="이 팀 질문을 알아볼 낱말 5~12개")
    members: list[ProposedMember] = Field(description="조직도 — 첫 줄이 팀장(team_lead), 나머지는 팀장 아래 팀원 2~5명")
    tools_needed: list[str] = Field(description="이 일에 필요한 도구·자료 (사람이 준비할 것 포함, 예: 판례 검색 API)")
    review_criteria: list[str] = Field(description="결과를 검토할 때 꼭 볼 기준 2~4개")


# ─── 부서 소개 (회의 프롬프트·낱말 비교용) ─────────────────────────────


def dept_brief(dept, teams: dict[str, Team]) -> str:
    """'- travel (✈️ 여행부, 부장 관점: …): 소개 / 팀: 국내 여행팀(다루는 질문: …; 다루지 않음: …), …'"""
    from core.dept_heads import head_of

    head = head_of(dept)
    parts = []
    for tid in dept.teams:
        t = teams[tid]
        scope = t.scope
        if scope and scope.accept_all:
            scope_text = "범용 창구 — 일반 지식으로 끝나는 조사·정리·비교·짧은 코드·글 초안만"
        elif scope:
            scope_text = scope.summary + (f"; 다루지 않음: {', '.join(scope.rejects)}" if scope.rejects else "")
        else:
            scope_text = t.description
        tasks = ", ".join(x.title for x in t.tasks[:4])
        parts.append(f"{t.name}(다루는 질문: {scope_text}" + (f"; 업무: {tasks}" if tasks else "") + ")")
    return (f"- {dept.id} ({dept.emoji} {dept.name}, {head.name} 관점: {head.persona})\n"
            f"  소개: {dept.description}\n  팀: " + " / ".join(parts))


def meeting_messages(question: str, depts: list, teams: dict[str, Team]):
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "당신은 회사 1층 🛎 인포메이션 센터의 진행자입니다. 아래 부서장들이 모인 '부서장 회의'를 대신 진행합니다.\n"
        "각 부서장의 관점에서 '이 요청이 우리 부서 팀들이 맡을 일인가'를 부서마다 짧게 판단하세요. 요청을 직접 처리하지 마세요.\n"
        "판단 규칙:\n"
        "- 팀의 '다루는 질문'·'업무'에 맞으면 할 수 있음. 전문 팀이 있는 일은 그 팀 부서의 fit 을 가장 높게.\n"
        "- 범용 창구 팀은 일반 지식으로 끝나는 가벼운 조사·정리만 맡는다. 꾸준히 반복될 전문 업무(전용 자료·도구·자격이 필요한 일,"
        " 예: 세무 신고서 작성, 계약서 법률 검토, 의료 판단, 회사 내부 시스템 운영)는 범용 창구로 맡기지 말고 할 수 없음으로.\n"
        f"- 할 수 있는 부서가 없거나 가장 높은 fit 이 {MIN_FIT} 미만이면 best_dept_id 를 빈 문자열로.\n\n"
        "[부서 목록]\n" + "\n".join(dept_brief(d, teams) for d in depts)
    )
    return [SystemMessage(content=system), HumanMessage(content=question)]


# ─── 1) 부서장 회의 ─────────────────────────────────────────────────


def _keyword_hits(question: str, team: Team) -> int:
    from core.scope import keyword_hits

    return len(keyword_hits(question, team))


def keyword_meeting(question: str, depts: list, teams: dict[str, Team]) -> dict:
    """모델 없이 — 부서마다 소속 팀 scope.keywords 가 맞은 개수. 가장 많은 부서, 없으면 범용 기본 팀의 부서."""
    verdicts = []
    for d in depts:
        hits = sum(_keyword_hits(question, teams[t]) for t in d.teams if teams[t].scope and not teams[t].scope.accept_all)
        verdicts.append({"dept_id": d.id, "can_handle": hits > 0, "fit": min(10, hits * 3),
                         "reason": f"팀 분야 낱말 {hits}개가 맞습니다." if hits else "맞는 분야 낱말이 없습니다."})
    best = max(verdicts, key=lambda v: v["fit"], default=None)
    if best and best["fit"] > 0:
        return {"dept_id": best["dept_id"], "verdicts": verdicts, "analysis": ""}
    general = next((d for d in depts for t in d.teams if teams[t].scope and teams[t].scope.accept_all), None)
    if general:
        for v in verdicts:
            if v["dept_id"] == general.id:
                v.update(can_handle=True, fit=MIN_FIT, reason="분명한 전문 부서가 없어 범용 기본 팀이 받습니다.")
        return {"dept_id": general.id, "verdicts": verdicts, "analysis": ""}
    return {"dept_id": "", "verdicts": verdicts, "analysis": ""}


def pick_best(result: MeetingResult, dept_ids: list[str]) -> str:
    """모델이 고른 부서를 그대로 믿지 않고 확인 — 목록에 있고 할 수 있음·fit ≥ MIN_FIT 인 부서 중 fit 이 가장 높은 곳."""
    ok = [v for v in result.verdicts if v.dept_id in dept_ids and v.can_handle and v.fit >= MIN_FIT]
    if not ok:
        return ""
    top = max(v.fit for v in ok)
    preferred = next((v.dept_id for v in ok if v.dept_id == result.best_dept_id and v.fit == top), None)
    return preferred or next(v.dept_id for v in ok if v.fit == top)


def meeting(question: str, depts: list, teams: list[Team], models, choice: str = "") -> dict:
    """🛎 부서장 회의 — {dept_id(""=맡을 부서 없음), verdicts, analysis, how: llm|keyword, model, note}.

    choice = 인포메이션 센터가 쓸 모델 ("provider:model", 없으면 models.yaml utility — core/dept_heads.assist_chain)."""
    by_id = {t.id: t for t in teams}
    ids = [d.id for d in depts]
    from core import dept_heads, llm_factory, usage  # 무거운 import 는 실제로 부를 때만

    note = ""
    chain = dept_heads.assist_chain(models, choice)  # 가벼운 안내 — 부장 기본(Opus 구독)이 아니라 utility
    try:
        with usage.scope(purpose="인포메이션 센터"):
            call = llm_factory.invoke_structured(chain, models, MeetingResult, meeting_messages(question, depts, by_id),
                                                 "🛎 인포메이션 센터")
        result: MeetingResult = call.parsed
        got = {v.dept_id for v in result.verdicts}
        verdicts = [v.model_dump() for v in result.verdicts if v.dept_id in ids]
        verdicts += [{"dept_id": i, "can_handle": False, "fit": 0, "reason": "(판단 없음)"} for i in ids if i not in got]
        verdicts.sort(key=lambda v: ids.index(v["dept_id"]))
        return {"dept_id": pick_best(result, ids), "verdicts": verdicts, "analysis": result.analysis, "how": "llm",
                "model": call.spec.label, "note": ""}
    except Exception as e:  # noqa: BLE001 — 키 없음·한도 초과: 낱말 비교로라도 안내는 끝낸다
        note = f"모델을 쓰지 못해 낱말 비교로 정했습니다 ({type(e).__name__})."
    out = keyword_meeting(question, depts, by_id)
    return {**out, "how": "keyword", "model": "", "note": note}


# ─── 3) 새 부서 제안 ────────────────────────────────────────────────


def slug(text: str, fallback: str) -> str:
    """영문 소문자·숫자·밑줄 id (한글만 있으면 fallback)."""
    s = re.sub(r"[^a-z0-9_]+", "_", (text or "").lower()).strip("_")
    s = re.sub(r"_+", "_", s)
    return s if re.fullmatch(r"[a-z][a-z0-9_]{1,30}", s or "") else fallback


def proposal_messages(question: str, depts: list, teams: dict[str, Team], rank_ids: list[str]):
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "회사에 아래 부서들이 있지만, 사람의 요청을 맡을 부서가 없습니다. 이 일을 꾸준히 맡을 새 부서와 첫 팀을 설계하세요.\n"
        f"- 직급 id 는 {', '.join(rank_ids)} 중에서만. 첫 팀원은 팀장(team_lead), 그 아래 팀원 2~5명.\n"
        "- 기존 부서와 겹치지 않게. 이름은 한국어, id 는 영문 소문자·밑줄.\n"
        "- tools_needed 에는 실제로 필요한 자료·도구를 적되, 웹 검색처럼 밖으로 나가는 도구는 '필요하면 사람이 켬'이라고 적는다.\n\n"
        "[지금 부서]\n" + "\n".join(f"- {d.id}: {d.emoji} {d.name} — {d.description}" for d in depts)
    )
    return [SystemMessage(content=system), HumanMessage(content=question)]


def fallback_proposal(question: str, taken: set[str]) -> NewDeptProposal:
    """모델 없이 만드는 기본 초안 — 사람이 미리 보기에서 고쳐 쓴다."""
    short = " ".join(question.split())[:30]
    dept_id = _free_id("new_dept", taken)
    return NewDeptProposal(
        analysis="모델 없이 만든 기본 초안입니다 — 이름·하는 일·팀 구성을 고쳐 주세요.",
        dept_id=dept_id, dept_name="새 부서", emoji="🆕", dept_description=f"{short} 같은 요청을 맡는 부서",
        head_persona="요청한 내용에 답했는지, 근거 없는 단정이 없는지를 본다.",
        team_id=_free_id(dept_id + "_team", taken), team_name="🆕 새 팀", team_description=f"{short} 같은 요청을 처리한다",
        scope_summary=short, accepts=[short], rejects=["다른 부서가 있는 질문 (→ 그 부서)"], keywords=[],
        members=[ProposedMember(id="lead", rank="team_lead", name="팀장", specialty="업무 총괄"),
                 ProposedMember(id="researcher", rank="deputy_gm", name="차장·리서치", specialty="자료 조사"),
                 ProposedMember(id="writer", rank="assistant_manager", name="대리·작성", specialty="초안 작성")],
        tools_needed=["(사람이 정함)"], review_criteria=["요청한 내용에 빠짐없이 답했는가", "근거 없는 단정이 없는가"],
    )


def _free_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def propose(question: str, depts: list, teams: list[Team], models, rank_ids: list[str], taken: set[str],
            choice: str = "") -> dict:
    """{proposal(dict), how: llm|fallback, model, note} — 새 부서·팀 초안. taken = 이미 쓰는 부서·팀 id (겹치면 뒤에 _2)."""
    from core import dept_heads, llm_factory, usage

    note = ""
    chain = dept_heads.assist_chain(models, choice)  # 가벼운 안내 — 부장 기본(Opus 구독)이 아니라 utility
    try:
        with usage.scope(purpose="새 부서 제안"):
            call = llm_factory.invoke_structured(chain, models, NewDeptProposal,
                                                 proposal_messages(question, depts, {t.id: t for t in teams}, rank_ids),
                                                 "🛎 새 부서 제안")
        p: NewDeptProposal = call.parsed
        how, model = "llm", call.spec.label
    except Exception as e:  # noqa: BLE001
        p, how, model = fallback_proposal(question, taken), "fallback", ""
        note = f"모델을 쓰지 못해 기본 초안을 만들었습니다 ({type(e).__name__})."
    data = p.model_dump()
    data["dept_id"] = _free_id(slug(data["dept_id"], "new_dept"), taken)
    data["team_id"] = _free_id(slug(data["team_id"], data["dept_id"] + "_team"), taken | {data["dept_id"]})
    data["members"] = _clean_members(data["members"], rank_ids)
    return {"proposal": data, "how": how, "model": model, "note": note}


def _clean_members(members: list[dict], rank_ids: list[str]) -> list[dict]:
    """조직도 정리 — 첫 줄은 팀장, 없는 직급은 사원으로, id 겹침은 뒤에 숫자."""
    lead_rank = "team_lead" if "team_lead" in rank_ids else rank_ids[-1]
    low = rank_ids[0]
    out, seen = [], set()
    for i, m in enumerate(members[:8] or [{"id": "lead", "rank": lead_rank, "name": "팀장", "specialty": "업무 총괄"}]):
        rank = lead_rank if i == 0 else (m.get("rank") if m.get("rank") in rank_ids and m.get("rank") != lead_rank else low)
        mid = _free_id(slug(m.get("id", ""), "lead" if i == 0 else f"member{i}"), seen)
        seen.add(mid)
        out.append({"id": mid, "rank": rank, "name": str(m.get("name") or mid), "specialty": str(m.get("specialty") or "")})
    if len(out) == 1:
        out.append({"id": "member1", "rank": low, "name": "실무자", "specialty": "초안 작성"})
    return out


# ─── [나중에] — 추후 진행 목록 ─────────────────────────────────────────


class LaterStore:
    """data/new_dept_later.json — [{id, question, proposal, created_at}] (최근 것이 앞)."""

    def __init__(self, data_dir: Path | str):
        self.path = Path(data_dir) / LATER_FILE

    def items(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return data if isinstance(data, list) else []

    def _write(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, question: str, proposal: dict) -> dict:
        item = {"id": uuid.uuid4().hex[:10], "question": question, "proposal": proposal,
                "created_at": time.strftime("%Y-%m-%d %H:%M")}
        self._write([item, *self.items()])
        return item

    def get(self, item_id: str) -> dict | None:
        return next((i for i in self.items() if i["id"] == item_id), None)

    def remove(self, item_id: str) -> None:
        self._write([i for i in self.items() if i["id"] != item_id])

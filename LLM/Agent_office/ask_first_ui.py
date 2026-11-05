"""❓ 질문 먼저 화면 — 업무 양식으로 맡긴 뒤, 시작 전에 결과 모양을 크게 바꾸는 판단을 1~3개 묻는다 (core/ask_first.py).

팀 설정(team.yaml 의 ask_first)이 있는 팀만 — 지금은 👔 포트폴리오·자소서 팀. 설정이 없거나 질문이 없으면 request 를 그대로 돌려준다
(다른 팀은 지금과 똑같이 바로 시작). 질문은 코드가 만든다 (모델 호출 없음).

  step : 질문 상자 — '✅ 이 답으로 시작' (답을 업무 요청에 넣고 시작) · '⏭ 묻지 않고 시작' (모델이 원문을 바꾸지 않는 쪽으로 쓰고
         needs_check 맨 위에 '결정이 필요한 질문'을 남긴다) · '✖ 취소'. 기다리는 동안은 빈 dict (업무를 시작하지 않는다).
"""

import streamlit as st

from core import ask_first
from core.schemas import Team

CUSTOM = "직접 입력"


def step(request: dict, team: Team, task_id: str) -> dict:
    key = f"ask_first_{team.id}_{task_id}"
    if request:
        items = ask_first.questions_for(team, task_id, request.get("tool_inputs") or {})
        if not items:
            st.session_state.pop(key, None)
            return request
        nonce = (st.session_state.get(key) or {}).get("nonce", 0) + 1  # 다시 맡기면 답 칸을 새로
        st.session_state[key] = {"request": request, "items": [i.as_dict() for i in items], "nonce": nonce}
    pending = st.session_state.get(key)
    if not pending:
        return {}
    wkey = f"{key}_{pending['nonce']}"
    with st.container(border=True):
        st.markdown("##### ❓ 시작 전에 정할 것 — 결과 모양을 크게 바꾸는 판단")
        st.caption("모델이 혼자 정하지 않도록 먼저 묻습니다. 고른 답은 업무 요청에 '먼저 답한 질문'으로 들어가고 모든 팀원이 따릅니다. "
                   "질문은 코드가 올린 자료·추가 요청을 보고 만들었습니다 (모델 호출 없음).")
        answers = []
        for i, item in enumerate(pending["items"]):
            choice = st.radio(f"{i + 1}. {item['question']}", [*item["options"], CUSTOM], key=f"{wkey}_q{i}")
            if item.get("why"):
                st.caption(item["why"])
            if choice == CUSTOM:
                choice = st.text_input("답 (직접 입력)", key=f"{wkey}_t{i}").strip()
            answers.append({"id": item["id"], "question": item["question"], "answer": choice})
        c1, c2, c3 = st.columns(3)
        go = c1.button("✅ 이 답으로 시작", type="primary", key=f"{wkey}_go")
        skip = c2.button("⏭ 묻지 않고 시작", key=f"{wkey}_skip",
                         help="모델이 원문을 바꾸지 않는 쪽(원래 형식·구분, 합치지 않음)으로 쓰고, 결정이 필요한 질문을 결과의 확인 필요 맨 위에 남깁니다.")
        if c3.button("✖ 취소 (업무 맡기지 않음)", key=f"{wkey}_cancel"):
            st.session_state.pop(key, None)
            st.rerun()
        if go:
            if any(not a["answer"] for a in answers):
                st.warning("'직접 입력'을 고른 질문의 답이 비어 있습니다.")
                return {}
            st.session_state.pop(key, None)
            return ask_first.apply_answers(pending["request"], answers)
        if skip:
            st.session_state.pop(key, None)
            return pending["request"]
    return {}

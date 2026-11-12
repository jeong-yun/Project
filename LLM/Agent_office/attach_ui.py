"""📎 첨부 칸 — 💬 자유 질문(app.py) · 📮 접수 창구(reception_ui.py) · 👔 부서장 창구(building_ui.head_desk)가 함께 쓴다.

  uploader    : 파일 칸 + 안내 한 줄 (이 PC 에만 저장 · 내용은 고른 모델로 전송)
  blocked     : 맡기기 전 — 읽지 못하는 파일이 있으면 어느 파일이 왜 안 됐고 무엇을 하면 되는지 알리고 막는다
  carry       : 창구에서 올린 파일을 저장해 다음 화면으로 넘긴다 (접수 창구 → 부서장 창구 → 팀)
  collect     : 넘어온 첨부 + 지금 올린 파일(저장) → 업무의 툴 입력 {attachments, attachments_names}
  vision_step : 그림·스캔 PDF 가 있으면 글자 확인 단계 — 커리어부의 확인 상자(career_ui.vision_step)를 그대로 쓴다

파일을 읽어 근거 자료로 만드는 일은 core/tools/attachments.py (모든 팀 공통 툴)가 한다.
"""

from pathlib import Path

import streamlit as st

from core.tasks import store_uploads
from core.tools import attachments

KEY = attachments.KEY
CARRY_KEY = "carried_attachments"  # 다음 화면으로 넘길 첨부 [{path, name}] — 업무가 시작되면 비운다
LABEL = "📎 첨부 파일 (선택, 여러 개) — 문서·그림·자료 파일"
HELP = ("문서(pdf·docx·hwp·hwpx·md·txt) · 자료·코드(csv·json·yaml·log·py·sql·tf) · 그림(png·jpg·webp)을 받습니다. "
        "파일 하나 30,000자, 합쳐서 60,000자까지 팀의 근거 자료로 넣고, 넘치면 잘렸다고 표시합니다. "
        "그림·스캔 PDF 는 그림 속 글자만 옮겨 읽으며, 시작 전에 옮긴 글을 확인합니다. "
        "엑셀·파워포인트는 아직 읽지 못합니다 — PDF 나 csv 로 저장해 올려 주세요.")
NOTICE = ("🔒 올린 파일은 이 PC 의 `data/uploads/` 에만 저장되고 GitHub 에 올라가지 않습니다. "
          "다만 내용은 고른 AI 모델(외부 API 포함)로 전송됩니다.")
VISION_NOTICE = ("🔒 그림은 고른 모델의 회사(OpenAI · Google · Anthropic)로 전송됩니다. 그림 속 **글자만** 옮겨 읽습니다 "
                 "(글자가 없는 사진·도표의 모양은 아직 읽지 못합니다). 보내기 싫으면 '취소'하고 글로 붙여 넣거나 "
                 "PDF(글자 층)·Word 로 올려 주세요.")


def uploader(key: str) -> list:
    files = st.file_uploader(LABEL, type=attachments.TYPES, accept_multiple_files=True, key=key, help=HELP)
    st.caption(NOTICE)
    return list(files or [])


def blocked(files: list) -> bool:
    """읽지 못하는 파일이 있으면 알리고 True (맡기지 않는다)."""
    problems = attachments.check(files)
    if problems:
        st.warning("📎 읽지 못하는 첨부가 있어 맡기지 않았습니다 — 아래 파일을 빼거나 바꿔 올린 뒤 다시 눌러 주세요.\n\n"
                   + "\n".join(f"- {p}" for p in problems))
    return bool(problems)


def inputs_of(items: list[dict]) -> dict:
    """[{path, name}] → 툴 입력 조각 (없으면 빈 dict)."""
    return {KEY: [i["path"] for i in items], f"{KEY}_names": [i["name"] for i in items]} if items else {}


def carried() -> list[dict]:
    return list(st.session_state.get(CARRY_KEY) or [])


def hint(files: list) -> str:
    """배정·범위 판단에 붙일 한 줄 — 넘어온 첨부 + 지금 올린 파일의 이름만."""
    return attachments.hint(inputs_of([*carried(), *({"path": f.name, "name": f.name} for f in files or [])]))


def _save(files: list, data_dir: Path) -> list[dict]:
    if not files:
        return []
    out = store_uploads({KEY: list(files)}, Path(data_dir) / "uploads")
    return [{"path": p, "name": n} for p, n in zip(out[KEY], out[f"{KEY}_names"])]


def carry(files: list, data_dir: Path, replace: bool = False) -> list[dict]:
    """올린 파일을 저장해 다음 화면으로 넘긴다. replace = 앞서 넘어온 것은 버리고 새로 (📮 접수 창구의 새 질문)."""
    items = ([] if replace else carried()) + _save(files, data_dir)
    if items:
        st.session_state[CARRY_KEY] = items
    else:
        st.session_state.pop(CARRY_KEY, None)
    return items


def collect(files: list, data_dir: Path) -> dict:
    """넘어온 첨부 + 지금 올린 파일(저장) → 이번 업무의 툴 입력."""
    return inputs_of(carried() + _save(files, data_dir))


def clear() -> None:
    st.session_state.pop(CARRY_KEY, None)


def show_carried(key: str) -> None:
    """'📎 넘어온 첨부 N개 …' 한 줄 + 빼기 (없으면 아무것도 그리지 않는다)."""
    items = carried()
    if not items:
        return
    c1, c2 = st.columns([5, 1])
    c1.caption(f"📎 넘어온 첨부 {len(items)}개 (이 요청과 함께 맡깁니다): " + ", ".join(i["name"] for i in items))
    c2.button("✖ 첨부 빼기", key=key, on_click=clear, use_container_width=True)


def vision_step(request: dict, team_id: str, slot: str, models, chains: list) -> dict:
    """첨부에 그림·스캔 PDF 가 있으면 글자 확인 단계. 없으면 request 그대로, 확인을 기다리는 동안은 빈 dict."""
    import career_ui  # 확인 상자는 커리어부 것을 그대로 (그림이 있을 때만 필요)

    return career_ui.vision_step(request, team_id, f"attach_{slot}", models, chains,
                                 find=attachments.vision_items, notice=VISION_NOTICE)

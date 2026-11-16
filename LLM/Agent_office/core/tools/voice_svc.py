"""🎤 노래 음색 변환 팀 · 🗣 일상 말 음성 변환 팀 자료 툴 — 코드가 정한 값만 근거 자료로 준다 (core/voice.py, 모델·외부 호출 없음).
두 팀이 함께 쓴다. 종류는 업무 양식의 fixed.voice_kind (song | speech) — 자유 질문은 종류와 상관없이 범위 밖 요청만 본다.

  범위 밖 요청(사칭·무단 복제)  → '돕지 않는다' 지시 + harness_options.voice_blocked (검증기 voice_guard 가 본다)
  📋 학습 계획 (voice_task=plan) → 학습 설정표(화자·epochs·batch_size)·원본 파일에 적힌 시간 어림값 + harness_options.voice_kit
  🔍 데이터 점검 (voice_task=check) → 올린 dataset_raw.zip 의 숫자 요약 표 + harness_options.voice_dataset
                                    음원 내용(소리 데이터)은 근거 자료에 넣지 않는다 — 모델로 가는 것은 숫자뿐
  자유 질문 → None (팀 스킬 글이 근거)
"""

from core import voice
from core.schemas import ModelsConfig, ToolResult

NAME = "voice_svc"


def _blocked(reason: str) -> ToolResult:
    return ToolResult(
        tool=NAME, title="🚫 범위 밖 요청",
        content=(f"**이 요청은 이 팀의 범위 밖이다 ({reason}).** 학습·변환 방법(명령·스크립트·폴더 구조)을 안내하지 말고, "
                 "돕지 않는 이유와 다음에 할 일만 쉬운 말로 답하라. 실존 인물의 이름을 되풀이하지 않는다.\n"
                 f"> {voice.refusal_message(reason)}"),
        harness_options={"voice_blocked": reason})


def run(question: str, inputs: dict, models: ModelsConfig) -> ToolResult | None:
    reason = voice.misuse(question)
    if reason:
        return _blocked(reason)
    task = inputs.get("voice_task")
    if task == "plan":
        kit = voice.kit_from_inputs(inputs)
        if kit is None:  # 화면이 맡기기 전에 막지만, 다른 길로 들어와도 계획을 만들지 않는다
            return _blocked("목소리 사용 확인이 없는 요청")
        return ToolResult(tool=NAME, title="🎛 학습 설정 (코드가 정함)", content=voice.plan_evidence(kit),
                          harness_options={"voice_kit": kit})
    if task == "check":
        name = inputs.get("dataset_name") or "dataset_raw.zip"
        try:
            stats = voice.inspect_zip(inputs.get("dataset") or "", name, voice.kind_of(inputs))
        except voice.VoiceError as e:
            return ToolResult(tool=NAME, title="🔍 데이터 점검 (실패)", notes=[str(e)],
                              content=(f"**데이터를 점검하지 못했다:** {e}\n클립 수·길이 같은 숫자를 지어내지 말고, "
                                       "왜 점검하지 못했는지와 다시 올리는 방법만 안내하라."))
        return ToolResult(
            tool=NAME, title=f"🔍 데이터 점검 — {name} (코드 계산)",
            content=("### 🔍 데이터 점검 결과 — 코드가 계산한 값. 숫자·판정을 바꿔 쓰지 않는다\n"
                     + voice.dataset_table(stats)
                     + "\n\n(음원 내용은 모델에 보내지 않았다 — 위 숫자 요약만 근거다.)"),
            harness_options={"voice_dataset": stats})
    return None

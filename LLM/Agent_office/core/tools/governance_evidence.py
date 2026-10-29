"""보안 거버넌스 증적 툴 — 증적 묶음 · 설정 drift(코드 diff) · 쓸 수 있는 통제 번호 목록을 근거 자료로 만든다.

외부 호출 없음 (웹 검색도 하지 않는다 — 내부 설정·정책이 검색어로 나가지 않게).
입력 (팀 업무 양식 · 자율 실행 작업 inputs)
  framework    : ISMS-P / ISO/IEC 27001:2022 / NIST CSF 2.0
  evidence     : 붙여 넣은 증적 글 (선택)
  use_sample   : 켜면 가상 회사 샘플 증적 폴더를 쓴다 (teams/security_governance/samples/sample_corp)
  evidence_dir : 증적 폴더 (자율 실행용, 앱 폴더 기준 상대 경로 가능)
  baseline_dir / before : drift 비교용 '이전' 설정 폴더 / 글  (있으면 현재와 줄 단위 diff)
하네스에 넘기는 값: control_framework(통제 번호 검증기) · evidence_text(인용 문구 검증기 — 입력 증적 원문 전부)
"""

from core import governance
from core.schemas import ModelsConfig, ToolResult

SAMPLE = "sample_corp"


def _sample_dirs(task_kind: str) -> tuple[str, str | None]:
    base = governance.SAMPLES_DIR / SAMPLE
    if task_kind == "drift":
        return str(base / "current" / "config"), str(base / "previous" / "config")
    return str(base / "current"), None


def run(question: str, inputs: dict, models: ModelsConfig) -> ToolResult | None:
    framework = str(inputs.get("framework") or "").strip()
    if not framework:
        return None
    fw = governance.load_framework(framework)
    notes: list[str] = []
    evidence_dir, baseline_dir = inputs.get("evidence_dir"), inputs.get("baseline_dir")
    if inputs.get("use_sample"):  # 통제 항목 점검: 샘플 회사의 현재 증적 전체
        evidence_dir, _ = _sample_dirs("check")
    if inputs.get("use_sample_drift"):  # 설정 drift 점검: 샘플 회사의 지난주·이번 주 설정
        evidence_dir, baseline_dir = _sample_dirs("drift")
    files = governance.read_evidence_dir(evidence_dir) if evidence_dir else {}
    before_files = governance.read_evidence_dir(baseline_dir) if baseline_dir else {}
    pasted = str(inputs.get("evidence") or "")
    before = str(inputs.get("before") or "")
    if before.strip():  # 붙여 넣은 이전·현재 설정 — 같은 이름으로 두어야 줄 단위로 비교된다
        before_files["붙여 넣은 설정"] = before
        files = {**files, "붙여 넣은 설정": pasted}
    is_drift = bool(before_files)

    parts = [governance.framework_prompt(fw)]
    drift = ""
    if is_drift and before_files:
        drift = governance.config_drift(before_files, files)
        added, removed = governance.drift_lines(drift)
        parts.append(f"### 설정 drift (코드가 계산한 줄 단위 diff) — 추가 {added}줄 · 삭제 {removed}줄\n"
                     + (f"```diff\n{drift}\n```" if drift else "바뀐 줄 없음"))
    bundle = governance.evidence_bundle(files)
    if len(bundle) > governance.MAX_EVIDENCE_CHARS:
        notes.append(f"증적이 {len(bundle):,}자라 앞 {governance.MAX_EVIDENCE_CHARS:,}자만 넣었습니다.")
        bundle = bundle[: governance.MAX_EVIDENCE_CHARS]
    if files:
        parts.append(f"### 증적 파일 {len(files)}개 (인용은 이 원문에서 글자 그대로)\n" + bundle)
    if not (files or pasted.strip()):
        notes.append("증적이 없습니다 — 모든 항목이 '확인 불가'가 됩니다. 증적을 붙여 넣거나 샘플 증적을 켜세요.")

    # 인용 문구 검증기가 비교할 원문 = 붙여 넣은 글 + 증적 파일 + 이전 설정 (drift 의 '이전 값' 인용도 허용)
    source = "\n\n".join([pasted, *files.values(), *before_files.values()])
    return ToolResult(
        tool="governance_evidence",
        title=f"🛡️ 증적·통제 목록 ({fw.name})",
        content="\n\n".join(parts),
        notes=notes,
        harness_options={"control_framework": fw.name, "evidence_text": source},
    )

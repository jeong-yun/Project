"""IaC 점검 툴 — Terraform·쿠버네티스 매니페스트·Dockerfile 을 내장 규칙(+ 설치돼 있으면 외부 스캐너)으로 검사해 근거 자료로 만든다.

웹 검색·외부 전송 없음 (설정 파일이 검색어로 나가지 않게). 외부 스캐너도 이 PC 에서만 돈다.
입력 (팀 업무 양식 · 자율 실행 작업 inputs)
  framework  : 통제 대응을 보여 줄 프레임워크 (ISMS-P / ISO/IEC 27001:2022 / NIST CSF 2.0, 비우면 ISMS-P)
  use_sample : 켜면 가상 회사 샘플 IaC (teams/infra_security/samples/sample_corp)
  iac_dir    : IaC 폴더 (자율 실행용, 앱 폴더 기준 상대 경로 가능)
  iac        : 붙여 넣은 IaC 글 (iac_name 으로 파일 이름, 비우면 내용으로 종류를 짐작)
  external   : false 면 외부 스캐너를 쓰지 않는다 (기본: 설치돼 있으면 쓴다)
하네스에 넘기는 값: iac_files(수정안 재검사 검증기 iac_fix — 원본 전부) · control_framework(통제 번호 검증기) · iac_source
"""

import os
import re

from core import iac
from core.schemas import ModelsConfig, ToolResult

SAMPLE = "sample_corp"
FRAMEWORKS = ("ISMS-P", "ISO/IEC 27001:2022", "NIST CSF 2.0")

DIFF_RULES = """### 수정안(diff) 작성 규칙 — 코드가 사본에 적용해 다시 검사한다
- 파일마다 ```diff 블록 하나: 머리줄 `--- a/<경로>` · `+++ b/<경로>` (경로는 아래 '원본 파일'의 이름 그대로), 그 아래 `@@ -줄,수 +줄,수 @@` hunk
- ' '(문맥)·'-'(지울) 줄은 원본을 글자 그대로 (들여쓰기 포함) 복사한다. 문맥은 바꿀 곳 앞뒤 2~3줄이면 된다
- 리소스·컨테이너·파일을 지워서 '고친' 것은 인정되지 않는다 — 설정 값을 바꾼다
- 수정 뒤 같은 파일에 같거나 더 높은 심각도의 새 문제가 생기거나 파일이 문법상 깨지면 그 파일 수정안 전체가 불합격이다
- 고치지 못하거나 사람이 정해야 하는 것(예: 실제로 필요한 IAM 권한 목록)은 diff 대신 not_attempted 에 이유를 적는다"""


def _guess_name(text: str) -> str | None:
    if re.search(r'^\s*(resource|provider|variable|module|data|terraform)\s+("|\{)', text, re.M):
        return "pasted.tf"
    if re.search(r"^\s*apiVersion\s*:", text, re.M):
        return "pasted.yaml"
    if re.search(r"^\s*FROM\s+\S", text, re.M | re.I):
        return "Dockerfile"
    return None


def collect(inputs: dict) -> tuple[dict[str, str], str, list[str]]:
    """입력 → ({상대 경로: 내용}, 출처 설명, 메모)."""
    notes: list[str] = []
    files: dict[str, str] = {}
    source = ""
    folder = str(iac.SAMPLES_DIR / SAMPLE) if inputs.get("use_sample") else inputs.get("iac_dir")
    if folder:
        files = iac.read_dir(folder)
        source = "teams/infra_security/samples/sample_corp" if inputs.get("use_sample") else str(folder)
    pasted = str(inputs.get("iac") or "")
    if pasted.strip():
        name = str(inputs.get("iac_name") or "").strip() or _guess_name(pasted)
        if name and iac.file_kind(name):
            files[name] = pasted
            source = source or "붙여 넣은 글"
        else:
            notes.append("붙여 넣은 글의 종류를 모르겠습니다 — 파일 이름(main.tf / deploy.yaml / Dockerfile)을 적어 주세요.")
    for name in [n for n, t in files.items() if len(t) > iac.MAX_FILE_CHARS]:
        notes.append(f"{name}: {len(files[name]):,}자라 빼고 검사했습니다 (상한 {iac.MAX_FILE_CHARS:,}자).")
        files.pop(name)
    return files, source, notes


def run(question: str, inputs: dict, models: ModelsConfig) -> ToolResult | None:
    if not any(inputs.get(k) for k in ("use_sample", "iac_dir", "iac")):
        return None
    framework = str(inputs.get("framework") or "").strip() or FRAMEWORKS[0]
    files, source, notes = collect(inputs)
    if not files:
        return ToolResult(tool="iac_scan", title="🏗️ IaC 점검", content="", notes=notes or ["검사할 IaC 파일이 없습니다."])
    scan = iac.scan_files(files)
    counts = scan.counts()
    sev = " · ".join(f"{s} {counts[s]}" for s in reversed(iac.SEVERITIES) if counts[s])
    parts = [f"### 🏗️ IaC 점검 결과 — 코드(내장 규칙 {len(iac.rules())}개)가 찾은 것: 파일 {len(scan.files)}개, 발견 {counts['total']}건"
             + (f" ({sev})" if sev else "")
             + "\n발견 목록·ID·심각도·통제 대응은 코드가 정했다. 설명과 수정안은 이 표의 ID(F01…)로만 가리킨다.\n\n"
             + (iac.findings_table(scan, framework) if scan.findings else "발견 없음")]
    if scan.errors:
        parts.append("### 구문 오류 (이 파일은 규칙 검사를 못 했다)\n" + "\n".join(f"- {f}: {e}" for f, e in scan.errors.items()))
    use_external = inputs.get("external", True) is not False and os.getenv("AGENT_OFFICE_IAC_EXTERNAL", "1") != "0"
    tools = iac.available_scanners() if use_external else []
    if tools:
        ext, ext_notes = iac.run_external(files, tools)
        notes += ext_notes
        rows = [f"| {f.source} | {f.rule} | {f.severity} | {f.file}:{f.line} | {f.resource} | {f.title[:80]} | {f.same_as or '-'} |"
                for f in ext[:40]]
        parts.append(f"### 외부 스캐너 (참고, {', '.join(tools)}) — {len(ext)}건. 수정안 재검사 판정은 내장 규칙으로 한다\n"
                     "| 도구 | 규칙 | 심각도 | 파일:줄 | 리소스 | 내용 | 같은 내장 규칙 |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    else:
        notes.append("외부 스캐너(checkov·trivy·kube-linter·hadolint)가 없어 내장 규칙으로만 검사했습니다." if use_external
                     else "외부 스캐너를 끄고 내장 규칙으로만 검사했습니다.")
    parts.append(DIFF_RULES)
    parts.append(f"### 원본 파일 {len(files)}개 (diff 는 이 원문 기준)\n" + iac.files_bundle(files))
    return ToolResult(
        tool="iac_scan",
        title=f"🏗️ IaC 점검 ({counts['total']}건)",
        content="\n\n".join(parts),
        notes=notes,
        harness_options={"iac_files": files, "control_framework": framework, "iac_source": source},
    )

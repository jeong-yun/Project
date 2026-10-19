"""결정론적 검증 하네스의 뼈대.

왜 필요한가?
  LLM 이 쓴 코드·쿼리가 맞는지 다시 LLM 에게 물으면 또 환각이 생길 수 있다.
  대신 실제 인터프리터·DB(샌드박스)에 돌려 보고, 에러가 나면 '0점 + 에러 로그'를 그대로 돌려줘
  다시 쓰게 한다 (RLVR 의 '검증 가능한 보상' 아이디어, Backtracking).

새 검증기 추가 방법
  1) core/harness/validators.py 에 Validator 를 상속한 클래스를 만들고 @register 를 붙인다.
  2) name 을 정하고 check(answer, options) 에서 HarnessResult 를 반환한다.
  3) team.yaml 의 harness.validators 에 name 을 적는다.
"""

import re
from abc import ABC, abstractmethod

from core.schemas import HarnessResult

# 이름 → 검증기 인스턴스. @register 가 채운다.
REGISTRY: dict[str, "Validator"] = {}


class Validator(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def check(self, answer: str, options: dict) -> HarnessResult:
        """answer(LLM 답변 전문)를 검사한다. options 는 team.yaml 의 harness.options."""

    def ok(self, log: str = "") -> HarnessResult:
        return HarnessResult(validator=self.name, passed=True, score=1.0, log=log)

    def fail(self, log: str) -> HarnessResult:
        return HarnessResult(validator=self.name, passed=False, score=0.0, log=log)


def register(cls: type[Validator]) -> type[Validator]:
    """클래스 데코레이터 — 검증기를 REGISTRY 에 등록한다."""
    REGISTRY[cls.name] = cls()
    return cls


def extract_code_blocks(text: str, *langs: str) -> list[str]:
    """마크다운 코드 블록(```lang ... ```) 중 지정한 언어의 내용만 뽑는다."""
    pattern = r"```[ \t]*(" + "|".join(map(re.escape, langs)) + r")[ \t]*\n(.*?)```"
    return [m.group(2) for m in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE)]


def run_validators(names: list[str], answer: str, options: dict) -> list[HarnessResult]:
    """여러 검증기를 차례로 실행한다. 등록되지 않은 이름은 설정 검증 단계에서 걸러진다."""
    return [REGISTRY[name].check(answer, options) for name in names if name in REGISTRY]


def all_passed(results: list[HarnessResult]) -> bool:
    return all(r.passed for r in results)


def harness_feedback(results: list[HarnessResult]) -> str:
    """실패한 검증 결과를 LLM 에게 돌려줄 피드백 문장으로 만든다 (결정론적 피드백)."""
    failed = [r for r in results if not r.passed]
    lines = [f"- [{r.validator}] 0점 — {r.log}" for r in failed]
    return (
        "자동 검증(하네스) 결과 다음 오류가 발견되었다. 점수 0점.\n"
        + "\n".join(lines)
        + "\n에러 로그를 참고해 오류를 고친 완성 답변을 같은 형식으로 다시 작성하라."
    )

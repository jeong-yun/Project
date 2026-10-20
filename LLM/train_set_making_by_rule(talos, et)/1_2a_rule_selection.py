"""[1.2a 1.2 보조] 룰 선별 조건 로더 — 선별조건.md 의 표를 읽어서 '어떤 룰을 추출할지' 판단한다.

조건 자체는 이 파일이 아니라 `docs/코드참조/선별조건.md` 에 있다.
사람이 md 표의 '선택' 열(O/X)을 고치면, 1_2_parse_rules.py 가 다음 실행 때
조건 변경을 감지(fingerprint 비교)해서 자동으로 다시 파싱한다.

[카테고리 정하는 법]
  - Talos : 룰 파일 이름 (.rules 제외)   예) server-webapp.rules → "server-webapp"
  - ET    : msg 의 두 번째 단어           예) "ET WEB_SERVER ..." → "WEB_SERVER"
  - custom: msg 로 추정 (custom_category)  예) "CUSTOM.1.00001 AI_Agent_..." → "ai", "PUA-OTHER ..." → "pua-other"
            "GPL SQL ..." 같은 옛 커뮤니티 룰도 같은 방식 → "SQL"

[판단 규칙]
  1. '## 개별 룰' 표에 그 룰(소스:gid:sid)이 있으면 그 O/X 를 따른다. (카테고리와 상관없이)
  2. 아니면 카테고리 표를 위에서부터 비교해서 '처음 맞는 줄'의 O/X 를 따른다. 맞는 줄이 없으면 제외.
"""

import fnmatch
import hashlib
import re
from pathlib import Path

SELECTION_FILE = Path(__file__).resolve().parent / "docs" / "코드참조" / "선별조건.md"

_ET_CATEGORY_RE = re.compile(r"^(?:ET|GPL)\s+([A-Z0-9_]+)")
_YES = {"O", "o", "○", "Y", "y", "선택"}
_NO = {"X", "x", "×", "N", "n", "제외"}


def _load(path=SELECTION_FILE):
    """md 파일에서 '## Talos', '## ET' 아래의 표를 읽는다.

    반환: {"talos": [(패턴, 선택여부, 이유), ...], "et": [(패턴, 선택여부, 대응Talos, 이유), ...],
           "dedup": {"enabled": bool, "keep": "talos"|"et", "exact": bool, "cve_uri": bool}}
    표가 없거나 선택 값이 O/X 가 아니면 오류를 낸다. (잘못된 조건으로 조용히 진행하지 않도록)
    """
    if not path.exists():
        raise FileNotFoundError(f"선별 조건 파일이 없습니다: {path.name}")
    rules = {"talos": [], "et": [], "custom": [], "dedup": {}, "rules": {}}
    section = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s.startswith("## "):
            title = s[3:].strip().lower()
            section = ("talos" if title.startswith("talos") else "et" if title.startswith("et")
                       else "dedup" if title.startswith("중복") else "rules" if title.startswith("개별")
                       else "custom" if title.startswith("사용자") else None)
            continue
        if section is None or not s.startswith("|"):
            continue
        cells = [c.strip().strip("`").strip() for c in s.strip("|").split("|")]
        # 머리글 줄과 구분선(---) 은 건너뜀
        if not cells or set(cells[0]) <= set("-: ") or cells[0] in ("파일 패턴", "카테고리", "항목", "룰 키", "카테고리 패턴"):
            continue
        if len(cells) < 2:
            continue
        if section == "dedup":
            _read_dedup_row(rules["dedup"], cells[0], cells[1], path.name, lineno)
            continue
        mark = cells[1]
        if mark in _YES:
            chosen = True
        elif mark in _NO:
            chosen = False
        else:
            raise ValueError(f"{path.name}:{lineno} - '선택' 열은 O 또는 X 여야 합니다: {mark!r}")
        if section == "rules":
            # 개별 룰: 키 = 소스:gid:sid (예: talos:1:66203). 카테고리 판단보다 우선
            rules["rules"][cells[0].lower()] = (chosen, cells[2] if len(cells) > 2 else "")
        elif section in ("talos", "custom"):
            rules[section].append((cells[0].lower(), chosen, cells[2] if len(cells) > 2 else ""))
        else:
            rules["et"].append((cells[0].upper() if cells[0] != "*" else "*", chosen,
                                cells[2] if len(cells) > 2 else "", cells[3] if len(cells) > 3 else ""))
    for src in ("talos", "et"):
        if not rules[src]:
            raise ValueError(f"{path.name} 에서 '## {'Talos' if src == 'talos' else 'ET'}' 표를 찾지 못했습니다")
    # 중복 제거 표가 없거나 일부 항목이 빠졌으면 기본값 (사용 O, talos 우선, 두 기준 모두 O)
    d = rules["dedup"]
    d.setdefault("enabled", True)
    d.setdefault("keep", "talos")
    d.setdefault("exact", True)
    d.setdefault("cve_uri", True)
    return rules


def _read_dedup_row(d, item, value, fname, lineno):
    """'## 중복 제거' 표의 한 줄을 읽는다."""
    def yn(v):
        if v in _YES:
            return True
        if v in _NO:
            return False
        raise ValueError(f"{fname}:{lineno} - '{item}' 값은 O 또는 X 여야 합니다: {v!r}")

    if item.startswith("사용"):
        d["enabled"] = yn(value)
    elif item.startswith("남길"):
        v = value.lower()
        if v not in ("talos", "et"):
            raise ValueError(f"{fname}:{lineno} - '남길 소스'는 talos 또는 et 여야 합니다: {value!r}")
        d["keep"] = v
    elif item.startswith("기준1"):
        d["exact"] = yn(value)
    elif item.startswith("기준2"):
        d["cve_uri"] = yn(value)


def dedup_options():
    """중복 제거 설정 {"enabled", "keep", "exact", "cve_uri"}."""
    return dict(_rules()["dedup"])


def dedup_fingerprint():
    """중복 제거 설정의 지문. 바뀌면 통합 결과를 다시 만든다."""
    return hashlib.sha1(repr(sorted(dedup_options().items())).encode("utf-8")).hexdigest()[:12]


_RULES = None   # 처음 쓸 때 읽는다 (1_2_parse_rules.py 는 시작할 때 reload() 로 미리 읽어 오류를 로그에 남김)


def reload():
    """md 파일을 (다시) 읽는다. 형식이 잘못됐으면 ValueError / FileNotFoundError."""
    global _RULES
    _RULES = _load()


def _rules():
    if _RULES is None:
        reload()
    return _RULES


def fingerprint():
    """조건의 지문. 패턴과 O/X 만으로 계산하므로, '이유' 문구만 바꾼 경우는 같은 값이 나온다."""
    rules = _rules()   # (중복 제거 설정은 파싱과 무관하므로 여기 포함하지 않음 → dedup_fingerprint)
    core = repr([(p, c) for p, c, *_ in rules["talos"]] + [(p, c) for p, c, *_ in rules["et"]]
                + sorted((k, v[0]) for k, v in rules["rules"].items())
                + [(p, c) for p, c, *_ in rules["custom"]])
    return hashlib.sha1(core.encode("utf-8")).hexdigest()[:12]


_TALOS_PREFIX_RE = re.compile(r"^([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\s")
_IGR_TOKEN_RE = re.compile(r"^IGR\w*\.\S+\s+([A-Za-z0-9\-]+)[_,\s]")


def custom_category(msg):
    """사용자 룰(사내 탐지정책) msg → 카테고리.

      'PUA-OTHER XMRMiner ...'                       → pua-other   (Talos 형식 머리말)
      'CUSTOM.1.00001 AI_Agent_User-Agent_...'        → ai          (코드 뒤 첫 단어)
      'CUSTOM.1.00001 server-other_CrushFTP_...'      → server-other
      'CUSTOM.1.00001 Hidden_Cobra A Network Trojan was detected' → malware-other ('Trojan was detected')
      그 외 → etc
    """
    msg = (msg or "").strip()
    if "trojan was detected" in msg.lower():
        return "malware-other"
    m = _TALOS_PREFIX_RE.match(msg)
    if m:
        return m.group(1).lower()
    m = _IGR_TOKEN_RE.match(msg + " ")
    return m.group(1).lower() if m else "etc"


def rule_category(rule):
    """룰의 카테고리. Talos=파일 이름(확장자 제외), ET=msg 두 번째 단어, 사용자 룰=msg 로 추정. 모르면 'OTHER'."""
    if rule["source"] == "custom":
        # 응답·방향 불명 룰은 클라이언트 측(브라우저·파일) 공격이 대부분 → 방향을 카테고리로 (Talos 의 browser-*/file-* 제외와 같은 기준)
        side = (rule.get("http") or {}).get("side")
        if side in ("response", "unknown"):
            return f"{side}-side"
        return custom_category(rule.get("msg"))
    if rule["source"] == "talos":
        return rule["file"].rsplit(".rules", 1)[0]
    m = _ET_CATEGORY_RE.match(rule.get("msg") or "")
    return m.group(1) if m else "OTHER"


def _match(source, category):
    """표에서 처음 맞는 줄을 찾는다. 없으면 None."""
    for row in _rules().get(source, []):
        if fnmatch.fnmatchcase(category.lower() if source != "et" else category, row[0]):
            return row
    return None


def rule_key(rule):
    """룰 키 '소스:gid:sid'. 키가 없으면 None."""
    if rule.get("key"):
        return rule["key"].lower()
    if rule.get("sid") is not None:
        return f"{rule['source']}:{rule.get('gid', 1)}:{rule['sid']}"
    return None


def individual_rules():
    """'## 개별 룰' 표 {키: (선택, 이유)}."""
    return dict(_rules()["rules"])


def is_selected(rule):
    """선별 조건에 해당하는 룰이면 True. '## 개별 룰' 표에 있으면 그것이 카테고리 판단보다 우선."""
    key = rule_key(rule)
    if key and key in _rules()["rules"]:
        return _rules()["rules"][key][0]
    row = _match(rule["source"], rule_category(rule))
    return bool(row and row[1])


def category_selected(source, category):
    row = _match(source, category)
    return bool(row and row[1])


def reason(source, category):
    """보고서용: 이 카테고리를 선택/제외한 이유 (md 표의 '이유' 열)."""
    row = _match(source, category)
    if row is None:
        return "조건 표에 없음 (기본 제외)"
    if source == "et":
        _, _, talos, why = row
        return f"[Talos {talos}] {why}" if talos and talos != "-" else why
    return row[2]

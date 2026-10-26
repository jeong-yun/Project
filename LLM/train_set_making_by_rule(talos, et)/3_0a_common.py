"""[3.0a 3.x 보조] LLM 데이터 생성 단계(3_1 ~ 3_4)가 함께 쓰는 기능. 직접 실행하지 않는다.

  - 경로·로그·.env·설정 md(LLM_설정.md) 읽기
  - LLM 호출 (provider: upstage / openai / anthropic / mock)
  - LLM 응답에서 JSON 꺼내기
  - 코드 검사(local_check): 샘플을 룰 조건(content·정규식·부정 조건)과 기계적으로 대조

다른 3_x 파일에서 불러 쓰는 방법 (파일 이름이 숫자로 시작해 일반 import 불가):
    common = load_common()
"""

import collections
import contextlib
import functools
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from urllib.parse import unquote, unquote_plus

BASE_DIR = Path(__file__).resolve().parent
LLM_INPUT_DIR = BASE_DIR / "05_llm_input"
LLM_OUTPUT_DIR = BASE_DIR / "07_llm_output"
SETTINGS_FILE = BASE_DIR / "docs" / "코드참조" / "LLM_설정.md"
GEN_PROMPT_FILE = BASE_DIR / "docs" / "코드참조" / "LLM_프롬프트.md"
REV_PROMPT_FILE = BASE_DIR / "docs" / "코드참조" / "LLM_검토_프롬프트.md"
ENV_FILE = BASE_DIR / ".env"
LOG_DIR = BASE_DIR / "logs"

# 07_llm_output/<날짜>/ 안의 단계별 폴더 (1_4 는 맨 위의 *.jsonl 만 읽으므로 '_' 로 시작하는 폴더는 안 읽음)
REQ_DIR, GEN_DIR, REV_DIR = "_1_requests", "_2_generated", "_3_reviewed"

# 실행 잠금·중지 요청 (UI·1.0·3.x 공용, 0_0a_runtime.py)
_rt_spec = importlib.util.spec_from_file_location("runtime", BASE_DIR / "0_0a_runtime.py")
RT = importlib.util.module_from_spec(_rt_spec); _rt_spec.loader.exec_module(RT)
LOCK_OWNER_ENV = "RULEDATA_LOCK_OWNER"     # 3_0 이 잠금을 잡고 자식 단계(3.1~3.4)를 부를 때 넣는 표시


def generation_guard(log, who):
    """학습데이터 생성(API 비용)은 한 번에 하나만 돌게 한다.
    3_0 이 잡은 잠금 아래(자식 단계)면 통과. 단독 실행이면 직접 잠금을 잡고, 다른 생성이 돌고 있으면 False."""
    if os.environ.get(LOCK_OWNER_ENV) == "generation":
        return True
    holder = RT.acquire_lock("generation", by=who)
    if holder:
        log.error("다른 학습데이터 생성이 실행 중입니다 (PID %s, %s 시작, %s) - 끝난 뒤 다시 실행하세요",
                  holder.get("pid"), holder.get("started_at"), holder.get("by", ""))
        return False
    import atexit
    atexit.register(RT.release_lock, "generation")
    return True

# Snort pcre 를 파이썬 re 로 컴파일할 때 나오는 FutureWarning(예: "Possible nested set") 은 동작과 무관하므로 숨긴다
warnings.filterwarnings("ignore", category=FutureWarning, module="re")

LABELS = {"anomalies", "normal"}
ATTACK_TYPES = {
    "sql_injection", "xss", "xxe", "ssrf", "deserialization", "code_injection", "command_injection",
    "file_inclusion", "path_traversal", "file_upload", "webshell", "auth_bypass", "rce", "buffer_overflow",
    "info_disclosure", "scanner", "dos", "csrf", "open_redirect", "informational", "other",
}


# ════════════════════════════════════════════════════════════════════════
#  준비
# ════════════════════════════════════════════════════════════════════════

def setup_logging(name):
    """logs/<name>.log 와 화면에 로그. 같은 이름 로거를 두 번 설정하지 않는다."""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    if sys.stdout is not None:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


def load_env(path=ENV_FILE):
    """.env 의 KEY=VALUE 를 환경변수로 (이미 있으면 유지). 키 값은 로그에 남기지 않는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.replace("\t", " ").split(" #")[0].strip().strip('"').strip("'"))   # 값 뒤 ' #메모' 는 주석


def md_tables(path):
    """md 파일의 '## 제목' 별 표를 {제목: [[셀, ...], ...]} 로 읽는다. (머리글·구분선 제외)"""
    tables, section, header_seen = {}, None, False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("## "):
            section, header_seen = s[3:].strip(), False
            tables[section] = []
            continue
        if section is None or not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if not header_seen:
            header_seen = True
            continue
        tables[section].append(cells)
    return tables


def load_settings(path=SETTINGS_FILE):
    """LLM_설정.md → (roles {"생성": cfg, "검토": cfg}, providers, run 설정 dict)."""
    t = md_tables(path)
    roles = {}
    for row in t.get("모델", []):
        if len(row) >= 5:
            roles[row[0]] = {"provider": row[1], "model": row[2],
                             "temperature": float(row[3]) if row[3] else None,
                             "max_tokens": int(row[4] or 8000)}
    providers = {row[0]: {"url": row[1], "key": row[2], "style": row[3]}
                 for row in t.get("provider", []) if len(row) >= 4}
    run = {row[0]: row[1] for row in t.get("실행", []) if len(row) >= 2}
    if "생성" not in roles:
        raise ValueError(f"{path.name} 의 '## 모델' 표에 '생성' 줄이 없습니다")
    return roles, providers, run


def run_int(run, key, default):
    try:
        return int(run.get(key, default))
    except (TypeError, ValueError):
        return default


def role_config(roles, role, override):
    """역할(생성/검토) 설정. 명령줄 'provider:model' 이 있으면 우선. provider 가 none 이면 None."""
    cfg = dict(roles.get(role) or {"temperature": None, "max_tokens": 8000, "model": ""})
    if override:
        provider, _, model = override.partition(":")
        cfg["provider"] = provider
        if model:
            cfg["model"] = model
    if cfg.get("provider") in (None, "", "none", "-"):
        return None
    return cfg


def input_dirs():
    """완성된 05_llm_input 날짜 폴더 (오래된 순). 1.3 이 쓰는 중인 임시 폴더(.<날짜>.tmp…)와
    summary.json(마지막에 씀)이 없는 폴더는 뺀다 → 수집이 도는 중에도 반쯤 쓴 입력을 읽지 않는다."""
    return sorted(d for d in LLM_INPUT_DIR.glob("*") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name)
                  and (d / "batches").is_dir() and (d / "summary.json").exists())


def latest_input_dir(date=None):
    dirs = input_dirs()
    if date:
        dirs = [d for d in dirs if d.name == date]
    return dirs[-1] if dirs else None


def output_dir(date):
    return LLM_OUTPUT_DIR / date


def output_dates():
    """07_llm_output 아래 날짜 폴더 목록 (오래된 순)."""
    if not LLM_OUTPUT_DIR.is_dir():
        return []
    return sorted(d for d in LLM_OUTPUT_DIR.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name))


def rule_hash(r):
    """룰 내용 지문 (원본 룰 텍스트, 공백 정리). 같은 룰 id 라도 rev·조건이 바뀌면 달라진다."""
    return hashlib.sha1(" ".join(r.get("rule", "").split()).encode("utf-8")).hexdigest()[:16]


def generated_rules(exclude=None):
    """다른 날짜 폴더에서 생성(3.2)까지 끝난 룰 → {룰 id: (룰 내용 지문, 날짜)}.

    입력 날짜가 바뀌면(1_0 이 새 05_llm_input/<날짜> 를 만들면) 3.x 는 새 07_llm_output/<날짜> 에서 일한다.
    이때 이미 만든 룰을 다시 만들지 않도록 3.1 이 이 목록을 본다. 생성 응답에서 빠진 룰(rules_missing)은 넣지 않는다.
    exclude: 빼고 볼 날짜 폴더 (지금 작업 중인 폴더 - 그 안의 생성 결과는 3.1 이 따로 본다)
    """
    done = {}
    for base in output_dates():
        if exclude is not None and base.resolve() == Path(exclude).resolve():
            continue
        req_dir, gen_dir = base / REQ_DIR, base / GEN_DIR
        for g in sorted(gen_dir.glob("*.jsonl")) if gen_dir.is_dir() else []:
            req = req_dir / g.name
            if not req.exists():
                continue
            meta = gen_dir / f"{g.stem}.meta.json"
            missing = set(json.loads(meta.read_text(encoding="utf-8")).get("rules_missing", [])) if meta.exists() else set()
            for r in read_jsonl(req):
                if r["id"] not in missing:
                    done[r["id"]] = (rule_hash(r), base.name)
    return done


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")      # PID: 여러 프로세스가 같은 파일을 써도 임시 파일이 겹치지 않게
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    tmp.replace(path)


def fill_prompt(text, run):
    """지시문의 {ANOMALIES}, {NORMAL} 을 LLM_설정.md 의 샘플 수로 바꾼다."""
    return (text.replace("{ANOMALIES}", str(run_int(run, "생성 anomalies 수", 10)))
                .replace("{NORMAL}", str(run_int(run, "생성 normal 수", 5))))


# ════════════════════════════════════════════════════════════════════════
#  LLM 호출
# ════════════════════════════════════════════════════════════════════════

class LLMError(Exception):
    pass


def _post(url, headers, body, timeout):
    """→ (응답 JSON, 응답 헤더 {소문자 이름: 값})."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers=dict(headers, **{"Content-Type": "application/json"}), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")), {k.lower(): v for k, v in r.headers.items()}


class Throttle:
    """API 에 동시에 보내는 요청 수를 속도 제한에 맞춰 스스로 조절한다 (3.2·3.3 이 역할마다 하나씩).
    처음에는 start 개로 시작해 성공할 때마다 1개씩 늘려 최대(ceiling)까지 간다.
    속도 제한(429)·과부하(503)가 오면 반으로 줄이고 2분 동안은 늘리지 않는다 (한꺼번에 여러 개가 와도 30초에 한 번만 줄임)."""

    def __init__(self, ceiling, log, name, start=4):
        self.ceiling = max(1, ceiling)
        self.limit = max(1, min(self.ceiling, start))
        self.active, self.calm_until, self.cut_at = 0, 0.0, 0.0
        self.cv = threading.Condition()
        self.log, self.name = log, name

    def __enter__(self):
        with self.cv:
            while self.active >= self.limit:
                self.cv.wait()
            self.active += 1
        return self

    def __exit__(self, *exc):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()
        return False

    def ok(self):
        with self.cv:
            if self.limit < self.ceiling and time.time() >= self.calm_until:
                self.limit += 1
                self.log.info("  %s 동시 요청 %d → %d", self.name, self.limit - 1, self.limit)
                self.cv.notify_all()

    def limited(self):
        with self.cv:
            now = time.time()
            if self.limit > 1 and now - self.cut_at > 30:
                old, self.limit, self.cut_at = self.limit, max(1, self.limit // 2), now
                self.log.warning("  %s 동시 요청 %d → %d (속도 제한·과부하)", self.name, old, self.limit)
            self.calm_until = now + 120


_LIMITS_SEEN = set()


def _log_limits(cfg, hdr, log):
    """계정의 API 한도 (응답 헤더 x-ratelimit-*)를 처음 한 번 로그에 남긴다 → 동시 요청 수를 정하는 근거."""
    key = f"{cfg['provider']}:{cfg['model']}"
    if key in _LIMITS_SEEN:
        return
    _LIMITS_SEEN.add(key)
    get = lambda k: hdr.get(f"x-ratelimit-{k}")
    if get("limit-requests") or get("limit-tokens"):
        log.info("  API 한도 %s: 분당 요청 %s (남은 %s), 분당 토큰 %s (남은 %s)", key, get("limit-requests") or "-",
                 get("remaining-requests") or "-", get("limit-tokens") or "-", get("remaining-tokens") or "-")
    else:
        log.info("  API 한도 %s: 응답에 한도 정보 없음", key)


RATE_LIMIT_WAITS = 6       # 속도 제한(429)은 '재시도 횟수'와 따로 이만큼 기다렸다 다시 보낸다 (동시 요청 때 흔함)


def _retry_after(e):
    """429 응답의 Retry-After(초). 없거나 이상하면 None."""
    try:
        return max(1, min(120, int(float(e.headers.get("Retry-After", "")))))
    except (TypeError, ValueError, AttributeError):
        return None


def call_llm(cfg, providers, system, user, run, log):
    """한 번 호출해 (응답 텍스트, 사용량 dict) 를 돌려준다. 429·5xx·네트워크 오류는 재시도.
    사용량 dict 의 finish_reason: 'stop' = 정상 종료, 'length' = 출력 한도에서 잘림 (반복 루프 등)."""
    p = providers.get(cfg["provider"])
    if p is None:
        raise LLMError(f"LLM_설정.md 의 provider 표에 '{cfg['provider']}' 가 없습니다")
    gate = cfg.get("throttle") or contextlib.nullcontext()      # 동시 요청 수 자동 조절 (Throttle)
    if p["style"] == "mock":
        with gate:
            time.sleep(float(os.environ.get("RULEDATA_MOCK_DELAY", "0") or 0))   # 동시 실행 시험용 (호출 시간 흉내)
        if cfg.get("throttle"):
            cfg["throttle"].ok()
        return mock_response(user, run), {"input_tokens": 0, "output_tokens": 0, "finish_reason": "stop"}
    key = os.environ.get(p["key"], "").strip()
    if not key:
        raise LLMError(f".env 에 {p['key']} 가 없습니다")
    if not key.isascii() or " " in key:   # 키 뒤에 메모를 붙여 쓴 경우 (HTTP 헤더에 못 넣음)
        raise LLMError(f".env 의 {p['key']} 값에 키가 아닌 글자(한글·공백 등)가 섞여 있습니다 - 메모는 # 주석 줄로 옮기세요")
    retries, timeout = run_int(run, "재시도 횟수", 3), run_int(run, "응답 대기 시간", 300)

    if p["style"] == "openai":
        headers = {"Authorization": f"Bearer {key}"}
        body = {"model": cfg["model"], "messages": [{"role": "system", "content": system},
                                                    {"role": "user", "content": user}]}
        # OpenAI 최신 모델은 max_completion_tokens (추론 토큰 포함), 호환 API(UpStage 등)는 max_tokens
        body["max_completion_tokens" if cfg["provider"] == "openai" else "max_tokens"] = cfg["max_tokens"]
        if cfg.get("temperature") is not None:
            body["temperature"] = cfg["temperature"]
        if cfg.get("reasoning_effort"):           # 추론 모델(gpt-5 등)의 추론 강도 (LLM_설정.md '검토 추론 강도')
            body["reasoning_effort"] = cfg["reasoning_effort"]
    elif p["style"] == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        body = {"model": cfg["model"], "max_tokens": cfg["max_tokens"], "system": system,
                "messages": [{"role": "user", "content": user}]}
        if cfg.get("temperature") is not None:
            body["temperature"] = cfg["temperature"]
    else:
        raise LLMError(f"알 수 없는 형식: {p['style']}")

    last, attempt, limited = None, 0, 0
    while True:
        try:
            with gate:
                data, hdr = _post(p["url"], headers, body, timeout)
            if cfg.get("throttle"):
                cfg["throttle"].ok()
            _log_limits(cfg, hdr, log)
            if p["style"] == "openai":
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
                u = data.get("usage", {})
                return text, {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                              "finish_reason": choice.get("finish_reason") or "stop"}
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            u = data.get("usage", {})
            return text, {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                          "finish_reason": "length" if data.get("stop_reason") == "max_tokens" else "stop"}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code not in (408, 409, 429) and e.code < 500:
                raise LLMError(f"HTTP {e.code} ({cfg['provider']}:{cfg['model']}): {detail}") from None
            last = f"HTTP {e.code}: {detail}"
            if e.code in (429, 503, 529) and cfg.get("throttle"):
                cfg["throttle"].limited()               # 동시 요청을 줄인다
            if e.code == 429 and limited < RATE_LIMIT_WAITS:
                limited += 1
                wait = _retry_after(e) or min(60, 15 * limited)
                log.warning("  속도 제한(429) - %ds 후 다시 보냄 (%d/%d, 자주 나오면 LLM_설정.md 의 동시 요청 수를 줄이세요)",
                            wait, limited, RATE_LIMIT_WAITS)
                time.sleep(wait + random.uniform(0, 3))   # 동시에 기다린 요청들이 한꺼번에 다시 몰리지 않게
                continue
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
            last = str(e)
        attempt += 1
        if attempt >= retries:
            break
        wait = 10 * attempt
        log.warning("  LLM 호출 실패 (%d/%d, %ds 후 재시도): %s", attempt, retries, wait, last)
        time.sleep(wait)
    raise LLMError(f"LLM 호출 실패 ({cfg['provider']}:{cfg['model']}): {last}")


def add_usage(usage, role, u):
    usage.setdefault(role, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    usage[role]["calls"] += 1
    usage[role]["input_tokens"] += u.get("input_tokens", 0) or 0
    usage[role]["output_tokens"] += u.get("output_tokens", 0) or 0


def mock_response(user, run):
    """키 없이 흐름을 시험하는 가짜 응답. 생성 요청이면 룰 조건으로 샘플을, 검토 요청이면 전부 pass."""
    items = [json.loads(l) for l in user.splitlines() if l.startswith("{")]
    out = []
    n_a = run_int(run, "생성 anomalies 수", 10)
    for item in items:
        if "generated" in item:                               # 검토 요청
            out.append({"id": item["rule"]["id"], "attack_type": item["generated"]["attack_type"],
                        "reviews": [{"i": s["i"], "verdict": "pass"} for s in item["generated"]["samples"]]})
            continue
        r = item                                              # 생성 요청
        val = lambda where: [unescape_value(c["value"]) for c in r["conditions"]
                             if c["type"] == "content" and c["in"] in where and not c.get("negated")]
        uri = "".join(val(("uri", "raw_uri")))
        uri = uri if uri.startswith("/") else "/" + uri
        hdr, body = val(("header",)), "".join(val(("body",)))
        method = r["methods"][0] if r["methods"] != ["ANY"] else r.get("method_hint", ["GET"])[0]
        method = "POST" if body else method
        samples = []
        for k in range(n_a):
            q = ("&" if "?" in uri else "?") + f"mock={k}"
            msg = f"{method} {uri}{q} HTTP/1.1\nHost: 192.168.20.20\n" + "".join(h + "\n" for h in hdr)
            msg += (f"Content-Length: {len(body)}\n\n{body}" if body else "\n")
            samples.append({"label": "anomalies", "attack_type": r["attack_type"], "http": msg, "note": "mock"})
        samples.append({"label": "normal", "attack_type": "none",
                        "http": f"{method} /index.html HTTP/1.1\nHost: 192.168.20.20\n\n", "note": "mock"})
        out.append({"id": r["id"], "attack_type": r["attack_type"], "samples": samples})
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in out)


_JSON_ESC = re.compile(r'\\(\\|x([0-9A-Fa-f]{2})|.)', re.S)


def _fix_json_escapes(s):
    """JSON 에 없는 이스케이프를 고친다. 모델이 바이너리를 \\x03 처럼 쓰면 객체 전체가 깨지기 때문.
    \\xHH → \\u00HH (같은 바이트), 그 밖의 잘못된 \\c → 글자 그대로 (\\\\c). 올바른 이스케이프는 그대로 둔다."""
    def fix(m):
        if m.group(1) == "\\" or m.group(1) in '"/bfnrtu':
            return m.group(0)
        if m.group(2):
            return "\\u00" + m.group(2)
        return "\\\\" + m.group(1)
    return _JSON_ESC.sub(fix, s)


def parse_json_lines(text):
    """LLM 응답에서 JSON 객체들을 꺼낸다.

    허용: ``` 코드 블록, 배열 하나, 한 줄에 객체 하나(JSON Lines), 객체 하나가 여러 줄에 걸친 형태(들여쓰기),
          앞뒤 설명 문장. 문자열 안의 '{' 는 건너뛴다 (json 해석기로 객체 단위로 읽음).
    """
    s = _fix_json_escapes(re.sub(r"```(?:json|jsonl)?", "", text).strip())
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else [obj]
    except ValueError:
        pass
    dec = json.JSONDecoder()
    out, i = [], 0
    while True:
        i = s.find("{", i)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(s, i)      # i 위치부터 객체 하나를 끝까지 읽는다 (여러 줄 가능)
            if isinstance(obj, dict):
                out.append(obj)
            i = end
        except ValueError:
            i += 1                               # 객체 시작이 아니면 다음 '{' 로
    return out


# 긴 반복 표기 (LLM_프롬프트.md 생성 규칙 11): 생성 모델이 '%00%00…', '..\..\…' 를 직접 반복하다 멈추지 못하고
# 출력 한도까지 가는 '반복 루프'가 잦아서(파일럿에서 요청의 약 30%), 긴 반복은 {{REPEAT:문자열:횟수}} 로 쓰게 하고 코드가 펼친다.
REPEAT_RE = re.compile(r"\{\{REPEAT:(.{1,40}?):(\d{1,6})\}\}", re.S)
REPEAT_MAX_CHARS = 20000          # 한 표기를 펼친 최대 길이 (과도한 값 방지)


def expand_repeats(text):
    """'{{REPEAT:A:2000}}' → 'A' 2000개. 표기가 없으면 그대로."""
    if not isinstance(text, str) or "{{REPEAT:" not in text:
        return text
    def rep(m):
        unit = m.group(1)
        return unit * min(int(m.group(2)), REPEAT_MAX_CHARS // len(unit))
    return REPEAT_RE.sub(rep, text)


def repeat_tail(text, span=240, max_unit=8):
    """응답이 짧은 문자열의 반복으로 끝났는지 (반복 루프로 출력 한도까지 간 경우). 반복 단위 또는 None."""
    t = (text or "").rstrip()
    if len(t) < span:
        return None
    seg = t[-span:]
    for k in range(1, max_unit + 1):
        if all(seg[i] == seg[i - k] for i in range(k, span)):
            start = len(t) - span                 # 반복이 시작된 곳을 찾아 그 순서대로 (예: '00%' 가 아니라 '%00')
            while start > 0 and t[start - 1] == t[start - 1 + k]:
                start -= 1
            return t[start:start + k]
    return None


def parallel_settings(run):
    """동시 처리 설정 (LLM_설정.md '## 실행'): (생성 동시 요청 수, 검토 동시 요청 수, 생성·검토 동시 진행)."""
    return (max(1, min(16, run_int(run, "생성 동시 요청 수", 10))), max(1, min(32, run_int(run, "검토 동시 요청 수", 12))),
            run_yes(run, "생성·검토 동시 진행"))


# ════════════════════════════════════════════════════════════════════════
#  코드 검사 (local_check)
# ════════════════════════════════════════════════════════════════════════

def unescape_value(v):
    """LLM 입력의 '\\xNN' 표기를 실제 문자로."""
    return re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), v)


def split_http(http):
    """HTTP 메시지 → 검사 위치('in') 별 텍스트."""
    s = http.replace("\r\n", "\n")
    head, _, body = s.partition("\n\n")
    lines = head.split("\n")
    first = lines[0].split(" ")
    is_resp = lines[0].startswith("HTTP/")
    headers = "\n".join(lines[1:])
    cookie = "\n".join(l.split(":", 1)[1].strip() for l in lines[1:] if l.lower().startswith("cookie:"))
    parts = {
        "method": first[0] if not is_resp else "",
        "uri": first[1] if len(first) > 1 and not is_resp else "",
        "header": headers, "cookie": cookie, "body": body,
        "status_code": first[1] if is_resp and len(first) > 1 else "",
        "status_msg": " ".join(first[2:]) if is_resp else "",
        "response_body": body if is_resp else "",
        "payload": s,
    }
    parts["raw_uri"], parts["raw_header"], parts["raw_cookie"] = parts["uri"], parts["header"], parts["cookie"]
    return parts


def _variants(text, where):
    """URL 디코딩 값(Snort 는 http_uri 를 정규화해서 검사)과 CRLF 버전(실제 패킷 줄바꿈)도 함께 본다."""
    out = [text]
    if "\n" in text:
        out.append(text.replace("\n", "\r\n"))
    if where in ("uri", "body", "cookie", "payload"):
        for f in (unquote, unquote_plus):
            try:
                d = f(text)
                if d not in out:
                    out.append(d)
            except Exception:
                pass
    return out


@functools.lru_cache(maxsize=None)
def _compile(pattern, flags):
    """정규식 컴파일 캐시 (룰 수만 개를 반복 대조할 때 re 기본 캐시(512개)로는 부족). 오류면 None."""
    try:
        return re.compile(pattern, flags)
    except re.error:
        return None


# 역추적이 폭발할 수 있는 모양의 정규식 (2026-10-26 추가). 예: et:1:2072556 의 (?:Content-Disposition\x3a\x20form-data.{,4096}){100,}
# 는 form-data 부분이 몰린 요청(35개)에서 경우의 수가 2^35 쯤 되어 끝나지 않아 3.1a 가 멈췄다. Snort 는 PCRE 검사 한도로 멈추지만
# 파이썬 re 에는 한도가 없다. 그래서 '반복 안의 반복'·'반복 안의 선택(|)' 이 있는 정규식(지금 룰의 약 7.5%)만 시간 제한이 되는
# regex 모듈(pip install regex)로 검사하고, 제한 시간을 넘으면 그 조건은 판단 불가(None, 정규식을 해석하지 못할 때와 같음)로 둔다.
# 그 밖의 정규식은 지금처럼 re 로 검사한다 (결과 그대로). regex 모듈이 없으면 위험한 모양의 정규식은 검사하지 않고 판단 불가.
try:
    import regex as _regex
except ImportError:
    _regex = None
REGEX_TIMEOUT = 1.0                         # 위험한 모양 정규식의 검사 1번 최대 시간(초)
REGEX_TIMEOUTS = collections.Counter()      # 시간 제한으로 판단 불가가 된 정규식 → 횟수 (단계가 끝날 때 로그)
_SKIP = object()


def _risky(items, in_rep=False):
    """정규식 구문 트리에 반복 안의 반복, 또는 반복 안의 선택(|)이 있는지."""
    reps = (_sre_c.MAX_REPEAT, _sre_c.MIN_REPEAT, getattr(_sre_c, "POSSESSIVE_REPEAT", None))
    for op, av in items:
        if op in reps:
            many = av[1] == _sre_c.MAXREPEAT or av[1] > 1
            if (many and in_rep) or _risky(av[2], in_rep or many):
                return True
        elif op is _sre_c.SUBPATTERN:
            if _risky(av[-1], in_rep):
                return True
        elif op is _sre_c.BRANCH:
            if in_rep or any(_risky(b, in_rep) for b in av[1]):
                return True
        elif op in (_sre_c.ASSERT, _sre_c.ASSERT_NOT):
            if _risky(av[1], in_rep):
                return True
        elif op is getattr(_sre_c, "ATOMIC_GROUP", None):
            if _risky(av, in_rep):
                return True
    return False


@functools.lru_cache(maxsize=None)
def _guarded(pattern, flags):
    """위험한 모양이면 시간 제한 검사기(regex 모듈), regex 모듈이 없으면 _SKIP, 아니면 None (re 로 검사)."""
    try:
        if not _risky(_sre_parse.parse(pattern, flags).data):
            return None
    except Exception:
        return None
    if _regex is None:
        return _SKIP
    fl = _regex.V0
    for a, b in ((re.I, _regex.I), (re.S, _regex.S), (re.M, _regex.M), (re.X, _regex.X)):
        if flags & a:
            fl |= b
    try:
        return _regex.compile(pattern, fl)
    except Exception:
        return _SKIP


def regex_search(rx, pattern, flags, text):
    """정규식 검사 한 번 → 맞으면 True, 아니면 False, 시간 제한을 넘었거나 검사하지 않으면 None."""
    g = _guarded(pattern, flags)
    if g is None:
        return rx.search(text) is not None
    if g is _SKIP:
        return None
    try:
        return g.search(text, timeout=REGEX_TIMEOUT) is not None
    except TimeoutError:
        REGEX_TIMEOUTS[pattern] += 1
        return None


def check_condition(c, parts):
    """조건 하나를 만족하면 True, 아니면 False, 판단 불가(정규식 오류·검사 시간 초과)면 None."""
    where = c.get("in", "payload")
    texts = _variants(parts.get(where, parts["payload"]), where)
    if c["type"] == "content":
        v = unescape_value(c.get("value", ""))
        found = any((v.lower() in t.lower()) if c.get("nocase") else (v in t) for t in texts)
    else:
        flags = 0
        for ch, fl in (("i", re.I), ("s", re.S), ("m", re.M), ("x", re.X)):
            if ch in c.get("flags", ""):
                flags |= fl
        rx = _compile(c["pattern"], flags)
        if rx is None:
            return None
        found, unknown = False, False
        for t in texts:
            r = regex_search(rx, c["pattern"], flags, t)
            if r:
                found = True
                break
            unknown = unknown or r is None
        if not found and unknown:
            return None
    return (not found) if c.get("negated") else found


def local_check(sample, rule):
    """샘플 검사 → {"result": "ok"|"fail"|"unknown", "reasons": [...]}

    anomalies: 모든 조건을 만족해야 ok. normal: 조건을 하나라도 어겨야 ok (모두 만족하면 탐지될 수 있으므로 fail).
    위치(depth·distance 등)는 보지 않는 근사 검사.
    """
    http = sample.get("http", "")
    if "|||" in http:
        return {"result": "fail", "reasons": ["'|||' 포함 (로그 구분자)"]}
    if not re.match(r"^([A-Z][A-Z\-_]*) \S+ HTTP/\d|^HTTP/\d", http.lstrip()):
        return {"result": "fail", "reasons": [BAD_LINE]}
    parts = split_http(http)
    results = [(c, check_condition(c, parts)) for c in rule.get("conditions", [])]
    failed = [c for c, ok in results if ok is False]
    unknown = [c for c, ok in results if ok is None]
    if sample.get("label") == "anomalies":
        if failed:
            return {"result": "fail", "reasons": [
                f"조건 불일치: {c['type']} in {c['in']} {'(negated) ' if c.get('negated') else ''}"
                f"{(c.get('value') or c.get('pattern', ''))[:60]}" for c in failed]}
        return {"result": "unknown" if unknown else "ok", "reasons": ["정규식 해석 불가"] if unknown else []}
    if results and not failed and not unknown:
        return {"result": "fail", "reasons": ["normal 샘플이 룰 조건을 모두 만족 (탐지될 수 있음)"]}
    if not failed and unknown:   # 어긴 조건이 확인되지 않음 → 룰이 정상이라고 하는지 알 수 없음
        return {"result": "unknown", "reasons": ["정규식 해석 불가"]}
    return {"result": "ok", "reasons": []}


def valid_sample(s):
    return isinstance(s, dict) and s.get("label") in LABELS and isinstance(s.get("http"), str) and s["http"].strip()


BAD_LINE = "HTTP 요청/응답 줄 형식이 아님"
_INVISIBLE = dict.fromkeys(map(ord, "﻿​‌‍⁠"), None)
_REQ_LINE_LOOSE = re.compile(r"^([A-Za-z]{3,10})\s+(.*?)\s*(?:%20|\s)(HTTP/\d(?:\.\d)?)\s*$")


def repair_http(http):
    """요청 줄만 고친다. 생성 모델이 URI 에 공백을 그대로 쓰거나(`?q=<SCRIPT SRC=…>`), 보이지 않는 글자(BOM 등)를 넣거나,
    마지막 공백까지 인코딩(`…%20HTTP/1.1`)해 요청 줄 형식이 깨진 경우 → 실제 클라이언트처럼 URI 의 공백을 %20 으로."""
    head, sep, rest = http.partition("\n")
    m = _REQ_LINE_LOOSE.match(head.translate(_INVISIBLE).replace(" ", " ").strip())
    if not m:
        return http
    method, uri, ver = m.groups()
    fixed = f"{method} {uri.strip().replace(' ', '%20')} {ver}"
    return fixed + sep + rest if fixed != head else http


def repair_sample(s, rule):
    """요청 줄 형식 때문에 코드 검사에 실패하는 샘플만 고친다 (고친 뒤 형식이 맞을 때만, note 에 표시)."""
    if not valid_sample(s) or BAD_LINE not in local_check(s, rule).get("reasons", []):
        return s
    fixed = dict(s, http=repair_http(s["http"]))
    if fixed["http"] == s["http"] or BAD_LINE in local_check(fixed, rule).get("reasons", []):
        return s
    return dict(fixed, note=(str(s.get("note") or "") + " [요청 줄 자동 수정]").strip())


def norm_payload(p):
    """기존 로그 payload(줄바꿈 = 두 글자 '\\n') 와 생성 샘플(실제 줄바꿈)을 같은 형태로 맞춘다."""
    return p.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n").strip()


def payload_hash(p):
    """payload 지문 (기존 학습 데이터와 '똑같은' 샘플인지 비교할 때 사용)."""
    return hashlib.sha1(norm_payload(p).encode("utf-8", "replace")).hexdigest()[:16]


def load_existing_hashes(base):
    """3.1a 가 만든 기존 학습 로그 payload 지문 집합. 없으면 빈 집합."""
    f = Path(base) / "_0_existing" / "payload_hashes.txt"
    return set(f.read_text(encoding="utf-8").split()) if f.exists() else set()


def stable_seed(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest(), 16)


# ── 정규식 예시 (생성 모델이 정규식을 잘못 읽지 않게) ─────────────────────
# 생성 실패의 큰 부분이 정규식 오해였다 (예: XSS 룰 52개의 `<.+\/script` 는 보통의 `</script>` 와 맞지 않고,
# RFI 룰 58개의 `=\s*(https?|ftps?|php)\:\/` 에 `ftp:\\`·`data://` 를 씀). 그래서 정규식마다 '실제로 맞는' 짧은 문자열을
# 코드로 만들어(정규식 구문 트리를 따라 생성 → re.search 로 확인) 생성 요청에 붙인다 (3_2 의 regex_examples).

try:                                                    # Python 3.11+
    import re._parser as _sre_parse
    import re._constants as _sre_c
except ImportError:                                     # 이전 버전
    import sre_parse as _sre_parse
    import sre_constants as _sre_c

_NICE = "abcdefghijklmnopqrstuvwxyz0123456789"
_PUNCT = "-_.~=&;:/?%+,!*'()@$#[]<>\"{}|\\^` "
_SPACE = " \t\n\r\f\v"
_MAX_GEN = 6000                                         # 만들 문자열의 최대 길이 (그 이상은 포기)


class _NoExample(Exception):
    pass


def _cat_has(cat, ch):
    c = _sre_c
    word = ch.isalnum() or ch == "_"
    table = {c.CATEGORY_DIGIT: ch.isdigit(), c.CATEGORY_NOT_DIGIT: not ch.isdigit(),
             c.CATEGORY_SPACE: ch in _SPACE, c.CATEGORY_NOT_SPACE: ch not in _SPACE,
             c.CATEGORY_WORD: word, c.CATEGORY_NOT_WORD: not word,
             c.CATEGORY_LINEBREAK: ch == "\n", c.CATEGORY_NOT_LINEBREAK: ch != "\n"}
    if cat not in table:
        raise _NoExample(cat)
    return table[cat]


def _in_class(ch, items):
    neg, hit = False, False
    for op, av in items:
        if op == _sre_c.NEGATE:
            neg = True
        elif op == _sre_c.LITERAL:
            hit = hit or ord(ch) == av
        elif op == _sre_c.RANGE:
            hit = hit or av[0] <= ord(ch) <= av[1]
        elif op == _sre_c.CATEGORY:
            hit = hit or _cat_has(av, ch)
        else:
            raise _NoExample(op)
    return hit != neg


def _pick_class(items, rng, rand):
    """문자 집합([...], \\d 등)에서 글자 하나. 사람이 읽기 쉬운 글자(영문·숫자)를 먼저."""
    listed = [chr(av) for op, av in items if op == _sre_c.LITERAL]
    listed += [chr(av[0]) for op, av in items if op == _sre_c.RANGE]
    pool = [ch for ch in dict.fromkeys(listed + list(_NICE) + list(_PUNCT)) if _in_class(ch, items)]
    if not pool:
        raise _NoExample("빈 문자 집합")
    nice = [ch for ch in pool if ch.isalnum()] or pool
    return rng.choice(nice) if rand else pool[0]


def _is_space(data):
    """반복 대상이 공백 하나(\\s, ' ', [\\s...] 처럼 공백만 허용)인지."""
    if len(data) != 1:
        return False
    op, av = data[0]
    if op == _sre_c.LITERAL:
        return chr(av) in _SPACE
    if op == _sre_c.IN:
        try:
            return all(not _in_class(ch, av) for ch in _NICE) and any(_in_class(ch, av) for ch in _SPACE)
        except _NoExample:
            return False
    return False


def _gen(data, rng, rand, groups, depth=0):
    if depth > 60:
        raise _NoExample("너무 깊음")
    out = []
    for op, av in data:
        if op == _sre_c.LITERAL:
            out.append(chr(av))
        elif op == _sre_c.NOT_LITERAL:
            out.append("x" if av != ord("x") else "y")
        elif op == _sre_c.ANY:
            out.append(rng.choice(_NICE) if rand else "x")
        elif op == _sre_c.IN:
            out.append(_pick_class(av, rng, rand))
        elif op == _sre_c.BRANCH:
            alts = av[1]
            out.append(_gen((rng.choice(alts) if rand else alts[0]).data, rng, rand, groups, depth + 1))
        elif op == _sre_c.SUBPATTERN:
            s = _gen(av[-1].data, rng, rand, groups, depth + 1)
            if av[0]:
                groups[av[0]] = s
            out.append(s)
        elif op in (_sre_c.MAX_REPEAT, _sre_c.MIN_REPEAT) or op == getattr(_sre_c, "POSSESSIVE_REPEAT", None):
            lo, hi, sub = av
            if lo > _MAX_GEN:
                raise _NoExample("반복이 너무 큼")
            if _is_space(sub.data):
                n = lo                                         # 없어도 되는 공백은 넣지 않음 (URI 에 공백이 들어가지 않게)
            elif not rand:
                n = lo if lo else (0 if hi == 1 else 1)        # ? 는 빼고, * 는 한 번
            elif lo == 0 and hi == 1:
                n = rng.randint(0, 1)
            else:
                n = min(hi, lo + rng.randint(0 if lo else 1, 2))
            out.append("".join(_gen(sub.data, rng, rand, groups, depth + 1) for _ in range(n)))
        elif op == _sre_c.AT:
            pass                                               # ^ $ \b 등 위치 표시: 글자를 만들지 않음
        elif op == _sre_c.GROUPREF:
            out.append(groups.get(av, ""))
        elif op == _sre_c.GROUPREF_EXISTS:
            group, yes, no = av
            branch = yes if group in groups else no
            out.append(_gen(branch.data, rng, rand, groups, depth + 1) if branch is not None else "")
        elif op in (_sre_c.ASSERT, _sre_c.ASSERT_NOT):
            pass                                               # 앞뒤 확인(lookaround): 만든 뒤 re.search 로 확인
        elif op == getattr(_sre_c, "ATOMIC_GROUP", None):
            out.append(_gen(av.data, rng, rand, groups, depth + 1))
        else:
            raise _NoExample(op)
        if sum(map(len, out)) > _MAX_GEN:
            raise _NoExample("너무 김")
    return "".join(out)


_RUN = re.compile(r"(.{1,8}?)\1{19,}", re.S)


def compact_repeats(s):
    """같은 문자열이 20번 넘게 이어지면 {{REPEAT:문자열:횟수}} 로 줄인다 (생성 규칙 11 과 같은 표기)."""
    return _RUN.sub(lambda m: "{{REPEAT:%s:%d}}" % (m.group(1), len(m.group(0)) // len(m.group(1))), s)


@functools.lru_cache(maxsize=None)
def regex_examples(pattern, flags="", n=3):
    """이 정규식에 실제로 맞는 짧은 문자열 n개 (코드로 만들고 re.search 로 확인, 긴 반복은 {{REPEAT}} 표기).
    첫 번째는 가장 짧은 선택, 나머지는 무작위 선택 (룰마다 같은 결과). 만들 수 없으면 빈 tuple."""
    fl = sum(f for ch, f in (("i", re.I), ("s", re.S), ("m", re.M), ("x", re.X)) if ch in flags)
    rx = _compile(pattern, fl)
    if rx is None:
        return ()
    try:
        tree = _sre_parse.parse(pattern, fl)
    except Exception:
        return ()
    rng = random.Random(stable_seed("regex_examples", pattern, flags))
    found = []
    for k in range(16):
        try:
            s = _gen(tree.data, rng, k > 0, {})
        except (_NoExample, RecursionError, ValueError, OverflowError):
            if k == 0:
                continue
            break
        if s and s not in found and regex_search(rx, pattern, fl, s):
            found.append(s)
            if len(found) >= n:
                break
    out = [compact_repeats(s) for s in found]
    return tuple(s for s in out if len(s) <= 240)


def rule_regex_examples(rule, n=3):
    """룰의 정규식 조건마다 예시 → [{"cond": 조건 번호, "in": 위치, "examples": [...]}] (없으면 [])."""
    out = []
    for i, c in enumerate(rule.get("conditions", [])):
        if c.get("type") == "regex" and not c.get("negated") and c.get("pattern"):
            ex = regex_examples(c["pattern"], c.get("flags", ""), n)
            if ex:
                out.append({"cond": i, "in": c.get("in", "payload"), "examples": list(ex)})
    return out


# ── 여러 룰 한꺼번에 대조 (3.1a·3.4·4.1 공용) ─────────────────────────────

# 코드 검사가 평가하지 못하는 옵션: content·정규식이 맞아도 실제 탐지는 '가능성' 으로만 본다
UNCHECKED_RE = re.compile(r"\b(byte_test|byte_jump|byte_extract|byte_math|isdataat|dsize|urilen|"
                          r"detection_filter|stream_size)\s*:")
_PORT_VARS_OK = ("any", "$HTTP_PORTS", "$FILE_DATA_PORTS")


def dst_port_ok(raw, ports):
    """룰의 목적지 포트가 ports(기본 80, 8080)를 포함하는지. 변수·any 면 포함으로 본다."""
    head = raw.lstrip("# ").split("(", 1)[0].split()
    if len(head) < 7:
        return True
    dport = head[6]
    if any(v in dport for v in _PORT_VARS_OK) or dport.startswith("$"):
        return not dport.startswith("!")
    nums = set()
    for part in dport.strip("[]").split(","):
        part = part.strip()
        if ":" in part:                                    # 범위 1024:65535
            lo_, _, hi_ = part.partition(":")
            lo_ = int(lo_) if lo_.isdigit() else 0
            hi_ = int(hi_) if hi_.isdigit() else 65535
            if any(lo_ <= p <= hi_ for p in ports):
                return True
        elif part.isdigit():
            nums.add(int(part))
    return bool(nums & set(ports))


class RuleMatcher:
    """HTTP 하나를 룰 수만 개와 빠르게 대조한다.

    룰마다 가장 긴 content(4자 이상) 안의 '가장 드문 4글자' 로 색인 → 그 4글자가 있는 룰만 후보 →
    content 가 모두 있는지 먼저 보고 → 전체 조건 검사. content 가 없는(정규식만 있는) 룰은 매번 전부 검사한다.
    match() 결과: [(룰 id, "match" | "possible")]
      match    = 모든 조건 만족
      possible = 어긴 조건은 없지만 확인 못 한 부분이 있음 (정규식 해석 불가, byte_test 등 평가 못 하는 옵션)
    앞선 요청이 필요한 룰(flowbits)과 목적지 포트가 ports 를 포함하지 않는 룰은 뺀다.
    """

    def __init__(self, rules, ports=(80, 8080)):
        self.rules, self.rest = [], []
        gram_freq = collections.Counter()
        for r in rules:
            if r.get("requires_state") or not r.get("conditions") or not dst_port_ok(r.get("rule", ""), ports):
                continue
            need = [unescape_value(c["value"]).lower() for c in r["conditions"]
                    if c["type"] == "content" and not c.get("negated") and c.get("in") != "method"]
            longs = [v for v in need if len(v) >= 4]
            item = {"id": r["id"], "conditions": r["conditions"], "need": need,
                    "partial": bool(UNCHECKED_RE.search(r.get("rule", ""))),
                    "anchor": max(longs, key=len) if longs else None}
            self.rules.append(item)
            if item["anchor"]:
                a = item["anchor"]
                gram_freq.update({a[k:k + 4] for k in range(len(a) - 3)})
        self.by_gram = collections.defaultdict(list)
        for i, it in enumerate(self.rules):
            a = it["anchor"]
            if a is None:
                self.rest.append(i)
                continue
            self.by_gram[min((a[k:k + 4] for k in range(len(a) - 3)), key=lambda x: gram_freq[x])].append(i)

    def match(self, http, exclude=()):
        text = http.replace("\r\n", "\n")
        low = text.lower()
        try:
            low2 = unquote_plus(low)
        except Exception:
            low2 = low
        grams = {low[k:k + 4] for k in range(len(low) - 3)}
        if low2 != low:
            grams |= {low2[k:k + 4] for k in range(len(low2) - 3)}
        cands = set(self.rest)
        for g in grams & self.by_gram.keys():
            cands.update(self.by_gram[g])
        parts, out = None, []
        for i in cands:
            it = self.rules[i]
            if it["id"] in exclude or not all(v in low or v in low2 for v in it["need"]):
                continue
            if parts is None:
                parts = split_http(text)
            res = [check_condition(c, parts) for c in it["conditions"]]
            if False in res:
                continue
            out.append((it["id"], "possible" if (None in res or it["partial"]) else "match"))
        return out


# ── 헤더 현실화 (3.2·3.3·3.4 공용) ──────────────────────────────────────
# 검토 의견(2026-10-16): 샘플이 짧고 User-Agent 가 실제 같지 않음 (`Mozilla/5.0` 만, `curl/8.0` 처럼 없는 버전).
# 지시문으로 먼저 막고, 남은 것은 여기서 실제 클라이언트 값으로 고친다. 룰이 검사하는 헤더는 건드리지 않는다.

# (가중치, 플랫폼, 형식) - 실제 브라우저의 UA 형식 (Chrome 110 이후는 'NNN.0.0.0' 축약 형식이 실제 값)
_BROWSER_TEMPLATES = [
    (30, "windows", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{c}.0.0.0 Safari/537.36"),
    (10, "windows", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{c}.0.0.0 Safari/537.36 Edg/{c}.0.{eb}.{ep}"),
    (6, "windows", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{c}.0.0.0 Whale/3.{w}.{wb}.{wp} Safari/537.36"),
    (6, "windows", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{f}.0) Gecko/20100101 Firefox/{f}.0"),
    (8, "mac", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{c}.0.0.0 Safari/537.36"),
    (6, "mac", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{s} Safari/605.1.15"),
    (3, "mac", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:{f}.0) Gecko/20100101 Firefox/{f}.0"),
    (4, "linux", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{c}.0.0.0 Safari/537.36"),
    (2, "linux", "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:{f}.0) Gecko/20100101 Firefox/{f}.0"),
    (10, "android", "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{c}.0.0.0 Mobile Safari/537.36"),
    (6, "android", "Mozilla/5.0 (Linux; Android {a}; {m}) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/{sb}.0 Chrome/{sc}.0.0.0 Mobile Safari/537.36"),
    (9, "iphone", "Mozilla/5.0 (iPhone; CPU iPhone OS {i} like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{iv} Mobile/15E148 Safari/604.1"),
]
_MOBILE_MODELS = ["SM-S918N", "SM-S921N", "SM-S911N", "SM-S926N", "SM-A546S", "SM-F946N"]
_IOS = [("16_6", "16.6"), ("17_3_1", "17.3.1"), ("17_4_1", "17.4.1"), ("17_5_1", "17.5.1"), ("18_0", "18.0")]
_SAMSUNG = [(23, 115), (24, 117), (25, 121)]          # SamsungBrowser 버전 ↔ 기반 Chrome 버전
TOOL_UAS = ["curl/7.88.1", "curl/8.5.0", "curl/8.7.1", "python-requests/2.31.0", "python-requests/2.28.2",
            "Go-http-client/1.1", "Wget/1.21.4", "okhttp/4.12.0", "Java/17.0.10", "axios/1.6.8",
            "Apache-HttpClient/4.5.14 (Java/17.0.10)", "PostmanRuntime/7.37.3"]


def browser_ua(rng, hint="", desktop=False):
    """실제 형식의 브라우저 UA 하나. hint(windows·mac·linux·android·iphone)가 있으면 그 플랫폼에서,
    desktop=True 면 PC 브라우저에서만 고른다 (API·JSON 요청에 모바일 UA 는 어색함)."""
    pool = [t for t in _BROWSER_TEMPLATES if (not hint or t[1] == hint)
            and (hint or not desktop or t[1] in ("windows", "mac", "linux"))] or _BROWSER_TEMPLATES
    tpl = rng.choices([t[2] for t in pool], weights=[t[0] for t in pool])[0]
    i, iv = rng.choice(_IOS)
    sb, sc = rng.choice(_SAMSUNG)
    return tpl.format(c=rng.randint(116, 126), eb=rng.randint(2045, 2592), ep=rng.randint(40, 140),
                      w=rng.randint(24, 26), wb=rng.randint(200, 300), wp=rng.randint(10, 30),
                      f=rng.randint(115, 128), s=rng.choice(["16.6", "17.3.1", "17.4.1", "17.5", "18.0"]),
                      a=rng.choice([13, 14]), m=rng.choice(_MOBILE_MODELS), sb=sb, sc=sc, i=i, iv=iv)


_PLATFORM_HINTS = [("iphone", "iphone"), ("ipad", "iphone"), ("android", "android"), ("windows", "windows"),
                   ("macintosh", "mac"), ("mac os", "mac"), ("x11", "linux"), ("linux", "linux")]
_TOOL_FIX = [(re.compile(rf"^(?:{p})$", re.I), c) for p, c in [   # 버전이 잘렸거나 없는 도구 UA → 실제 문자열
    (r"curl(?:/\d+(?:\.\d+)?)?", ["curl/7.88.1", "curl/8.5.0", "curl/8.7.1"]),
    (r"python-requests(?:/\d+(?:\.(?:\d+|x))?)?", ["python-requests/2.31.0", "python-requests/2.28.2"]),
    (r"python(?:/[\d.]+)?|python-urllib(?:/\d+)?", ["Python-urllib/3.11", "Python-urllib/3.10"]),
    (r"wget(?:/\d+(?:\.\d+)?)?", ["Wget/1.21.4", "Wget/1.21.3"]),
    (r"go-http-client(?:/\d)?", ["Go-http-client/1.1"]),
    (r"okhttp(?:/\d+(?:\.\d+)?)?", ["okhttp/4.12.0"]),
    (r"java(?:/\d+(?:\.\d+)?)?", ["Java/17.0.10", "Java/11.0.22"]),
    (r"axios(?:/\d+(?:\.\d+)?)?", ["axios/1.6.8"]),
    (r"postman(?:runtime)?(?:/\d+(?:\.\d+)?)?", ["PostmanRuntime/7.37.3"]),
    (r"sqlmap(?:/[\w.#]+)?", ["sqlmap/1.7.2#stable (https://sqlmap.org)", "sqlmap/1.8#stable (https://sqlmap.org)"]),
    (r"nikto(?:[/ ][\d.]+)?", ["Mozilla/5.00 (Nikto/2.1.6) (Evasions:None) (Test:Port Check)"]),
    (r"nmap(?:[/ ][\d.]+)?(?: scripting engine)?", ["Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org/book/nse.html)"]),
    (r"masscan(?:/[\d.]+)?", ["masscan/1.3 (https://github.com/robertdavidgraham/masscan)"]),
]]
_MOZ_ONLY = re.compile(r"^mozilla/\d\.\d+\s*(?:\(([^)]*)\))?\s*$", re.I)
_BROWSER_UA_RE = re.compile(r"^Mozilla/5\.0 \((?:Windows|Macintosh|X11|Linux|iPhone|iPad)[^)]*\) (?:AppleWebKit|Gecko)", re.I)


def real_ua(ua, rng, desktop=False):
    """잘렸거나 없는 버전의 UA → 실제 문자열. 이미 실제 형식이거나 모르는 UA(도구·봇 고유 문자열)는 그대로."""
    s = ua.strip()
    m = _MOZ_ONLY.match(s)
    if m:
        inside = (m.group(1) or "").lower()
        if "compatible" in inside:              # (compatible; Nmap ...), (compatible; MSIE ...) 는 실제 UA
            return ua
        hint = next((h for k, h in _PLATFORM_HINTS if k in inside), None)
        if not inside or hint:                  # 'Mozilla/5.0' 만, 또는 플랫폼만 있고 엔진·브라우저가 빠진 것
            return browser_ua(rng, hint or "", desktop)
        return ua                               # (Hydra) 같은 도구 UA 는 그대로
    fixed = re.sub(r"^Mozilla/5\.0 \((Linux (?:x86_64|i686|aarch64))\)", r"Mozilla/5.0 (X11; \1)", s)
    if fixed != s:                              # 'X11;' 이 빠진 리눅스 데스크톱 UA (실제 형식이 아님)
        return fixed
    for rx, cands in _TOOL_FIX:
        if rx.match(s):
            return rng.choice(cands)
    return ua


def guarded_headers(rule):
    """룰 조건이 건드리는 헤더 이름(소문자). 이 헤더는 바꾸거나 새로 넣지 않는다."""
    names = set()
    for c in rule.get("conditions", []):
        if c.get("in") not in ("header", "raw_header", "cookie", "raw_cookie", "payload"):
            continue
        text = (c.get("value") or c.get("pattern") or "").lower()
        for h in ["host", "user-agent", "referer", "cookie", "accept", "accept-language", "accept-encoding",
                  "connection", "cache-control", "content-type", "content-length", "authorization", "origin",
                  "transfer-encoding"]:
            if h in text:
                names.add(h)
        if c.get("in") in ("cookie", "raw_cookie"):
            names.add("cookie")
    return names


def realistic_http(http, rule, label, seed):
    """HTTP 요청의 헤더를 실제 클라이언트처럼 고친다. 같은 입력이면 같은 결과.
      1. 잘린 UA(`Mozilla/5.0`, `Mozilla/5.0 (X11; Linux x86_64)`)·없는 버전(`curl/8.0`, `Python/3.9`) → 실제 UA
         UA 가 없으면 넣는다 (normal 은 브라우저, anomalies 는 브라우저 또는 도구)
      2. 클라이언트 기본 헤더가 없으면 넣는다 (브라우저: Accept·Accept-Encoding·Accept-Language·Connection,
         curl: Accept, python-requests: Accept-Encoding·Accept·Connection)
      3. 본문이 있으면 Content-Length 를 실제 길이로 (Transfer-Encoding 이 있으면 넣지 않음)
    룰이 검사하는 헤더(guarded_headers)는 새로 넣지 않는다. 잘린 UA 고치기는 룰이 UA 를 검사해도 하되,
    고친 뒤 룰 판정이 나빠지면 realistic_sample 이 원래대로 되돌린다. 응답(HTTP/…)은 그대로."""
    rng = random.Random(seed)
    guard = guarded_headers(rule)
    s = http.replace("\r\n", "\n")
    head, _, body = s.partition("\n\n")
    lines = head.split("\n")
    first, headers = lines[0], [h for h in lines[1:] if h.strip()]
    if first.startswith("HTTP/"):
        return http

    def names():
        return [h.split(":", 1)[0].strip().lower() for h in headers]

    parts = first.split(" ")
    path = parts[1].lower() if len(parts) > 1 else "/"
    ctype = next((h.split(":", 1)[1].strip().lower() for h in headers if h.lower().startswith("content-type:")), "")
    api = "json" in ctype or "xml" in ctype or "/api" in path or "/rest" in path or parts[0] in ("PUT", "PATCH", "DELETE")
    if "user-agent" in names():
        k = names().index("user-agent")
        old = headers[k].split(":", 1)[1].strip() if ":" in headers[k] else ""
        new = real_ua(old, rng, desktop=api)
        if new != old:
            headers[k] = f"User-Agent: {new}"
    elif "user-agent" not in guard:
        ua = browser_ua(rng, desktop=api) if (label == "normal" or rng.random() < 0.5) else rng.choice(TOOL_UAS)
        headers.insert(names().index("host") + 1 if "host" in names() else 0, f"User-Agent: {ua}")
    ua = next((h.split(":", 1)[1].strip() for h in headers if h.lower().startswith("user-agent:")), "")
    if _BROWSER_UA_RE.match(ua):
        extra = [("Accept", "application/json, text/plain, */*" if api else
                  "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"),
                 ("Accept-Encoding", "gzip, deflate, br"),
                 ("Accept-Language", rng.choice(["ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7", "ko-KR,ko;q=0.9",
                                                 "ko,en-US;q=0.9,en;q=0.8"])),
                 ("Connection", "keep-alive")]
    elif ua.lower().startswith("curl/"):
        extra = [("Accept", "*/*")]
    elif ua.lower().startswith("python-requests/"):
        extra = [("Accept-Encoding", "gzip, deflate"), ("Accept", "*/*"), ("Connection", "keep-alive")]
    elif ua.lower().startswith("go-http-client/"):
        extra = [("Accept-Encoding", "gzip")]
    else:
        extra = []
    pos = names().index("user-agent") + 1 if "user-agent" in names() else len(headers)
    for name, value in extra:
        if name.lower() not in names() and name.lower() not in guard:
            headers.insert(pos, f"{name}: {value}")
            pos += 1
    b = body.rstrip("\n")
    if b and "content-length" not in guard and "transfer-encoding" not in names():
        cl = f"Content-Length: {len(b.encode('utf-8'))}"
        if "content-length" in names():
            headers[names().index("content-length")] = cl
        else:
            headers.append(cl)
    return first + "\n" + "".join(h + "\n" for h in headers) + "\n" + body


def realistic_sample(s, rule):
    """샘플 하나에 헤더 현실화를 적용한다. 바꾼 뒤 룰 판정이 나빠지면(ok → fail) 원래대로 둔다."""
    if not valid_sample(s):
        return s
    new = realistic_http(s["http"], rule, s["label"], stable_seed(rule.get("id", ""), s["http"]))
    if new == s["http"]:
        return s
    if local_check(dict(s, http=new), rule)["result"] == "fail" and local_check(s, rule)["result"] != "fail":
        return s
    return dict(s, http=new)


def run_yes(run, key, default="예"):
    """LLM_설정.md 의 예/아니오 값."""
    return run.get(key, default).strip().lower() in ("예", "yes", "y", "true", "1", "on")

"""[3.0b 참고 예시 마스킹] 기존 학습데이터 payload 를 생성 요청의 참고 예시(existing_examples)로 붙이기 전에
민감값을 자리표시 값으로 바꾼다.  (3_1_prepare_requests.py 가 부름. 직접 실행하면 측정만 한다)

[왜]  참고 예시는 외부 LLM(생성·검토 모델)에 그대로 전달된다. 예시에 필요한 것은 공격 부분의 모양(경로·파라미터·페이로드
      구조)이지 Host·쿠키·인증값·IP 같은 환경 값이 아니다. (docs/ai_governance/07_마스킹_제안.md, 위험 등록부 R05)

[바꾸는 값]  자리표시 값은 생성 지시문의 안전 규칙(문서용 IP·example.com)과 같다
  - Host 헤더            → www.example.com (포트는 유지)
  - Cookie·Set-Cookie    → 이름은 두고 값만 x
  - Authorization·Proxy-Authorization → 방식만 남김 (Bearer TOKEN, Basic dXNlcjpwYXNz = user:pass)
  - Referer·Origin 의 호스트 → example.com (경로는 유지)
  - IPv4·IPv6 (어디든)    → 192.0.2.n / 2001:db8::n (한 예시 안에서 같은 값은 같은 값으로).
                           루프백·링크 로컬·0.0.0.0·문서용 주소는 그대로 (공격 내용인 경우가 많고 민감하지 않음)
  - 이메일                → user@example.com
  - JWT                   → eyJ0eXAiOiJKV1QifQ.e30.x
  - 이름이 비밀인 파라미터(쿼리·폼·JSON: token, api_key, key, secret, session, sid, password, passwd, pwd, auth 와
    access_token 같은 '_' 붙은 이름) 값 → 같은 길이의 x (최대 16자)
  - 32자 이상의 16진수 덩어리 → 0 32개, 32자 이상의 base64 같은 무작위 값 → x 32개

[원칙]  실패하면 막는다 (fail closed)
  - 마스킹 중 오류가 나면 그 예시는 보내지 않는다.
  - 마스킹 뒤 룰 조건 중 원래 만족하던 것이 하나라도 깨지면 그 예시는 보내지 않는다. 원래 값을 그대로 보내는 길은 없다.
  - 원래 값은 어디에도 남기지 않는다. 기록은 종류별 개수만.

[측정]  python 3_0b_mask.py --measure            # 가장 최근 3.1a 결과(coverage.json)의 예시 전체를 마스킹해 개수만 출력
        python 3_0b_mask.py --measure --date 2026-10-29
"""

import argparse
import collections
import importlib.util
import ipaddress
import json
import re
import sys
from pathlib import Path

VERSION = 1          # 바꾸는 규칙을 고치면 올린다 (3.1 조건 지문에 들어가 생성 전 요청을 다시 만든다)

HOST = "www.example.com"
REF_HOST = "example.com"
EMAIL = "user@example.com"
JWT = "eyJ0eXAiOiJKV1QifQ.e30.x"
AUTH = {"bearer": "Bearer TOKEN", "basic": "Basic dXNlcjpwYXNz"}
HEX_DUMMY = "0" * 32
B64_DUMMY = "x" * 32
KINDS = ("host", "cookie", "authorization", "referer_origin", "ipv4", "ipv6", "email", "jwt", "secret_param",
         "long_random")

_DOC_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "2001:db8::/32")]
_SECRET_NAME = re.compile(r"(?i)^(?:[\w.\-]*[_.\-])?(?:token|api_?key|key|secret|session(?:_?id)?|sid|"
                          r"passw(?:or)?d|pwd|auth)$")
_PARAM = re.compile(r"(?<![\w.\-\[\]])([A-Za-z_][\w.\-\[\]]{0,63})=([^&\s#]+)")
_JSON_PARAM = re.compile(r'"([A-Za-z_][\w.\-]{0,63})"\s*:\s*"((?:[^"\\]|\\.)*)"')
_JWT = re.compile(r"eyJ[\w-]{5,}\.[\w-]{5,}\.[\w-]*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_IPV4 = re.compile(r"(?<![\w.])(?<![A-Za-z]/)(\d{1,3}(?:\.\d{1,3}){3})(?![\w.])")   # Chrome/124.0.0.0 같은 버전은 뺌
_IPV6 = re.compile(r"(?<![\w:.])((?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4})(?![\w:.])")
_HEX = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{32,}(?![0-9A-Za-z])")
_B64 = re.compile(r"(?<![A-Za-z0-9+/_\-])[A-Za-z0-9+/_\-]{32,}={0,2}(?![A-Za-z0-9+/_\-=])")
_URL_HOST = re.compile(r"^(\s*[A-Za-z][\w+.\-]*://)(?:[^@/\s]*@)?(\[[^\]]*\]|[^:/\s?#]+)")


class MaskError(Exception):
    pass


def _keep_ip(ip):
    """그대로 둘 주소: 문서용·루프백·링크 로컬(메타데이터 주소 등)·0.0.0.0·멀티캐스트."""
    return (ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast
            or any(ip.version == n.version and ip in n for n in _DOC_NETS))


def _is_placeholder_host(h):
    h = h.lower().strip("[]")
    return h == REF_HOST or h.endswith("." + REF_HOST)


def _random_like(s):
    """base64 같은 무작위 값인지 (경로·반복 문자열이 아닌지). '/'·'-' 가 여럿이고 '+'·'=' 가 없으면 경로·글 제목으로 본다."""
    if s.count("/") + s.count("-") >= 3 and "+" not in s and not s.endswith("="):
        return False
    return (sum(c.isupper() for c in s) >= 2 and sum(c.islower() for c in s) >= 2 and sum(c.isdigit() for c in s) >= 2
            and len(set(s)) >= 12)


class _Masker:
    def __init__(self, text):
        self.text = text
        self.n = collections.Counter()
        self.ip_map = {}

    def _ip(self, raw):
        """원래 IP → 같은 예시 안에서 늘 같은 문서용 IP."""
        if raw not in self.ip_map:
            v6 = ":" in raw
            k = sum(1 for o in self.ip_map if (":" in o) == v6) + 1
            while True:
                cand = f"2001:db8::{k:x}" if v6 else f"192.0.2.{k}"
                if cand not in self.text and cand not in self.ip_map.values():
                    break
                k += 1
            if not v6 and k > 254:
                raise MaskError("IP 가 너무 많음")
            self.ip_map[raw] = cand
        return self.ip_map[raw]

    # ---- 헤더 ----
    def header(self, line):
        name, sep, value = line.partition(":")
        if not sep or not re.fullmatch(r"[A-Za-z0-9\-]+", name.strip()):
            return line
        key = name.strip().lower()
        lead = value[:len(value) - len(value.lstrip())]
        v = value.strip()
        if key == "host":
            m = re.fullmatch(r"(\[[^\]]*\]|[^:]*)(:\d+)?", v)
            host, port = (m.group(1), m.group(2) or "") if m else (v, "")
            if host and host.lower() != HOST:
                self.n["host"] += 1
                return f"{name}:{lead}{HOST}{port}"
            return line
        if key in ("cookie", "set-cookie"):
            def one(m):
                if m.group(2) in ("x", ""):
                    return m.group(0)
                self.n["cookie"] += 1
                return f"{m.group(1)}=x"
            return f"{name}:{lead}" + re.sub(r"([^=;,\s]+)=([^;,]*)", one, v)
        if key in ("authorization", "proxy-authorization"):
            scheme = v.split(None, 1)[0] if v else ""
            new = AUTH.get(scheme.lower(), f"{scheme} x" if scheme else "x")
            if v and v != new:
                self.n["authorization"] += 1
                return f"{name}:{lead}{new}"
            return line
        if key in ("referer", "origin"):
            m = _URL_HOST.match(v)
            if m and not _is_placeholder_host(m.group(2)):
                self.n["referer_origin"] += 1
                return f"{name}:{lead}{m.group(1)}{REF_HOST}{v[m.end():]}"
            return line
        return line

    # ---- 어디든 ----
    def anywhere(self, s):
        def jwt(m):
            if m.group(0) == JWT:
                return m.group(0)
            self.n["jwt"] += 1
            return JWT
        s = _JWT.sub(jwt, s)

        def email(m):
            if m.group(0).lower().endswith("@" + REF_HOST):
                return m.group(0)
            self.n["email"] += 1
            return EMAIL
        s = _EMAIL.sub(email, s)

        def param(m):
            name, val = m.group(1), m.group(2)
            if not _SECRET_NAME.match(name) or set(val) <= {"x"}:
                return m.group(0)
            self.n["secret_param"] += 1
            return f"{name}=" + "x" * min(len(val), 16)
        s = _PARAM.sub(param, s)

        def jparam(m):
            name, val = m.group(1), m.group(2)
            if not _SECRET_NAME.match(name) or not val or set(val) <= {"x"}:
                return m.group(0)
            self.n["secret_param"] += 1
            return m.group(0)[:m.start(2) - m.start(0)] + "x" * min(len(val), 16) + '"'
        s = _JSON_PARAM.sub(jparam, s)

        def hexv(m):
            if len(set(m.group(0).lower())) < 6:
                return m.group(0)
            self.n["long_random"] += 1
            return HEX_DUMMY
        s = _HEX.sub(hexv, s)

        def b64(m):
            if not _random_like(m.group(0)):
                return m.group(0)
            self.n["long_random"] += 1
            return B64_DUMMY
        s = _B64.sub(b64, s)

        def v4(m):
            raw = m.group(1)
            try:
                ip = ipaddress.IPv4Address(raw)
            except ValueError:
                return raw
            if _keep_ip(ip):
                return raw
            self.n["ipv4"] += 1
            return self._ip(raw)
        s = _IPV4.sub(v4, s)

        def v6(m):
            raw = m.group(1)
            if raw.count(":") < 2 or not re.search(r"[0-9A-Fa-f]", raw):
                return raw
            try:
                ip = ipaddress.IPv6Address(raw)
            except ValueError:
                return raw
            if _keep_ip(ip):
                return raw
            self.n["ipv6"] += 1
            return self._ip(raw.lower())
        s = _IPV6.sub(v6, s)
        return s

    def run(self):
        text = self.text
        lines = text.split("\n")
        out, in_head = [], True
        for k, line in enumerate(lines):
            bare = line.rstrip("\r")
            if in_head and k > 0 and bare == "":
                in_head = False
            if in_head and k > 0:
                cr = line[len(bare):]
                line = self.header(bare) + cr
            out.append(line)
        return self.anywhere("\n".join(out))


def mask_example(text):
    """예시 하나 → (바꾼 글, {종류: 바꾼 개수}). 처리할 수 없으면 MaskError."""
    if not isinstance(text, str):
        raise MaskError("글이 아님")
    try:
        m = _Masker(text)
        return m.run(), dict(m.n)
    except MaskError:
        raise
    except Exception as e:          # 예상 못 한 오류도 '보내지 않음' 으로 (fail closed)
        raise MaskError(f"{type(e).__name__}") from None


def _cond_results(C, text, rule):
    parts = C.split_http(text)
    return [C.check_condition(c, parts) for c in rule.get("conditions", [])]


def rule_kept(C, original, masked, rule):
    """원래 만족하던 룰 조건이 마스킹 뒤에도 모두 만족하는지 (C = 3_0a_common)."""
    before, after = _cond_results(C, original, rule), _cond_results(C, masked, rule)
    return all(a is True for b, a in zip(before, after) if b is True)


def safe_examples(C, examples, rule, n):
    """참고 예시 후보 → 보낼 예시(최대 n개, 마스킹됨)와 통계.
    통계: {"in": 본 후보 수, "sent": 보낸 수, "drop_error": 오류로 뺌, "drop_rule": 룰 조건이 깨져 뺌, "masked": {종류: 개수}}"""
    sent, st, masked = [], collections.Counter(), collections.Counter()
    for ex in examples:
        if len(sent) >= n:
            break
        st["in"] += 1
        try:
            new, cnt = mask_example(ex)
        except MaskError:
            st["drop_error"] += 1
            continue
        try:
            ok = rule_kept(C, ex, new, rule)
        except Exception:
            ok = False
        if not ok:
            st["drop_rule"] += 1
            continue
        sent.append(new)
        masked.update(cnt)
    return sent, {"in": st["in"], "sent": len(sent), "drop_error": st["drop_error"], "drop_rule": st["drop_rule"],
                  "masked": dict(masked)}


def load_common():
    spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    return C


def measure(date=None, n=None):
    """3.1a 결과의 참고 예시 후보 전체(또는 룰당 n개)를 마스킹해 개수만 센다. 파일은 쓰지 않는다."""
    C = load_common()
    in_dir = C.latest_input_dir(date)
    if in_dir is None:
        raise SystemExit("05_llm_input 이 없습니다")
    cov_file = C.output_dir(in_dir.name) / "_0_existing" / "coverage.json"
    if not cov_file.exists():
        raise SystemExit(f"3.1a 결과가 없습니다: {in_dir.name}")
    _, _, run = C.load_settings()
    lo = C.run_int(run, "기존 데이터 참고 기준", 50)
    n = n or C.run_int(run, "참고 예시 수", 3)
    cov = json.loads(cov_file.read_text(encoding="utf-8"))
    rules = {r["id"]: r for r in C.read_jsonl(in_dir / "rules_all.jsonl")}
    tot, kinds, per_kind_examples = collections.Counter(), collections.Counter(), collections.Counter()
    for rid, c in cov.items():
        if rid not in rules or not 1 <= c.get("count", 0) <= lo:
            continue
        tot["rules"] += 1
        exs = c.get("examples", [])
        tot["candidates"] += len(exs)
        tot["unmasked_would_send"] += min(len(exs), n)
        for ex in exs:          # 예시마다 바뀐 종류 (예시 하나에 그 종류가 있었는지)
            try:
                _, cnt = mask_example(ex)
                per_kind_examples.update(k for k, v in cnt.items() if v)
                tot["examples_with_any"] += bool(cnt)
            except MaskError:
                pass
        sent, st = safe_examples(C, exs, rules[rid], n)
        tot["sent"] += st["sent"]
        tot["drop_error"] += st["drop_error"]
        tot["drop_rule"] += st["drop_rule"]
        tot["rules_without_example"] += not sent
        kinds.update(st["masked"])
    return {"input_date": in_dir.name, "examples_per_rule": n, "reference_band": f"1~{lo}",
            "totals": dict(tot), "masked_values_sent": {k: kinds[k] for k in KINDS if kinds[k]},
            "candidate_examples_with_kind": {k: per_kind_examples[k] for k in KINDS if per_kind_examples[k]}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="참고 예시 마스킹 (측정만: 개수 출력, 원래 값은 출력하지 않음)")
    ap.add_argument("--measure", action="store_true", help="3.1a 결과의 예시를 마스킹해 종류별 개수만 출력")
    ap.add_argument("--date", help="05_llm_input 날짜 폴더 (기본: 가장 최근)")
    args = ap.parse_args(argv)
    if not args.measure:
        ap.print_help()
        return 0
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(measure(args.date), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

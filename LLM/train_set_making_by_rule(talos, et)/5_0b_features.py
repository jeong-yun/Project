"""[5.0b 지도학습 피처] 표준모델 피처 + 신규 후보 피처 9개 + 라벨 검토용 강한 시그니처.  (5.x 공용 보조 모듈, 직접 실행하지 않음)

표준모델 피처
  - 표준모델 피처 파일(노트북을 변환한 .py, 경로는 11_supervised/versions.json 의 feature_code)을 읽기만 한다.
    그 파일의 '# 1) 피처 생성' 앞부분(정규식 사전 regex_dict_comb/count/ratio, generate_features, ko_ratio)만 실행한다.
  - 피처 이름·정규식은 이 코드에 적지 않는다 (파일에서 읽음 → 표준모델이 바뀌면 파일만 바꾸면 됨).
  - 원본 generate_features 는 행마다 DataFrame 을 만들어 느리므로, 같은 정규식·같은 규칙(소문자, 줄바꿈·탭 → 공백,
    한글 비율 0.1 이상이면 0)으로 열 단위 계산을 한다. verify() 로 원본 함수와 결과가 같은지 확인한다.
신규 후보 (표준모델 v26.03.03 4단계 +ALL+R): 헤더에서 생기는 오탐을 줄이려고 적용 범위를 요청 대상으로 나눈다.
  uri = 요청 줄의 대상(경로+쿼리), path = uri 의 '?' 앞, target = uri + 본문(빈 줄 뒤), target_dec = target 을 URL 디코딩 1회
  F1 취약 경로 사전 (vuln_path_comb, path_confusion_comb)   F2 디렉터리 요청 (dir_req_comb, cms_component_comb)
  F3 인코딩 특수문자 (enc_special_count, html_entity_count)  F4 파일 다운로드 파라미터 (file_param_traversal_comb, file_param_sensitive_comb)
  R  인코딩 안 된 특수문자 수 (raw_special_count)
payload 변환: 학습 로그(수집 형식, 줄바꿈이 '\\n' 두 글자) → 학습 파일 형식(실제 줄바꿈). 요청 줄 안의 '\\n'(공격 문자열)은 그대로.
"""

import hashlib
import re
import warnings
from pathlib import Path
from urllib.parse import unquote, unquote_plus

import numpy as np
import pandas as pd

# 정규식에 묶음 ( ) 이 있으면 str.contains 가 줄마다 경고를 낸다 (결과는 같음)
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
V2_VERSION = "v26.03.03-4+R"        # 신규 후보 피처를 바꾸면 올린다 (피처 캐시를 새로 만듦)
_MARKER = "# 1) 피처 생성"
_legacy = {}                         # 경로 → (수정 시각, 이름 공간)


# ── 표준모델 피처 ─────────────────────────────────────────────────────────

def load_legacy(path) -> dict:
    """표준모델 피처 파일의 함수·정규식 사전 (generate_features, ko_ratio, regex_dict_comb/count/ratio)."""
    path = Path(path)
    mt = path.stat().st_mtime
    hit = _legacy.get(str(path))
    if hit and hit[0] == mt:
        return hit[1]
    src = path.read_text(encoding="utf-8")
    if _MARKER not in src:
        raise ValueError(f"표준모델 피처 파일에 '{_MARKER}' 표시가 없습니다: {path.name}")
    ns = {}
    exec(src[:src.index(_MARKER)], ns)
    for k in ("regex_dict_comb", "regex_dict_count", "regex_dict_ratio", "ko_ratio"):
        if k not in ns:
            raise ValueError(f"표준모델 피처 파일에 {k} 가 없습니다: {path.name}")
    _legacy[str(path)] = (mt, ns)
    return ns


def code_hash(path) -> str:
    """피처 파일 내용 + 신규 후보 버전 → 피처 캐시 이름 (둘 중 하나가 바뀌면 다시 계산)."""
    return hashlib.sha1(Path(path).read_bytes() + V2_VERSION.encode()).hexdigest()[:12]


def legacy_names(ns) -> list:
    return sorted(list(ns["regex_dict_comb"]) + list(ns["regex_dict_count"]) + list(ns["regex_dict_ratio"]))


def generate_fast(payload: pd.Series, ns) -> pd.DataFrame:
    """원본 generate_features + 한글 비율 마스크와 같은 결과를 열 단위로 계산 (열 이름 = 정규식 사전의 키)."""
    p = payload.fillna("").astype(str)
    proc = p.str.lower().str.replace("\n", " ", regex=False).str.replace("\r", " ", regex=False).str.replace("\t", " ", regex=False)
    out = {}
    for name, rxs in ns["regex_dict_comb"].items():
        hit = pd.Series(False, index=p.index)
        for rx in rxs:
            hit |= proc.str.contains(rx, regex=True)
        out[name] = hit.astype(int)
    for name, rxs in ns["regex_dict_count"].items():
        out[name] = sum(proc.str.count(rx) for rx in rxs)
    n = proc.str.len()
    for name, rxs in ns["regex_dict_ratio"].items():
        tmp = proc
        for rx in rxs:
            tmp = tmp.str.replace(rx, "", regex=True)
        out[name] = np.where(n > 0, (n - tmp.str.len()) / n.replace(0, 1), 0.0)
    f = pd.DataFrame(out, index=p.index)[sorted(out)]
    keep = p.map(ns["ko_ratio"]) < 0.1
    return f.where(np.repeat(keep.to_numpy()[:, None], f.shape[1], axis=1), 0.0)


def verify(payload: pd.Series, ns, n: int = 2000, seed: int = 0) -> float:
    """원본 generate_features 와 generate_fast 의 최대 차이 (샘플 n 행). 0 이어야 한다."""
    s = payload.sample(min(n, len(payload)), random_state=seed)
    slow = pd.concat([ns["generate_features"](x, ns["regex_dict_comb"], ns["regex_dict_count"], ns["regex_dict_ratio"])
                      for x in s], ignore_index=True)
    keep = (s.map(ns["ko_ratio"]) < 0.1).to_numpy()
    slow = slow.where(np.repeat(keep[:, None], slow.shape[1], axis=1), 0.0)
    fast = generate_fast(s, ns).reset_index(drop=True)
    return float((slow[fast.columns].astype(float) - fast.astype(float)).abs().to_numpy().max())


def raw_to_payload(s) -> str:
    """학습 로그 payload(줄바꿈이 '\\n' 두 글자) → 학습 파일 payload(실제 줄바꿈).
    기존 학습 데이터 규칙과 같게 요청 줄 안의 '\\n'(공격 문자열)은 그대로 두고, 요청 줄 끝(HTTP/x.x 뒤)부터만 바꾼다."""
    if not isinstance(s, str):
        return ""
    m = re.search(r" HTTP/\d(\.\d)?(?=\\n|$)", s)
    if not m:
        return s
    head, tail = s[:m.end()], s[m.end():]
    return head + tail.replace("\\r\\n", "\n").replace("\\n", "\n")


# ── 신규 후보 피처 (F1~F4 + R) ────────────────────────────────────────────

# F1 취약 경로 사전 (공개적으로 알려진 공격 대상 엔드포인트, path 에 적용)
VULN_PATH = [
    r"web\.config|trace\.axd|elmah\.axd|/_vti_|/aspnet_client/",
    r"/(jmx-console|web-console|invoker/|jolokia|actuator|manager/html|host-manager|axis2|solr/|druid/|nacos/|hudson|jenkins)",
    r"/(libs/granite|libs/cq|crx/|bin/querybuilder|system/console|etc/packages)",
    r"/(catalog-portal|vsphere-client|ui/vropspluginui|carbon/|cmspages/|mantisbt|mantis/|autodiscover)|^/(owa|ecp)/",
    r"/(remote/login|dana-na|mgmt/tm|tmui|sslvpn|boaform|gponform|hnap1|setup\.cgi|cgi-bin/luci|goform/)",
    r"/(vendor/phpunit|_ignition|telescope|horizon/api|adminer|server-status|server-info|phpinfo)",
    r"/(fckeditor|kindeditor|ueditor|elfinder|filemanager|plupload|uploadify)",
    r"/(-/snippets|api/v4/|users/sign_in|\.git/|\.svn/|\.ds_store|web-inf/|meta-inf/)",
]
PATH_CONFUSION = [
    r";[^/?]*\.(jar|js|css|png|jpg|gif|ico|html?)",   # /admin.jsp;.jar  (;jsessionid= 는 제외)
    r"/\.\.;/|/;/|%3b/",
]
# F2 디렉터리 요청
CMS_COMPONENT = [r"/components/com_|/modules/mod_|/administrator/|/wp-content/(plugins|themes)/[a-z0-9_.-]+/?$"]
# F3 인코딩 특수문자 (" ' < > ; ( ) ` | $ { } \ NUL CR LF, 이중 인코딩 포함)
ENC_SPECIAL = r"%(25)?(22|27|3c|3e|3b|28|29|60|7c|24|7b|7d|5c|00|0a|0d)"
HTML_ENTITY = r"&(amp|lt|gt|quot|apos|#x?[0-9a-f]+);"
# F4 파일 다운로드/포함 파라미터 (디코딩 1회 값에 적용)
FILE_PARAM = r"(?:^|[?&;/])(?:file|filename|fname|filepath|path|download|dn|doc|attach|f|page|template|include|inc|dir|folder)="
FILE_TRAVERSAL = FILE_PARAM + r"[^&]*(\.\.|%uff0e|%u2216|%u002e|%c0%ae|%e0%80%ae|\.%00)"
FILE_ABSOLUTE = FILE_PARAM + r"(/|\\|[a-z]:[\\/])"
FILE_SENSITIVE = FILE_PARAM + (r"[^&]*(\.(ini|conf|cfg|config|log|php\d?|jsp|aspx?|xml|properties|ya?ml|env|sql|bak|sh|py)(&|\s|$)"
                               r"|passwd|shadow|web\.config|error_log|\{file\})")
# R 인코딩되지 않은 특수문자 " ' < > ; ( ) ` | $ { } \ 개수 (HTML 엔티티의 ';' 는 빼고 셈)
RAW_SPECIAL = r"[\"'<>;()`|${}\\]"
V2_NAMES = ["vuln_path_comb", "path_confusion_comb", "dir_req_comb", "cms_component_comb", "enc_special_count",
            "html_entity_count", "file_param_traversal_comb", "file_param_sensitive_comb", "raw_special_count"]


def v2_cols(dev: str) -> list:
    return [f"{dev}_payload_{c}" for c in V2_NAMES]


def _any(s: pd.Series, regex_list) -> pd.Series:
    out = pd.Series(False, index=s.index)
    for rx in regex_list:
        out |= s.str.contains(rx, regex=True)
    return out.astype(int)


def _safe_unquote(x: str) -> str:
    try:
        return unquote(x)
    except Exception:
        return x


def parts(payload: pd.Series) -> pd.DataFrame:
    """payload → uri / path / query / target / target_dec."""
    p = payload.fillna("")
    uri = p.str.extract(r"^\s*[A-Za-z_-]+\s+(\S+)", expand=False).fillna("").str.lower()
    path = uri.str.split("?", n=1).str[0]
    query = uri.str.split("?", n=1).str[1].fillna("")
    body = p.str.split(r"\r?\n\r?\n", n=1, regex=True).str[1].fillna("").str.lower()
    target = uri + " " + body
    return pd.DataFrame({"uri": uri, "path": path, "query": query, "target": target,
                         "target_dec": target.map(_safe_unquote)})


def generate_v2(payload: pd.Series, dev: str) -> pd.DataFrame:
    """신규 후보 피처 9개 (열 이름: {장비}_payload_<이름>)."""
    t = parts(payload)
    f = pd.DataFrame(index=payload.index)
    f["vuln_path_comb"] = _any(t["path"], VULN_PATH)
    f["path_confusion_comb"] = _any(t["path"], PATH_CONFUSION)
    f["dir_req_comb"] = ((t["path"].str.len() > 1) & t["path"].str.endswith("/") & (t["query"] == "")).astype(int)
    f["cms_component_comb"] = _any(t["path"], CMS_COMPONENT)
    f["enc_special_count"] = t["target"].str.count(ENC_SPECIAL)
    f["html_entity_count"] = t["target"].str.count(HTML_ENTITY)
    f["file_param_traversal_comb"] = (_any(t["target"], [FILE_TRAVERSAL]) | _any(t["target_dec"], [FILE_TRAVERSAL, FILE_ABSOLUTE])).astype(int)
    f["file_param_sensitive_comb"] = _any(t["target_dec"], [FILE_SENSITIVE])
    f["raw_special_count"] = t["target"].str.replace(HTML_ENTITY, "", regex=True).str.count(RAW_SPECIAL)
    return f[V2_NAMES].add_prefix(f"{dev}_payload_")


# ── 라벨 검토용 강한 시그니처 · 데이터 흔적 ───────────────────────────────

STRONG_SIGS = {   # 사람이 봐도 공격인 강한 패턴 (URL 디코딩 2회 + 소문자에 적용). 모델 피처와 별개
    "SQLi:union select": r"union(\s|/\*.*?\*/)+(all(\s|/\*.*?\*/)+)?select",
    "SQLi:시간지연": r"(sleep|pg_sleep)\s*\(\s*\d|benchmark\s*\(\s*\d|waitfor\s+delay",
    "SQLi:xp_cmdshell/into outfile": r"xp_cmdshell|into\s+(out|dump)file",
    "SQLi:or 1=1": r"['\"]\s*(or|and)\s*['\"]?\d+['\"]?\s*=\s*['\"]?\d+|\bor\s+1\s*=\s*1\b",
    "SQLi:information_schema": r"information_schema|sysobjects|@@version",
    "XSS:script태그": r"<\s*script[\s>/]",
    "XSS:이벤트핸들러": r"<[a-z]+[^>]*\s(onerror|onload|onmouseover|onfocus|ontoggle)\s*=",
    "XSS:javascript:/alert": r"javascript\s*:|alert\s*\(|document\.cookie|prompt\s*\(\s*\d",
    "경로조작:../ 반복": r"(\.\.[/\\]){2,}",
    "경로조작:시스템파일": r"/etc/(passwd|shadow|hosts)|win\.ini|boot\.ini|/proc/self/",
    "명령실행:셸/명령": r"/bin/(ba)?sh\b|cmd\.exe|powershell|[;|`]\s*(wget|curl|id|whoami|uname|cat)(\s|;|\||`|$)|\$\(\s*(wget|curl|id|whoami)",
    "Log4Shell:jndi": r"\$\{\s*jndi\s*:",
    "PHP:코드실행": r"<\?php|php://(input|filter)|base64_decode\s*\(|eval\s*\(\s*\$_",
    "OGNL/SSTI": r"%\{\s*#|\$\{\s*#|\{\{\s*\d+\s*\*\s*\d+\s*\}\}|java\.lang\.runtime",
    "스캐너 UA": r"user-agent:[^\n]*(sqlmap|nikto|nmap|acunetix|nessus|zgrab|masscan|gobuster|dirbuster|wpscan|nuclei|arachni|openvas)",
}
_SIGS = {k: re.compile(v) for k, v in STRONG_SIGS.items()}


def match_sigs(p) -> str:
    """강한 시그니처 이름들 ('|' 로 연결, 없으면 빈 문자열)."""
    if not isinstance(p, str):
        return ""
    s = p
    for _ in range(2):
        try:
            s = unquote_plus(s)
        except Exception:
            break
    s = s.lower()
    return "|".join(k for k, rx in _SIGS.items() if rx.search(s))


UA_TOOLS = ["sqlmap", "curl", "python", "go-http", "wget", "스캐너"]


def ua_category(payload: pd.Series) -> pd.Series:
    ua = payload.str.extract(r"(?im)^user-agent:[ \t]*(.*)$", expand=False).fillna("").str.lower()
    cats = [("(없음)", ua == ""), ("sqlmap", ua.str.contains("sqlmap")), ("curl", ua.str.startswith("curl")),
            ("python", ua.str.contains("python")), ("go-http", ua.str.contains("go-http")), ("wget", ua.str.contains("wget")),
            ("스캐너", ua.str.contains("nikto|nmap|masscan|zgrab|gobuster|dirbuster|nuclei|acunetix|zap")),
            ("브라우저", ua.str.startswith("mozilla"))]
    out = pd.Series("기타", index=payload.index)
    for name, m in reversed(cats):
        out[m] = name
    return out


def python_requests_ua(payload: pd.Series) -> pd.Series:
    return payload.str.contains(r"(?im)^user-agent:[ \t]*python-requests")


def header_count(payload: pd.Series) -> pd.Series:
    head = payload.str.split(r"\r?\n\r?\n", n=1, regex=True).str[0]
    return head.str.count("\n")


def norm_payload(s) -> str:
    """중복 판단용 정규화: 소문자, 공백 정리, 숫자열을 0으로."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\d+", "0", s)
    return s.strip()


def hash_series(s: pd.Series) -> np.ndarray:
    """문자열 → uint64 지문 (pandas 고정 키라 실행마다 같음)."""
    return pd.util.hash_pandas_object(s.fillna("").astype(str), index=False).to_numpy()


def compute_chunk(dev: str, code_path: str, raw_payloads: list) -> pd.DataFrame:
    """학습 로그 payload 묶음 → 표준모델 피처 + 신규 후보 9 + 중복 지문 + 흔적 (5.5 가 샘플마다 한 번만 계산해 캐시)."""
    ns = load_legacy(code_path)
    p = pd.Series(raw_payloads, dtype=object).map(raw_to_payload)
    # 표준모델 피처는 float64 그대로 (확정할 때 학습 파일에 원본 코드와 같은 값으로 씀), 신규 후보는 float32
    f = pd.concat([generate_fast(p, ns).astype("float64"), generate_v2(p, dev).astype("float32")], axis=1)
    f["h_exact"] = hash_series(p)
    f["h_norm"] = hash_series(p.map(norm_payload))
    f["sigs"] = p.map(match_sigs)
    f["ua"] = ua_category(p)
    f["py_ua"] = python_requests_ua(p)
    f["hdr"] = header_count(p).astype("int16")
    f["plen"] = p.str.len().astype("int32")
    return f.reset_index(drop=True)

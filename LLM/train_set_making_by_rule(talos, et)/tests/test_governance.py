"""생성 이력(R11)·거부 편향 경보(R01) 시험. 가짜 모델(mock)·합성 데이터만, 임시 폴더에서 (API 호출·실제 데이터 없음).

    python -m unittest discover -s tests -v
"""

import hashlib
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


G = load("gen_g", "3_2_generate.py")
Rv = load("rev_g", "3_3_review.py")
P5 = load("prog_g", "3_5_progress.py")
LOG = logging.getLogger("gov_t")
LOG.handlers[:] = [logging.NullHandler()]
LOG.propagate = False

RUN = {"생성 anomalies 수": "3", "생성 normal 수": "1", "정규식 예시": "아니오", "헤더 현실화": "아니오", "재시도 횟수": "1"}
PROVIDERS = {"mock": {"url": "-", "key": "-", "style": "mock"}}


def rule(rid, mask=None, examples=None):
    r = {"id": rid, "attack_type": "rce", "methods": ["GET"], "rule": f"alert http ({rid})",
         "conditions": [{"type": "content", "value": "/x.php", "in": "uri"}]}
    if examples is not None:
        r["existing_examples"] = examples
    if mask is not None:
        r["_mask"] = mask
    return r


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gov_t_"))
        self.gen_dir, self.rev_dir = self.tmp / G.C.GEN_DIR, self.tmp / Rv.C.REV_DIR
        self.gen_dir.mkdir()
        self.rev_dir.mkdir()
        self.cfg = {"provider": "mock", "model": "m1", "temperature": 0.8, "max_tokens": 1000, "reasoning_effort": "low"}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def gen_meta(self, name):
        return json.loads((self.gen_dir / f"{name}.meta.json").read_text(encoding="utf-8"))

    def test_generate_meta(self):
        rules = [rule("sid:1", {"on": True, "values": 4, "dropped": 1}, ["e1", "e2"]), rule("sid:2")]
        G.generate("r1", rules, dict(self.cfg), PROVIDERS, RUN, "생성 지시문", self.gen_dir, LOG)
        m = self.gen_meta("r1")
        p = m["provenance"]
        self.assertEqual((p["model"], p["temperature"], p["max_tokens"], p["effort"], p["prompt_file"]),
                         ("mock:m1", 0.8, 1000, "low", "LLM_프롬프트.md"))
        self.assertEqual(p["prompt_sha256"], hashlib.sha256("생성 지시문".encode()).hexdigest())
        self.assertEqual((m["existing_examples_sent"], m["masked_values"]), (2, 4))

    def test_masked_values_none_when_no_record(self):
        G.generate("r2", [rule("sid:1", None, ["e1"])], dict(self.cfg), PROVIDERS, RUN, "p", self.gen_dir, LOG)
        self.assertEqual((self.gen_meta("r2")["existing_examples_sent"], self.gen_meta("r2")["masked_values"]), (1, None))
        G.generate("r3", [rule("sid:1")], dict(self.cfg), PROVIDERS, RUN, "p", self.gen_dir, LOG)
        self.assertEqual((self.gen_meta("r3")["existing_examples_sent"], self.gen_meta("r3")["masked_values"]), (0, 0))

    def test_review_meta(self):
        rules = [rule("sid:1")]
        G.generate("r1", rules, dict(self.cfg), PROVIDERS, RUN, "생성 지시문", self.gen_dir, LOG)
        gen = Rv.C.read_jsonl(self.gen_dir / "r1.jsonl")
        cfg = dict(self.cfg, fallback={"provider": "mock", "model": "fb"})
        Rv.review("r1", rules, gen, cfg, PROVIDERS, RUN, "검토 지시문", self.rev_dir, LOG)
        m = json.loads((self.rev_dir / "r1.meta.json").read_text(encoding="utf-8"))
        p = m["provenance"]
        self.assertEqual((p["generator"], p["model"], p["substitute"], p["effort"]), ("mock:m1", "mock:m1", "mock:fb", "low"))
        self.assertEqual(p["generator_prompt_sha256"], hashlib.sha256("생성 지시문".encode()).hexdigest())
        self.assertEqual(p["prompt_sha256"], hashlib.sha256("검토 지시문".encode()).hexdigest())
        self.assertEqual(m["review_by_type"], {"rce": [3, 0, 0, 0]})


class BiasTest(unittest.TestCase):
    def test_type_counts(self):
        a = {"label": "anomalies", "http": "GET / HTTP/1.1"}
        n = {"label": "normal", "http": "GET / HTTP/1.1"}
        gen = [{"id": "s1", "attack_type": "rce", "samples": [a, a, a, a, n]},
               {"id": "s2", "attack_type": "xss", "samples": [a, n]}]
        rej = [{"id": "s1", "i": 0, "stage": "검토", "reason": "실행 가능한 페이로드라 안전하게 제공할 수 없음"},
               {"id": "s1", "i": 1, "stage": "검토", "reason": "중복"},
               {"id": "s1", "i": 2, "stage": "코드 검사", "reason": "조건 불일치"},
               {"id": "s1", "i": 4, "stage": "검토", "reason": "안전"},              # 정상 샘플은 세지 않음
               {"id": "s2", "i": 0, "stage": "검토 거절", "reason": "x"}]
        self.assertEqual(Rv.type_counts(gen, rej), {"rce": [4, 2, 1, 0], "xss": [1, 0, 0, 1]})

    def test_bias_summary_threshold_and_min(self):
        out, alerts = P5.bias_summary({"rce": [200, 50, 30, 10], "xss": [50, 30, 30, 0], "sqli": [1000, 10, 2, 0]}, 5, 100)
        self.assertEqual(alerts, ["rce"])                      # xss 는 표본이 적어 빠짐, sqli 는 0.2%
        self.assertEqual(out["rce"]["refusal_pct"], 20.0)
        self.assertEqual(out["rce"]["drop_pct"], 30.0)

    def test_review_bias_latest_and_all(self):
        bias = {"2026-01-01": {"rce": [1000, 400, 400, 0]}, "2026-01-02": {"rce": [1000, 0, 0, 0]}}
        rb = P5.review_bias(bias, {"거부 편향 경보 기준": "5", "거부 편향 최소 샘플": "100"})
        self.assertEqual((rb["all"]["alerts"], rb["latest_date"], rb["latest"]["alerts"]), (["rce"], "2026-01-02", []))
        self.assertEqual(rb["all"]["by_type"]["rce"]["refusal_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()

"""화면 속도용 캐시 (core/memo.py) — 파일이 그대로면 다시 읽지 않고, 바뀌면 바로 다시 읽는지.

캐시는 '방금 바뀐 파일'(2초 안)은 캐시하지 않으므로, 테스트에서는 os.utime 으로 수정 시각을 과거로 돌려
'오래된 파일'을 흉내 낸다.
"""

import os
import time

from core import catalog as cat
from core import config
from core.memo import RACY_SEC, SigCache, file_sig
from core.storage import Storage


def age(path, seconds: float = 60) -> None:
    """파일(폴더)의 수정 시각을 seconds 초 전으로 — 매번 다른 시각이 되도록 조금씩 어긋나게."""
    when = time.time() - seconds
    os.utime(path, (when, when))


def test_file_sig_skips_recent_files_and_tracks_changes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("1", encoding="utf-8")
    assert file_sig(f) is None  # 방금 바뀜 → 캐시하지 말 것
    age(f)
    sig = file_sig(f)
    assert sig is not None and file_sig(f) == sig
    assert file_sig(tmp_path / "없음.txt") == (None,)
    f.write_text("22", encoding="utf-8")
    age(f, 30)
    assert file_sig(f) != sig
    assert RACY_SEC > 0


def test_sig_cache_recomputes_when_signature_changes():
    cache, calls = SigCache(maxsize=2), []

    def compute():
        calls.append(1)
        return len(calls)

    assert cache.get("k", (1,), compute) == 1
    assert cache.get("k", (1,), compute) == 1  # 같은 지문 → 저장한 값
    assert cache.get("k", (2,), compute) == 2  # 지문이 바뀜 → 다시 계산
    assert cache.get("k", None, compute) == 3  # 지문 없음(방금 바뀜) → 매번 계산
    cache.get("a", (1,), compute), cache.get("b", (1,), compute)  # 크기 넘치면 비우고 다시 채운다
    assert cache.get("k", (2,), compute) == 6


def test_load_yaml_rereads_changed_file_and_returns_copies(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text("a: [1, 2]\n", encoding="utf-8")
    age(f)
    first = config.load_yaml(f)
    first["a"].append(3)  # 돌려받은 값을 고쳐도
    assert config.load_yaml(f) == {"a": [1, 2]}  # 캐시는 그대로
    f.write_text("a: [9]\n", encoding="utf-8")
    age(f, 30)
    assert config.load_yaml(f) == {"a": [9]}
    f.write_text("b: 1\n", encoding="utf-8")  # 방금 바뀐 파일도 바로 반영
    assert config.load_yaml(f) == {"b": 1}


def test_avatar_lookup_sees_added_and_removed_files(tmp_path, monkeypatch, ranks):
    monkeypatch.setattr(config, "TEAMS_DIR", tmp_path)
    folder = tmp_path / "t" / "avatars"
    folder.mkdir(parents=True)
    age(folder)
    rank = ranks[0]
    before = config.resolve_avatar("t", rank)
    assert before is None or folder not in before.parents
    (folder / f"{rank.id}.PNG").write_bytes(b"x")  # 대문자 확장자도 (Windows 처럼 대소문자 무시)
    age(folder, 30)
    assert config.resolve_avatar("t", rank) == folder / f"{rank.id}.png"
    (folder / f"{rank.id}.PNG").unlink()
    age(folder, 20)
    assert config.resolve_avatar("t", rank) == before


def test_storage_reads_are_cached_but_see_writes_from_any_instance(tmp_path):
    db = tmp_path / "runs.db"
    a, b = Storage(db), Storage(db)
    a.create_run("r1", "general", "lead", "질문")
    age(db)
    assert [r["id"] for r in a.list_runs()] == ["r1"]
    assert [r["id"] for r in a.list_runs()] == ["r1"]  # 캐시에서
    b.create_run("r2", "general", "lead", "질문2")  # 다른 Storage 객체가 써도
    assert {r["id"] for r in a.list_runs()} == {"r1", "r2"}
    age(db, 40)
    assert {r["id"] for r in a.list_runs()} == {"r1", "r2"}  # 이 지문으로 캐시됨
    b.create_run("r3", "general", "lead", "질문3")
    age(db, 35)  # 쓴 뒤 시간이 지나도(지문이 달라) 새로 읽는다
    assert {r["id"] for r in a.list_runs()} == {"r1", "r2", "r3"}
    age(db, 30)
    b.set_price_override("m", 1.0, 2.0)
    assert a.price_overrides() == {"m": {"input": 1.0, "output": 2.0}}
    age(db, 20)
    assert a.price_overrides() == {"m": {"input": 1.0, "output": 2.0}}
    b.clear_price_override("m")
    assert a.price_overrides() == {}


def test_find_related_with_empty_input_is_empty():
    catalog = cat.Catalog(root=None, entries=[cat.Entry("AI", "[x]그룹 정책(GRPO) (작성일: 26.01.01)", "본문", "x", "그룹 정책(GRPO)")])
    assert cat.find_related(catalog, "", "") == []
    assert cat.find_related(catalog, "  ", "\n") == []
    assert [e.name for e, _ in cat.find_related(catalog, "GRPO 변형")] == ["그룹 정책(GRPO)"]

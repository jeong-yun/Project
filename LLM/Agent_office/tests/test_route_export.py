"""여행 경로 지도(core/route.py) · 채택 결과 저장(core/exporter.py) 테스트 — 네트워크·API 없이."""

import pytest

from core import route as route_mod
from core.brief import DesiredState
from core.config import load_team
from core.exporter import export_adopted, list_results, slugify
from core.live import new_run
from core.route import Itinerary, Stop, build_route, distance_km, geocode_stops, map_html
from core.schemas import HumanDecision
from tests.fakes import Script


def stop(day, order, name, category, lat=34.69, lon=135.50, time="오전"):
    return Stop(day=day, order=order, time=time, name=name, search_name=f"{name}, Osaka", category=category,
                lat=lat, lon=lon, note="메모")


ITIN = Itinerary(destination="오사카", stops=[
    stop(1, 2, "도톤보리", "walk"), stop(1, 1, "오사카성", "culture", 34.687, 135.526),
    stop(2, 1, "USJ", "theme", 34.666, 135.433, "오후"),
])


@pytest.fixture
def fake_geo(monkeypatch):
    """가짜 좌표 검색: 이름에 따라 고정 좌표. 호출 기록은 calls 에."""
    calls = []
    table = {"오사카성, Osaka": (34.6874, 135.5259), "도톤보리, Osaka": (34.6689, 135.5011),
             "USJ, Osaka": (35.0, 139.0)}  # USJ 는 일부러 먼 곳(동명이인) → 거부되어야 함

    def fake(query):
        calls.append(query)
        return table.get(query)

    monkeypatch.setattr(route_mod, "fetch_geocode", fake)
    return calls


def test_distance_km():
    assert 28 < distance_km(34.69, 135.50, 34.43, 135.22) < 40  # 난바 ↔ 간사이공항 약 35km


def test_geocode_accepts_near_rejects_far_and_caches(fake_geo, tmp_path):
    itin = ITIN.model_copy(deep=True)
    cache = tmp_path / "cache.json"
    result = geocode_stops(itin, cache_path=cache, pause=0)
    by_name = {s.name: s for s in itin.stops}
    assert result.verified["1-1"] is True and by_name["오사카성"].lat == 34.6874  # 가까움 → OSM 좌표 사용
    assert result.verified["2-1"] is False and by_name["USJ"].lat == 34.666  # 300km 이상 → 추정 좌표 유지
    assert any("USJ" in n for n in result.notes)

    calls_before = len(fake_geo)
    geocode_stops(ITIN.model_copy(deep=True), cache_path=cache, pause=0)
    assert len(fake_geo) == calls_before  # 두 번째는 캐시에서


def test_build_route_extracts_and_sorts(fake_geo, use_script, models, monkeypatch):
    monkeypatch.setattr(route_mod.time, "sleep", lambda s: None)  # OSM 대기 생략
    use_script(Script(Itinerary=ITIN.model_copy(deep=True)))
    result = build_route("1일차 오사카성 → 도톤보리, 2일차 USJ", models)
    assert [(s.day, s.order) for s in result.itinerary.stops] == [(1, 1), (1, 2), (2, 1)]  # 일차·순서로 정렬


def test_map_html_has_icons_animated_path_play_and_legend(fake_geo):
    result = geocode_stops(ITIN.model_copy(deep=True), pause=0)
    html = map_html(result, trip_start="2026-11-20")
    assert "antPath" in html  # 움직이는 경로선
    assert "timeDimension" in html  # ▶ 재생
    assert "🏛" in html and "🚶" in html and "🎢" in html  # 종류별 기호
    assert "범례" in html and "1일차" in html and "2026-11-20T09" in html  # 재생 시각 = 여행 날짜 + 시간대
    assert "fitBounds" in html
    day2 = map_html(result, day=2)
    assert "🎢" in day2 and "🏛" not in day2.split("범례")[0]  # 2일차만


# ─── 채택 결과 저장 ────────────────────────────────────────────────────


def test_slugify_keeps_korean():
    assert slugify("오사카 3박 4일 여행 계획을 세워줘!") == "오사카_3박_4일_여행_계획을_세워줘"
    assert slugify("???") == "result"


def test_export_adopted_writes_markdown_and_map(tmp_path):
    run = new_run("r1", "오사카 여행 계획 세워줘", "lead")
    run["brief"] = DesiredState(goal="3박 4일 계획서", context="", success_criteria=["모든 날 일정"], constraints=["예산 250만원"],
                                output_format="", assumptions=[])
    decision = HumanDecision(action="adopt", adopted_node_id="lead", final_answer="1일차 오사카성", edited=True, score=5, comment="좋음")
    paths = export_adopted(run, decision, "travel", "여행 팀", {"lead": "팀장·여행기획"}, tmp_path, "요금은 바뀔 수 있음", "<html>map</html>")

    md, html = paths
    assert md.parent == tmp_path / "travel" and md.suffix == ".md" and html.name.endswith("_지도.html")
    text = md.read_text(encoding="utf-8")
    assert "팀장·여행기획 의 안 (사람이 직접 수정)" in text and "1일차 오사카성" in text
    assert "[완료 기준] 모든 날 일정" in text and "> 요금은 바뀔 수 있음" in text
    assert list_results(tmp_path, "travel")[0].suffix in (".md", ".html")
    assert list_results(tmp_path, "stock") == []


def test_travel_team_has_route_map_feature():
    assert "route_map" in load_team("travel_domestic").features and "route_map" in load_team("travel_abroad").features
    assert "route_map" not in load_team("stock").features

"""여행 경로 지도 — 여행 계획서 → 방문지 목록 → 좌표 → 움직이는 지도.

1) 방문지 뽑기  : 보조 모델이 계획서에서 '일차·순서·시간대·장소·종류(문화/걷기/음식…)·대략 좌표'를 구조화 출력으로 뽑는다
2) 좌표 확인    : OpenStreetMap(Nominatim, 키 불필요)으로 장소를 검색해 좌표를 바로잡는다
                  - 모델이 추정한 좌표와 30km 이상 떨어지면 동명이인 장소로 보고 버린다 (추정 좌표 유지, '추정' 표시)
                  - 검색 결과는 data/geocode_cache.json 에 저장해 같은 장소를 다시 묻지 않는다 (OSM 이용 규칙: 초당 1회)
3) 지도 그리기  : folium(Leaflet)
                  - 종류별 기호 마커(🏛 문화, 🚶 걷기, 🍜 음식 …) + 일차-순서 번호
                  - 일차별 색의 움직이는 경로선(AntPath — 점선이 이동 방향으로 흐름)
                  - ▶ 재생 버튼(TimestampedGeoJson) — 시간 순서대로 경로가 그려지며 이동을 보여 줌
                  - 범례(기호·일차 색)
"""

import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm_factory
from core.llm_factory import NoAvailableModelError
from core.schemas import ModelsConfig

# 종류 → (기호, 이름, 색)
CATEGORIES: dict[str, tuple[str, str, str]] = {
    "culture": ("🏛", "문화·역사", "#6554C0"),
    "walk": ("🚶", "걷기·산책", "#36B37E"),
    "food": ("🍜", "음식", "#FF5630"),
    "cafe": ("☕", "카페·디저트", "#8D6E63"),
    "shopping": ("🛍", "쇼핑", "#FF8B00"),
    "nature": ("🌳", "자연·공원", "#00875A"),
    "theme": ("🎢", "테마파크·체험", "#E91E63"),
    "view": ("🌃", "야경·전망", "#172B4D"),
    "stay": ("🏨", "숙소", "#0052CC"),
    "transport": ("🚆", "교통·이동", "#5E6C84"),
}
Category = Literal["culture", "walk", "food", "cafe", "shopping", "nature", "theme", "view", "stay", "transport"]
DAY_COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1", "#6D4C41", "#3949AB"]
SLOT_HOURS = {"오전": 9, "점심": 12, "오후": 14, "저녁": 18, "밤": 20}
MAX_DISTANCE_KM = 30  # 모델 추정 좌표와 OSM 검색 좌표가 이보다 멀면 다른 장소로 본다
LEGEND_PAD_X = 190  # 오른쪽 위 범례 폭만큼 비워 두고 경로를 맞춘다 (마커가 범례에 가리지 않게)
USER_AGENT = "AgentOffice/0.1 (personal travel planner; https://github.com/jeong-yun/Project)"


class Stop(BaseModel):
    day: int = Field(description="몇 일차인지 (1부터)")
    order: int = Field(description="그날 방문 순서 (1부터)")
    time: str = Field(description="시간대: 오전 / 점심 / 오후 / 저녁 / 밤 중 하나, 또는 'HH:MM'")
    name: str = Field(description="장소 이름 (한국어)")
    search_name: str = Field(description="지도 검색용 정식 명칭 + 도시 (현지어 또는 영어. 예: 'Osaka Castle, Osaka')")
    category: Category = Field(description="종류")
    lat: float = Field(description="대략적인 위도 (모르면 도시 중심)")
    lon: float = Field(description="대략적인 경도 (모르면 도시 중심)")
    note: str = Field(description="한 줄 메모 (할 일·이동 방법 등). 없으면 빈 문자열")


class Itinerary(BaseModel):
    destination: str = Field(description="여행 목적지 (도시·지역)")
    stops: list[Stop] = Field(description="방문지 목록 (일차·순서대로). 구체적인 장소가 있는 일정만")


class RouteResult(BaseModel):
    """화면·저장용 결과 — 방문지 + 좌표 출처 + 기록."""

    itinerary: Itinerary
    verified: dict[str, bool] = Field(default_factory=dict)  # "day-order" → OSM 좌표 확인 여부
    notes: list[str] = Field(default_factory=list)


# ─── 1) 방문지 뽑기 ────────────────────────────────────────────────────


def extract_itinerary(plan_text: str, models: ModelsConfig) -> tuple[Itinerary, list[str]]:
    categories = ", ".join(f"{k}({v[1]})" for k, v in CATEGORIES.items())
    messages = [
        SystemMessage(
            "\n".join(
                [
                    "너는 여행 계획서를 지도로 옮기는 담당이다. 계획서에서 실제로 방문하는 구체적인 장소만 일차·순서대로 뽑는다.",
                    f"- category 는 다음 중 하나: {categories}",
                    "- '휴식', '자유 시간'처럼 장소가 없는 일정은 뺀다. 숙소는 위치가 적혀 있으면 stay 로 넣는다.",
                    "- 공항·역 이동은 transport, 산책로·거리 걷기는 walk, 박물관·사찰·성은 culture, 식당은 food.",
                    "- search_name 은 지도에서 검색되도록 정식 명칭과 도시를 함께 쓴다 (현지어 또는 영어).",
                    "- 좌표는 아는 만큼 대략 쓴다 (나중에 지도 검색으로 바로잡는다).",
                ]
            )
        ),
        HumanMessage(plan_text[:12000]),
    ]
    call = llm_factory.invoke_structured(models.utility, models, Itinerary, messages, "여행 경로 추출")
    itinerary: Itinerary = call.parsed
    itinerary.stops.sort(key=lambda s: (s.day, s.order))
    return itinerary, call.errors


# ─── 2) 좌표 확인 (OpenStreetMap) ──────────────────────────────────────


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이 거리 (하버사인 공식)."""
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_geocode(query: str) -> tuple[float, float] | None:
    """Nominatim 검색 1회 (테스트에서 가짜로 바꿀 수 있게 함수로 분리)."""
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT, "Accept-Language": "ko,en"},
        timeout=10,
    )
    resp.raise_for_status()
    rows = resp.json()
    return (float(rows[0]["lat"]), float(rows[0]["lon"])) if rows else None


def geocode_stops(itinerary: Itinerary, cache_path: Path | None = None, pause: float = 1.1) -> RouteResult:
    """각 방문지 좌표를 OSM 검색으로 바로잡는다. 실패·의심스러우면 모델 추정 좌표를 유지한다."""
    cache: dict = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    result = RouteResult(itinerary=itinerary)
    for stop in itinerary.stops:
        key = f"{stop.day}-{stop.order}"
        found = None
        for query in dict.fromkeys([stop.search_name, f"{stop.name} {itinerary.destination}"]):
            if query in cache:
                found = tuple(cache[query]) if cache[query] else None
            else:
                try:
                    found = fetch_geocode(query)
                except Exception as e:  # noqa: BLE001 — 네트워크 오류는 지도 생성을 막지 않는다
                    result.notes.append(f"좌표 검색 실패 ({query}): {type(e).__name__}")
                    found = None
                cache[query] = list(found) if found else None
                time.sleep(pause)  # OSM 이용 규칙: 초당 1회 이하
            if found:
                break
        estimated = abs(stop.lat) > 0.01 or abs(stop.lon) > 0.01
        if found and (not estimated or distance_km(stop.lat, stop.lon, *found) <= MAX_DISTANCE_KM):
            stop.lat, stop.lon = found
            result.verified[key] = True
        else:
            result.verified[key] = False
            if found:
                result.notes.append(f"'{stop.name}' 검색 좌표가 추정 위치와 {MAX_DISTANCE_KM}km 이상 달라 추정 좌표 사용")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    # 좌표가 전혀 없는 장소(0,0)는 지도에서 뺀다
    itinerary.stops = [s for s in itinerary.stops if abs(s.lat) > 0.01 or abs(s.lon) > 0.01]
    return result


def build_route(plan_text: str, models: ModelsConfig, cache_path: Path | None = None) -> RouteResult:
    """계획서 → 방문지 → 좌표 확인까지 한 번에. 모델 호출이 모두 실패하면 NoAvailableModelError."""
    itinerary, errors = extract_itinerary(plan_text, models)
    result = geocode_stops(itinerary, cache_path)
    result.notes = errors + result.notes
    return result


# ─── 3) 지도 그리기 ────────────────────────────────────────────────────


def _stop_time(trip_start: str | None, stop: Stop) -> datetime:
    """재생 애니메이션용 시각: 여행 시작일 + (일차-1)일 + 시간대."""
    base = datetime.fromisoformat(trip_start) if trip_start else datetime(2000, 1, 1)
    hour = SLOT_HOURS.get(stop.time.strip()[:2], None)
    if hour is None and ":" in stop.time:
        try:
            hour = int(stop.time.split(":")[0])
        except ValueError:
            hour = None
    hour = hour if hour is not None else 9
    return base + timedelta(days=stop.day - 1, hours=hour, minutes=stop.order)  # 같은 시간대는 순서대로 1분씩


def _marker_html(stop: Stop, day_color: str) -> str:
    emoji, _, color = CATEGORIES[stop.category]
    return (
        f"<div style='position:relative;width:38px;height:38px;border-radius:50%;background:#fff;"
        f"border:3px solid {color};box-shadow:0 1px 4px rgba(0,0,0,.35);display:flex;align-items:center;"
        f"justify-content:center;font-size:20px'>{emoji}"
        f"<span style='position:absolute;right:-8px;top:-8px;background:{day_color};color:#fff;border-radius:9px;"
        f"padding:0 5px;font-size:11px;font-weight:700;line-height:17px'>{stop.day}-{stop.order}</span></div>"
    )


def _legend_html(itinerary: Itinerary) -> str:
    used = list(dict.fromkeys(s.category for s in itinerary.stops))
    days = sorted({s.day for s in itinerary.stops})
    rows = "".join(f"<div>{CATEGORIES[c][0]} {CATEGORIES[c][1]}</div>" for c in used)
    day_rows = "".join(
        f"<div><span style='display:inline-block;width:18px;height:4px;background:{DAY_COLORS[(d - 1) % len(DAY_COLORS)]};"
        f"vertical-align:middle;margin-right:6px'></span>{d}일차</div>"
        for d in days
    )
    return (
        "<div style='position:fixed;top:12px;right:12px;z-index:9999;background:rgba(255,255,255,.95);"
        "padding:8px 12px;border-radius:10px;box-shadow:0 1px 6px rgba(0,0,0,.3);font-size:13px;line-height:1.6'>"
        f"<b>범례</b>{rows}<hr style='margin:4px 0'>{day_rows}"
        "<div style='color:#666;font-size:11px;margin-top:4px'>숫자 = 일차-순서 · ▶ 로 이동 재생</div></div>"
    )


def build_map(route: RouteResult, day: int | None = None, trip_start: str | None = None):
    """folium 지도 객체. day 를 주면 그날만 그린다."""
    import folium
    from folium.plugins import AntPath, TimestampedGeoJson

    stops = [s for s in route.itinerary.stops if day is None or s.day == day]
    if not stops:
        return None
    center = [sum(s.lat for s in stops) / len(stops), sum(s.lon for s in stops) / len(stops)]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)

    # 일차별 움직이는 경로선 + 기호 마커
    for d in sorted({s.day for s in stops}):
        color = DAY_COLORS[(d - 1) % len(DAY_COLORS)]
        day_stops = [s for s in stops if s.day == d]
        if len(day_stops) > 1:
            AntPath([[s.lat, s.lon] for s in day_stops], color=color, weight=5, delay=900, dash_array=[12, 18],
                    tooltip=f"{d}일차 경로").add_to(m)
        for s in day_stops:
            emoji, label, _ = CATEGORIES[s.category]
            verified = route.verified.get(f"{s.day}-{s.order}", False)
            folium.Marker(
                [s.lat, s.lon],
                icon=folium.DivIcon(html=_marker_html(s, color), icon_size=(38, 38), icon_anchor=(19, 19)),
                tooltip=f"{s.day}일차 {s.order}번 · {s.time} · {s.name} ({emoji} {label})",
                popup=folium.Popup(
                    f"<b>{s.day}일차 {s.order}. {s.name}</b><br>{emoji} {label} · {s.time}<br>{s.note}"
                    + ("" if verified else "<br><i style='color:#999'>위치는 추정값</i>"),
                    max_width=260,
                ),
            ).add_to(m)

    # ▶ 재생: 시간 순서대로 경로가 그려지며 이동
    times = [_stop_time(trip_start, s).isoformat() for s in stops]
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[s.lon, s.lat] for s in stops]},
            "properties": {"times": times, "style": {"color": "#FF1744", "weight": 3, "opacity": 0.8}},
        },
        *[
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [s.lon, s.lat]},
                "properties": {
                    "time": t,
                    "popup": f"{s.day}일차 {s.order}. {s.name}",
                    "icon": "circle",
                    "iconstyle": {"fillColor": CATEGORIES[s.category][2], "fillOpacity": 0.9, "stroke": True,
                                  "color": "#fff", "radius": 7},
                },
            }
            for s, t in zip(stops, times)
        ],
    ]
    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT1H", duration=None, add_last_point=True, auto_play=False, loop=False,
        transition_time=150, speed_slider=False,  # 1시간 = 0.15초 (iframe 안에서 속도 슬라이더가 NaN 으로 보여 숨김)
        loop_button=True, date_options="YYYY-MM-DD HH:mm", time_slider_drag_update=True,
    ).add_to(m)

    # 경로 전체가 보이게 맞춘다. 재생 플러그인이 첫 지점으로 확대해 버리므로, 모든 스크립트가 실행된 뒤 한 번 더 맞춘다
    bounds = [[min(s.lat for s in stops), min(s.lon for s in stops)], [max(s.lat for s in stops), max(s.lon for s in stops)]]
    m.fit_bounds(bounds, padding_top_left=(30, 30), padding_bottom_right=(LEGEND_PAD_X, 70))
    # (지도 칸의 크기가 아직 0 이면 Leaflet 이 최대 확대로 계산하므로, 크기가 생길 때까지 잠깐씩 기다렸다가 맞춘다)
    name = m.get_name()
    m.get_root().script.add_child(folium.Element(
        "(function(){var tries=0;function fitAll(){var mp=" + name + ";mp.invalidateSize();"
        "if(mp.getSize().x===0&&tries++<20){return setTimeout(fitAll,250);}"
        "mp.fitBounds(" + json.dumps(bounds) + ",{paddingTopLeft:[30,30],paddingBottomRight:[" + str(LEGEND_PAD_X) + ",70],maxZoom:15});}"
        "setTimeout(fitAll,300);window.addEventListener('resize',fitAll);})();"
    ))
    m.get_root().html.add_child(folium.Element(_legend_html(route.itinerary)))
    return m


def map_html(route: RouteResult, day: int | None = None, trip_start: str | None = None) -> str:
    """지도를 HTML 문자열로 (화면 표시·파일 저장용)."""
    m = build_map(route, day, trip_start)
    return m.get_root().render() if m else ""

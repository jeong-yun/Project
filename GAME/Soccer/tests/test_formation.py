"""v4.2 테스트: 포메이션 바꾸기(킥오프 전 · 경기 중 · 퇴장 후 · AI 전술) + 엔진/AI 분리.

실행:  python -m pytest tests -q
"""
from collections import Counter

import pytest

import soccer as s
import soccer_ai
import soccer_engine
from soccer import L, W
from test_rules import make_world, player, put_ball


def roles(w, team):
    return Counter(p.role for p in w.players if p.team == team and p.active)

def shape(name):
    return Counter(r for r, _, _ in s.FORMATIONS[name])


# ---------------- 포메이션 데이터 ----------------
@pytest.mark.parametrize("name", list(s.FORMATIONS))
def test_every_formation_is_valid(name):
    slots = s.FORMATIONS[name]
    assert len(slots) == 11 and sum(1 for r, _, _ in slots if r == 'GK') == 1
    assert all(0 < fy < 1 and 0 < fx < 0.8 for _, fx, fy in slots)
    parts = [int(x) for x in name.split('-')]               # 이름 = 수비-…-공격 인원
    assert sum(parts) == 10 and parts[0] == shape(name)['DEF'] and parts[-1] == shape(name)['FWD']

def test_world_starts_in_chosen_formations():
    s.random.seed(1)
    w = s.World(formations=('4-3-3', '3-5-2'))
    assert roles(w, 0) == shape('4-3-3') and roles(w, 1) == shape('3-5-2')
    for p in w.players:                                    # 킥오프: 자기 진영
        assert (p.pos[0] < L / 2) if p.team == 0 else (p.pos[0] > L / 2)
    with pytest.raises(ValueError):
        s.World(formations=('4-4-2', '2-3-5'))

def test_default_442_keeps_v4_positions():
    # 4-4-2는 ROLE_BASE_FX와 같은 깊이라 v4와 같은 자리(동작 변화 없음)
    w = make_world()
    w.owner = None; w.ball.last_team = 1; w.ball.pos = [52.5, W / 2, 0.0]
    for p in w.players:
        if p.role != 'GK':
            assert p.base_fx == s.ROLE_BASE_FX[p.role]

def test_formation_depth_offsets_change_home_positions():
    # 4-2-3-1: 수비형 미드(0.38)는 공격형 미드(0.55)보다 뒤에 선다
    s.random.seed(2)
    w = s.World(formations=('4-2-3-1', '4-4-2'))
    w.owner = None; w.ball.last_team = 1; w.ball.pos = [52.5, W / 2, 0.0]
    mids = [p for p in w.players if p.team == 0 and p.role == 'MID']
    dm = [w.home_pos(p, w.ball)[0] for p in mids if p.base_fx == 0.38]
    am = [w.home_pos(p, w.ball)[0] for p in mids if p.base_fx == 0.55]
    assert len(dm) == 2 and len(am) == 3 and max(dm) + 10 < min(am)


# ---------------- 바꾸기 ----------------
def test_set_formation_before_kickoff_repositions():
    w = make_world()
    before = {p.pid: list(p.pos) for p in w.players if p.team == 0}
    ok, _ = s.set_formation(w, 0, '3-5-2')
    assert ok and w.formation[0] == '3-5-2' and roles(w, 0) == shape('3-5-2')
    assert any(before[p.pid] != p.pos for p in w.players if p.team == 0)   # 새 자리로 섰다
    assert not w.formation_log                                            # 킥오프 전은 기록 안 함

def test_set_formation_mid_match_keeps_players_and_logs():
    w = s.simulate(1800, seed=3)
    on_before = {p.pid for p in w.players if p.team == 1 and p.active}
    defs_before = {p.pid for p in w.players if p.team == 1 and p.role == 'DEF'}
    ok, _ = s.set_formation(w, 1, '4-3-3')
    assert ok and roles(w, 1) == shape('4-3-3')
    assert {p.pid for p in w.players if p.team == 1 and p.active} == on_before   # 같은 11명
    assert {p.pid for p in w.players if p.team == 1 and p.role == 'DEF'} == defs_before   # 수비는 그대로
    assert w.formation_log[-1][1:] == (1, '4-3-3') and w.events.get("FORMATION 4-3-3") == 1
    for _ in range(600):                                   # 바꾼 뒤에도 경기는 정상
        s.step(w, s.DT, human_mode=False)
    assert not s.set_formation(w, 1, '1-1-8')[0]

def test_red_card_reshapes_and_drops_a_forward_slot():
    w = make_world()
    d = player(w, 0, 'DEF', 1)
    s.send_off(w, d)
    put_ball(w, 40.0, -0.4, last_team=1)                   # 멈춤 → AI 감독이 다시 배정
    s.rules(w)
    r = roles(w, 0)
    assert sum(r.values()) == 10 and r['DEF'] == 4          # 뒷줄 구멍을 메우고
    assert r['FWD'] == 1                                    # 가장 앞 자리를 비운다

def test_lineup_with_other_formation_from_squad_file():
    w = make_world(3)
    d = s.squad_dict(w)
    assert d['teams'][0]['formation'] == '4-4-2'
    d['teams'][0]['formation'] = '4-3-3'
    d['teams'][1]['formation'] = '5-3-2'
    w2 = s.World(squad=d)
    assert w2.formation == ['4-3-3', '5-3-2']
    assert roles(w2, 0) == shape('4-3-3') and roles(w2, 1) == shape('5-3-2')
    # 4-3-3엔 공격 자리가 3개인데 선발 공격수는 2명 → 미드 1명이 공격 자리(주 포지션은 그대로)
    moved = [p for p in w2.players if p.team == 0 and p.role != p.pos_role]
    assert len(moved) == 1 and moved[0].pos_role == 'MID' and moved[0].role == 'FWD'


# ---------------- AI 전술 ----------------
def test_ai_switches_to_attacking_formation_when_losing_late():
    w = make_world(); w.match_len = 60.0
    w.score = [0, 1]
    w.time = 60.0 * 72 / 90                                 # 72분
    s.on_stoppage(w)
    assert w.formation[0] == w.sub_policy[0]['chase_formation'] == '4-3-3'
    assert w.formation[1] == '4-4-2'                        # 이기고 있지만 아직 80분 전
    w.time = 60.0 * 82 / 90
    s.on_stoppage(w)
    assert w.formation[1] == '5-3-2'
    s.set_formation(w, 0, '4-4-2')                          # 한 번만 바꾼다(되돌려도 다시 안 바꿈)
    s.on_stoppage(w)
    assert w.formation[0] == '4-4-2'

def test_ai_tactics_can_be_turned_off():
    w = make_world(); w.match_len = 60.0
    w.score = [0, 2]; w.time = 60.0 * 85 / 90
    w.sub_policy[0]['tactics'] = False
    s.on_stoppage(w)
    assert w.formation[0] == '4-4-2'


# ---------------- 엔진 / AI 분리 ----------------
def test_engine_and_ai_are_separate_modules():
    assert 'ai_intent' not in vars(soccer_engine) and 'pass_options' not in vars(soccer_engine)
    assert soccer_ai.ai_intent is s.ai_intent and soccer_engine.step is s.step
    assert not any(n in vars(soccer_engine) for n in ('draw', 'game_loop', 'draw_card'))
    assert 'TOUCH_TIME' in vars(soccer_ai) and 'TOUCH_TIME' not in vars(soccer_engine)

def test_engine_runs_without_ui_module():
    assert 'soccer_ui' not in vars(soccer_engine) and 'soccer_ui' not in vars(soccer_ai)
    w = soccer_engine.World(20.0)
    while not w.finished:
        soccer_engine.step(w, soccer_engine.DT, human_mode=False)
    assert w.events.get("FULL TIME") == 1

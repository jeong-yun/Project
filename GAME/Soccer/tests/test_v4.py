"""v4 테스트: AI 판단(패스 길목·수비 라인) · 세트피스 루틴 · 경기 시간/피로 · 지표 범위.

실행:  python -m pytest tests -q      (GAME/Soccer 폴더에서)
"""
import random

import pytest

import soccer as s
from soccer import L, W
from test_rules import make_world, park, player, put_ball


# ---------------- 1. AI 판단 ----------------
def lane_setup():
    # 0팀 패서(40,34) · 앞쪽 동료 B(56,34) · 옆쪽 동료 C(44,20). 나머지는 멀리
    w = make_world()
    park(w, 0, 10.0); park(w, 1, 95.0)
    a = player(w, 0, 'MID', 0); a.pos = [40.0, 34.0]
    b = player(w, 0, 'MID', 1); b.pos = [56.0, 34.0]
    c = player(w, 0, 'MID', 2); c.pos = [44.0, 20.0]
    put_ball(w, 40.8, 34.0)
    w.owner = a; w.ball.owner = a
    return w, a, b, c

def test_open_lane_prefers_forward_pass():
    w, a, b, c = lane_setup()
    best = s.pass_options(w, a)[0]
    assert best[2] is b and best[1] > 0.9                 # 길이 열려 있으면 앞쪽 동료

def test_blocked_lane_makes_ai_choose_another_option():
    w, a, b, c = lane_setup()
    blocker = player(w, 1, 'MID', 0); blocker.pos = [48.0, 34.4]   # 패스 길 한가운데
    assert s.lane_risk(w, a, b.pos[0], b.pos[1], s.pass_speed(16), b) > 0.9
    opts = s.pass_options(w, a)
    assert opts[0][2] is c                                # 막히면 다른 동료를 고른다
    w.owner_t = 1.0                                       # 공 잡고 고개 든 뒤
    it = s.ai_on_ball(w, a)
    assert it.action == 'pass' and it.target[2] is c

def test_obvious_offside_runner_is_not_picked():
    w, a, b, c = lane_setup()
    b.pos = [92.0, 34.0]                                  # 1팀 수비(x=95) 라인 근처는 괜찮지만
    for q in w.players:
        if q.team == 1 and q.role != 'GK':
            q.pos[0] = 70.0                               # 라인을 70으로 당기면 B는 22m 오프사이드
    assert s.offside_obvious(w, a, b)
    assert all(o[2] is not b for o in s.pass_options(w, a))

def test_defensive_line_moves_with_ball():
    w = make_world()
    w.ball.last_team = 1                                  # 0팀 수비 중
    w.owner = None
    d = player(w, 0, 'DEF', 1)
    xs = []
    for bx in (15.0, 52.5, 90.0):
        w.ball.pos = [bx, W / 2, 0.0]
        xs.append(w.home_pos(d, w.ball)[0])
    assert xs[0] < xs[1] < xs[2]                          # 공이 올라가면 라인도 올라간다
    assert xs[0] == pytest.approx(s.LINE_DEF_MIN * L)     # 박스 근처: 가장 깊게(10.5m)
    assert xs[2] == pytest.approx(s.LINE_DEF_MAX * L)     # 공이 상대 진영 끝: 최대 47m
    w.ball.last_team = 0                                  # 공을 가지면 라인을 더 올린다
    w.ball.pos = [52.5, W / 2, 0.0]
    assert w.home_pos(d, w.ball)[0] > xs[1]

def test_defensive_line_is_mirrored_for_team_1():
    w = make_world()
    w.owner = None
    d0, d1 = player(w, 0, 'DEF', 1), player(w, 1, 'DEF', 1)
    for bx in (20.0, 60.0):
        w.ball.pos = [bx, W / 2, 0.0]; w.ball.last_team = 1
        x0 = w.home_pos(d0, w.ball)[0]
        w.ball.pos = [L - bx, W / 2, 0.0]; w.ball.last_team = 0
        x1 = w.home_pos(d1, w.ball)[0]
        assert x1 == pytest.approx(L - x0)

def test_xg_is_mirrored_and_sane():
    # v4 중 발견한 버그: 1팀 xG 각도가 ±π를 넘어 하프라인에서 xG 0.98이었다
    w = make_world()
    park(w, 0, 10.0); park(w, 1, 95.0)
    for p in w.players:
        p.pos = [-50.0, -50.0]                             # 슛 길 막는 선수 없음
    assert s.shot_xg(w, 1, 51.0, W / 2) == pytest.approx(s.shot_xg(w, 0, L - 51.0, W / 2))
    assert s.shot_xg(w, 1, 51.0, W / 2) < 0.01
    assert 0.25 < s.shot_xg(w, 0, L - 11.0, W / 2) < 0.40    # 페널티 스폿 근처 ≈ 0.33


# ---------------- 2. 세트피스 ----------------
def run_until_kick(w, max_steps=600):
    snap = None
    for _ in range(max_steps):
        if w.setpiece is None:
            break
        snap = [(p.team, p.pos[0], p.pos[1]) for p in w.players if p.active]
        s.step(w, s.DT, human_mode=False)
    return snap

def test_corner_produces_box_runs_and_aimed_cross():
    w = make_world(4)
    for _ in range(300):
        s.step(w, s.DT, human_mode=False)
    s.restart(w, 0, L - 0.5, 0.5, 'corner')
    assert w.setpiece is not None and w.setpiece['kind'] == 'corner'
    snap = run_until_kick(w)
    assert w.setpiece is None
    in_box = sum(1 for t, x, y in snap if t == 0 and s.in_box(x, y, 1))
    assert in_box >= 3                                    # 침투조가 박스에 들어갔다
    assert w.match['crosses'][0] == 1 and w.aerial is not None
    tx, ty = s.aerial_point(w.ball)                       # 공이 머리 높이로 떨어지는 곳 = 박스 근처
    assert L - tx < s.PBOX_DEPTH + 3 and abs(ty - W / 2) < s.PBOX_HALF + 3   # (조준 오차 몇 m 허용)

def test_direct_free_kick_has_wall_and_shot():
    w = make_world(6)
    for _ in range(120):
        s.step(w, s.DT, human_mode=False)
    s.restart(w, 0, L - 22.0, W / 2 + 3, 'freekick', push=True)
    assert s.fk_direct(w, w.setpiece)
    bx, by = w.ball.pos[0], w.ball.pos[1]
    snap = run_until_kick(w)
    assert w.match['shots'][0] == 1 and w.events.get("FREE KICK SHOT") == 1
    wall = [1 for t, x, y in snap if t == 1 and abs(s.vlen(x - bx, y - by) - 9.3) < 1.5
            and x > bx]                                    # 공과 골대 사이 9.15m 거리
    assert len(wall) >= 2

def test_corner_kicked_straight_out_is_goal_kick():
    # 재개한 팀이 마지막 터치 → 코너킥이 그대로 골라인을 넘으면 골킥(예전엔 코너가 반복될 수 있었음)
    w = make_world()
    s.restart(w, 0, L - 0.5, 0.5, 'corner')
    w.setpiece = None
    put_ball(w, L + 0.3, 10.0, last_team=w.ball.last_team)
    s.rules(w)
    assert w.events.get("GOAL KICK") == 1


# ---------------- 3. 경기 시간 · 피로 · 교체 ----------------
def test_match_ends_at_full_time():
    w = s.simulate(10 ** 6, seed=3, match_len=20.0)
    assert w.finished and w.time == pytest.approx(20.0, abs=2 * s.DT)
    assert w.events.get("HALF TIME") == 1 and w.events.get("FULL TIME") == 1
    assert s.game_minute(w) == pytest.approx(90.0, abs=0.1)
    before = (w.time, list(w.ball.pos))
    s.step(w, s.DT, human_mode=False)                     # 종료 후엔 아무것도 안 움직인다
    assert (w.time, w.ball.pos) == before

def test_stamina_lowers_accuracy():
    w = make_world()
    park(w, 1, 95.0)
    p = player(w, 0, 'FWD'); p.pos = [60.0, W / 2]; p.vel = [0.0, 0.0]
    fresh_pass, fresh_shot = s.pass_error_deg(p), s.shot_error_deg(w, p)
    p.stamina, p.energy = 20.0, 0.6                       # 지친 선수
    assert s.pass_error_deg(p) > fresh_pass * 1.3
    assert s.shot_error_deg(w, p) > fresh_shot * 1.3
    # 실제로 찬 공의 방향 퍼짐도 커진다(같은 난수열로 비교)
    def spread(stamina, energy, n=300):
        p.stamina, p.energy = stamina, energy
        random.seed(11); devs = []
        for _ in range(n):
            put_ball(w, p.pos[0] + 0.8, p.pos[1]); w.owner = p
            s.do_shot(w, p, 0.7)
            devs.append(abs(w.ball.vel[1] / w.ball.vel[0]))
        return sum(devs) / n
    assert spread(20.0, 0.6) > spread(100.0, 1.0)

def test_energy_drains_and_tired_players_are_substituted():
    w = s.simulate(10 ** 6, seed=5, match_len=60.0)
    assert w.finished
    assert min(p.energy for p in w.players if p.active) < 0.95   # 90분 동안 지친다
    w2 = make_world(); w2.match_len = 60.0; w2.time = 45.0     # 67.5분
    tired = player(w2, 0, 'MID', 2); tired.energy = 0.5
    old_pid = tired.pid
    s.maybe_subs(w2)
    assert w2.match['subs'][0] == 1 and w2.events.get("SUBSTITUTION") == 1
    assert old_pid not in [p.pid for p in w2.players]
    w3 = make_world(); w3.match_len = 60.0; w3.time = 10.0     # 15분: 아직 교체 안 함
    player(w3, 0, 'MID', 2).energy = 0.5
    s.maybe_subs(w3)
    assert w3.match['subs'] == [0, 0]


# ---------------- 4. 지표가 실제 축구 범위인가(시드 여러 개) ----------------
def test_metrics_within_realistic_range():
    # 90분 경기(150초로 압축) × 시드 8개 ≈ 45초. 슛 표본이 25개 안팎이라 범위는 넉넉하게 잡았다
    # (정확한 값은 10경기 × 90분 측정 — READ_ME 9절)
    tot = {k: [0, 0] for k in s.MATCH_KEYS}; goals = 0
    for seed in range(1, 9):
        w = s.simulate(10 ** 6, seed=seed, match_len=150.0)
        assert w.finished
        for k in s.MATCH_KEYS:
            tot[k][0] += w.match[k][0]; tot[k][1] += w.match[k][1]
        goals += sum(w.score)
    S = lambda k: sum(tot[k])
    pass_pct = S('passes_ok') / S('passes')
    poss0 = tot['poss_time'][0] / S('poss_time')
    assert 0.68 <= pass_pct <= 0.90, pass_pct             # 실제 75~85%
    # 슛 수는 짧은 경기 8개로는 10~24개로 크게 흔들린다(같은 코드·같은 기대값).
    # 그래서 여기선 '공격이 죽지 않았나'(8개 이상)와 'v3처럼 골이 쏟아지지 않나'(30% 이하)만 본다.
    # 슛→골 8~15% 같은 정확한 범위는 10경기 × 90분 측정(READ_ME 9.4)이 기준
    assert S('shots') >= 8 and S('saves') >= 2
    assert goals <= 0.30 * S('shots'), goals / S('shots')          # 실제 약 10%, v3 19%
    assert S('offsides') >= 1                             # v3: 10경기 0회
    assert 0.35 <= poss0 <= 0.65, poss0

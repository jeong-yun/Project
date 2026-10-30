"""v4.1 테스트: 스쿼드·선발 · 경기 중 교체 규칙 · AI 자동 교체 · 선수 카드 · 조작 선수 고정.

엔진 함수만 부른다(화면 없이). 실행:  python -m pytest tests -q
"""
import json

import pytest

import soccer as s
from soccer import L, W
from test_rules import make_world, park, player, put_ball


def stoppage(w, team=0):
    # 공이 멈추는 상황 하나(스로인) — 대기 중 교체가 여기서 적용된다
    put_ball(w, 40.0, -0.4, last_team=1 - team)
    s.rules(w)


# ---------------- 선수 선택(스쿼드 · 선발) ----------------
def test_auto_squad_has_names_numbers_and_bench():
    w = make_world()
    for team in (0, 1):
        on = [p for p in w.players if p.team == team]
        assert len(on) == 11 and len(w.bench[team]) == 7
        assert [p.pos_role for p in w.bench[team]] == list(s.BENCH_ROLES)
        names = [p.name for p in on + w.bench[team]]
        assert len(set(names)) == 18 and all(names)
        assert sorted(p.number for p in on + w.bench[team]) == list(range(1, 19))

def test_lineup_from_squad_file_sets_the_right_11(tmp_path):
    w = make_world(3)
    d = s.squad_dict(w)
    t0 = d['teams'][0]
    # 벤치 수비(번호 13)를 선발 수비 3번 자리 대신 넣는다
    bench_def = next(e for e in t0['players'] if e['number'] == 13)
    assert bench_def['role'] == 'DEF'
    t0['lineup'] = [n for n in t0['lineup'] if n != 3] + [13]
    t0['players'][0]['name'] = 'Keeper'
    path = tmp_path / "squad.json"
    path.write_text(json.dumps(d), encoding='utf-8')
    w2 = s.World(squad=s.load_squad(path))
    on = [p for p in w2.players if p.team == 0]
    assert sorted(p.number for p in on) == sorted(t0['lineup'])
    assert on[0].name == 'Keeper' and on[0].role == 'GK'
    assert [p.role for p in on] == [r for r, _, _ in s.FORMATION]      # 4-4-2 자리에 역할대로
    assert 3 in [p.number for p in w2.bench[0]]                       # 빠진 선수는 벤치로
    p13 = next(p for p in on if p.number == 13)
    assert p13.stats.physical == pytest.approx(bench_def['stats']['physical'])

def test_bad_lineup_is_rejected():
    d = s.squad_dict(make_world())
    d['teams'][1]['lineup'] = d['teams'][1]['lineup'][:10] + [12]    # FWD 대신 GK(12번)
    with pytest.raises(ValueError):
        s.World(squad=d)

def test_squad_file_match_runs_headless(tmp_path):
    d = s.squad_dict(make_world(8))
    w = s.simulate(3000, seed=8, squad=d)
    assert w.time == pytest.approx(50.0, abs=0.1)

def test_swap_lineup_before_kickoff_only():
    w = make_world()
    d = player(w, 0, 'DEF', 1)
    idx = next(i for i, p in enumerate(w.bench[0]) if p.pos_role == 'DEF')
    newcomer = w.bench[0][idx]
    ok, _ = s.swap_lineup(w, 0, d.pid, idx)
    assert ok and newcomer in w.players and d in w.bench[0]
    assert (newcomer.base_fx, newcomer.base_fy) == (d.base_fx, d.base_fy)
    fwd_idx = next(i for i, p in enumerate(w.bench[0]) if p.pos_role == 'FWD')
    assert not s.swap_lineup(w, 0, newcomer.pid, fwd_idx)[0]          # 포지션 다르면 안 됨
    w.time = 1.0
    assert not s.swap_lineup(w, 0, newcomer.pid, idx)[0]              # 킥오프 뒤엔 교체로

def test_control_lock_and_switch():
    w = make_world()
    keys_none = {}
    m = player(w, 0, 'MID', 2)
    w.human_pick = m.pid
    s.select_controlled(w)
    assert w.controlled is m                                           # 고정한 선수
    s.switch_control(w)                                                # Q
    assert w.controlled is not m and w.human_pick == w.controlled.pid
    w.human_pick = None                                                # L: 자동
    s.select_controlled(w)
    b = w.ball
    nearest = min((p for p in w.players if p.team == 0 and p.role != 'GK'),
                  key=lambda p: s.vlen(b.pos[0] - p.pos[0], b.pos[1] - p.pos[1]))
    assert w.controlled is nearest


# ---------------- 선수 정보 ----------------
def test_player_info_matches_engine_state():
    w = s.simulate(3600, seed=2)
    p = max((q for q in w.players if q.team == 0), key=lambda q: w.pstats.get(q.pid, {}).get('passes', 0))
    info = s.player_info(w, p)
    assert info['name'] == p.name and info['number'] == p.number and info['where'] == 'pitch'
    assert info['stats']['pace'] == round(p.stats.pace) and info['stats']['keeping'] == round(p.stats.keeping)
    assert info['stamina'] == round(p.stamina) and info['energy'] == round(p.energy * 100)
    assert info['match'] == w.pstats[p.pid] and info['match']['passes'] > 0
    assert info['yellow'] == w.cards.get(p.pid, 0)
    # 선수별 패스 합 = 팀 기록
    for team in (0, 1):
        pids = {q.pid for q in w.players + w.bench[team] + w.subbed_off if q.team == team}
        assert sum(w.pstats.get(i, {}).get('passes', 0) for i in pids) == w.match['passes'][team]
    assert s.player_info(w, w.bench[1][0])['where'] == 'bench'

def test_goal_is_credited_to_scorer():
    w = make_world()
    park(w, 0, 10.0); park(w, 1, 30.0)
    f = player(w, 0, 'FWD')
    put_ball(w, L - 3.0, W / 2)
    s.kick(w, 20.0, 0.0, 0.3, 0, f)
    put_ball(w, L + 0.3, W / 2, z=0.5, last_team=0)
    s.rules(w)
    assert w.score == [1, 0] and w.pstats[f.pid]['goals'] == 1


# ---------------- 경기 중 교체 ----------------
def test_sub_applies_only_at_a_stoppage():
    w = make_world()
    out = player(w, 0, 'MID', 1)
    ok, why = s.request_sub(w, 0, out.pid, 3)                          # 벤치 3번 = MID
    inn = w.bench[0][3]
    assert ok and w.sub_queue
    for _ in range(30):                                                # 경기 진행 중엔 그대로
        w.ball.pos = [L / 2, W / 2, 0.0]; w.ball.vel = [0.0, 0.0, 0.0]
        s.step(w, s.DT, human_mode=False)
    assert out in w.players and w.match['subs'] == [0, 0]
    stoppage(w)
    assert out not in w.players and inn in w.players
    assert (inn.base_fx, inn.base_fy, inn.role) == (out.base_fx, out.base_fy, out.role)
    assert inn.stamina == 100.0 and inn.energy == 1.0
    assert w.match['subs'] == [1, 0] and w.events.get("SUBSTITUTION") == 1
    assert w.sub_log[-1][1:] == (0, out.name, inn.name)

def test_sub_limit_is_enforced():
    w = make_world()
    outs = [p for p in w.players if p.team == 0 and p.role != 'GK']
    for k in range(s.SUBS_MAX):
        assert s.request_sub(w, 0, outs[k].pid, 0)[0]
        stoppage(w)
    assert w.match['subs'][0] == s.SUBS_MAX
    ok, why = s.request_sub(w, 0, outs[s.SUBS_MAX].pid, 0)
    assert not ok and "no subs left" in why

def test_queued_subs_count_toward_limit():
    w = make_world()
    outs = [p for p in w.players if p.team == 1 and p.role != 'GK']
    for k in range(s.SUBS_MAX):
        assert s.request_sub(w, 1, outs[k].pid, k)[0]
    assert not s.request_sub(w, 1, outs[s.SUBS_MAX].pid, 5)[0]

def test_subbed_off_player_cannot_return():
    w = make_world()
    out = player(w, 0, 'FWD', 0)
    s.request_sub(w, 0, out.pid, 5)
    stoppage(w)
    assert out in w.subbed_off and out not in w.bench[0]               # 벤치로 안 돌아감
    ok, why = s.request_sub(w, 0, out.pid, 0)
    assert not ok and "already substituted" in why

def test_red_carded_player_cannot_be_replaced(monkeypatch):
    w = make_world()
    d = player(w, 1, 'DEF', 0)
    s.send_off(w, d)
    ok, why = s.request_sub(w, 1, d.pid, 1)
    assert not ok and "sent-off" in why
    assert sum(1 for p in w.players if p.team == 1 and p.active) == 10
    # 대기 중이던 교체 대상이 그 사이 퇴장당하면 교체는 취소
    m = player(w, 1, 'MID', 0)
    assert s.request_sub(w, 1, m.pid, 3)[0]
    s.send_off(w, m)
    stoppage(w, team=1)
    assert w.match['subs'][1] == 0 and sum(1 for p in w.players if p.team == 1 and p.active) == 9

def test_ai_auto_sub_on_low_energy_in_90_minute_sim():
    # 90분(120초 압축) 시뮬. 기준을 0.85로 올린 정책(설정 가능)으로 — 기본 0.75면 경기마다 0~3명
    s.random.seed(5)
    w = s.World(match_len=120.0)
    for t in (0, 1):
        w.sub_policy[t]['energy'] = 0.85
    while not w.finished:
        s.step(w, s.DT, human_mode=False)
    assert sum(w.match['subs']) >= 2 and w.sub_log
    for mnt, team, out, inn in w.sub_log:
        assert mnt >= s.SUB_AFTER_MIN                                  # 55분 이후에만
    gone = list(w.subbed_off)
    assert all(p.energy < 0.85 or w.cards.get(p.pid) for p in gone)
    assert all(p not in w.players for p in gone)

def test_auto_sub_policy_can_be_turned_off():
    w = make_world(); w.match_len = 60.0; w.time = 45.0               # 67.5분
    w.sub_policy[0]['auto'] = False
    player(w, 0, 'MID', 2).energy = 0.4
    player(w, 1, 'MID', 2).energy = 0.4
    s.maybe_subs(w)
    assert w.match['subs'] == [0, 1]                                   # 끈 팀(0)은 안 바꿈

def test_yellow_card_risk_sub():
    w = make_world(); w.match_len = 60.0; w.time = 42.0               # 63분
    d = player(w, 1, 'DEF', 2)
    w.cards[d.pid] = 1
    s.maybe_subs(w)
    assert d in w.subbed_off and w.sub_log[-1][2] == d.name

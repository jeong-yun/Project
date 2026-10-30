"""규칙·GK·어드밴티지 단위 테스트 + 시드 고정 전체 경기 테스트.

실행:  python -m pytest tests -q      (GAME/Soccer 폴더에서)
단위 테스트는 선수 위치를 직접 배치해 상황을 만든 뒤 규칙 함수를 부른다.
확률이 끼는 곳은 random.random 을 정해진 값 순서로 바꿔 결과를 고정한다.
"""
import random
import time

import pytest

import soccer as s
from soccer import L, W


# ---------------- 도우미 ----------------
def make_world(seed=0):
    random.seed(seed)
    return s.World()

def park(w, team, x):
    # 한 팀 전원을 x 줄에 세로로 세운다(공과 멀리 떼어 놓기용)
    ps = [p for p in w.players if p.team == team]
    for i, p in enumerate(ps):
        p.pos = [x, 3.0 + i * 6.0]; p.vel = [0.0, 0.0]

def player(w, team, role, k=0):
    return [p for p in w.players if p.team == team and p.role == role][k]

def put_ball(w, x, y, z=0.0, vel=(0.0, 0.0, 0.0), last_team=0):
    b = w.ball
    b.pos = [x, y, z]; b.vel = list(vel); b.cooldown = 0.0; b.last_team = last_team
    b.owner = None; w.owner = None

def fixed_random(monkeypatch, values):
    # random.random() 이 values 를 순서대로 돌려주게 한다(소진되면 0.99)
    it = iter(values)
    monkeypatch.setattr(s.random, "random", lambda: next(it, 0.99))


# ---------------- 득점 · 세트피스 ----------------
def test_goal_counts_and_kickoff():
    w = make_world()
    put_ball(w, L + 0.3, W / 2, z=0.5)
    s.rules(w)
    assert w.score == [1, 0]
    assert w.events.get("GOAL") == 1
    assert w.ball.pos[:2] == [L / 2, W / 2]

def test_over_crossbar_is_goal_kick():
    w = make_world()
    put_ball(w, L + 0.3, W / 2, z=3.0, last_team=0)      # 공격(0팀)이 내보냄
    s.rules(w)
    assert w.score == [0, 0]
    assert w.events.get("GOAL KICK") == 1
    assert w.ball.pos[0] == pytest.approx(L - 5.5)
    assert w.no_offside is True

def test_defender_last_touch_is_corner():
    w = make_world()
    put_ball(w, -0.3, 10.0, last_team=0)                  # 0팀 골라인, 0팀(수비)이 내보냄
    s.rules(w)
    assert w.events.get("CORNER") == 1
    assert w.match['corners'] == [0, 1]
    assert w.ball.pos[:2] == [1, 1]                       # 가까운 코너(restart가 1m 안쪽으로 클램프)

def test_sideline_is_throw_in_for_other_team():
    w = make_world()
    put_ball(w, 40.0, -0.4, last_team=1)
    s.rules(w)
    assert w.events.get("THROW-IN") == 1
    taker = min((p for p in w.players if p.team == 0 and p.role != 'GK'),
                key=lambda p: s.vlen(p.pos[0] - 40.0, p.pos[1] - 1.0))
    assert s.vlen(taker.pos[0] - w.ball.pos[0], taker.pos[1] - w.ball.pos[1]) < 1.0


# ---------------- 오프사이드 ----------------
def offside_setup(attacker_x):
    w = make_world()
    park(w, 0, 10.0)
    park(w, 1, L - 20.0)                                  # 수비 라인 x = L-20
    player(w, 1, 'GK').pos = [L - 1.0, W / 2]
    kicker = player(w, 0, 'MID')
    kicker.pos = [L / 2 + 5, W / 2]
    att = player(w, 0, 'FWD')
    att.pos = [attacker_x, W / 2 + 10]
    put_ball(w, kicker.pos[0] + 0.5, kicker.pos[1])
    s.kick(w, 20.0, 0.0, 0.0, 0, kicker)                  # 차는 순간 판정
    put_ball(w, att.pos[0] + 0.3, att.pos[1])             # 공이 공격수에게 도착
    w.attack_team = 0
    return w, att

def test_offside_called_when_beyond_line():
    w, att = offside_setup(L - 10.0)
    assert att.pid in w.offside_suspects
    s.update_possession(w)
    assert w.events.get("OFFSIDE") == 1
    assert w.match['offsides'] == [1, 0]

def test_onside_player_keeps_ball():
    w, att = offside_setup(L - 25.0)
    s.update_possession(w)
    assert "OFFSIDE" not in w.events
    assert w.owner is att

def test_set_piece_first_play_is_exempt():
    w = make_world()
    w.no_offside = True
    s.mark_offside(w, 0, None)
    assert w.offside_suspects == set() and w.no_offside is False


# ---------------- 골키퍼 ----------------
def shot_at_gk(w, offset_y=0.3, speed=24.0):
    # 1팀 골(x=L)을 향한 0팀 슛, GK 바로 앞에서 판정
    park(w, 0, 10.0); park(w, 1, 30.0)
    gk = player(w, 1, 'GK')
    gk.pos = [L - 2.0, W / 2]; gk.stats.keeping = 85
    put_ball(w, L - 2.5, W / 2 + offset_y, z=0.5, vel=(speed, 0.0, 0.0))
    w.shot = {'team': 0, 'pen': False, 'on_target': True, 'tried': set(), 'guess': 0}
    w.attack_team = 0
    return gk

def test_gk_catch_takes_possession_and_clears_offside(monkeypatch):
    w = make_world()
    gk = shot_at_gk(w)
    w.offside_suspects = {player(w, 0, 'FWD').pid}
    fixed_random(monkeypatch, [0.0, 0.0])                 # 선방 성공 → 캐치 성공
    s.gk_save(w)
    assert w.owner is gk and w.shot is None
    assert w.match['saves'] == [0, 1] and w.events.get("SAVE") == 1
    assert w.offside_suspects == set()

def test_gk_parry_rebound_to_offside_player_is_offside(monkeypatch):
    w = make_world()
    shot_at_gk(w)
    fwd = player(w, 0, 'FWD')
    w.offside_suspects = {fwd.pid}
    fixed_random(monkeypatch, [0.0, 0.99, 0.99])          # 선방 → 캐치 실패 → 옆으로 쳐냄
    s.gk_save(w)
    assert w.owner is None and w.events.get("SAVE") == 1
    assert fwd.pid in w.offside_suspects                  # 선방은 '수비의 의도적 플레이'가 아님
    fwd.pos = [w.ball.pos[0] - 0.3, w.ball.pos[1]]
    w.ball.cooldown = 0.0; w.ball.vel = [0.0, 0.0, 0.0]
    s.update_possession(w)
    assert w.events.get("OFFSIDE") == 1

def test_gk_miss_lets_ball_continue(monkeypatch):
    w = make_world()
    shot_at_gk(w)
    fixed_random(monkeypatch, [0.99])                     # 선방 실패
    s.gk_save(w)
    assert w.match['saves'] == [0, 0] and w.ball.vel[0] == pytest.approx(24.0)

def test_save_chance_drops_with_stretch_and_speed():
    # 같은 GK라도 몸 중심에서 멀고 빠른 슛일수록 선방률이 낮아야 한다
    def save_rate(offset, speed, n=400):
        saved = 0
        for i in range(n):
            w = make_world(i)
            shot_at_gk(w, offset_y=offset, speed=speed)
            s.gk_save(w)
            saved += w.match['saves'][1]
        return saved / n
    easy = save_rate(0.2, 20)
    hard = save_rate(1.0, 30)
    # v4: GK 기본 선방값 0.45 → 0.35(필드 슛 선방률을 실제 70% 근처로). 쉬운 슛 기준도 0.75 → 0.70
    assert easy > 0.70 and hard < easy - 0.25

def test_shot_block_by_defender(monkeypatch):
    w = make_world()
    park(w, 0, 10.0); park(w, 1, 30.0)
    d = player(w, 1, 'DEF')
    d.pos = [70.0, W / 2]
    put_ball(w, 69.5, W / 2, z=0.4, vel=(25.0, 0.0, 0.0))
    w.shot = {'team': 0, 'pen': False, 'on_target': True, 'tried': set(), 'guess': 0}
    fixed_random(monkeypatch, [0.0])
    s.update_possession(w)
    assert w.match['blocks'] == [0, 1] and w.shot is None
    assert s.vlen(w.ball.vel[0], w.ball.vel[1]) < 25.0 * 0.45

def run_penalty(w, attack_team, max_steps=600):
    # 페널티 1번: 슛이 나간 뒤 '살아있는 슛'이 끝날 때까지 진행 → 득점 여부
    before = w.score[attack_team]
    s.penalty(w, attack_team)
    shot_taken = False
    for _ in range(max_steps):
        s.step(w, s.DT, human_mode=False)
        if w.shot is not None:
            shot_taken = True
        elif shot_taken:
            break
    return w.score[attack_team] > before

def test_penalty_conversion_is_realistic():
    # 실제 축구 PK 성공률 ≈ 75~80%. GK 로직이 없을 땐 거의 100%였다
    goals = n = 0
    for seed in range(4):
        w = make_world(100 + seed)
        for i in range(30):
            goals += run_penalty(w, i % 2)
            n += 1
    rate = goals / n
    assert 0.55 <= rate <= 0.92, rate


# ---------------- 파울 · 어드밴티지 · 태클 ----------------
def foul_setup(victim_x):
    w = make_world()
    park(w, 0, 10.0); park(w, 1, 95.0)
    victim = player(w, 0, 'FWD'); victim.pos = [victim_x, W / 2 - 25]
    fouler = player(w, 1, 'DEF'); fouler.pos = [victim_x + 1.0, W / 2 - 25]
    put_ball(w, victim_x + 0.8, W / 2 - 25)
    w.owner = victim; w.ball.owner = victim
    return w, victim, fouler

def test_foul_in_box_gives_penalty(monkeypatch):
    w, victim, fouler = foul_setup(L - 8.0)
    victim.pos = [L - 8.0, W / 2]
    fixed_random(monkeypatch, [0.99])                     # 카드 없음
    s.award_foul(w, fouler, victim)
    assert w.events.get("PENALTY") == 1 and w.penalty_pending
    assert w.ball.pos[:2] == [L - s.PEN_SPOT, W / 2]

def test_foul_in_own_half_is_free_kick_with_wall_distance(monkeypatch):
    w, victim, fouler = foul_setup(40.0)
    fixed_random(monkeypatch, [0.99])
    s.award_foul(w, fouler, victim)
    assert w.events.get("FOUL") == 1 and "ADVANTAGE" not in w.events
    for p in w.players:
        if p.team == 1 and p.active:
            assert s.vlen(p.pos[0] - w.ball.pos[0], p.pos[1] - w.ball.pos[1]) >= 9.15 - 1e-6

def test_advantage_called_back_when_ball_lost(monkeypatch):
    w, victim, fouler = foul_setup(70.0)
    fixed_random(monkeypatch, [0.1, 0.99])                # 어드밴티지 선언, 카드 없음
    s.award_foul(w, fouler, victim)
    assert w.events.get("ADVANTAGE") == 1 and w.advantage is not None
    w.owner = fouler                                      # 1초 뒤 공을 뺏김
    s.update_advantage(w, 1.0)
    assert w.events.get("ADV CALLED BACK") == 1 and w.advantage is None
    assert w.ball.pos[:2] == pytest.approx([70.0, W / 2 - 25])

def test_advantage_played_on_when_ball_kept(monkeypatch):
    w, victim, fouler = foul_setup(70.0)
    fixed_random(monkeypatch, [0.1, 0.99])
    s.award_foul(w, fouler, victim)
    for _ in range(4):
        s.update_advantage(w, 1.0)
    assert w.advantage is None and "ADV CALLED BACK" not in w.events
    assert "FOUL" not in w.events

def test_tackle_has_cooldown():
    w, victim, fouler = foul_setup(40.0)
    s.do_tackle(w, fouler)
    assert w.match['tackles'][1] == 1
    w.owner = victim; w.ball.owner = victim; w.ball.cooldown = 0.0
    s.do_tackle(w, fouler)                                # 쿨다운 중 → 시도 안 됨
    assert w.match['tackles'][1] == 1

def test_pass_completion_is_counted():
    w = make_world()
    park(w, 0, 10.0); park(w, 1, 95.0)
    a = player(w, 0, 'MID', 0); a.pos = [40.0, 20.0]
    b = player(w, 0, 'MID', 1); b.pos = [50.0, 20.0]
    put_ball(w, 40.8, 20.0)
    w.owner = a; w.ball.owner = a
    s.do_pass(w, a, 0.5)
    assert w.match['passes'][0] == 1 and w.pending_pass == 0
    put_ball(w, b.pos[0] - 0.5, b.pos[1])
    s.update_possession(w)
    assert w.owner is b and w.match['passes_ok'][0] == 1


class HoldKeys:
    # pygame.key.get_pressed() 흉내: 지정한 키만 눌린 상태
    def __init__(self, *pressed): self.pressed = set(pressed)
    def __getitem__(self, k): return k in self.pressed

def test_human_mode_runs_and_controls_home_player():
    w = make_world(5)
    keys = HoldKeys(s.pygame.K_RIGHT, s.pygame.K_LSHIFT)
    for _ in range(900):                                  # 15초: 오른쪽으로 스프린트
        s.step(w, s.DT, human_keys=keys, human_mode=True)
    c = w.controlled
    assert c is not None and c.team == s.HUMAN_TEAM and c.role != 'GK'


# ---------------- 전체 경기(시드 고정) ----------------
def test_same_seed_same_match():
    w1 = s.simulate(1500, seed=7)
    w2 = s.simulate(1500, seed=7)
    assert w1.score == w2.score and w1.events == w2.events
    assert w1.ball.pos == w2.ball.pos

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_full_match_runs_and_is_sane(seed):
    t0 = time.perf_counter()
    w = s.simulate(6000, seed=seed)                       # 시뮬 100초
    ms = (time.perf_counter() - t0) / 6000 * 1000
    m = w.match
    assert ms < 8.0, f"step too slow: {ms:.2f}ms"         # 60FPS 예산(16.7ms)의 절반 이하
    # v4: '100초 안에 슛·선방이 꼭 나온다'는 조건은 뺐다. v3는 200초에 슛 27.6개(실제의 약 14배)라
    # 늘 성립했지만, 실제 축구는 100초에 슛 0.5개꼴이다. 슛·선방 수는 test_v4의 여러 경기 지표 테스트가 본다
    assert sum(m['passes']) > 0
    assert sum(m['on_target']) <= sum(m['shots'])
    assert sum(w.score) <= sum(m['on_target'])            # 골은 유효슛에서만(자책 제외 근사)
    assert sum(1 for p in w.players if p.active) >= 18

def test_possession_is_balanced_over_seeds():
    # 같은 조건의 두 팀이면 점유율이 한쪽으로 쏠리면 안 된다(예전 버그: 1팀 75~89%)
    t = [0.0, 0.0]; tackles = [0, 0]
    for seed in range(1, 6):
        w = s.simulate(4000, seed=seed)
        for i in (0, 1):
            t[i] += w.match['poss_time'][i]; tackles[i] += w.match['tackles'][i]
    share0 = t[0] / (t[0] + t[1])
    assert 0.38 <= share0 <= 0.62, share0
    assert tackles[0] > 0 and tackles[1] > 0

"""
탑다운 축구 프로토타입 (Pygame) v2 - 규칙 정교화
============================================================
설계 핵심:  Intent(의도) 레이어로 사람 / AI 컨트롤러를 분리한다.
            매 프레임 모든 선수는 Intent(이동방향, 패스/슛/태클, 파워)를
            내놓고, 엔진이 그것을 [스탯 + 물리 + 확률]로 해석한다.
            → 같은 엔진으로 두 모드를 모두 돌린다.
                - 사람 플레이 모드: 사람 1명 + AI 21명
                - 자동 시뮬 모드 : AI 22명

실사화 포인트:
    1) 공은 발에 붙지 않는 독립 물리 객체 (x,y,z + 중력/마찰/바운스/공기저항)
    2) 선수 관성: 가속/감속 + 고속에서 방향전환 제한(민첩)
    3) 결과 = 의도 + 스탯 + 상황 + 확률 (결정론 금지)
    4) 스태미나로 후반 능력치 저하
    5) 오프더볼 AI: 진형 슬라이드 + 압박 + 분산 + 오프사이드 라인 유지

규칙(이번 버전 추가):
    - 오프사이드: 패스 순간 '2nd-last 수비수 라인'을 기준으로 판정 → 간접 FK
    - 파울/카드: 태클 실패 시 반칙 확률 → 프리킥, 박스 안이면 페널티킥, 옐로/레드
    - 세트피스: 사이드라인=스로인, 골라인=마지막 터치에 따라 코너/골킥 구분

실행:  python soccer.py
조작:  WASD/화살표 이동 · Shift 스프린트 · Space 패스 · K(꾹) 슛
       J 태클 · M 모드전환 · R 리셋 · +/- 시뮬속도 · Esc 종료
"""

import sys, math, random, argparse
from dataclasses import dataclass, field

try:
    import pygame
except ImportError:
    print("pygame가 필요합니다:  pip install pygame")
    sys.exit(1)

# ---------------- 상수 ----------------
L, W = 105.0, 68.0          # 피치 길이/너비 (m)
GOAL_W = 7.32               # 골대 폭
CROSSBAR = 2.44             # 크로스바 높이
PBOX_DEPTH = 16.5           # 페널티 박스 깊이
PBOX_HALF = 20.16           # 페널티 박스 절반 폭
PEN_SPOT = 11.0             # 페널티 스폿 거리
GRAV = 9.81
GROUND_FRIC = 0.9
AIR_DRAG = 0.05
RESTITUTION = 0.55
CONTROL_RADIUS = 1.5
CONTROL_HEIGHT = 1.0
DRIBBLE_AHEAD = 1.0
KICK_COOLDOWN = 0.35
FPS = 60
DT = 1.0 / FPS
HUMAN_TEAM = 0

MARGIN = 60
PPM = 8.5
SCREEN_W = int(MARGIN * 2 + L * PPM)
SCREEN_H = int(MARGIN * 2 + W * PPM)
TEAM_COLORS = [(40, 110, 230), (230, 70, 60)]


# ---------------- 벡터/수학 ----------------
def vlen(x, y): return math.hypot(x, y)

def vnorm(x, y):
    d = math.hypot(x, y)
    return (x / d, y / d) if d > 1e-9 else (0.0, 0.0)

def clamp(v, a, b): return a if v < a else b if v > b else v

def logistic(z): return 1.0 / (1.0 + math.exp(-clamp(z, -20, 20)))

def in_box(x, y, defend_team):
    # defend_team의 페널티 박스 안인가
    if defend_team == 0:
        return x < PBOX_DEPTH and abs(y - W / 2) < PBOX_HALF
    return x > L - PBOX_DEPTH and abs(y - W / 2) < PBOX_HALF


# ---------------- 데이터 모델 ----------------
@dataclass
class Stats:
    pace: float = 70; accel: float = 70; agility: float = 70
    finishing: float = 70; passing: float = 70
    physical: float = 70; stamina: float = 70

def rand_stats(role):
    u = random.uniform
    s = Stats(u(60, 86), u(60, 86), u(60, 86), u(45, 82), u(58, 85), u(58, 86), u(60, 90))
    if role == 'FWD': s.finishing = u(72, 92); s.pace = u(76, 92)
    if role == 'DEF': s.physical = u(72, 92); s.finishing = u(35, 60)
    if role == 'GK':  s.physical = u(70, 90)
    return s

@dataclass
class Player:
    team: int
    role: str
    base_fx: float
    base_fy: float
    stats: Stats
    pos: list = field(default_factory=lambda: [0.0, 0.0])
    vel: list = field(default_factory=lambda: [0.0, 0.0])
    facing: float = 0.0
    stamina: float = 100.0
    pid: int = 0
    active: bool = True         # 퇴장 시 False

@dataclass
class Ball:
    pos: list = field(default_factory=lambda: [L / 2, W / 2, 0.0])
    vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    owner: object = None
    cooldown: float = 0.0
    last_team: int = 0

@dataclass
class Intent:
    move: tuple = (0.0, 0.0)
    sprint: bool = False
    action: str = None         # 'pass'|'shoot'|'tackle'
    power: float = 0.5


# ---------------- 월드 ----------------
FORMATION = [   # 4-4-2 (공격 +x 기준 정규화 좌표)
    ('GK', 0.05, 0.50),
    ('DEF', 0.22, 0.18), ('DEF', 0.22, 0.39), ('DEF', 0.22, 0.61), ('DEF', 0.22, 0.82),
    ('MID', 0.46, 0.18), ('MID', 0.46, 0.39), ('MID', 0.46, 0.61), ('MID', 0.46, 0.82),
    ('FWD', 0.68, 0.40), ('FWD', 0.68, 0.60),
]

class World:
    def __init__(self):
        self.players = []
        self.ball = Ball()
        self.score = [0, 0]
        self.time = 0.0
        self.owner = None
        self.controlled = None
        # 규칙 상태
        self.offside_suspects = set()   # 마지막 플레이에서 오프사이드 위치였던 pid
        self.attack_team = 0            # 마지막으로 공을 '찬' 팀
        self.no_offside = False         # 세트피스 직후 첫 플레이는 오프사이드 면제
        self.cards = {}                 # pid -> 경고 수
        self.event_text = ""
        self.event_timer = 0.0
        self.events = {}                # 이벤트 카운터(테스트용)
        pid = 0
        for team in (0, 1):
            for role, fx, fy in FORMATION:
                self.players.append(Player(team, role, fx, fy, rand_stats(role), pid=pid))
                pid += 1
        self.kickoff()

    def opp_goal(self, team): return (L, W / 2) if team == 0 else (0.0, W / 2)
    def own_goal(self, team): return (0.0, W / 2) if team == 0 else (L, W / 2)

    def home_pos(self, p, ball):
        bx = clamp(ball.pos[0] / L, 0, 1)
        by = clamp(ball.pos[1] / W, 0, 1)
        fx = clamp(p.base_fx + (bx - 0.5) * 0.5, 0.04, 0.92)
        fy = clamp(p.base_fy + (by - 0.5) * 0.25, 0.05, 0.95)
        if p.team == 0:
            return (fx * L, fy * W)
        return ((1 - fx) * L, (1 - fy) * W)

    def kickoff(self):
        self.ball.pos = [L / 2, W / 2, 0.0]
        self.ball.vel = [0.0, 0.0, 0.0]
        self.ball.owner = None
        self.ball.cooldown = 0.5
        self.owner = None
        self.offside_suspects = set()
        self.no_offside = False
        for p in self.players:
            if not p.active:
                continue
            hx, hy = self.home_pos(p, self.ball)
            p.pos = [hx, hy]; p.vel = [0.0, 0.0]


def set_event(world, text):
    world.event_text = text
    world.event_timer = 2.0
    world.events[text] = world.events.get(text, 0) + 1


# ---------------- 오프사이드 ----------------
def offside_line(world, team):
    # team이 공격할 때, 상대(수비)의 2nd-last 선수 x좌표 = 오프사이드 라인
    xs = sorted(p.pos[0] for p in world.players if p.team != team and p.active)
    if len(xs) < 2:
        return L if team == 0 else 0.0
    return xs[-2] if team == 0 else xs[1]

def mark_offside(world, team, kicker):
    world.attack_team = team
    if world.no_offside:                 # 세트피스 첫 플레이 면제
        world.no_offside = False
        world.offside_suspects = set()
        return
    line = offside_line(world, team)
    bx = world.ball.pos[0]
    s = set()
    for p in world.players:
        if p.team != team or p is kicker or not p.active or p.role == 'GK':
            continue
        if team == 0 and p.pos[0] > line and p.pos[0] > bx and p.pos[0] > L / 2:
            s.add(p.pid)
        elif team == 1 and p.pos[0] < line and p.pos[0] < bx and p.pos[0] < L / 2:
            s.add(p.pid)
    world.offside_suspects = s


# ---------------- 소유권 + 오프사이드 발동 ----------------
def update_possession(world):
    b = world.ball
    if b.cooldown > 0:
        world.owner = None; b.owner = None
        return
    best, bestd = None, CONTROL_RADIUS
    if b.pos[2] < CONTROL_HEIGHT:
        for p in world.players:
            if not p.active:
                continue
            d = vlen(b.pos[0] - p.pos[0], b.pos[1] - p.pos[1])
            if d < bestd:
                best, bestd = p, d
    if best is not None:
        if best.pid in world.offside_suspects and best.team == world.attack_team:
            world.offside_suspects = set()
            offside_freekick(world, best)
            return
        world.offside_suspects = set()
        b.last_team = best.team
    world.owner = best
    b.owner = best


# ---------------- 물리 ----------------
def step_player(world, p, intent, dt):
    s = p.stats
    fatigue = 0.55 + 0.45 * (p.stamina / 100.0)
    max_speed = (4.5 + s.pace / 99 * 4.5) * fatigue
    accel = (18 + s.accel / 99 * 22) * fatigue
    if intent.sprint and p.stamina > 1:
        max_speed *= 1.22
        p.stamina -= dt * (10 - s.stamina / 99 * 5)
    else:
        p.stamina += dt * 4
    p.stamina = clamp(p.stamina, 0, 100)

    desired = (intent.move[0] * max_speed, intent.move[1] * max_speed)
    dvx, dvy = desired[0] - p.vel[0], desired[1] - p.vel[1]
    turn_cap = 0.35 + 0.65 * (s.agility / 99)
    step = accel * dt * turn_cap
    dl = vlen(dvx, dvy)
    if dl > step:
        dvx *= step / dl; dvy *= step / dl
    p.vel[0] += dvx; p.vel[1] += dvy
    p.pos[0] += p.vel[0] * dt; p.pos[1] += p.vel[1] * dt
    p.pos[0] = clamp(p.pos[0], -2, L + 2)
    p.pos[1] = clamp(p.pos[1], -2, W + 2)
    if vlen(p.vel[0], p.vel[1]) > 0.3:
        p.facing = math.atan2(p.vel[1], p.vel[0])

def step_ball(world, dt):
    b = world.ball
    if b.cooldown > 0:
        b.cooldown -= dt
    if world.owner is not None and b.cooldown <= 0:   # 드리블 추종(붙지 않음)
        o = world.owner
        tx = o.pos[0] + math.cos(o.facing) * DRIBBLE_AHEAD
        ty = o.pos[1] + math.sin(o.facing) * DRIBBLE_AHEAD
        k = 10.0 - clamp(vlen(o.vel[0], o.vel[1]), 0, 9) * 0.55
        b.vel[0] = (tx - b.pos[0]) * k
        b.vel[1] = (ty - b.pos[1]) * k
        b.vel[2] = 0.0; b.pos[2] = 0.0
        b.pos[0] += b.vel[0] * dt; b.pos[1] += b.vel[1] * dt
        return
    if b.pos[2] > 0 or b.vel[2] != 0:
        b.vel[2] -= GRAV * dt
    for i in range(3):
        b.vel[i] *= (1 - AIR_DRAG * dt)
    b.pos[0] += b.vel[0] * dt; b.pos[1] += b.vel[1] * dt; b.pos[2] += b.vel[2] * dt
    if b.pos[2] <= 0:
        b.pos[2] = 0.0
        if b.vel[2] < 0:
            b.vel[2] = -b.vel[2] * RESTITUTION
            if b.vel[2] < 0.8:
                b.vel[2] = 0.0
        sp = vlen(b.vel[0], b.vel[1])
        if sp > 0:
            dec = min(GROUND_FRIC * dt * 3, sp)
            f = (sp - dec) / sp
            b.vel[0] *= f; b.vel[1] *= f


# ---------------- 액션 해석 ----------------
def kick(world, vx, vy, vz, team, kicker=None):
    b = world.ball
    b.vel = [vx, vy, vz]
    b.owner = None; world.owner = None
    b.cooldown = KICK_COOLDOWN; b.last_team = team
    mark_offside(world, team, kicker)

def do_pass(world, p, power):
    best, bestscore = None, -1e9
    gx, gy = world.opp_goal(p.team)
    for m in world.players:
        if m.team != p.team or m is p or m.role == 'GK' or not m.active:
            continue
        dx, dy = m.pos[0] - p.pos[0], m.pos[1] - p.pos[1]
        dist = vlen(dx, dy)
        if dist < 2 or dist > 45:
            continue
        fwd = ((gx - p.pos[0]) * dx + (gy - p.pos[1]) * dy) / (dist + 1)
        opp = min((vlen(m.pos[0] - o.pos[0], m.pos[1] - o.pos[1])
                   for o in world.players if o.team != p.team and o.active), default=99)
        score = fwd + opp * 1.5
        if score > bestscore:
            bestscore, best = score, m
    if best is None:
        dx, dy = vnorm(gx - p.pos[0], gy - p.pos[1])
        kick(world, dx * 18, dy * 18, 1.5, p.team, p); return
    dx, dy = best.pos[0] - p.pos[0], best.pos[1] - p.pos[1]
    dist = vlen(dx, dy)
    ang = math.atan2(dy, dx) + random.gauss(0, math.radians((1 - p.stats.passing / 99) * 9))
    speed = clamp(dist * 1.6, 8, 26) * (0.8 + 0.4 * power)
    kick(world, math.cos(ang) * speed, math.sin(ang) * speed,
         1.0 if dist > 22 else 0.3, p.team, p)

def do_shot(world, p, power):
    gx, gy = world.opp_goal(p.team)
    aim_y = gy + random.uniform(-GOAL_W * 0.42, GOAL_W * 0.42)
    ang = math.atan2(aim_y - p.pos[1], gx - p.pos[0])
    pressure = sum(1 for o in world.players if o.team != p.team and o.active
                   and vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) < 3)
    off_balance = clamp(vlen(p.vel[0], p.vel[1]) - 4, 0, 5)
    err = math.radians((1 - p.stats.finishing / 99) * 11 + pressure * 4 + off_balance)
    ang += random.gauss(0, err)
    speed = 20 + power * 10
    kick(world, math.cos(ang) * speed, math.sin(ang) * speed, 2.0 + power * 3.0, p.team, p)

def do_tackle(world, p):
    o = world.owner
    if o is None or o.team == p.team or not o.active:
        return
    if vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) > 2.0:
        return
    prob = 0.2 + 0.6 * logistic(0.08 * (p.stats.physical - o.stats.physical))
    if random.random() < prob:                       # 태클 성공 → 탈취
        b = world.ball
        dx, dy = vnorm(p.pos[0] - b.pos[0], p.pos[1] - b.pos[1])
        kick(world, dx * 4 + random.uniform(-2, 2),
             dy * 4 + random.uniform(-2, 2), 0.5, p.team, p)
        b.cooldown = 0.15
    else:                                            # 실패 → 반칙 확률
        foul_chance = 0.35 + 0.25 * clamp(vlen(p.vel[0], p.vel[1]) - 3, 0, 4) / 4
        if random.random() < foul_chance:
            award_foul(world, p, o)


# ---------------- 파울 / 카드 / 세트피스 ----------------
def send_off(world, p):
    p.active = False
    p.pos = [-3.0, -3.0]; p.vel = [0.0, 0.0]
    set_event(world, "RED CARD")

def give_card(world, p):
    sev = random.random()
    if sev < 0.04:
        send_off(world, p)
    elif sev < 0.25:
        world.cards[p.pid] = world.cards.get(p.pid, 0) + 1
        if world.cards[p.pid] >= 2:
            send_off(world, p)              # 경고 누적 퇴장
        else:
            set_event(world, "YELLOW CARD")

def award_foul(world, fouling, victim):
    if in_box(victim.pos[0], victim.pos[1], fouling.team):
        penalty(world, victim.team)
        set_event(world, "PENALTY")
    else:
        restart(world, victim.team, victim.pos[0], victim.pos[1], 'freekick', push=True)
        set_event(world, "FOUL")
    give_card(world, fouling)

def restart(world, team, x, y, kind, push=False):
    b = world.ball
    b.pos = [clamp(x, 1, L - 1), clamp(y, 1, W - 1), 0.0]
    b.vel = [0.0, 0.0, 0.0]
    b.owner = None; world.owner = None; b.cooldown = 0.5
    world.offside_suspects = set()
    world.no_offside = kind in ('throwin', 'corner', 'goalkick')
    cands = [p for p in world.players if p.team == team and p.active and p.role != 'GK']
    taker = min(cands, key=lambda p: vlen(p.pos[0] - b.pos[0], p.pos[1] - b.pos[1]),
                default=None)
    if taker:
        taker.pos = [b.pos[0] + (-0.8 if team == 0 else 0.8), b.pos[1]]
        taker.vel = [0.0, 0.0]
    if push:                                # 프리킥: 상대 9.15m 밀어내기
        for p in world.players:
            if p.team == team or not p.active:
                continue
            dx, dy = p.pos[0] - b.pos[0], p.pos[1] - b.pos[1]
            dd = vlen(dx, dy)
            if dd < 9.15:
                nx, ny = vnorm(dx, dy) if dd > 0 else (1.0, 0.0)
                p.pos[0] = b.pos[0] + nx * 9.15
                p.pos[1] = b.pos[1] + ny * 9.15

def penalty(world, attack_team):
    defend = 1 - attack_team
    spot_x = PEN_SPOT if defend == 0 else L - PEN_SPOT
    b = world.ball
    b.pos = [spot_x, W / 2, 0.0]; b.vel = [0.0, 0.0, 0.0]
    b.owner = None; world.owner = None; b.cooldown = 0.6
    world.offside_suspects = set(); world.no_offside = True
    cands = [p for p in world.players if p.team == attack_team and p.active and p.role != 'GK']
    taker = min(cands, key=lambda p: vlen(p.pos[0] - spot_x, p.pos[1] - W / 2), default=None)
    if taker:
        taker.pos = [spot_x + (-1.5 if attack_team == 0 else 1.5), W / 2]; taker.vel = [0, 0]
    for p in world.players:                 # 박스 정리
        if not p.active or p is taker:
            continue
        if p.role == 'GK':
            if p.team == defend:
                p.pos = [0.5 if defend == 0 else L - 0.5, W / 2]; p.vel = [0, 0]
            continue
        if in_box(p.pos[0], p.pos[1], defend):
            edge = (PBOX_DEPTH + 1) if defend == 0 else (L - PBOX_DEPTH - 1)
            p.pos = [edge, p.pos[1]]; p.vel = [0, 0]

def offside_freekick(world, offender):
    restart(world, 1 - offender.team, offender.pos[0], offender.pos[1], 'freekick', push=True)
    set_event(world, "OFFSIDE")


# ---------------- 오프더볼 AI ----------------
def separation(world, p):
    sx = sy = 0.0
    for q in world.players:
        if q is p or q.team != p.team or not q.active:
            continue
        dx, dy = p.pos[0] - q.pos[0], p.pos[1] - q.pos[1]
        d = vlen(dx, dy)
        if 0 < d < 3.5:
            sx += dx / d * (3.5 - d); sy += dy / d * (3.5 - d)
    return sx, sy

def move_to(world, p, tx, ty, sprint=False):
    mvx, mvy = tx - p.pos[0], ty - p.pos[1]
    sx, sy = separation(world, p)
    return Intent(move=vnorm(mvx + sx * 0.5, mvy + sy * 0.5), sprint=sprint)

def ai_intent(world, p):
    b = world.ball
    gx, gy = world.opp_goal(p.team)
    ogx, ogy = world.own_goal(p.team)
    my_poss = world.owner is not None and world.owner.team == p.team

    if p.role == 'GK':
        ty = clamp(b.pos[1], W / 2 - GOAL_W, W / 2 + GOAL_W)
        out = 8 if vlen(b.pos[0] - ogx, b.pos[1] - ogy) < 18 else 3.5
        it = move_to(world, p, ogx + (out if p.team == 0 else -out), ty)
        if world.owner is p:
            it.action = 'pass'
        return it

    if world.owner is p:                                   # 볼 소유
        dgoal = vlen(gx - p.pos[0], gy - p.pos[1])
        press = min((vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1])
                     for o in world.players if o.team != p.team and o.active), default=99)
        if dgoal < 22:
            return Intent(action='shoot', power=clamp(1 - dgoal / 30, 0.4, 1.0))
        if press < 2.3:
            return Intent(action='pass', power=0.6)
        dx, dy = vnorm(gx - p.pos[0], gy - p.pos[1])
        return Intent(move=(dx, dy), sprint=dgoal > 28)

    if my_poss:                                            # 공격 지원(오프사이드 유지)
        if p.role == 'FWD':
            line = offside_line(world, p.team)
            tx = (b.pos[0] + gx) / 2
            tx = min(tx, line - 1.0) if p.team == 0 else max(tx, line + 1.0)
            return move_to(world, p, tx, clamp(p.pos[1], 8, W - 8), sprint=True)
        hx, hy = world.home_pos(p, b)
        return move_to(world, p, hx, hy)

    # 수비: 최근접 1명 압박/태클, 나머지 진형 유지
    mates = [q for q in world.players if q.team == p.team and q.role != 'GK' and q.active]
    nearest = min(mates, key=lambda q: vlen(b.pos[0] - q.pos[0], b.pos[1] - q.pos[1]),
                  default=None)
    if p is nearest:
        it = move_to(world, p, b.pos[0], b.pos[1], sprint=True)
        o = world.owner
        if o is not None and o.team != p.team and \
           vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) < 2.0:
            it.action = 'tackle'
        return it
    hx, hy = world.home_pos(p, b)
    return move_to(world, p, hx, hy)


# ---------------- 사람 입력 ----------------
def human_intent(keys):
    mvx = mvy = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]: mvx -= 1
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: mvx += 1
    if keys[pygame.K_UP] or keys[pygame.K_w]: mvy -= 1
    if keys[pygame.K_DOWN] or keys[pygame.K_s]: mvy += 1
    return Intent(move=vnorm(mvx, mvy),
                  sprint=keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

def select_controlled(world):
    o = world.owner
    if o is not None and o.team == HUMAN_TEAM and o.active:
        world.controlled = o; return
    cands = [p for p in world.players if p.team == HUMAN_TEAM and p.role != 'GK' and p.active]
    world.controlled = min(cands, key=lambda p: vlen(
        world.ball.pos[0] - p.pos[0], world.ball.pos[1] - p.pos[1]), default=None)


# ---------------- 규칙(골/세트피스) ----------------
def rules(world):
    b = world.ball
    if b.pos[0] < 0 or b.pos[0] > L:                    # 골라인 통과
        if abs(b.pos[1] - W / 2) < GOAL_W / 2 and b.pos[2] < CROSSBAR:
            world.score[0 if b.pos[0] > L else 1] += 1
            set_event(world, "GOAL")
            world.kickoff(); return
        defend = 0 if b.pos[0] < 0 else 1
        line_x = 0.0 if b.pos[0] < 0 else L
        sgn = 1 if line_x == 0 else -1
        if b.last_team == defend:                       # 수비가 내보냄 → 코너
            cy = 0.5 if b.pos[1] < W / 2 else W - 0.5
            restart(world, 1 - defend, line_x + sgn * 0.5, cy, 'corner')
            set_event(world, "CORNER")
        else:                                           # 공격이 내보냄 → 골킥
            restart(world, defend, line_x + sgn * 5.5, W / 2, 'goalkick')
            set_event(world, "GOAL KICK")
        return
    if b.pos[1] < 0 or b.pos[1] > W:                    # 사이드라인 → 스로인
        ty = 0.5 if b.pos[1] < 0 else W - 0.5
        restart(world, 1 - b.last_team, b.pos[0], ty, 'throwin')
        set_event(world, "THROW-IN")


# ---------------- 시뮬 1스텝(사람·AI 공통) ----------------
def step(world, dt, human_keys=None, human_mode=True):
    update_possession(world)
    if human_mode:
        select_controlled(world)
    for p in world.players:
        if not p.active:
            continue
        if human_mode and p is world.controlled and human_keys is not None:
            intent = human_intent(human_keys)
        else:
            intent = ai_intent(world, p)
            if intent.action == 'pass' and world.owner is p:
                do_pass(world, p, intent.power)
            elif intent.action == 'shoot' and world.owner is p:
                do_shot(world, p, intent.power)
            elif intent.action == 'tackle':
                do_tackle(world, p)
        step_player(world, p, intent, dt)
    step_ball(world, dt)
    rules(world)
    if world.event_timer > 0:
        world.event_timer -= dt
    world.time += dt


# ---------------- 렌더링 ----------------
def w2s(x, y):
    return (int(MARGIN + x * PPM), int(MARGIN + y * PPM))

def draw(screen, font, bigfont, world, human_mode, sim_speed, charge):
    screen.fill((22, 130, 55))
    white = (235, 235, 235)
    for i in range(0, int(L), 6):
        if (i // 6) % 2 == 0:
            x, _ = w2s(i, 0)
            pygame.draw.rect(screen, (26, 140, 60), (x, MARGIN, int(6 * PPM), int(W * PPM)))

    def line(x1, y1, x2, y2, wd=2):
        pygame.draw.line(screen, white, w2s(x1, y1), w2s(x2, y2), wd)

    pygame.draw.rect(screen, white, (*w2s(0, 0), int(L * PPM), int(W * PPM)), 2)
    line(L / 2, 0, L / 2, W)
    pygame.draw.circle(screen, white, w2s(L / 2, W / 2), int(9.15 * PPM), 2)
    pygame.draw.circle(screen, white, w2s(L / 2, W / 2), 3)
    for gx in (0, L):
        sgn = 1 if gx == 0 else -1
        bx = gx + sgn * PBOX_DEPTH
        line(gx, W / 2 - PBOX_HALF, bx, W / 2 - PBOX_HALF)
        line(gx, W / 2 + PBOX_HALF, bx, W / 2 + PBOX_HALF)
        line(bx, W / 2 - PBOX_HALF, bx, W / 2 + PBOX_HALF)
        gax = gx + sgn * 5.5
        line(gx, W / 2 - 9.16, gax, W / 2 - 9.16)
        line(gx, W / 2 + 9.16, gax, W / 2 + 9.16)
        line(gax, W / 2 - 9.16, gax, W / 2 + 9.16)
        pygame.draw.circle(screen, white, w2s(gx + sgn * PEN_SPOT, W / 2), 3)  # 페널티 스폿
        gpx = gx - sgn * 1.5
        pygame.draw.rect(screen, (255, 230, 80),
                         (*w2s(min(gx, gpx), W / 2 - GOAL_W / 2),
                          int(1.5 * PPM), int(GOAL_W * PPM)))

    # 오프사이드 라인(공 가진 팀 기준)
    if world.owner is not None and world.owner.active:
        lx = offside_line(world, world.owner.team)
        xpx = int(MARGIN + lx * PPM)
        for yy in range(MARGIN, MARGIN + int(W * PPM), 14):
            pygame.draw.line(screen, (255, 170, 50), (xpx, yy), (xpx, yy + 7), 1)

    b = world.ball                              # 공 + 높이
    sx, sy = w2s(b.pos[0], b.pos[1])
    pygame.draw.ellipse(screen, (0, 60, 20), (sx - 4, sy - 2, 8, 4))
    hy = int(b.pos[2] * PPM * 0.7)
    r = 4 + int(b.pos[2] * 1.5)
    pygame.draw.circle(screen, (250, 250, 250), (sx, sy - hy), r)
    pygame.draw.circle(screen, (0, 0, 0), (sx, sy - hy), r, 1)

    for p in world.players:
        if not p.active:
            continue
        px, py = w2s(p.pos[0], p.pos[1])
        if human_mode and p is world.controlled:
            pygame.draw.circle(screen, (255, 240, 90), (px, py), 12, 3)
        pygame.draw.circle(screen, TEAM_COLORS[p.team], (px, py), 8)
        pygame.draw.circle(screen, (15, 15, 15), (px, py), 8, 1)
        if world.cards.get(p.pid):              # 경고 표시
            pygame.draw.rect(screen, (240, 220, 40), (px + 6, py - 12, 4, 6))
        pygame.draw.line(screen, (245, 245, 245), (px, py),
                         (px + int(math.cos(p.facing) * 12),
                          py + int(math.sin(p.facing) * 12)), 2)

    n0 = sum(1 for p in world.players if p.team == 0 and p.active)
    n1 = sum(1 for p in world.players if p.team == 1 and p.active)
    screen.blit(bigfont.render(f"{world.score[0]} : {world.score[1]}", True, white),
                (SCREEN_W // 2 - 30, 12))
    mode = "HUMAN PLAY (M:switch)" if human_mode else f"AUTO SIM x{sim_speed} (M:switch)"
    screen.blit(font.render(mode, True, white), (MARGIN, 14))
    screen.blit(font.render(f"{n0}v{n1}", True, white), (SCREEN_W // 2 - 14, 44))
    screen.blit(font.render(f"{int(world.time // 60):02d}:{int(world.time % 60):02d}",
                            True, white), (SCREEN_W - MARGIN - 56, 16))
    screen.blit(font.render(
        "Move:WASD/Arrows  Shift:Sprint  Space:Pass  K(hold):Shoot  J:Tackle  R:Reset",
        True, white), (MARGIN, SCREEN_H - 28))
    if human_mode and world.controlled:
        c = world.controlled
        screen.blit(font.render(
            f"{c.role}  STA:{int(c.stamina)}  FIN:{int(c.stats.finishing)} "
            f"PAC:{int(c.stats.pace)} PHY:{int(c.stats.physical)}",
            True, white), (MARGIN, SCREEN_H - 50))
        if charge > 0:
            pygame.draw.rect(screen, white, (MARGIN, SCREEN_H - 68, 104, 8), 1)
            pygame.draw.rect(screen, (255, 200, 60),
                             (MARGIN + 2, SCREEN_H - 66, int(charge * 100), 4))
    if world.event_timer > 0:                   # 이벤트 배너
        surf = bigfont.render(world.event_text, True, (255, 240, 120))
        screen.blit(surf, (SCREEN_W // 2 - surf.get_width() // 2, SCREEN_H // 2 - 18))


# ---------------- 메인 ----------------
def game_loop():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Top-Down Soccer Prototype")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,arial", 16)
    bigfont = pygame.font.SysFont("consolas,arial", 30, bold=True)

    world = World()
    human_mode = True
    sim_speed = 1
    charging = False
    charge = 0.0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_m:
                    human_mode = not human_mode
                elif e.key == pygame.K_r:
                    world.kickoff()
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    sim_speed = min(8, sim_speed + 1)
                elif e.key == pygame.K_MINUS:
                    sim_speed = max(1, sim_speed - 1)
                elif human_mode and world.controlled is not None:
                    c = world.controlled
                    if e.key == pygame.K_SPACE and world.owner is c:
                        do_pass(world, c, 0.6)
                    elif e.key == pygame.K_j:
                        do_tackle(world, c)
                    elif e.key == pygame.K_k and world.owner is c:
                        charging, charge = True, 0.0
            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_k and charging:
                    charging = False
                    if human_mode and world.controlled is not None \
                            and world.owner is world.controlled:
                        do_shot(world, world.controlled, clamp(charge, 0.2, 1.0))

        keys = pygame.key.get_pressed()
        if charging:
            charge = min(1.0, charge + DT * 1.5)
        for _ in range(1 if human_mode else sim_speed):
            step(world, DT, human_keys=keys if human_mode else None, human_mode=human_mode)
        draw(screen, font, bigfont, world, human_mode, sim_speed, charge if charging else 0)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


def run_test(steps):
    w = World()
    for _ in range(steps):
        step(w, DT, human_keys=None, human_mode=False)
    n0 = sum(1 for p in w.players if p.team == 0 and p.active)
    n1 = sum(1 for p in w.players if p.team == 1 and p.active)
    print(f"TEST OK  score={w.score}  players={n0}v{n1}  "
          f"events={dict(sorted(w.events.items()))}  time={w.time:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', type=int, default=0, help="헤드리스 시뮬 스텝 수")
    args = ap.parse_args()
    if args.test:
        run_test(args.test)
    else:
        game_loop()


if __name__ == "__main__":
    main()

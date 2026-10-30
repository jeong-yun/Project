"""
탑다운 축구 프로토타입 (Pygame) v3 - GK 선방 · 어드밴티지 · 경기 기록
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

v3 추가:
    - GK 선방: 반응 지연 → 각 좁히기 위치 → 다이브, 캐치/펀칭/크로스바 위로 쳐내기
               페널티는 차는 순간 방향을 찍어 다이브
    - 어드밴티지: 상대 진영 반칙 시 공격 계속, 3초 안에 뺏기면 원래 지점 프리킥
    - 태클 쿨다운(매 프레임 태클 방지), 슛 블록, GK 쳐낸 공 리바운드 오프사이드
    - 경기 기록(점유율·슛·유효슛·선방·패스 성공률·태클·파울) — Tab 패널
    - 공정성 수정: 1팀 진형 반대 이동 버그, 리스트 순서에 따른 0팀 선행 이득 제거

실행:  python soccer.py
테스트: python soccer.py --test 12000 --seed 1,2,3   /   python -m pytest tests
조작:  WASD/화살표 이동 · Shift 스프린트 · Space 패스 · K(꾹) 슛
       J 태클 · M 모드전환 · R 리셋 · +/- 시뮬속도 · Tab 경기기록 · Esc 종료
"""

import sys, math, random, argparse, time
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
TACKLE_CD = 0.9             # 태클 후 다음 태클까지 대기(초) — 매 프레임 태클 방지
SHOT_LIVE_SPEED = 9.0       # 이 속도 이상이면 '살아있는 슛'(필드 선수는 받지 못하고 블록만)
GK_REACH = 1.1              # GK 선 자세 손 닿는 거리(m)
GK_REACH_DIVE = 1.9         # 다이브 중 손 닿는 거리(m)
GK_TOP = 2.6                # GK가 닿는 최대 높이(m)
GK_DOWN = 0.7               # 다이브 후 일어나는 시간(초)
ADV_TIME = 3.0              # 어드밴티지 지켜보는 시간(초)
SHOT_BASE_ERR = 3.0         # 누구에게나 있는 슛 각도 오차(도). 없으면 유효슈팅률 80%로 비현실적
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
    keeping: float = 30         # GK 능력(반응·다이브·캐치). 필드 선수는 낮음

def rand_stats(role):
    u = random.uniform
    s = Stats(u(60, 86), u(60, 86), u(60, 86), u(45, 82), u(58, 85), u(58, 86), u(60, 90),
              u(15, 40))
    if role == 'FWD': s.finishing = u(72, 92); s.pace = u(76, 92)
    if role == 'DEF': s.physical = u(72, 92); s.finishing = u(35, 60)
    if role == 'GK':  s.physical = u(70, 90); s.keeping = u(70, 92)
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
    tackle_cd: float = 0.0      # 태클 쿨다운 남은 시간
    react: float = 0.0          # GK: 슛 반응 지연 남은 시간
    dive: float = 0.0           # GK: 다이브 남은 시간(조작 불가)
    down: float = 0.0           # GK: 다이브 후 쓰러져 있는 시간

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
        self.shot = None                # 날아가는 중인 슛 정보(dict) — GK 선방 판정용
        self.penalty_pending = False    # 페널티킥 대기 중(GK는 골라인에서 대기)
        self.advantage = None           # 어드밴티지 진행 중(dict)
        self.pending_pass = None        # 패스한 팀(받는 사람이 정해지면 성공/실패 집계)
        self.match = new_match_stats()  # 경기 기록
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
        if p.team == 1:              # 공 위치도 그 팀의 공격 방향 기준으로 뒤집는다
            bx, by = 1 - bx, 1 - by  # (예전엔 안 뒤집어서 1팀 진형이 반대로 움직였음)
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
        self.shot = None; self.penalty_pending = False
        self.advantage = None; self.pending_pass = None
        for p in self.players:
            if not p.active:
                continue
            hx, hy = self.home_pos(p, self.ball)
            p.pos = [hx, hy]; p.vel = [0.0, 0.0]
            p.dive = p.down = p.react = 0.0


MATCH_KEYS = ('shots', 'on_target', 'saves', 'blocks', 'passes', 'passes_ok',
              'tackles', 'tackles_won', 'fouls', 'offsides', 'corners', 'poss_time')

def new_match_stats():
    return {k: [0, 0] for k in MATCH_KEYS}

def possession_pct(world):
    t0, t1 = world.match['poss_time']
    tot = t0 + t1
    return (50, 50) if tot <= 0 else (round(100 * t0 / tot), 100 - round(100 * t0 / tot))

def match_summary(world):
    m = world.match
    p0, p1 = possession_pct(world)
    def pr(k): return f"{m[k][0]}-{m[k][1]}"
    pass_pct = [round(100 * m['passes_ok'][t] / m['passes'][t]) if m['passes'][t] else 0
                for t in (0, 1)]
    return (f"poss {p0}-{p1}%  shots {pr('shots')}  on-target {pr('on_target')}  "
            f"saves {pr('saves')}  blocks {pr('blocks')}  pass% {pass_pct[0]}-{pass_pct[1]}  "
            f"tackles {pr('tackles_won')}/{pr('tackles')}  fouls {pr('fouls')}")


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
    sh = world.shot
    if sh is not None and vlen(b.vel[0], b.vel[1]) < SHOT_LIVE_SPEED:
        world.shot = sh = None                     # 슛이 힘을 잃음 → 일반 볼
    if sh is not None:                             # 살아있는 슛: 받지 못하고 몸으로 막기만
        try_block(world, sh)
        world.owner = None; b.owner = None
        return
    best, bestd = None, CONTROL_RADIUS
    if b.pos[2] < CONTROL_HEIGHT:
        for p in world.players:
            if not p.active or p.dive > 0 or p.down > 0:
                continue
            d = vlen(b.pos[0] - p.pos[0], b.pos[1] - p.pos[1])
            if d < bestd:
                best, bestd = p, d
    if best is not None:
        if world.pending_pass is not None:         # 패스 성공/실패 집계
            if best.team == world.pending_pass:
                world.match['passes_ok'][best.team] += 1
            world.pending_pass = None
        if best.pid in world.offside_suspects and best.team == world.attack_team:
            world.offside_suspects = set()
            offside_freekick(world, best)
            return
        world.offside_suspects = set()
        b.last_team = best.team
    world.owner = best
    b.owner = best

def try_block(world, sh):
    # 슛 경로의 수비 필드 선수: 선수마다 슛 1번당 1회, 몸으로 막을 확률
    b = world.ball
    if b.pos[2] > 1.8:
        return
    for p in world.players:
        if not p.active or p.team == sh['team'] or p.role == 'GK' or p.pid in sh['tried']:
            continue
        if vlen(b.pos[0] - p.pos[0], b.pos[1] - p.pos[1]) > 1.0:
            continue
        sh['tried'].add(p.pid)
        if random.random() < 0.25 + 0.2 * p.stats.physical / 99:
            sp = vlen(b.vel[0], b.vel[1]) * random.uniform(0.15, 0.4)
            ang = random.uniform(0, 2 * math.pi)
            b.vel = [math.cos(ang) * sp, math.sin(ang) * sp, random.uniform(0, 2)]
            b.last_team = p.team; b.cooldown = 0.15
            world.shot = None
            world.match['blocks'][p.team] += 1
            return


# ---------------- 물리 ----------------
def step_player(world, p, intent, dt):
    s = p.stats
    p.tackle_cd = max(0.0, p.tackle_cd - dt)
    p.react = max(0.0, p.react - dt)
    if p.dive > 0:                         # GK 다이브: 조작 불가, 몸을 옆으로 날림
        p.dive -= dt
        p.pos[0] += p.vel[0] * dt; p.pos[1] += p.vel[1] * dt
        if p.dive <= 0:
            p.down = GK_DOWN; p.vel = [0.0, 0.0]
        return
    if p.down > 0:                         # 쓰러졌다 일어나는 중
        p.down -= dt; p.vel = [0.0, 0.0]
        return
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
    world.shot = None; world.pending_pass = None
    mark_offside(world, team, kicker)

def predict_cross(b, x_plane):
    # 공이 x = x_plane 평면을 지날 때의 (남은 시간, y, 높이) — 탄도 근사(바운드 무시)
    if abs(b.vel[0]) < 1e-6:
        return None
    t = (x_plane - b.pos[0]) / b.vel[0]
    if t < 0:
        return None
    y = b.pos[1] + b.vel[1] * t
    z = b.pos[2] + b.vel[2] * t - 0.5 * GRAV * t * t
    return t, y, max(z, 0.0)

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
    world.match['passes'][p.team] += 1
    if best is None:
        dx, dy = vnorm(gx - p.pos[0], gy - p.pos[1])
        kick(world, dx * 18, dy * 18, 1.5, p.team, p)
    else:
        dx, dy = best.pos[0] - p.pos[0], best.pos[1] - p.pos[1]
        dist = vlen(dx, dy)
        ang = math.atan2(dy, dx) + random.gauss(0, math.radians((1 - p.stats.passing / 99) * 9))
        speed = clamp(dist * 1.6, 8, 26) * (0.8 + 0.4 * power)
        kick(world, math.cos(ang) * speed, math.sin(ang) * speed,
             1.0 if dist > 22 else 0.3, p.team, p)
    world.pending_pass = p.team

def do_shot(world, p, power):
    gx, gy = world.opp_goal(p.team)
    aim_y = gy + random.uniform(-GOAL_W * 0.42, GOAL_W * 0.42)
    ang = math.atan2(aim_y - p.pos[1], gx - p.pos[0])
    pressure = sum(1 for o in world.players if o.team != p.team and o.active
                   and vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) < 3)
    off_balance = clamp(vlen(p.vel[0], p.vel[1]) - 4, 0, 5)
    err = math.radians(SHOT_BASE_ERR + (1 - p.stats.finishing / 99) * 11
                       + pressure * 4 + off_balance)
    ang += random.gauss(0, err)
    speed = 20 + power * 10
    pen = world.penalty_pending
    kick(world, math.cos(ang) * speed, math.sin(ang) * speed, 2.0 + power * 3.0, p.team, p)
    register_shot(world, p, pen)

def register_shot(world, p, pen):
    # 슛 기록 + 상대 GK에게 반응 지연(또는 페널티 방향 예측) 부여
    b = world.ball
    m = world.match
    m['shots'][p.team] += 1
    gx, _ = world.opp_goal(p.team)
    cr = predict_cross(b, gx)
    on = cr is not None and abs(cr[1] - W / 2) < GOAL_W / 2 and cr[2] < CROSSBAR
    if on:
        m['on_target'][p.team] += 1
    world.shot = {'team': p.team, 'pen': pen, 'on_target': on, 'tried': set(),
                  'guess': random.choices((-1, 0, 1), (0.45, 0.1, 0.45))[0]}
    world.penalty_pending = False
    if world.advantage is not None and world.advantage['team'] == p.team:
        world.advantage = None                       # 어드밴티지 살려서 슛까지 감
    for g in world.players:
        if g.role == 'GK' and g.active and g.team != p.team:
            # 페널티는 차는 순간 미리 방향을 정해 몸을 던진다(반응 지연 0)
            g.react = 0.0 if pen else 0.30 - 0.18 * g.stats.keeping / 99

def do_tackle(world, p):
    o = world.owner
    if o is None or o.team == p.team or not o.active or p.tackle_cd > 0:
        return
    if vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) > 2.0:
        return
    p.tackle_cd = TACKLE_CD
    world.match['tackles'][p.team] += 1
    prob = 0.2 + 0.6 * logistic(0.08 * (p.stats.physical - o.stats.physical))
    if random.random() < prob:                       # 태클 성공 → 탈취
        world.match['tackles_won'][p.team] += 1
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

def advantage_ok(world, victim):
    # 어드밴티지 조건: 반칙당한 선수가 공을 지켰고, 상대 진영에서 공격 중
    if world.owner is not victim:
        return False
    return victim.pos[0] > L / 2 if victim.team == 0 else victim.pos[0] < L / 2

def award_foul(world, fouling, victim):
    world.match['fouls'][fouling.team] += 1
    if in_box(victim.pos[0], victim.pos[1], fouling.team):
        penalty(world, victim.team)
        set_event(world, "PENALTY")
    elif advantage_ok(world, victim) and random.random() < 0.75:   # 심판 재량
        world.advantage = {'team': victim.team, 'x': victim.pos[0], 'y': victim.pos[1],
                           't': ADV_TIME}
        set_event(world, "ADVANTAGE")
    else:
        restart(world, victim.team, victim.pos[0], victim.pos[1], 'freekick', push=True)
        set_event(world, "FOUL")
    give_card(world, fouling)          # 단순화: 어드밴티지여도 카드는 즉시

def update_advantage(world, dt):
    # 어드밴티지 중 ADV_TIME 안에 상대에게 공을 뺏기면 → 원래 반칙 지점 프리킥
    adv = world.advantage
    if adv is None:
        return
    o = world.owner
    if o is not None and o.team != adv['team']:
        world.advantage = None
        restart(world, adv['team'], adv['x'], adv['y'], 'freekick', push=True)
        set_event(world, "ADV CALLED BACK")
        return
    adv['t'] -= dt
    if adv['t'] <= 0:
        world.advantage = None           # 이득을 봤다 → 그대로 진행

def restart(world, team, x, y, kind, push=False):
    b = world.ball
    b.pos = [clamp(x, 1, L - 1), clamp(y, 1, W - 1), 0.0]
    b.vel = [0.0, 0.0, 0.0]
    b.owner = None; world.owner = None; b.cooldown = 0.5
    world.offside_suspects = set()
    world.no_offside = kind in ('throwin', 'corner', 'goalkick')
    world.shot = None; world.advantage = None; world.pending_pass = None
    world.penalty_pending = False
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
    world.shot = None; world.advantage = None; world.pending_pass = None
    world.penalty_pending = True
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
                p.dive = p.down = p.react = 0.0
            continue
        if in_box(p.pos[0], p.pos[1], defend):
            edge = (PBOX_DEPTH + 1) if defend == 0 else (L - PBOX_DEPTH - 1)
            p.pos = [edge, p.pos[1]]; p.vel = [0, 0]

def offside_freekick(world, offender):
    world.match['offsides'][offender.team] += 1
    restart(world, 1 - offender.team, offender.pos[0], offender.pos[1], 'freekick', push=True)
    set_event(world, "OFFSIDE")


# ---------------- 골키퍼 선방 ----------------
def gk_save(world):
    # 살아있는 슛이 GK 손 닿는 범위에 들어오면 1회 선방 판정(스탯 + 거리 + 속도 + 높이)
    sh = world.shot
    if sh is None or world.owner is not None:
        return
    b = world.ball
    for g in world.players:
        if g.role != 'GK' or not g.active or g.team == sh['team'] or g.pid in sh['tried']:
            continue
        diving = g.dive > 0 or g.down > 0
        reach = GK_REACH_DIVE if diving else GK_REACH
        d = vlen(b.pos[0] - g.pos[0], b.pos[1] - g.pos[1])
        if d > reach or b.pos[2] > GK_TOP:
            continue
        sh['tried'].add(g.pid)
        k = g.stats.keeping / 99
        speed = vlen(b.vel[0], b.vel[1])
        # 공 진행선과 GK 사이 최단거리 = 실제로 손을 뻗어야 하는 거리
        perp = abs(b.vel[0] * (g.pos[1] - b.pos[1]) - b.vel[1] * (g.pos[0] - b.pos[0])) \
            / max(speed, 1e-6)
        stretch = clamp(perp / reach, 0.0, 1.0)   # 몸 중심에서 멀수록 어렵다
        prob = (0.45 + 0.55 * k - 0.40 * stretch - 0.012 * max(0.0, speed - 20)
                - (0.10 if b.pos[2] > 1.9 else 0.0)
                - (0.25 if sh['pen'] else 0.0))   # PK: 미리 몸을 던져 공을 끝까지 못 봄
        if random.random() >= clamp(prob, 0.05, 0.95):
            return                                # 못 막음 → 공은 그대로
        world.match['saves'][g.team] += 1
        world.shot = None
        catch = clamp(k * (1 - stretch) * (1.6 - speed / 25), 0.0, 0.9)
        if random.random() < catch:               # 잡음 → GK 소유(수비의 의도적 플레이)
            b.vel = [0.0, 0.0, 0.0]; b.pos = [g.pos[0], g.pos[1], 0.0]
            b.last_team = g.team; b.cooldown = 0.0
            world.offside_suspects = set()
            world.owner = g; b.owner = g
            g.dive = g.down = 0.0; g.vel = [0.0, 0.0]
        elif random.random() < 0.3:               # 크로스바 위로 쳐냄 → 코너킥
            b.vel = [b.vel[0] * random.uniform(0.25, 0.4), b.vel[1] * 0.3,
                     random.uniform(8, 10)]
            b.last_team = g.team; b.cooldown = 0.25
        else:                                     # 쳐냄 → 리바운드(오프사이드 용의자 유지)
            side = 1 if b.pos[1] > g.pos[1] else -1
            b.vel = [b.vel[0] * random.uniform(-0.35, 0.05),
                     side * random.uniform(3, 9), random.uniform(0.5, 3.0)]
            b.last_team = g.team; b.cooldown = 0.25
        set_event(world, "SAVE")
        return

def gk_shot_intent(world, p, sh):
    # 슛 대응: 반응 지연 → 예측 지점으로 이동 → 멀면 다이브
    b = world.ball
    cr = predict_cross(b, p.pos[0])
    if cr is None:
        return Intent()
    t, y, _ = cr
    if sh['pen']:
        y = W / 2 + sh['guess'] * 2.8              # 페널티: 미리 정한 방향
    if p.react > 0:
        return Intent()                           # 아직 반응 못함
    dy = clamp(y, W / 2 - GOAL_W, W / 2 + GOAL_W) - p.pos[1]
    if abs(dy) > 0.9 and (t < 0.5 or sh['pen']):
        dive_speed = 4.0 + 3.0 * p.stats.keeping / 99
        p.dive = min(0.45, abs(dy) / dive_speed + 0.05)
        p.vel = [0.0, math.copysign(dive_speed, dy)]
        return Intent()
    return Intent(move=vnorm(0.0, dy) if abs(dy) > 0.15 else (0.0, 0.0), sprint=True)


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
        if p.dive > 0 or p.down > 0:
            return Intent()
        sh = world.shot
        if sh is not None and sh['team'] != p.team:
            return gk_shot_intent(world, p, sh)
        if world.penalty_pending and in_box(b.pos[0], b.pos[1], p.team):
            return move_to(world, p, ogx + (0.5 if p.team == 0 else -0.5), ogy)  # 골라인 대기
        # 각 좁히기: 골대 중앙→공 방향 선 위, 공이 가까울수록 앞으로 나온다
        dist = vlen(b.pos[0] - ogx, b.pos[1] - ogy)
        ux, uy = vnorm(b.pos[0] - ogx, b.pos[1] - ogy)
        out = 2.0 + 3.0 * clamp(1 - dist / 35, 0, 1)
        it = move_to(world, p, ogx + ux * out, ogy + uy * out)
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
        if o is not None and o.team != p.team and p.tackle_cd <= 0 and \
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
            world.match['corners'][1 - defend] += 1
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
    update_advantage(world, dt)
    if world.owner is not None:
        world.match['poss_time'][world.owner.team] += dt
    if human_mode:
        select_controlled(world)
    # 1) 모든 선수가 같은 순간의 상태를 보고 의도를 정한다
    #    (예전엔 리스트 순서대로 판단+실행 → 0팀이 항상 먼저 패스해 1팀 태클이 0회였음)
    intents = []
    for p in world.players:
        if not p.active:
            continue
        if human_mode and p is world.controlled and human_keys is not None:
            intents.append((p, human_intent(human_keys)))
        else:
            intents.append((p, ai_intent(world, p)))
    # 2) 액션은 매 프레임 무작위 순서로 처리 → 어느 팀도 '먼저 움직이는' 이득이 없다
    actions = [(p, it) for p, it in intents if it.action]
    random.shuffle(actions)
    for p, it in actions:
        if it.action == 'pass' and world.owner is p:
            do_pass(world, p, it.power)
        elif it.action == 'shoot' and world.owner is p:
            do_shot(world, p, it.power)
        elif it.action == 'tackle':
            do_tackle(world, p)
    for p, it in intents:
        step_player(world, p, it, dt)
    step_ball(world, dt)
    gk_save(world)
    rules(world)
    if world.event_timer > 0:
        world.event_timer -= dt
    world.time += dt


# ---------------- 렌더링 ----------------
def w2s(x, y):
    return (int(MARGIN + x * PPM), int(MARGIN + y * PPM))

def draw_stats(screen, font, world):
    # 경기 기록 패널(Tab)
    m = world.match
    p0, p1 = possession_pct(world)
    def pct(t): return f"{round(100 * m['passes_ok'][t] / m['passes'][t])}%" \
        if m['passes'][t] else "-"
    rows = [("Possession", f"{p0}%", f"{p1}%"),
            ("Shots", m['shots'][0], m['shots'][1]),
            ("On target", m['on_target'][0], m['on_target'][1]),
            ("Saves", m['saves'][0], m['saves'][1]),
            ("Blocks", m['blocks'][0], m['blocks'][1]),
            ("Pass acc.", pct(0), pct(1)),
            ("Tackles won", f"{m['tackles_won'][0]}/{m['tackles'][0]}",
             f"{m['tackles_won'][1]}/{m['tackles'][1]}"),
            ("Fouls", m['fouls'][0], m['fouls'][1]),
            ("Offsides", m['offsides'][0], m['offsides'][1]),
            ("Corners", m['corners'][0], m['corners'][1])]
    w, h = 330, 30 + 22 * len(rows)
    x0, y0 = SCREEN_W // 2 - w // 2, 80
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    screen.blit(panel, (x0, y0))
    for i, (name, a, b) in enumerate(rows):
        y = y0 + 15 + i * 22
        screen.blit(font.render(str(a), True, TEAM_COLORS[0]), (x0 + 20, y))
        surf = font.render(name, True, (235, 235, 235))
        screen.blit(surf, (x0 + w // 2 - surf.get_width() // 2, y))
        sb = font.render(str(b), True, TEAM_COLORS[1])
        screen.blit(sb, (x0 + w - 20 - sb.get_width(), y))

def draw(screen, font, bigfont, world, human_mode, sim_speed, charge, show_stats=False):
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
        if p.dive > 0 or p.down > 0:            # 다이브/쓰러짐: 옆으로 누운 타원
            pygame.draw.ellipse(screen, TEAM_COLORS[p.team], (px - 6, py - 14, 12, 28))
            pygame.draw.ellipse(screen, (15, 15, 15), (px - 6, py - 14, 12, 28), 1)
            continue
        pygame.draw.circle(screen, TEAM_COLORS[p.team], (px, py), 8)
        pygame.draw.circle(screen, (240, 240, 240) if p.role == 'GK' else (15, 15, 15),
                           (px, py), 8, 2 if p.role == 'GK' else 1)
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
        "Move:WASD/Arrows  Shift:Sprint  Space:Pass  K(hold):Shoot  J:Tackle  R:Reset  "
        "Tab:Stats",
        True, white), (MARGIN, SCREEN_H - 28))
    if world.advantage is not None:             # 어드밴티지 진행 표시
        screen.blit(font.render(f"ADVANTAGE {world.advantage['t']:.1f}s", True,
                                (255, 240, 120)), (SCREEN_W // 2 + 60, 44))
    if show_stats:
        draw_stats(screen, font, world)
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
    show_stats = False
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
                elif e.key == pygame.K_TAB:
                    show_stats = not show_stats
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
        draw(screen, font, bigfont, world, human_mode, sim_speed, charge if charging else 0,
             show_stats)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


def simulate(steps, seed=None):
    # 헤드리스 자동 경기. seed를 주면 같은 결과가 재현된다(테스트용)
    if seed is not None:
        random.seed(seed)
    w = World()
    for _ in range(steps):
        step(w, DT, human_keys=None, human_mode=False)
    return w

def run_test(steps, seeds=(None,)):
    for seed in seeds:
        t0 = time.perf_counter()
        w = simulate(steps, seed)
        ms = (time.perf_counter() - t0) / steps * 1000
        n0 = sum(1 for p in w.players if p.team == 0 and p.active)
        n1 = sum(1 for p in w.players if p.team == 1 and p.active)
        print(f"TEST OK  seed={seed}  score={w.score}  players={n0}v{n1}  "
              f"events={dict(sorted(w.events.items()))}  time={w.time:.0f}s  "
              f"step={ms:.2f}ms")
        print(f"         {match_summary(w)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', type=int, default=0, help="헤드리스 시뮬 스텝 수")
    ap.add_argument('--seed', type=str, default=None,
                    help="난수 시드. 쉼표로 여러 개(예: 1,2,3) → 시드마다 한 경기")
    args = ap.parse_args()
    if args.test:
        seeds = [int(s) for s in args.seed.split(',')] if args.seed else [None]
        run_test(args.test, seeds)
    else:
        game_loop()


if __name__ == "__main__":
    main()

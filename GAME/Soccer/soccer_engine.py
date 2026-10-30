"""
축구 엔진 (v4.2) — 데이터 · 물리 · 규칙 · 경기 시간 · 스쿼드/교체 · 1스텝 통합(step)
=========================================================================
"무슨 일이 일어나는가"를 정한다: 선수·공 상태, 스탯 → 물리/확률, 득점·오프사이드·파울·세트피스,
GK 선방, 경기 시간·피로, 스쿼드·교체 규칙, 포메이션. 화면(pygame 창)은 쓰지 않는다.

"선수가 무엇을 하려는가"(판단·움직임)는 soccer_ai.py, 화면은 soccer_ui.py.
엔진이 AI를 부르는 곳은 step()의 의도·마킹·세트피스, 사람 패스 자동 조준(do_pass), 멈춤 때의 AI 감독(교체·전술) 뿐이다.
"""

import sys, math, random, argparse, time, json
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
RED_P, YELLOW_P = 0.01, 0.17   # 반칙 1번당 레드·옐로 확률
SHOT_BASE_ERR = 8.5         # 누구에게나 있는 슛 각도 오차(도). v3 3° → v4 8.5°(유효슈팅 v3 66% → 실제 33%에 가깝게)
GK_SAVE_BASE = 0.35         # GK 선방 확률 기본값(v3 0.45). 아래 gk_save 식 참고
PASS_BASE_ERR = 4.5         # 누구에게나 있는 패스 각도 오차(도). v3는 0 → 패스 성공 90%+로 너무 정확
TAKEON_K = 0.06             # 돌파 성공 로지스틱 기울기(능력치 1당)
TAKEON_BEATEN = 0.7         # 돌파당한 수비가 뒤처지는 시간(초)
TOUCH_EASY = 7.0            # 이보다 빠른(상대 속도) 공은 첫 터치가 튈 수 있다(m/s)
TOUCH_BAD_PER = 0.024       # 1 m/s 초과마다 첫 터치 실패 확률 +2.4%(기술 낮을수록 더)
# --- v4: 크로스 · 헤더(공중 경합) ---
HEADER_TOP = 2.6            # 머리로 닿는 최대 높이(m). CONTROL_HEIGHT(1.0)~이 사이가 공중 경합
HEADER_RADIUS = 1.2         # 공중 경합에 낄 수 있는 수평 거리(m)
HEADER_SHOT_DIST = 14.0     # 공격 쪽이 이 거리 안에서 이기면 헤더 슛, 멀면 떨궈 주기
HEADER_ERR = 7.0            # 헤더 슛 기본 각도 오차(도) — 발 슛(5°)보다 부정확
CROSS_Z = 1.7               # 크로스가 목표 지점에 도착하는 높이(머리 높이)
SETPIECE_WAIT = 6.0         # 코너·공격 지역 프리킥: 자리 잡기 최대 대기(초). 침투조가 다 도착하면 바로 찬다
FK_ROUTINE_DIST = 35.0      # 상대 골대에서 이 거리 안 프리킥만 루틴(그 밖은 빠른 재개)
# 수비 라인 높이(자기 골대 기준 0~1). 라인 = SLOPE × 공 위치 + OFS, MIN~MAX로 제한
LINE_SLOPE = 0.75
LINE_DEF_OFS, LINE_DEF_MIN, LINE_DEF_MAX = -0.12, 0.10, 0.45
LINE_ATT_OFS, LINE_ATT_MIN, LINE_ATT_MAX = -0.05, 0.15, 0.55
ROLE_GAP = {'DEF': 0.0, 'MID': 0.17, 'FWD': 0.34}   # 수비 라인에서 각 라인까지 간격(수비 시)
ROLE_BASE_FX = {'DEF': 0.22, 'MID': 0.46, 'FWD': 0.68}  # v4.2: 포메이션 자리의 fx가 이 값과 다르면
                                                      # 그만큼 라인 간격에 더한다(4-2-3-1의 수비형·공격형 미드)
# --- v4: 경기 시간 · 피로 ---
MATCH_MINUTES = 90          # 표시 경기 시간(분). --length 초에 압축(기본 600초 = ×9)
ENERGY_PER_MIN = 0.0035     # 실제 경기 1분마다 경기 체력 감소(운동량 1, 스태미나 스탯 보정 전)
HALFTIME_RECOVER = 0.06     # 하프타임 경기 체력 회복
ACC_FATIGUE = 0.8           # 컨디션 1 → 0 사이에서 패스·슛·헤더 오차 최대 +80%, 첫 터치 실수·돌파도 비례
SUBS_MAX = 5                # 팀당 교체 인원(현행 규칙)
SUB_AFTER_MIN = 55          # 이 시간 이후부터 교체
SUB_ENERGY = 0.75           # 경기 체력이 이 아래면 교체 후보
SUB_POLICY = {'auto': True, 'after_min': SUB_AFTER_MIN, 'energy': SUB_ENERGY, 'yellow_risk': True,
              # v4.2 AI 전술: 지면 70분 뒤 4-3-3, 이기면 80분 뒤 5-3-2(한 번씩)
              'tactics': True, 'chase_after': 70, 'chase_formation': '4-3-3',
              'protect_after': 80, 'protect_formation': '5-3-2'}
BENCH_ROLES = ('GK', 'DEF', 'DEF', 'MID', 'MID', 'FWD', 'FWD')   # 벤치 7명
SQUAD_SIZE = 11 + len(BENCH_ROLES)
TEAM_NAMES = ('Blue', 'Red')
SURNAMES = ('Kim', 'Park', 'Lee', 'Choi', 'Jung', 'Kang', 'Cho', 'Yoon', 'Jang', 'Lim',
            'Han', 'Oh', 'Seo', 'Shin', 'Kwon', 'Hwang', 'Ahn', 'Song', 'Silva', 'Costa',
            'Muller', 'Rossi', 'Garcia', 'Martin', 'Smith', 'Jones', 'Dubois', 'Novak',
            'Jansen', 'Larsen', 'Moreau', 'Santos', 'Ferreira', 'Kowalski', 'Ito', 'Sato')
PSTAT_KEYS = ('passes', 'passes_ok', 'shots', 'goals', 'tackles', 'tackles_won', 'fouls')
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
    run: float = 0.0            # 공격수: 뒷공간 침투 남은 시간
    beaten: float = 0.0         # 돌파당해 뒤처진 남은 시간(못 움직임)
    energy: float = 1.0         # 경기 체력(1 → 경기 내내 천천히 줄어듦). stamina는 스프린트용 단기 체력
    name: str = ''              # v4.1: 이름·등번호·주 포지션(교체로 다른 자리에 들어가도 그대로)
    number: int = 0
    pos_role: str = ''

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
    target: object = None      # AI가 고른 패스 후보(pass_options 한 줄). None이면 실행 때 고름


# ---------------- 월드 ----------------
FORMATION = [   # 4-4-2 (공격 +x 기준 정규화 좌표) — 기본
    ('GK', 0.05, 0.50),
    ('DEF', 0.22, 0.18), ('DEF', 0.22, 0.39), ('DEF', 0.22, 0.61), ('DEF', 0.22, 0.82),
    ('MID', 0.46, 0.18), ('MID', 0.46, 0.39), ('MID', 0.46, 0.61), ('MID', 0.46, 0.82),
    ('FWD', 0.68, 0.40), ('FWD', 0.68, 0.60),
]

# v4.2: 고를 수 있는 포메이션(역할, 깊이 fx, 폭 fy). GK 1 + 필드 10. 깊이는 ROLE_BASE_FX와의 차이만 쓴다
FORMATIONS = {
    '4-4-2': FORMATION,
    '4-3-3': [('GK', 0.05, 0.50),
              ('DEF', 0.22, 0.18), ('DEF', 0.22, 0.39), ('DEF', 0.22, 0.61), ('DEF', 0.22, 0.82),
              ('MID', 0.44, 0.30), ('MID', 0.44, 0.50), ('MID', 0.44, 0.70),
              ('FWD', 0.66, 0.18), ('FWD', 0.70, 0.50), ('FWD', 0.66, 0.82)],
    '4-2-3-1': [('GK', 0.05, 0.50),
                ('DEF', 0.22, 0.18), ('DEF', 0.22, 0.39), ('DEF', 0.22, 0.61), ('DEF', 0.22, 0.82),
                ('MID', 0.38, 0.38), ('MID', 0.38, 0.62),
                ('MID', 0.55, 0.18), ('MID', 0.55, 0.50), ('MID', 0.55, 0.82),
                ('FWD', 0.70, 0.50)],
    '4-5-1': [('GK', 0.05, 0.50),
              ('DEF', 0.22, 0.18), ('DEF', 0.22, 0.39), ('DEF', 0.22, 0.61), ('DEF', 0.22, 0.82),
              ('MID', 0.46, 0.12), ('MID', 0.44, 0.32), ('MID', 0.42, 0.50), ('MID', 0.44, 0.68),
              ('MID', 0.46, 0.88),
              ('FWD', 0.70, 0.50)],
    '3-5-2': [('GK', 0.05, 0.50),
              ('DEF', 0.22, 0.28), ('DEF', 0.21, 0.50), ('DEF', 0.22, 0.72),
              ('MID', 0.48, 0.10), ('MID', 0.44, 0.32), ('MID', 0.42, 0.50), ('MID', 0.44, 0.68),
              ('MID', 0.48, 0.90),
              ('FWD', 0.68, 0.40), ('FWD', 0.68, 0.60)],
    '5-3-2': [('GK', 0.05, 0.50),
              ('DEF', 0.24, 0.10), ('DEF', 0.22, 0.30), ('DEF', 0.21, 0.50), ('DEF', 0.22, 0.70),
              ('DEF', 0.24, 0.90),
              ('MID', 0.44, 0.30), ('MID', 0.44, 0.50), ('MID', 0.44, 0.70),
              ('FWD', 0.68, 0.40), ('FWD', 0.68, 0.60)],
}
FORMATION_NAMES = tuple(FORMATIONS)

class World:
    def __init__(self, match_len=None, squad=None, formations=None):
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
        self.pass_to = None             # 패스 받을 선수 pid(그 선수는 공 경로로 마중 나간다)
        self.owner_t = 0.0              # 지금 소유자가 공을 가진 시간(초)
        self.marks = {}                 # 수비 pid -> 마킹할 상대(매 프레임 배정)
        self.aerial = None              # 날아가는 크로스(dict) — 머리 높이에서 공중 경합 1회
        self.setpiece = None            # 진행 중인 세트피스 루틴(dict): 코너·공격 지역 프리킥
        self.match = new_match_stats()  # 경기 기록
        self.match_len = match_len      # 경기 길이(시뮬 초). None이면 끝없이(v3 방식)
        self.half = 1
        self.finished = False
        # v4.1: 스쿼드(선발 11 + 벤치 7) · 교체 · 선수별 기록 · 조작 선수 고정
        self.team_names = list(TEAM_NAMES)
        self.bench = [[], []]           # 팀별 벤치(경기장 밖 Player)
        self.subbed_off = []            # 교체로 나간 선수(다시 못 들어옴)
        self.sub_queue = []             # (팀, 나갈 pid, 들어올 pid) — 다음 멈춤에서 적용
        self.sub_log = []               # (분, 팀, 나간 이름, 들어온 이름)
        self.sub_policy = [dict(SUB_POLICY), dict(SUB_POLICY)]   # 팀별 AI 교체·전술 정책
        self.formation = list(formations or ('4-4-2', '4-4-2'))  # v4.2: 팀별 포메이션 이름
        self.formation_log = []         # (분, 팀, 포메이션) — 경기 중 바꾼 기록
        self.tactic_state = [{}, {}]    # AI 전술 변경 이력(한 번씩만)
        for f in self.formation:
            if f not in FORMATIONS:
                raise ValueError(f"unknown formation {f!r}; choose from {', '.join(FORMATION_NAMES)}")
        self.pstats = {}                # pid -> 선수별 경기 기록
        self.reds = set()               # 퇴장당한 pid
        self.human_pick = None          # 사람이 고정한 조작 선수 pid(None = 공 가까운 선수 자동)
        self.last_kick = None           # (팀, pid) 마지막으로 찬 선수 — 골 기록용
        self.last_passer = None
        if squad is None:
            pid = 0
            for team in (0, 1):
                for role, fx, fy in FORMATIONS[self.formation[team]]:
                    self.players.append(Player(team, role, fx, fy, rand_stats(role), pid=pid))
                    pid += 1
        else:
            load_lineup(self, squad)
        self.first_kick = 0 if random.random() < 0.5 else 1   # 동전 던지기(예전엔 늘 0팀)
        if squad is None:                                     # 벤치는 선발·동전 뒤에 만든다
            for team in (0, 1):
                for role in BENCH_ROLES:
                    self.bench[team].append(Player(team, role, 0.0, 0.0, rand_stats(role),
                                                   pos=[-5.0, -5.0], pid=len(self.players)
                                                   + sum(map(len, self.bench))))
            name_squad(self)
        self.next_pid = len(self.players) + sum(map(len, self.bench))
        self.kickoff(self.first_kick)

    def opp_goal(self, team): return (L, W / 2) if team == 0 else (0.0, W / 2)
    def own_goal(self, team): return (0.0, W / 2) if team == 0 else (L, W / 2)

    def attacking(self, team):
        # 공을 가졌거나(패스가 날아가는 중이면) 마지막으로 찬 팀 = 공격 중
        o = self.owner
        return (o.team if o is not None else self.ball.last_team) == team

    def def_line(self, team, ball, attacking=None):
        # 수비 라인 높이(자기 골대 0 ~ 상대 골대 1). 공이 올라가면 같이 올라가고, 내려오면 내려앉는다.
        bn = clamp(ball.pos[0] / L, 0, 1)
        if team == 1:
            bn = 1 - bn
        if attacking is None:
            attacking = self.attacking(team)
        if attacking:
            return clamp(LINE_SLOPE * bn + LINE_ATT_OFS, LINE_ATT_MIN, LINE_ATT_MAX)
        return clamp(LINE_SLOPE * bn + LINE_DEF_OFS, LINE_DEF_MIN, LINE_DEF_MAX)

    def home_pos(self, p, ball):
        bx = clamp(ball.pos[0] / L, 0, 1)
        by = clamp(ball.pos[1] / W, 0, 1)
        if p.team == 1:              # 공 위치도 그 팀의 공격 방향 기준으로 뒤집는다
            bx, by = 1 - bx, 1 - by  # (예전엔 안 뒤집어서 1팀 진형이 반대로 움직였음)
        if p.role == 'GK':
            fx = p.base_fx
        else:                        # v4: 라인 하나(수비 라인)를 기준으로 미드·공격 라인이 따라붙는다
            att = self.attacking(p.team)
            gap = (ROLE_GAP[p.role] + (p.base_fx - ROLE_BASE_FX[p.role])) * (1.15 if att else 1.0)
            fx = clamp(self.def_line(p.team, ball, att) + gap, 0.04, 0.92)
        width = 1.0 if p.role == 'GK' or self.attacking(p.team) else 0.8   # 수비 땐 좁게
        fy = clamp(0.5 + (p.base_fy - 0.5) * width + (by - 0.5) * 0.25, 0.05, 0.95)
        if p.team == 0:
            return (fx * L, fy * W)
        return ((1 - fx) * L, (1 - fy) * W)

    def kickoff(self, team=0):
        # team = 킥오프하는 팀(실점한 팀). v3까지는 늘 같은 진형으로 시작해 0팀이 먼저 공을 잡았다
        if self.time > 0:
            on_stoppage(self)                              # 골·하프타임 = 공이 멈춤 → 교체
        self.ball.pos = [L / 2, W / 2, 0.0]
        self.ball.vel = [0.0, 0.0, 0.0]
        self.ball.owner = None
        self.ball.cooldown = 0.5
        self.ball.last_team = team
        self.owner = None
        self.offside_suspects = set()
        self.no_offside = False
        self.shot = None; self.penalty_pending = False
        self.advantage = None; self.pending_pass = None; self.pass_to = None
        self.aerial = None; self.setpiece = None
        for p in self.players:
            if not p.active:
                continue
            hx, hy = self.home_pos(p, self.ball)
            hx = min(hx, L / 2 - 1.0) if p.team == 0 else max(hx, L / 2 + 1.0)   # 킥오프: 자기 진영
            p.pos = [hx, hy]; p.vel = [0.0, 0.0]
            p.dive = p.down = p.react = p.run = p.beaten = 0.0
        taker = next((p for p in self.players if p.team == team and p.role == 'FWD' and p.active),
                     None)
        if taker:                                          # 킥오프 키커는 공 옆에
            taker.pos = [L / 2 + (-0.6 if team == 0 else 0.6), W / 2]


MATCH_KEYS = ('shots', 'on_target', 'saves', 'blocks', 'passes', 'passes_ok',
              'tackles', 'tackles_won', 'fouls', 'offsides', 'corners', 'crosses', 'headers',
              'dribbles', 'dribbles_won', 'subs',
              'poss_time')

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
            f"tackles {pr('tackles_won')}/{pr('tackles')}  fouls {pr('fouls')}  "
            f"offsides {pr('offsides')}  crosses {pr('crosses')}  subs {pr('subs')}")


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
    if world.aerial is not None and CONTROL_HEIGHT <= b.pos[2] <= HEADER_TOP:
        if aerial_contest(world):                  # 크로스: 머리 높이에서 공중 경합
            return
    best, bestd = None, CONTROL_RADIUS
    o = world.owner
    if o is not None and o.active and vlen(b.pos[0] - o.pos[0], b.pos[1] - o.pos[1]) < CONTROL_RADIUS:
        best = o                                   # v4: 가진 사람은 태클로만 뺏긴다(가까이 붙기만 해선 안 됨)
    elif b.pos[2] < CONTROL_HEIGHT:
        for p in world.players:
            if not p.active or p.dive > 0 or p.down > 0:
                continue
            d = vlen(b.pos[0] - p.pos[0], b.pos[1] - p.pos[1])
            if d < bestd:
                best, bestd = p, d
    if best is not None and best is not world.owner and bad_touch(world, best):
        return                                     # 첫 터치가 튐 → 공은 아직 자유
    if best is not None:
        if world.pending_pass is not None:         # 패스 성공/실패 집계
            if best.team == world.pending_pass:
                world.match['passes_ok'][best.team] += 1
                pstat(world, world.last_passer, 'passes_ok')
            world.pending_pass = None
        world.pass_to = None
        if best.pid in world.offside_suspects and best.team == world.attack_team:
            world.offside_suspects = set()
            offside_freekick(world, best)
            return
        world.offside_suspects = set()
        b.last_team = best.team
    if best is not world.owner:
        world.owner_t = 0.0
    world.owner = best
    b.owner = best

def bad_touch(world, p):
    # v4 첫 터치: 빠른 공(선수 기준 상대 속도)일수록, 기술(passing)이 낮을수록, 상대가 붙어 있을수록
    # 공이 발에서 튄다. 튀면 공은 2~5 m/s로 옆으로 흘러 다시 경합(받는 쪽 실수 = 실제 패스 실패의 큰 몫)
    b = world.ball
    rel = vlen(b.vel[0] - p.vel[0], b.vel[1] - p.vel[1])
    if rel <= TOUCH_EASY:
        return False
    tight = any(o.team != p.team and o.active and
                vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) < 2.0 for o in world.players)
    prob = (rel - TOUCH_EASY) * TOUCH_BAD_PER * (1.5 - p.stats.passing / 99) * condition_mult(p) \
        + (0.06 if tight else 0.0)
    if random.random() >= prob:
        return False
    sp = random.uniform(2, 5)
    ang = math.atan2(b.vel[1], b.vel[0]) + random.uniform(-1.2, 1.2)
    b.vel = [math.cos(ang) * sp, math.sin(ang) * sp, 0.0]
    b.last_team = p.team; b.cooldown = 0.2
    world.owner = None; b.owner = None; world.pass_to = None
    return True

def aerial_contest(world):
    # 크로스가 머리 높이로 들어올 때 1회 경합: 가까운 선수들(피지컬 가중) + 자기 박스 안 GK(손, 키핑 가중)
    b = world.ball
    ae = world.aerial
    cands = []
    for p in world.players:
        if not p.active or p.dive > 0 or p.down > 0 or p.pid in ae['tried']:
            continue
        d = vlen(b.pos[0] - p.pos[0], b.pos[1] - p.pos[1])
        if p.role == 'GK':
            if p.team == ae['team'] or not in_box(p.pos[0], p.pos[1], p.team) or d > 1.6 \
                    or b.pos[2] > GK_TOP:
                continue
            w = 0.4 + 1.2 * p.stats.keeping / 99           # 손을 쓰니 유리
        else:
            if d > HEADER_RADIUS:
                continue
            w = (0.3 + p.stats.physical / 99) * (1 - 0.5 * d / HEADER_RADIUS)
        cands.append((w, p))
    if not cands:
        return False
    for _, p in cands:
        ae['tried'].add(p.pid)
    r = random.random() * sum(w for w, _ in cands)
    for w, win in cands:
        r -= w
        if r <= 0:
            break
    m = world.match
    world.aerial = None
    if world.pending_pass is not None:               # 크로스 성공 = 우리 편이 먼저 닿음
        if win.team == world.pending_pass:
            m['passes_ok'][win.team] += 1
            pstat(world, world.last_passer, 'passes_ok')
        world.pending_pass = None
    world.pass_to = None
    if win.role == 'GK':                             # GK가 공중에서 잡음
        b.vel = [0.0, 0.0, 0.0]; b.pos = [win.pos[0], win.pos[1], 0.0]
        b.last_team = win.team; b.cooldown = 0.0
        world.offside_suspects = set()
        world.owner = win; b.owner = win; world.owner_t = 0.0
        set_event(world, "GK CLAIM")
        return True
    m['headers'][win.team] += 1
    if win.pid in world.offside_suspects and win.team == world.attack_team:
        world.offside_suspects = set()
        offside_freekick(world, win)
        return True
    world.offside_suspects = set()
    gx, gy = world.opp_goal(win.team)
    if win.team == ae['team'] and vlen(gx - b.pos[0], gy - b.pos[1]) < HEADER_SHOT_DIST:
        do_header(world, win)
    elif win.team == ae['team']:                     # 떨궈 주기: 골 쪽으로 약하게
        ux, uy = vnorm(gx - b.pos[0], gy - b.pos[1])
        kick(world, ux * 4 + random.uniform(-2, 2), uy * 4 + random.uniform(-2, 2), 1.0,
             win.team, win)
    else:                                            # 수비 헤더: 우리 골대 반대쪽으로 걷어냄
        ogx, ogy = world.own_goal(win.team)
        ux, uy = vnorm(b.pos[0] - ogx, b.pos[1] - ogy)
        ang = math.atan2(uy, ux) + random.uniform(-0.6, 0.6)
        sp = random.uniform(10, 15)
        kick(world, math.cos(ang) * sp, math.sin(ang) * sp, random.uniform(3, 5), win.team, win)
    return True

def do_header(world, p):
    # 헤더 슛: 골대 안 무작위 지점, 오차 = 기본 7° + 결정력 + 피로. 공은 아래로(바운드 노림)
    b = world.ball
    gx, gy = world.opp_goal(p.team)
    aim_y = gy + random.uniform(-GOAL_W * 0.4, GOAL_W * 0.4)
    ang = math.atan2(aim_y - b.pos[1], gx - b.pos[0])
    err = (HEADER_ERR + (1 - p.stats.finishing / 99) * 8) * condition_mult(p)
    ang += random.gauss(0, math.radians(err))
    sp = 11 + 5 * p.stats.physical / 99
    kick(world, math.cos(ang) * sp, math.sin(ang) * sp, random.uniform(-1.5, 1.5), p.team, p)
    register_shot(world, p, False)

def lofted_kick(world, p, tx, ty, err_deg, z_end=CROSS_Z):
    # 높게 띄워 (tx, ty)에 z_end 높이로 도착하게: 체공 시간 T → 수직 속도 = (z_end + ½gT²)/T
    bx, by = world.ball.pos[0], world.ball.pos[1]
    d = vlen(tx - bx, ty - by)
    vh = clamp(10 + 0.35 * d, 14, 22)
    t = d / vh
    vz = (z_end - world.ball.pos[2] + 0.5 * GRAV * t * t) / t
    vh *= (1 + AIR_DRAG * t / 2) * random.gauss(1.0, 0.04)   # 공기저항 보정 + 세기 오차
    ang = math.atan2(ty - by, tx - bx) + random.gauss(0, math.radians(err_deg))
    kick(world, math.cos(ang) * vh, math.sin(ang) * vh, vz, p.team, p)

def do_cross(world, p, tx, ty):
    # 크로스(패스 기록에도 포함 — 실제 기록처럼): 머리 높이 도착 + 공중 경합 대기
    world.match['passes'][p.team] += 1
    world.match['crosses'][p.team] += 1
    pstat(world, p.pid, 'passes'); world.last_passer = p.pid
    lofted_kick(world, p, tx, ty, pass_error_deg(p) * 1.3)
    world.aerial = {'team': p.team, 'tried': set()}
    world.pending_pass = p.team

def condition(p):
    # 컨디션 0~1: 단기 체력(stamina, 스프린트하면 줄고 쉬면 참) × 경기 체력(energy)
    return (0.5 + 0.5 * p.stamina / 100.0) * p.energy

def condition_mult(p):
    # 피로 → 정확도 저하 배수. 쌩쌩하면 1, 컨디션 0.7이면 1 + 0.8×0.3 = 1.24
    return 1.0 + ACC_FATIGUE * (1.0 - condition(p))

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
    p.run = max(0.0, p.run - dt)
    if p.dive > 0:                         # GK 다이브: 조작 불가, 몸을 옆으로 날림
        p.dive -= dt
        p.pos[0] += p.vel[0] * dt; p.pos[1] += p.vel[1] * dt
        if p.dive <= 0:
            p.down = GK_DOWN; p.vel = [0.0, 0.0]
        return
    if p.beaten > 0:                       # 돌파당함: 몸이 반대로 쏠려 잠깐 못 따라감
        p.beaten -= dt
        p.vel[0] *= 0.9; p.vel[1] *= 0.9
        p.pos[0] += p.vel[0] * dt; p.pos[1] += p.vel[1] * dt
        return
    if p.down > 0:                         # 쓰러졌다 일어나는 중
        p.down -= dt; p.vel = [0.0, 0.0]
        return
    fatigue = (0.55 + 0.45 * (p.stamina / 100.0)) * (0.8 + 0.2 * p.energy)
    max_speed = (4.5 + s.pace / 99 * 4.5) * fatigue
    accel = (18 + s.accel / 99 * 22) * fatigue
    if intent.sprint and p.stamina > 1:
        max_speed *= 1.22
        p.stamina -= dt * (10 - s.stamina / 99 * 5)
    else:
        p.stamina += dt * 4
    p.stamina = clamp(p.stamina, 0, 100)
    if world.match_len:                    # 경기 체력: 실제 경기 1분마다 ENERGY_PER_MIN × 운동량 × 스태미나 스탯
        game_min = dt * MATCH_MINUTES / world.match_len
        work = 1.6 if intent.sprint else 1.0
        p.energy = max(0.3, p.energy - game_min * ENERGY_PER_MIN * work * (1.3 - 0.6 * s.stamina / 99))

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
    if kicker is not None:
        world.last_kick = (team, kicker.pid)
    b.vel = [vx, vy, vz]
    b.owner = None; world.owner = None
    b.cooldown = KICK_COOLDOWN; b.last_team = team
    world.shot = None; world.pending_pass = None; world.pass_to = None; world.aerial = None
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

def do_pass(world, p, power, opt=None):
    # 패스: 후보 중 기대값 최고(사람의 Space도 같은 자동 조준) → 패스 정확도 오차
    if opt is None:
        opts = ai.pass_options(world, p)
        opt = opts[0] if opts else None
    world.match['passes'][p.team] += 1
    pstat(world, p.pid, 'passes'); world.last_passer = p.pid
    if opt is None:
        gx, gy = world.opp_goal(p.team)
        dx, dy = vnorm(gx - p.pos[0], gy - p.pos[1])
        kick(world, dx * 18, dy * 18, 1.5, p.team, p)
    else:
        _, _, m, tx, ty, speed = opt
        dx, dy = tx - p.pos[0], ty - p.pos[1]
        ang = math.atan2(dy, dx) + random.gauss(0, math.radians(pass_error_deg(p)))
        speed *= 0.9 + 0.2 * power
        kick(world, math.cos(ang) * speed, math.sin(ang) * speed,
             1.0 if vlen(dx, dy) > 25 else 0.3, p.team, p)
        world.pass_to = m.pid
    world.pending_pass = p.team

def do_clear(world, p):
    # 걷어내기: 상대가 적은 쪽 측면으로 높고 길게(패스 기록엔 안 넣음 — 실제 기록도 따로 셈)
    gx, gy = world.opp_goal(p.team)
    tx = L * 0.62 if p.team == 0 else L * 0.38
    side = 1 if p.pos[1] > W / 2 else -1
    ty = W / 2 + side * random.uniform(8, 24)
    dx, dy = vnorm(tx - p.pos[0], ty - p.pos[1])
    ang = math.atan2(dy, dx) + random.gauss(0, math.radians(pass_error_deg(p) * 2))
    sp = random.uniform(20, 25)
    kick(world, math.cos(ang) * sp, math.sin(ang) * sp, random.uniform(8, 11), p.team, p)

def pass_error_deg(p):
    # 패스 각도 오차 σ(도): 패스 능력
    return (PASS_BASE_ERR + (1 - p.stats.passing / 99) * 9) * condition_mult(p)

def do_shot(world, p, power):
    gx, gy = world.opp_goal(p.team)
    aim_y = gy + random.uniform(-GOAL_W * 0.42, GOAL_W * 0.42)
    ang = math.atan2(aim_y - p.pos[1], gx - p.pos[0])
    ang += random.gauss(0, math.radians(shot_error_deg(world, p)))
    speed = 20 + power * 10
    pen = world.penalty_pending
    kick(world, math.cos(ang) * speed, math.sin(ang) * speed, 2.0 + power * 3.0, p.team, p)
    register_shot(world, p, pen)

def shot_error_deg(world, p):
    # 슛 각도 오차 σ(도): 기본 + 결정력 + 압박(3m 안 상대 수) + 불균형(빠르게 달리며 차기)
    pressure = sum(1 for o in world.players if o.team != p.team and o.active
                   and vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) < 3)
    off_balance = clamp(vlen(p.vel[0], p.vel[1]) - 4, 0, 5)
    return (SHOT_BASE_ERR + (1 - p.stats.finishing / 99) * 11 + pressure * 4 + off_balance) \
        * condition_mult(p)

def register_shot(world, p, pen):
    # 슛 기록 + 상대 GK에게 반응 지연(또는 페널티 방향 예측) 부여
    b = world.ball
    m = world.match
    m['shots'][p.team] += 1
    pstat(world, p.pid, 'shots')
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
    pstat(world, p.pid, 'tackles')
    prob = 0.2 + 0.6 * logistic(0.08 * (p.stats.physical - o.stats.physical))
    if random.random() < prob:                       # 태클 성공 → 탈취
        world.match['tackles_won'][p.team] += 1
        pstat(world, p.pid, 'tackles_won')
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
    world.reds.add(p.pid)
    p.active = False
    p.pos = [-3.0, -3.0]; p.vel = [0.0, 0.0]
    set_event(world, "RED CARD")

def give_card(world, p):
    # v4: 레드 4% → 1%, 옐로 21% → 17%(실제 경기당 반칙 ~22 · 옐로 ~4 · 레드 ~0.1).
    # v3 비율로 90분을 돌리면 10경기 레드 7장이었다
    sev = random.random()
    if sev < RED_P:
        send_off(world, p)
    elif sev < RED_P + YELLOW_P:
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
    pstat(world, fouling.pid, 'fouls')
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
    on_stoppage(world)                      # 공이 멈췄을 때만 교체
    b = world.ball
    b.pos = [clamp(x, 1, L - 1), clamp(y, 1, W - 1), 0.0]
    b.vel = [0.0, 0.0, 0.0]
    b.owner = None; world.owner = None; b.cooldown = 0.5
    world.offside_suspects = set()
    world.no_offside = kind in ('throwin', 'corner', 'goalkick')
    world.shot = None; world.advantage = None; world.pending_pass = None; world.pass_to = None
    world.aerial = None
    world.penalty_pending = False
    cands = [p for p in world.players if p.team == team and p.active and p.role != 'GK']
    taker = min(cands, key=lambda p: vlen(p.pos[0] - b.pos[0], p.pos[1] - b.pos[1]),
                default=None)
    if taker:
        taker.pos = [b.pos[0] + (-0.8 if team == 0 else 0.8), b.pos[1]]
        taker.vel = [0.0, 0.0]
    b.last_team = team                      # 재개하는 팀이 공을 가진 것(코너킥이 그대로 나가면 골킥)
    world.setpiece = None
    gx, gy = world.opp_goal(team)
    if taker and (kind == 'corner' or
                  (kind == 'freekick' and vlen(gx - b.pos[0], gy - b.pos[1]) < FK_ROUTINE_DIST)):
        # v4 세트피스 루틴: 기다리는 동안 박스 침투·마킹·수비벽 → 키커가 조준해서 찬다
        world.setpiece = {'kind': kind, 'team': team, 'taker': taker, 't': SETPIECE_WAIT}
        b.cooldown = SETPIECE_WAIT + 1.0
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
    on_stoppage(world)
    defend = 1 - attack_team
    spot_x = PEN_SPOT if defend == 0 else L - PEN_SPOT
    b = world.ball
    b.pos = [spot_x, W / 2, 0.0]; b.vel = [0.0, 0.0, 0.0]
    b.owner = None; world.owner = None; b.cooldown = 0.6
    world.offside_suspects = set(); world.no_offside = True; world.setpiece = None
    world.shot = None; world.advantage = None; world.pending_pass = None; world.pass_to = None
    world.aerial = None
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
        prob = (GK_SAVE_BASE + 0.55 * k - 0.40 * stretch - 0.012 * max(0.0, speed - 20)
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

def takeon_prob(p, d):
    # 돌파 성공 확률: 공격(민첩·속도 평균) − 수비(민첩·피지컬 평균). 같으면 45%(실제 리그 평균 근처)
    att = (p.stats.agility + p.stats.pace) / 2
    dfd = (d.stats.agility + d.stats.physical) / 2
    return (0.15 + 0.6 * logistic(TAKEON_K * (att - dfd))) / condition_mult(p)

def do_takeon(world, p, d):
    # 성공: 수비는 TAKEON_BEATEN초 뒤처짐(못 움직이고 태클 못 함), 공격수는 골 쪽으로 가속
    # 실패: 수비가 공을 빼앗음(태클 성공처럼)
    if d is None or not d.active or vlen(d.pos[0] - p.pos[0], d.pos[1] - p.pos[1]) > 2.5:
        return
    p.tackle_cd = 1.0
    m = world.match
    m['dribbles'][p.team] += 1
    if random.random() < takeon_prob(p, d):
        m['dribbles_won'][p.team] += 1
        d.beaten = TAKEON_BEATEN; d.tackle_cd = max(d.tackle_cd, TAKEON_BEATEN)
        gx, gy = world.opp_goal(p.team)
        ux, uy = vnorm(gx - p.pos[0], gy - p.pos[1])
        sp = max(vlen(p.vel[0], p.vel[1]), 6.0)
        p.vel = [ux * sp, uy * sp]; p.facing = math.atan2(uy, ux)
    else:
        b = world.ball
        dx, dy = vnorm(d.pos[0] - b.pos[0], d.pos[1] - b.pos[1])
        kick(world, dx * 4, dy * 4, 0.3, d.team, d)
        b.cooldown = 0.15


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
    hp = find_player(world, world.human_pick)
    if hp is not None and hp.team == HUMAN_TEAM and hp.active and hp.role != 'GK':
        world.controlled = hp; return            # v4.1: 사람이 고정한 선수
    o = world.owner
    if o is not None and o.team == HUMAN_TEAM and o.active and o.role != 'GK':
        world.controlled = o; return             # 우리 GK가 잡은 공은 AI GK가 처리(예전엔 사람이 GK를 몰고 나감)
    cands = [p for p in world.players if p.team == HUMAN_TEAM and p.role != 'GK' and p.active]
    world.controlled = min(cands, key=lambda p: vlen(
        world.ball.pos[0] - p.pos[0], world.ball.pos[1] - p.pos[1]), default=None)


# ---------------- 규칙(골/세트피스) ----------------
def rules(world):
    b = world.ball
    if b.pos[0] < 0 or b.pos[0] > L:                    # 골라인 통과
        if abs(b.pos[1] - W / 2) < GOAL_W / 2 and b.pos[2] < CROSSBAR:
            scorer = 0 if b.pos[0] > L else 1
            world.score[scorer] += 1
            if world.last_kick and world.last_kick[0] == scorer:   # 골 = 마지막으로 찬 같은 팀 선수
                pstat(world, world.last_kick[1], 'goals')
            set_event(world, "GOAL")
            world.kickoff(1 - scorer); return
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
    if world.finished:                      # 경기 종료: 상태 고정
        return
    update_possession(world)
    update_advantage(world, dt)
    if world.owner is not None:
        world.match['poss_time'][world.owner.team] += dt
        world.owner_t += dt
    world.marks = ai.assign_marks(world)
    if world.setpiece is not None:
        world.setpiece['t'] -= dt
        if world.setpiece['t'] <= 0 or ai.setpiece_ready(world):
            ai.take_setpiece(world)
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
            intents.append((p, ai.ai_intent(world, p)))
    # 2) 액션은 매 프레임 무작위 순서로 처리 → 어느 팀도 '먼저 움직이는' 이득이 없다
    actions = [(p, it) for p, it in intents if it.action]
    random.shuffle(actions)
    for p, it in actions:
        if it.action == 'pass' and world.owner is p:
            do_pass(world, p, it.power, it.target)
        elif it.action == 'shoot' and world.owner is p:
            do_shot(world, p, it.power)
        elif it.action == 'takeon' and world.owner is p:
            do_takeon(world, p, it.target)
        elif it.action == 'clear' and world.owner is p:
            do_clear(world, p)
        elif it.action == 'cross' and world.owner is p:
            _, m, tx, ty = it.target
            do_cross(world, p, tx, ty)
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
    update_clock(world)


# ---------------- 경기 시간 · 하프타임 · 교체 ----------------
def game_minute(world):
    # 표시용 경기 시간(분). 경기 길이가 있으면 90분으로 늘려 보여준다(배속)
    if world.match_len:
        return world.time / world.match_len * MATCH_MINUTES
    return world.time / 60.0

def update_clock(world):
    if not world.match_len:
        return
    if world.half == 1 and world.time >= world.match_len / 2:
        world.half = 2
        set_event(world, "HALF TIME")
        for p in world.players:                # 라커룸 휴식: 단기 체력 회복 + 경기 체력 조금
            p.stamina = 100.0
            p.energy = min(1.0, p.energy + HALFTIME_RECOVER)
        world.kickoff(1 - world.first_kick)       # 킥오프 직전에 교체(on_stoppage)    # 후반은 상대 팀 킥오프(진영 교체는 단순화로 생략)
    elif world.time >= world.match_len:
        world.finished = True
        set_event(world, "FULL TIME")
        world.event_timer = 1e9                # 배너 계속 표시

def on_stoppage(world):
    # 공이 멈춘 순간(재개·페널티·골 뒤 킥오프·하프타임): 대기 중 교체 적용 → AI 교체 → AI 전술
    apply_subs(world)
    ai.maybe_subs(world)
    ai.maybe_tactics(world)

# ---------------- v4.2: 포메이션 ----------------
def assign_slots(players, slots):
    # 선수들을 자리 목록에 배정 → slots 순서대로 선수 리스트. 비용이 낮은 쌍부터(욕심 배정):
    #   지금 맡은 역할과 같으면 0, 주 포지션과 같으면 1, 다르면 2 + 폭(fy)·깊이(fx) 차이
    #   GK 자리는 GK끼리만. 선수가 자리보다 적으면(퇴장) 가장 앞쪽(fx 큰) 자리부터 비운다
    gk_slots = [i for i, sl in enumerate(slots) if sl[0] == 'GK']
    gks = [p for p in players if p.role == 'GK' or (p.pos_role == 'GK' and p.role not in ROLE_GAP)]
    out = [None] * len(slots)
    for i, p in zip(gk_slots, gks):
        out[i] = p
    field = [p for p in players if p not in gks]
    free = sorted((i for i, sl in enumerate(slots) if sl[0] != 'GK'), key=lambda i: slots[i][1])
    free = free[:len(field)]
    def cost(p, i):
        role, fx, fy = slots[i]
        c = 0 if p.role == role else 1 if p.pos_role == role else 2
        return c + abs(p.base_fy - fy) * 0.8 + abs(p.base_fx - fx) * 0.3
    pairs = sorted((cost(p, i), k, i) for k, p in enumerate(field) for i in free)
    used_p, used_i = set(), set()
    for c, k, i in pairs:
        if k in used_p or i in used_i:
            continue
        out[i] = field[k]; used_p.add(k); used_i.add(i)
    return out

def set_formation(world, team, name):
    # 포메이션 바꾸기(킥오프 전 · 경기 중 아무 때나 — 실제로도 전술 변경은 멈춤이 필요 없다)
    # 뛰고 있는 선수만 새 자리에 배정. 퇴장으로 줄었으면 가장 앞 자리부터 비운다
    if name not in FORMATIONS:
        return False, f"unknown formation {name!r}"
    slots = FORMATIONS[name]
    on = [p for p in world.players if p.team == team and p.active]
    for sl, p in zip(slots, assign_slots(on, slots)):
        if p is not None:
            p.role, p.base_fx, p.base_fy = sl
    changed = world.formation[team] != name
    world.formation[team] = name
    if world.time > 0:
        if changed:
            world.formation_log.append((game_minute(world), team, name))
            set_event(world, "FORMATION " + name)
    else:                                   # 킥오프 전: 새 자리로 바로 세운다
        world.kickoff(world.first_kick if hasattr(world, 'first_kick') else 0)
    return True, name


def subs_left(world, team):
    queued = sum(1 for q in world.sub_queue if q[0] == team)
    return SUBS_MAX - world.match['subs'][team] - queued

def find_player(world, pid):
    if pid is None:
        return None
    return next((p for p in world.players if p.pid == pid), None)

def request_sub(world, team, out_pid, bench_idx):
    # 교체 요청(사람·AI 공통). 규칙 검사 후 대기열에 넣고 다음 멈춤에서 적용 → (성공 여부, 이유)
    if world.finished:
        return False, "match finished"
    if subs_left(world, team) <= 0:
        return False, f"no subs left ({SUBS_MAX} used)"
    if out_pid in world.reds:
        return False, "sent-off player cannot be replaced"
    if any(p.pid == out_pid for p in world.subbed_off):
        return False, "already substituted"
    out = find_player(world, out_pid)
    if out is None or out.team != team or not out.active:
        return False, "player not on the pitch"
    if not 0 <= bench_idx < len(world.bench[team]):
        return False, "no such bench player"
    inp = world.bench[team][bench_idx]
    if any(q[1] == out_pid or q[2] == inp.pid for q in world.sub_queue):
        return False, "already queued"
    world.sub_queue.append((team, out_pid, inp.pid))
    return True, "queued"

def apply_subs(world):
    # 대기 중 교체 적용: 들어온 선수는 나간 선수의 포메이션 자리·위치, 체력은 새것
    queue, world.sub_queue = world.sub_queue, []
    for team, out_pid, in_pid in queue:
        out = find_player(world, out_pid)
        inp = next((p for p in world.bench[team] if p.pid == in_pid), None)
        if out is None or not out.active or inp is None:    # 그 사이 퇴장 등 → 취소
            continue
        inp.role, inp.base_fx, inp.base_fy = out.role, out.base_fx, out.base_fy
        inp.pos = list(out.pos); inp.vel = [0.0, 0.0]
        inp.stamina, inp.energy, inp.active = 100.0, 1.0, True
        world.players[world.players.index(out)] = inp
        world.bench[team].remove(inp)
        world.subbed_off.append(out)
        world.offside_suspects.discard(out.pid)
        if world.controlled is out:
            world.controlled = None
        if world.human_pick == out.pid:
            world.human_pick = None
        world.match['subs'][team] += 1
        world.sub_log.append((game_minute(world), team, out.name, inp.name))
        set_event(world, "SUBSTITUTION")

# ---------------- v4.1: 스쿼드 · 선발 · 선수 정보 ----------------
def name_squad(world):
    # 이름은 난수를 안 쓰는 고정 목록에서(경기 재현성 그대로). 선발 1~11번, 벤치 12~18번
    for team in (0, 1):
        squad = [p for p in world.players if p.team == team] + world.bench[team]
        for i, p in enumerate(squad):
            p.name = SURNAMES[(i + team * SQUAD_SIZE) % len(SURNAMES)]
            p.number = i + 1
            p.pos_role = p.pos_role or p.role

def load_lineup(world, squad):
    # squad(dict): {"teams": [{"name", "formation"(선택), "players": [{"name","number","role","stats"}...],
    #                          "lineup": [등번호 11개]} × 2]}
    # 선발 11명(GK 정확히 1명)을 포메이션 자리에 배정: 같은 주 포지션 자리 우선(assign_slots)
    pid = 0
    pool = []
    for team, t in enumerate(squad['teams']):
        world.team_names[team] = t.get('name', TEAM_NAMES[team])
        if t.get('formation'):
            if t['formation'] not in FORMATIONS:
                raise ValueError(f"unknown formation {t['formation']!r}")
            world.formation[team] = t['formation']
        by_num = {}
        for e in t['players']:
            st = Stats(**e['stats'])
            p = Player(team, e['role'], 0.0, 0.0, st, pos=[-5.0, -5.0], name=e['name'],
                       number=int(e['number']), pos_role=e['role'])
            by_num[p.number] = p
        lineup = [by_num[int(n)] for n in t['lineup']]
        n_gk = sum(1 for p in lineup if p.pos_role == 'GK')
        if len(lineup) != 11 or n_gk != 1:
            raise ValueError(f"team {team} lineup must be 11 players with exactly 1 GK, "
                             f"got {len(lineup)} players / {n_gk} GK")
        slots = FORMATIONS[world.formation[team]]
        for (role, fx, fy), p in zip(slots, assign_slots(lineup, slots)):
            p.role, p.base_fx, p.base_fy, p.pid = role, fx, fy, pid
            world.players.append(p); pid += 1
        pool.append([p for p in by_num.values() if p not in lineup])
    for team in (0, 1):
        for p in pool[team]:
            p.pid = pid; pid += 1
            world.bench[team].append(p)

def squad_dict(world):
    # 지금 스쿼드를 파일 형식(dict)으로 — --export-squad
    out = {'teams': []}
    for team in (0, 1):
        on = [p for p in world.players if p.team == team]
        rows = []
        for p in on + world.bench[team] + [q for q in world.subbed_off if q.team == team]:
            rows.append({'name': p.name, 'number': p.number, 'role': p.pos_role,
                         'stats': {k: round(v, 1) for k, v in vars(p.stats).items()}})
        out['teams'].append({'name': world.team_names[team], 'formation': world.formation[team],
                             'players': rows,
                             'lineup': [p.number for p in on]})
    return out

def load_squad(path):
    with open(path, encoding='utf-8') as fp:
        return json.load(fp)

def swap_lineup(world, team, starter_pid, bench_idx):
    # 킥오프 전 선발 바꾸기(교체 횟수 아님): 같은 주 포지션끼리만
    if world.time > 0:
        return False, "match already started"
    a = find_player(world, starter_pid)
    if a is None or a.team != team or not 0 <= bench_idx < len(world.bench[team]):
        return False, "bad selection"
    b = world.bench[team][bench_idx]
    if a.pos_role != b.pos_role:
        return False, f"position differs ({a.pos_role} vs {b.pos_role})"
    b.role, b.base_fx, b.base_fy, b.pos, b.vel = a.role, a.base_fx, a.base_fy, list(a.pos), [0.0, 0.0]
    a.pos = [-5.0, -5.0]
    world.players[world.players.index(a)] = b
    world.bench[team][bench_idx] = a
    if world.human_pick == a.pid:
        world.human_pick = None
    return True, "swapped"

def pstat(world, pid, key):
    if pid is None:
        return
    d = world.pstats.setdefault(pid, dict.fromkeys(PSTAT_KEYS, 0))
    d[key] += 1

def player_info(world, p):
    # 선수 카드 내용(엔진 상태 그대로). UI는 이걸 그리기만 한다
    st = world.pstats.get(p.pid, dict.fromkeys(PSTAT_KEYS, 0))
    where = ('pitch' if p in world.players else 'bench' if p in world.bench[p.team]
             else 'subbed off')
    return {'name': p.name, 'number': p.number, 'team': world.team_names[p.team],
            'position': p.pos_role or p.role, 'slot': p.role, 'where': where,
            'stats': {k: round(v) for k, v in vars(p.stats).items()},
            'stamina': round(p.stamina), 'energy': round(p.energy * 100),
            'match': dict(st), 'yellow': world.cards.get(p.pid, 0),
            'red': p.pid in world.reds}

def switch_control(world):
    # Q: 공에 (지금 선수 다음으로) 가까운 동료로 바꾸고 고정
    b = world.ball
    cands = sorted((p for p in world.players if p.team == HUMAN_TEAM and p.active
                    and p.role != 'GK' and p is not world.controlled),
                   key=lambda p: vlen(b.pos[0] - p.pos[0], b.pos[1] - p.pos[1]))
    if cands:
        world.human_pick = cands[0].pid
        world.controlled = cands[0]


# AI 모듈은 맨 끝에서 불러온다(서로 부르는 순환 import — 이름은 호출할 때 찾음)
import soccer_ai as ai  # noqa: E402

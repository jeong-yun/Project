"""
탑다운 축구 프로토타입 (Pygame) v4 - AI 판단 · 세트피스 루틴 · 경기 시간/피로
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

v4 추가:
    - AI 판단: xG(기대득점)로 슛 결정, 패스 길목 검사 + 기대값으로 패스 선택, 수비 라인 높이,
               맨마킹·지연 수비·패스 끊기, 1대1 돌파, 압박 시 걷어내기, 첫 터치 실수, 침투·지원 움직임
    - 크로스·헤더(공중 경합, GK 공중볼 캐치)
    - 세트피스 루틴: 코너·공격 지역 프리킥 → 박스 침투·마킹·수비벽 → 키커 조준(크로스/직접 슛)
    - 경기 시간: 90분 배속(기본 600초), 하프타임, 종료 휘슬, 경기 체력(energy), 교체(팀당 5명)
    - 피로 → 패스·슛·헤더 정확도, 첫 터치, 돌파 성공률 저하
    - 공정성 수정: 1팀 xG 각도 계산 버그, 킥오프를 실점한 팀이 하도록

실행:  python soccer.py                 (--length 600 = 90분을 600초에, 0이면 끝없이)
테스트: python soccer.py --test 12000 --seed 1,2,3   /   python soccer.py --match --seed 1,2
       python -m pytest tests
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
RED_P, YELLOW_P = 0.01, 0.17   # 반칙 1번당 레드·옐로 확률
SHOT_BASE_ERR = 8.5         # 누구에게나 있는 슛 각도 오차(도). v3 3° → v4 8.5°(유효슈팅 v3 66% → 실제 33%에 가깝게)
GK_SAVE_BASE = 0.35         # GK 선방 확률 기본값(v3 0.45). 아래 gk_save 식 참고

# --- v4: AI 판단 상수 (값의 근거는 READ_ME 4절) ---
PASS_MIN, PASS_MAX = 4.0, 40.0   # 패스 후보 거리(m)
PASS_BASE_ERR = 4.5         # 누구에게나 있는 패스 각도 오차(도). v3는 0 → 패스 성공 90%+로 너무 정확
PASS_SPEED_BASE = 8.0       # 패스 공 속도 = BASE + PER_M × 거리 (m/s)
PASS_SPEED_PER_M = 0.55
PASS_SPEED_MAX = 22.0       # v3는 거리×1.6(최대 26 m/s) → 너무 빨라 받는 사람 지나침
OPP_REACT = 0.2             # 상대가 패스를 보고 움직이기 시작하는 시간(초)
OPP_CLOSE_SPEED = 3.0       # 상대가 패스 길목으로 다가서는 평균 속도(m/s, 정지→가속 포함)
LANE_END = 0.85             # 길목 끝 15%는 받는 사람이 마중 나와 지킨다
LANE_MARGIN = 1.5           # 길목 위험이 0이 되는 여유 거리(m)
PASS_LONG_RISK = 0.012      # 15m 넘는 패스는 1m마다 성공 추정 1.2%↓
VAL_PROG, VAL_XG = 1.5, 2.0  # 위치 가치 = VAL_PROG × 전진도² + VAL_XG × xG
LOSS_COST = 0.20            # 공을 뺏길 때 손해(가치 단위). 우리 골대 앞 ×1.5 ~ 상대 골대 앞 ×0.5
MARK_ZONE = 40.0            # 우리 골대에서 이 거리 안의 상대만 맨마킹(그 밖은 지역 방어)
OFFSIDE_JUDGE = 2.0         # 패서가 '확실한 오프사이드'로 알아보는 거리(m). 이보다 아슬아슬하면 못 알아봄
SHOT_XG_MIN = 0.035         # 기대득점(xG)이 이 이상이면 슛
SHOT_XG_PRESSED = 0.02      # 상대가 붙으면(PRESS_DIST 안) 이 정도만 돼도 슛
PRESS_DIST = 2.3
BLOCKER_XG = 0.8            # 슛 길 위 수비 1명마다 xG ×0.8
HOLD_PATIENCE = 4.0         # 공을 끄는 가치가 이 시간(초)에 걸쳐 30%까지 줄어든다 → 패스·크로스 선택
STALL_TIME = 2.5            # 공 가진 채 이만큼 막혀 있으면 낮은 xG(PRESSED 기준)도 슛
RUN_RATE = 0.35             # 공격수 뒷공간 침투 시작 빈도(초당)
RUN_TIME = 1.6              # 침투 지속 시간(초)
RUN_DEPTH = 8.0             # 침투 목표: 오프사이드 라인 뒤 몇 m
TOUCH_TIME = 0.8            # 공을 받고 고개 들어 패스/슛하기까지 걸리는 시간(초). 압박 받으면 40%
MARK_RANGE = 15.0           # 수비가 자기 자리에서 이 거리 안 상대를 맨마킹
MARK_DIST = 1.8             # 마킹: 상대에서 우리 골대·공 쪽으로 몇 m
INTERCEPT_RANGE = 5.0       # 상대 패스 경로가 이 거리 안을 지나면 끊으러 나간다
JOCKEY_DIST = 1.0           # 압박 수비가 공 앞(우리 골대 쪽) 몇 m에 서나
TACKLE_REACH = 1.6          # 태클 가능 거리(m)
TACKLE_RATE = 1.5           # 붙어 있을 때 초당 태클 시도율(v3: 2m 안이면 쿨다운마다 무조건)
TAKEON_ZONE = 0.40          # 상대 진영 쪽(전진도 40% 이상)에서만 돌파 시도
TAKEON_K = 0.06             # 돌파 성공 로지스틱 기울기(능력치 1당)
TAKEON_BEATEN = 0.7         # 돌파당한 수비가 뒤처지는 시간(초)
CLEAR_ZONE = 0.35           # 자기 골대 쪽 35% 안에서 압박받고 패스 길이 없으면 걷어낸다
TOUCH_EASY = 7.0            # 이보다 빠른(상대 속도) 공은 첫 터치가 튈 수 있다(m/s)
TOUCH_BAD_PER = 0.024       # 1 m/s 초과마다 첫 터치 실패 확률 +2.4%(기술 낮을수록 더)
# --- v4: 크로스 · 헤더(공중 경합) ---
HEADER_TOP = 2.6            # 머리로 닿는 최대 높이(m). CONTROL_HEIGHT(1.0)~이 사이가 공중 경합
HEADER_RADIUS = 1.2         # 공중 경합에 낄 수 있는 수평 거리(m)
HEADER_SHOT_DIST = 14.0     # 공격 쪽이 이 거리 안에서 이기면 헤더 슛, 멀면 떨궈 주기
HEADER_ERR = 7.0            # 헤더 슛 기본 각도 오차(도) — 발 슛(5°)보다 부정확
CROSS_Z = 1.7               # 크로스가 목표 지점에 도착하는 높이(머리 높이)
SETPIECE_WAIT = 6.0         # 코너·공격 지역 프리킥: 자리 잡기 최대 대기(초). 침투조가 다 도착하면 바로 찬다
SETPIECE_MIN = 1.5          # 최소 대기(초)
FK_ROUTINE_DIST = 35.0      # 상대 골대에서 이 거리 안 프리킥만 루틴(그 밖은 빠른 재개)
FK_SHOT_DIST = 28.0         # 이 안 + 중앙 쪽이면 직접 슛, 아니면 박스로 크로스
WALL_N = 3                  # 직접 프리킥 수비벽 인원
CROSS_ZONE_X = 0.72         # 상대 골라인 쪽 28% 안 + 측면(박스 폭 밖)이면 크로스 지역
# 수비 라인 높이(자기 골대 기준 0~1). 라인 = SLOPE × 공 위치 + OFS, MIN~MAX로 제한
LINE_SLOPE = 0.75
LINE_DEF_OFS, LINE_DEF_MIN, LINE_DEF_MAX = -0.12, 0.10, 0.45
LINE_ATT_OFS, LINE_ATT_MIN, LINE_ATT_MAX = -0.05, 0.15, 0.55
ROLE_GAP = {'DEF': 0.0, 'MID': 0.17, 'FWD': 0.34}   # 수비 라인에서 각 라인까지 간격(수비 시)
# --- v4: 경기 시간 · 피로 ---
MATCH_MINUTES = 90          # 표시 경기 시간(분). --length 초에 압축(기본 600초 = ×9)
ENERGY_PER_MIN = 0.0035     # 실제 경기 1분마다 경기 체력 감소(운동량 1, 스태미나 스탯 보정 전)
HALFTIME_RECOVER = 0.06     # 하프타임 경기 체력 회복
ACC_FATIGUE = 0.8           # 컨디션 1 → 0 사이에서 패스·슛·헤더 오차 최대 +80%, 첫 터치 실수·돌파도 비례
SUBS_MAX = 5                # 팀당 교체 인원(현행 규칙)
SUB_AFTER_MIN = 55          # 이 시간 이후부터 교체
SUB_ENERGY = 0.75           # 경기 체력이 이 아래면 교체 후보
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
FORMATION = [   # 4-4-2 (공격 +x 기준 정규화 좌표)
    ('GK', 0.05, 0.50),
    ('DEF', 0.22, 0.18), ('DEF', 0.22, 0.39), ('DEF', 0.22, 0.61), ('DEF', 0.22, 0.82),
    ('MID', 0.46, 0.18), ('MID', 0.46, 0.39), ('MID', 0.46, 0.61), ('MID', 0.46, 0.82),
    ('FWD', 0.68, 0.40), ('FWD', 0.68, 0.60),
]

class World:
    def __init__(self, match_len=None):
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
        self.next_pid = 2 * len(FORMATION)   # 교체 선수 번호
        pid = 0
        for team in (0, 1):
            for role, fx, fy in FORMATION:
                self.players.append(Player(team, role, fx, fy, rand_stats(role), pid=pid))
                pid += 1
        self.first_kick = 0 if random.random() < 0.5 else 1   # 동전 던지기(예전엔 늘 0팀)
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
            gap = ROLE_GAP[p.role] * (1.15 if att else 1.0)
            fx = clamp(self.def_line(p.team, ball, att) + gap, 0.04, 0.92)
        width = 1.0 if p.role == 'GK' or self.attacking(p.team) else 0.8   # 수비 땐 좁게
        fy = clamp(0.5 + (p.base_fy - 0.5) * width + (by - 0.5) * 0.25, 0.05, 0.95)
        if p.team == 0:
            return (fx * L, fy * W)
        return ((1 - fx) * L, (1 - fy) * W)

    def kickoff(self, team=0):
        # team = 킥오프하는 팀(실점한 팀). v3까지는 늘 같은 진형으로 시작해 0팀이 먼저 공을 잡았다
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
    lofted_kick(world, p, tx, ty, pass_error_deg(p) * 1.3)
    world.aerial = {'team': p.team, 'tried': set()}
    world.pending_pass = p.team

def box_spots(world, team, near_y):
    # 박스 침투 지점(공 쪽이 니어): 니어 포스트, 파 포스트, 페널티 스폿, 골문 앞, 박스 모서리(세컨드 볼)
    gx, gy = world.opp_goal(team)
    sgn = -1 if team == 0 else 1                     # 골라인에서 경기장 안쪽 방향
    side = 1 if near_y > gy else -1
    return [(gx + sgn * 5.5, gy + side * 2.5), (gx + sgn * 6.5, gy - side * 3.5),
            (gx + sgn * 11, gy + side * 0.5), (gx + sgn * 8, gy - side * 0.5),
            (gx + sgn * 18, gy - side * 6)]

def cross_target(world, p):
    # 박스 안 동료 중 상대가 가장 적게 붙은 선수(공중 경합 승산) → (기대값, 동료, x, y)
    best = None
    for m in world.players:
        if m.team != p.team or m is p or m.role == 'GK' or not m.active:
            continue
        if not in_box(m.pos[0], m.pos[1], 1 - p.team) or offside_obvious(world, p, m):
            continue
        rivals = sum(1 for o in world.players if o.team != p.team and o.active
                     and vlen(o.pos[0] - m.pos[0], o.pos[1] - m.pos[1]) < 2.5)
        ok = 1.0 / (1.0 + rivals)
        ev = ok * pos_value(world, p.team, m.pos[0], m.pos[1]) - (1 - ok) * loss_cost(p.team, p.pos[0])
        if best is None or ev > best[0]:
            best = (ev, m, m.pos[0], m.pos[1])
    return best

def in_cross_zone(world, p):
    prog = p.pos[0] / L if p.team == 0 else 1 - p.pos[0] / L
    return prog > CROSS_ZONE_X and abs(p.pos[1] - W / 2) > PBOX_HALF - 4

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

# ---------------- v4: 판단 도구(xG · 패스 길목 · 위치 가치) ----------------
def shot_xg(world, team, x, y, shooter=None):
    # 기대득점 근사: 거리·골대가 보이는 각도 로지스틱 × 슛 길을 막은 수비 수
    #   중앙 11m ≈ 0.33, 중앙 20m ≈ 0.07, 중앙 25m ≈ 0.03 (실제 xG 모델과 비슷한 크기)
    gx, gy = world.opp_goal(team)
    dist = vlen(gx - x, gy - y)
    # 두 골대 기둥을 보는 각도(벡터 사이각 — atan2 차이는 1팀(−x 방향)에서 ±π를 넘어 틀린다)
    u1x, u1y = gx - x, gy - GOAL_W / 2 - y
    u2x, u2y = gx - x, gy + GOAL_W / 2 - y
    angle = abs(math.atan2(u1x * u2y - u1y * u2x, u1x * u2x + u1y * u2y))
    xg = logistic(-0.35 - 0.15 * dist + 2.0 * angle)
    ux, uy = vnorm(gx - x, gy - y)
    for o in world.players:                      # 슛 길(골대 중앙 방향) 위의 수비 필드 선수
        if o.team == team or not o.active or o.role == 'GK' or o is shooter:
            continue
        ox, oy = o.pos[0] - x, o.pos[1] - y
        along = ox * ux + oy * uy
        if 0 < along < dist and abs(ox * uy - oy * ux) < 0.5 + 0.08 * along:
            xg *= BLOCKER_XG
    return xg

def pass_speed(dist):
    return clamp(PASS_SPEED_BASE + PASS_SPEED_PER_M * dist, 9.0, PASS_SPEED_MAX)

def lane_risk(world, p, tx, ty, speed, recv):
    # 패스 길목 위험(0~1): 공 경로 옆 상대가 공이 오기 전에 닿을 수 있나 + 목표점에 먼저 닿나
    px, py = p.pos
    dx, dy = tx - px, ty - py
    d2 = dx * dx + dy * dy
    d = math.sqrt(d2)
    if d < 1e-6:
        return 1.0
    rd = vlen(recv.pos[0] - tx, recv.pos[1] - ty)
    risk = 0.0
    for o in world.players:
        if o.team == p.team or not o.active:
            continue
        ox, oy = o.pos[0] - px, o.pos[1] - py
        t = (ox * dx + oy * dy) / d2
        if t <= 0:
            continue
        if t < LANE_END:                         # 길목: 공이 그 지점에 오는 시간 vs 상대가 다가서는 거리
            perp = abs(ox * dy - oy * dx) / d
            tb = t * d / speed
            # 발끝에선 정말 공 경로 위(0.6m)여야 막고, 공이 멀리 갈수록 다가설 시간이 생긴다
            reach = min(CONTROL_RADIUS, 0.6 + 0.6 * t * d) + OPP_CLOSE_SPEED * max(0.0, tb - OPP_REACT)
            r = 1 - (perp - reach) / LANE_MARGIN
        else:                                    # 길목 끝·목표점: 받는 사람보다 상대가 가까우면 위험
            od = vlen(o.pos[0] - tx, o.pos[1] - ty)
            r = (rd - od + 1.0) / 3.0
        risk = max(risk, clamp(r, 0.0, 1.0))
    return risk

def loss_cost(team, x):
    # 뺏겼을 때 손해: 우리 골대 가까울수록 크다(상대 진영에선 과감하게)
    prog = x / L if team == 0 else 1 - x / L
    return LOSS_COST * (1.5 - prog)

def pos_value(world, team, x, y):
    # 위치 가치: 상대 골대 쪽으로 갈수록 가파르게(제곱) + 슛 가능성(xG). 직선이면 옆·뒤 패스만 돌림
    prog = x / L if team == 0 else 1 - x / L
    return VAL_PROG * prog * prog + VAL_XG * shot_xg(world, team, x, y)

def offside_obvious(world, p, m):
    # 패서 눈에도 확실한 오프사이드(라인보다 OFFSIDE_JUDGE m 이상 앞) → 그 선수에겐 안 준다
    line = offside_line(world, p.team)
    bx = world.ball.pos[0]
    if p.team == 0:
        return m.pos[0] > max(line, bx) + OFFSIDE_JUDGE and m.pos[0] > L / 2
    return m.pos[0] < min(line, bx) - OFFSIDE_JUDGE and m.pos[0] < L / 2

def pass_options(world, p):
    # 패스 후보: (기대값, 성공 추정, 받는 사람, 목표 x, 목표 y, 공 속도)
    opts = []
    for m in world.players:
        if m.team != p.team or m is p or m.role == 'GK' or not m.active:
            continue
        d0 = vlen(m.pos[0] - p.pos[0], m.pos[1] - p.pos[1])
        if d0 < PASS_MIN or d0 > PASS_MAX or offside_obvious(world, p, m):
            continue
        lead = d0 / pass_speed(d0)                       # 받는 사람이 움직이는 쪽으로 앞에 준다
        tx = clamp(m.pos[0] + m.vel[0] * lead, 1, L - 1)
        ty = clamp(m.pos[1] + m.vel[1] * lead, 1, W - 1)
        dist = vlen(tx - p.pos[0], ty - p.pos[1])
        speed = pass_speed(dist)
        ok = (1 - lane_risk(world, p, tx, ty, speed, m)) * \
            (1 - PASS_LONG_RISK * max(0.0, dist - 15))
        ev = ok * pos_value(world, p.team, tx, ty) - (1 - ok) * loss_cost(p.team, p.pos[0])
        opts.append((ev, ok, m, tx, ty, speed))
    opts.sort(key=lambda o: o[0], reverse=True)
    return opts

def do_pass(world, p, power, opt=None):
    # 패스: 후보 중 기대값 최고(사람의 Space도 같은 자동 조준) → 패스 정확도 오차
    if opt is None:
        opts = pass_options(world, p)
        opt = opts[0] if opts else None
    world.match['passes'][p.team] += 1
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
    maybe_subs(world)                       # 공이 멈췄을 때만 교체
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

def intercept_point(b, p, run_speed=7.0):
    # 굴러가는 공을 가장 빨리 만날 수 있는 지점(0.1초 간격으로 공 궤적을 앞으로 굴려 본다)
    x, y, vx, vy = b.pos[0], b.pos[1], b.vel[0], b.vel[1]
    for i in range(1, 26):
        sp = vlen(vx, vy)
        if sp > 0:
            f = max(0.0, sp - GROUND_FRIC * 3 * 0.1) / sp
            vx *= f; vy *= f
        x += vx * 0.1; y += vy * 0.1
        if vlen(x - p.pos[0], y - p.pos[1]) - CONTROL_RADIUS <= run_speed * i * 0.1:
            break
    return clamp(x, 0.5, L - 0.5), clamp(y, 0.5, W - 0.5)

def near_ball_path(b, p, r):
    # 빠른 사전 검사: 공 진행 방향 반직선에서 r m 안에 있나(intercept_point는 비싸서 걸러 낸다)
    vx, vy = b.vel[0], b.vel[1]
    ox, oy = p.pos[0] - b.pos[0], p.pos[1] - b.pos[1]
    sp = vlen(vx, vy)
    if sp < 1e-6:
        return vlen(ox, oy) < r
    along = (ox * vx + oy * vy) / sp
    if along < -r:
        return False
    return abs(ox * vy - oy * vx) / sp < r

def ai_on_ball(world, p):
    # 볼 소유 판단: ① 슛 가치(xG) ② 패스 후보 기대값 vs 지키며 드리블 ③ 드리블(상대 피해서)
    gx, gy = world.opp_goal(p.team)
    dgoal = vlen(gx - p.pos[0], gy - p.pos[1])
    opps = [o for o in world.players if o.team != p.team and o.active]
    press = min((vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) for o in opps), default=99)
    pressed = press < PRESS_DIST
    ready = world.owner_t >= TOUCH_TIME * (0.4 if pressed else 1.0)
    if not ready:                                          # 공 잡는 중: 가장 가까운 상대 반대쪽으로 지킨다
        near = min(opps, key=lambda o: vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]))
        ax, ay = vnorm(p.pos[0] - near.pos[0], p.pos[1] - near.pos[1])
        dx, dy = vnorm(gx - p.pos[0], gy - p.pos[1])
        return Intent(move=vnorm(dx + ax * 1.5, dy + ay * 1.5) if pressed else (dx * 0.5, dy * 0.5))
    if p.role != 'GK':
        xg = shot_xg(world, p.team, p.pos[0], p.pos[1], p)
        if xg >= SHOT_XG_MIN or ((pressed or world.owner_t > STALL_TIME) and xg >= SHOT_XG_PRESSED):
            return Intent(action='shoot', power=clamp(1 - dgoal / 30, 0.4, 1.0))
    opts = pass_options(world, p)
    hold_ok = clamp((press - 1.0) / (PRESS_DIST + 0.5), 0.0, 1.0)   # 상대가 붙을수록 지키기 어렵다
    patience = clamp(1 - world.owner_t / HOLD_PATIENCE, 0.3, 1.0)   # 오래 끌수록 지키기 가치↓
    hold_ev = hold_ok * patience * pos_value(world, p.team, p.pos[0], p.pos[1]) \
        - (1 - hold_ok) * loss_cost(p.team, p.pos[0])
    if p.role == 'GK' and opts:
        return Intent(action='pass', power=0.6, target=opts[0])
    best_ev = opts[0][0] if opts else -1e9
    if in_cross_zone(world, p):                            # 측면 깊은 곳: 박스 안 동료에게 크로스
        cr = cross_target(world, p)
        if cr is not None and cr[0] > max(best_ev, hold_ev):
            return Intent(action='cross', target=cr)
    prog = p.pos[0] / L if p.team == 0 else 1 - p.pos[0] / L
    d = front_defender(world, p)
    if d is not None and prog > TAKEON_ZONE and p.tackle_cd <= 0:   # 1대1 돌파 기대값
        ux, uy = vnorm(gx - p.pos[0], gy - p.pos[1])
        win = takeon_prob(p, d)
        t_ev = win * pos_value(world, p.team, p.pos[0] + ux * 5, p.pos[1] + uy * 5)             - (1 - win) * loss_cost(p.team, p.pos[0])
        if t_ev > max(best_ev, hold_ev):
            return Intent(action='takeon', target=d)
    if opts and best_ev > hold_ev:
        return Intent(action='pass', power=0.6, target=opts[0])
    if pressed and prog < CLEAR_ZONE:                      # 우리 진영에서 막혔다 → 멀리 걷어낸다
        return Intent(action='clear')
    dx, dy = vnorm(gx - p.pos[0], gy - p.pos[1])
    ax = ay = 0.0
    for o in opps:                                         # 앞을 막은 상대를 옆으로 비켜 간다
        ox, oy = p.pos[0] - o.pos[0], p.pos[1] - o.pos[1]
        d = vlen(ox, oy)
        if 0 < d < 7 and (o.pos[0] - p.pos[0]) * dx + (o.pos[1] - p.pos[1]) * dy > 0:
            ax += ox / d * (7 - d) / 7; ay += oy / d * (7 - d) / 7
    return Intent(move=vnorm(dx + ax * 0.8, dy + ay * 0.8), sprint=press > 4 and dgoal > 25)

def front_defender(world, p):
    # 골 쪽 앞 2m 안에서 길을 막은 상대(돌파 대상)
    gx, gy = world.opp_goal(p.team)
    ux, uy = vnorm(gx - p.pos[0], gy - p.pos[1])
    best, bd = None, 2.0
    for o in world.players:
        if o.team == p.team or not o.active or o.role == 'GK' or o.beaten > 0:
            continue
        ox, oy = o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]
        d = vlen(ox, oy)
        if d < bd and ox * ux + oy * uy > 0:
            best, bd = o, d
    return best

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

def forward_run(world, p):
    # 공격수: 평소엔 오프사이드 라인 1m 안쪽, 가끔 라인 뒤로 침투(타이밍이 이르면 오프사이드)
    b = world.ball
    gx, gy = world.opp_goal(p.team)
    line = offside_line(world, p.team)
    sgn = 1 if p.team == 0 else -1
    if p.run <= 0 and random.random() < RUN_RATE * DT:
        p.run = RUN_TIME
    if p.run > 0:
        return move_to(world, p, line + sgn * RUN_DEPTH,
                       clamp(p.pos[1] + (gy - p.pos[1]) * 0.3, 8, W - 8), sprint=True)
    tx = (b.pos[0] + gx) / 2
    tx = min(tx, line - 1.0) if p.team == 0 else max(tx, line + 1.0)
    return move_to(world, p, tx, clamp(p.pos[1], 8, W - 8), sprint=True)

def box_run_spot(world, p):
    # 볼 가진 동료가 크로스 지역이면 공격수 2명 + 먼 쪽 미드 1명이 박스 지점으로(세트피스도 같은 지점)
    o = world.owner
    if o is None or o is p or not in_cross_zone(world, o):
        return None
    runners = [q for q in world.players if q.team == p.team and q.active and q is not o
               and (q.role == 'FWD' or (q.role == 'MID' and (q.pos[1] - W / 2) * (o.pos[1] - W / 2) < 0))]
    runners.sort(key=lambda q: q.pid)
    if p not in runners:
        return None
    return box_spots(world, p.team, o.pos[1])[runners.index(p) % 5]

def aerial_point(b, z=2.0):
    # 높이 뜬 공이 내려오며 z 높이를 지나는 지점(공기저항 무시)
    disc = b.vel[2] ** 2 + 2 * GRAV * (b.pos[2] - z)
    if disc < 0:
        return None
    t = (b.vel[2] + math.sqrt(disc)) / GRAV
    return b.pos[0] + b.vel[0] * t, b.pos[1] + b.vel[1] * t

def support_player(world, team):
    # 볼 가진 선수에게 가장 가까운 동료 미드필더 1명이 짧은 패스 길을 만들어 준다
    o = world.owner
    mids = [q for q in world.players if q.team == team and q.role == 'MID' and q.active
            and q is not o]
    return min(mids, key=lambda q: vlen(q.pos[0] - o.pos[0], q.pos[1] - o.pos[1]), default=None)

def open_spot(world, carrier, p):
    # 볼 가진 선수 주변 11m, 7방향 중 상대와 멀고 패스 길이 열린 곳(앞쪽 가산)
    gx, gy = world.opp_goal(p.team)
    base = math.atan2(gy - carrier.pos[1], gx - carrier.pos[0])
    best, bs = (p.pos[0], p.pos[1]), -1e9
    for deg in (-120, -75, -35, 0, 35, 75, 120):
        a = base + math.radians(deg)
        x = clamp(carrier.pos[0] + math.cos(a) * 11, 2, L - 2)
        y = clamp(carrier.pos[1] + math.sin(a) * 11, 2, W - 2)
        space = lane = 9.0
        for o in world.players:
            if o.team == p.team or not o.active:
                continue
            space = min(space, vlen(o.pos[0] - x, o.pos[1] - y))
            ux, uy = x - carrier.pos[0], y - carrier.pos[1]
            ox, oy = o.pos[0] - carrier.pos[0], o.pos[1] - carrier.pos[1]
            t = clamp((ox * ux + oy * uy) / (ux * ux + uy * uy), 0, 1)
            lane = min(lane, vlen(ox - ux * t, oy - uy * t))
        score = min(space, 8) + min(lane, 4) + 0.3 * math.cos(math.radians(deg)) * 4
        if score > bs:
            bs, best = score, (x, y)
    return best

def fk_direct(world, sp):
    # 직접 프리킥 조건: 거리 FK_SHOT_DIST 안 + 박스 폭 안쪽
    b = world.ball
    gx, gy = world.opp_goal(sp['team'])
    return sp['kind'] == 'freekick' and vlen(gx - b.pos[0], gy - b.pos[1]) < FK_SHOT_DIST         and abs(b.pos[1] - gy) < PBOX_HALF

def setpiece_roles(world, sp):
    # 공격: 피지컬 높은 순 5명이 박스 지점, 수비 2명은 하프라인 근처 대기(역습 대비)
    # 수비: 직접 프리킥이면 공에 가까운 WALL_N명이 벽, 나머지는 마킹(assign_marks)
    team = sp['team']
    b = world.ball
    att = sorted((q for q in world.players if q.team == team and q.active and q.role != 'GK'
                  and q is not sp['taker']), key=lambda q: -q.stats.physical)
    runners = [q for q in att if q.role != 'DEF'][:3] + [q for q in att if q.role == 'DEF'][:2]
    wall = []
    if fk_direct(world, sp):
        dfd = [q for q in world.players if q.team != team and q.active and q.role != 'GK']
        wall = sorted(dfd, key=lambda q: vlen(q.pos[0] - b.pos[0], q.pos[1] - b.pos[1]))[:WALL_N]
        for q in wall:                                     # 주심이 벽을 세운다(실제 20초쯤 걸리는 정리를 생략)
            q.pos = list(wall_spot(world, sp, wall, q)); q.vel = [0.0, 0.0]
    return runners, wall

def wall_spot(world, sp, wall, q):
    # 공-우리 골대 선 위 9.3m, 옆으로 0.6m 간격
    b = world.ball
    ogx, ogy = world.own_goal(q.team)
    ux, uy = vnorm(ogx - b.pos[0], ogy - b.pos[1])
    k = wall.index(q) - (len(wall) - 1) / 2
    return b.pos[0] + ux * 9.3 - uy * 0.6 * k, b.pos[1] + uy * 9.3 + ux * 0.6 * k

def setpiece_intent(world, p):
    sp = world.setpiece
    b = world.ball
    if 'roles' not in sp:
        sp['roles'] = setpiece_roles(world, sp)
    runners, wall = sp['roles']
    if p is sp['taker']:
        return move_to(world, p, b.pos[0] + (-0.8 if p.team == 0 else 0.8), b.pos[1])
    if p.team == sp['team']:
        if p in runners and not fk_direct(world, sp):
            x, y = box_spots(world, p.team, b.pos[1])[runners.index(p)]
            return move_to(world, p, x, y, sprint=True)
        hx, hy = world.home_pos(p, b)
        return move_to(world, p, hx, hy)
    ogx, ogy = world.own_goal(p.team)
    if p in wall:                                          # 수비벽: 자리 유지(서로 밀어내는 분산은 끈다)
        x, y = wall_spot(world, sp, wall, p)
        return Intent(move=vnorm(x - p.pos[0], y - p.pos[1]) if vlen(x - p.pos[0], y - p.pos[1]) > 0.3
                      else (0.0, 0.0))
    m = world.marks.get(p.pid)
    if m is not None:
        ux, uy = vnorm(ogx - m.pos[0], ogy - m.pos[1])
        return move_to(world, p, m.pos[0] + ux * 1.0, m.pos[1] + uy * 1.0, sprint=True)
    hx, hy = world.home_pos(p, b)
    if vlen(hx - b.pos[0], hy - b.pos[1]) < 9.5:          # 9.15m 규칙
        ux, uy = vnorm(hx - b.pos[0], hy - b.pos[1])
        hx, hy = b.pos[0] + ux * 9.5, b.pos[1] + uy * 9.5
    return move_to(world, p, hx, hy)

def setpiece_ready(world):
    # 최소 대기 뒤, 박스 침투조가 모두 자기 지점 3.5m 안에 오면 준비 완료
    sp = world.setpiece
    if SETPIECE_WAIT - sp['t'] < SETPIECE_MIN or 'roles' not in sp:
        return False
    runners = sp['roles'][0] if not fk_direct(world, sp) else []
    spots = box_spots(world, sp['team'], world.ball.pos[1])
    return all(vlen(q.pos[0] - spots[i][0], q.pos[1] - spots[i][1]) < 3.5
               for i, q in enumerate(runners) if q.active)

def take_setpiece(world):
    # 키커가 찬다: 직접 프리킥(GK 반대쪽 구석 조준) 또는 박스 안 가장 덜 붙은 동료에게 크로스
    sp = world.setpiece
    world.setpiece = None
    p = sp['taker']
    if not p.active:
        return
    b = world.ball
    b.cooldown = 0.0
    world.owner = p; b.owner = p
    if fk_direct(world, sp):
        gx, gy = world.opp_goal(p.team)
        gk = next((g for g in world.players if g.role == 'GK' and g.team != p.team and g.active), None)
        side = 1 if gk is None or gk.pos[1] < gy else -1          # GK 반대쪽 구석
        aim_y = gy + side * GOAL_W * 0.38
        lofted_kick(world, p, gx, aim_y, shot_error_deg(world, p), z_end=1.6)
        register_shot(world, p, False)
        set_event(world, "FREE KICK SHOT")
        return
    cr = cross_target(world, p)
    if cr is not None:
        _, m, tx, ty = cr
    else:
        tx, ty = box_spots(world, p.team, b.pos[1])[2]
    do_cross(world, p, tx, ty)

def ai_intent(world, p):
    b = world.ball
    gx, gy = world.opp_goal(p.team)
    ogx, ogy = world.own_goal(p.team)
    my_poss = world.owner is not None and world.owner.team == p.team
    if world.setpiece is not None and p.role != 'GK':
        return setpiece_intent(world, p)

    if p.role == 'GK':
        if p.dive > 0 or p.down > 0:
            return Intent()
        sh = world.shot
        if sh is not None and sh['team'] != p.team:
            return gk_shot_intent(world, p, sh)
        ae = world.aerial
        if ae is not None and ae['team'] != p.team:        # 크로스: 골문 앞으로 떨어지면 나가서 잡는다
            pt = aerial_point(b)
            if pt is not None and abs(pt[0] - ogx) < 7 and abs(pt[1] - ogy) < 9:
                return move_to(world, p, pt[0], pt[1], sprint=True)
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
        return ai_on_ball(world, p)

    if world.owner is None and world.pass_to == p.pid:     # 나에게 오는 패스: 공 경로로 마중
        ix, iy = intercept_point(b, p)
        return move_to(world, p, ix, iy, sprint=True)

    if my_poss:                                            # 공격 지원
        spot = box_run_spot(world, p)
        if spot is not None:                               # 측면 크로스 각: 박스로 침투
            return move_to(world, p, spot[0], spot[1], sprint=True)
        if p.role == 'FWD':
            return forward_run(world, p)
        if p is support_player(world, p.team):
            sx, sy = open_spot(world, world.owner, p)
            return move_to(world, p, sx, sy, sprint=True)
        hx, hy = world.home_pos(p, b)
        return move_to(world, p, hx, hy)

    # 수비: 최근접 1명 압박/태클, 나머지 진형 유지
    mates = [q for q in world.players if q.team == p.team and q.role != 'GK' and q.active]
    nearest = min(mates, key=lambda q: vlen(b.pos[0] - q.pos[0], b.pos[1] - q.pos[1]),
                  default=None)
    if p is nearest:
        o = world.owner
        if o is not None and o.team != p.team:         # 지연: 공과 우리 골대 사이에 서서 기회를 본다
            ux, uy = vnorm(ogx - b.pos[0], ogy - b.pos[1])
            it = move_to(world, p, b.pos[0] + ux * JOCKEY_DIST, b.pos[1] + uy * JOCKEY_DIST,
                         sprint=True)
            if p.tackle_cd <= 0 and vlen(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) < TACKLE_REACH \
                    and random.random() < TACKLE_RATE * DT:
                it.action = 'tackle'
            return it
        tx, ty = intercept_point(b, p) if o is None else (b.pos[0], b.pos[1])
        return move_to(world, p, tx, ty, sprint=True)
    if world.owner is None and world.pending_pass is not None and world.pending_pass != p.team \
            and near_ball_path(b, p, INTERCEPT_RANGE + 2):
        ix, iy = intercept_point(b, p)                     # 상대 패스가 내 옆을 지나가면 끊으러 간다
        if vlen(ix - p.pos[0], iy - p.pos[1]) < INTERCEPT_RANGE:
            return move_to(world, p, ix, iy, sprint=True)
    m = world.marks.get(p.pid)
    if m is not None:                                      # 맨마킹: 상대의 골대·공 쪽 옆에 붙는다
        ux, uy = vnorm((ogx + b.pos[0]) / 2 - m.pos[0], (ogy + b.pos[1]) / 2 - m.pos[1])
        return move_to(world, p, m.pos[0] + ux * MARK_DIST, m.pos[1] + uy * MARK_DIST,
                       sprint=True)
    hx, hy = world.home_pos(p, b)
    return move_to(world, p, hx, hy)

def assign_marks(world):
    # 수비 팀 마킹 배정: 우리 골대에 가까운(위험한) 상대부터, 자기 자리에서 가장 가까운 수비가 맡는다
    o = world.owner
    att = o.team if o is not None else world.ball.last_team
    dfd = 1 - att
    ogx, ogy = world.own_goal(dfd)
    b = world.ball
    presser = min((q for q in world.players if q.team == dfd and q.role != 'GK' and q.active),
                  key=lambda q: vlen(b.pos[0] - q.pos[0], b.pos[1] - q.pos[1]), default=None)
    free = {q.pid: q for q in world.players
            if q.team == dfd and q.role != 'GK' and q.active and q is not presser}
    homes = {pid: world.home_pos(q, b) for pid, q in free.items()}
    threats = sorted((q for q in world.players if q.team == att and q.role != 'GK' and q.active
                      and q is not o and vlen(q.pos[0] - ogx, q.pos[1] - ogy) < MARK_ZONE),
                     key=lambda q: vlen(q.pos[0] - ogx, q.pos[1] - ogy))
    marks = {}
    for t in threats:
        best, bd = None, MARK_RANGE
        for pid in free:
            hx, hy = homes[pid]
            d = vlen(t.pos[0] - hx, t.pos[1] - hy)
            if d < bd:
                best, bd = pid, d
        if best is not None:
            marks[best] = t
            del free[best]
    return marks


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
    world.marks = assign_marks(world)
    if world.setpiece is not None:
        world.setpiece['t'] -= dt
        if world.setpiece['t'] <= 0 or setpiece_ready(world):
            take_setpiece(world)
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
        maybe_subs(world)
        world.kickoff(1 - world.first_kick)    # 후반은 상대 팀 킥오프(진영 교체는 단순화로 생략)
    elif world.time >= world.match_len:
        world.finished = True
        set_event(world, "FULL TIME")
        world.event_timer = 1e9                # 배너 계속 표시

def maybe_subs(world):
    # AI 감독: SUB_AFTER_MIN분 뒤, 경기 체력이 SUB_ENERGY 아래인 가장 지친 필드 선수를 교체(팀당 SUBS_MAX명)
    if not world.match_len or game_minute(world) < SUB_AFTER_MIN:
        return
    for team in (0, 1):
        if world.match['subs'][team] >= SUBS_MAX:
            continue
        tired = [p for p in world.players if p.team == team and p.active and p.role != 'GK'
                 and p.energy < SUB_ENERGY]
        if not tired:
            continue
        out = min(tired, key=lambda p: p.energy)
        new = Player(team, out.role, out.base_fx, out.base_fy, rand_stats(out.role),
                     pos=list(out.pos), pid=world.next_pid)
        world.next_pid += 1
        world.players[world.players.index(out)] = new
        world.offside_suspects.discard(out.pid)
        world.match['subs'][team] += 1
        set_event(world, "SUBSTITUTION")


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
            ("Corners", m['corners'][0], m['corners'][1]),
            ("Crosses", m['crosses'][0], m['crosses'][1]),
            ("Dribbles won", f"{m['dribbles_won'][0]}/{m['dribbles'][0]}",
             f"{m['dribbles_won'][1]}/{m['dribbles'][1]}"),
            ("Subs", m['subs'][0], m['subs'][1])]
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
    gm = game_minute(world)                    # 경기 시계(90분 배속) + 전/후반
    half = "" if not world.match_len else ("FT " if world.finished else f"{world.half}H ")
    screen.blit(font.render(f"{half}{int(gm):02d}:{int(gm * 60 % 60):02d}",
                            True, white), (SCREEN_W - MARGIN - 90, 16))
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
            f"{c.role}  STA:{int(c.stamina)} NRG:{int(c.energy * 100)}  FIN:{int(c.stats.finishing)} "
            f"PAC:{int(c.stats.pace)} PHY:{int(c.stats.physical)}",
            True, white), (MARGIN, SCREEN_H - 50))
        if charge > 0:
            pygame.draw.rect(screen, white, (MARGIN, SCREEN_H - 68, 104, 8), 1)
            pygame.draw.rect(screen, (255, 200, 60),
                             (MARGIN + 2, SCREEN_H - 66, int(charge * 100), 4))
    if world.event_timer > 0:                   # 이벤트 배너(Tab 패널이 켜져 있으면 그 아래)
        surf = bigfont.render(world.event_text, True, (255, 240, 120))
        by = SCREEN_H - 120 if show_stats else SCREEN_H // 2 - 18
        screen.blit(surf, (SCREEN_W // 2 - surf.get_width() // 2, by))


# ---------------- 메인 ----------------
def game_loop(match_len=600):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Top-Down Soccer Prototype")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,arial", 16)
    bigfont = pygame.font.SysFont("consolas,arial", 30, bold=True)

    world = World(match_len)
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
                elif e.key == pygame.K_r:              # 리셋(경기가 끝났으면 새 경기)
                    world = World(match_len) if world.finished else world
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


def simulate(steps, seed=None, match_len=None):
    # 헤드리스 자동 경기. seed를 주면 같은 결과가 재현된다(테스트용)
    # match_len(초)을 주면 90분 경기(배속)로 돌고 종료 휘슬에서 멈춘다
    if seed is not None:
        random.seed(seed)
    w = World(match_len)
    for _ in range(steps):
        if w.finished:
            break
        step(w, DT, human_keys=None, human_mode=False)
    return w

def run_test(steps, seeds=(None,), match_len=None):
    for seed in seeds:
        t0 = time.perf_counter()
        w = simulate(steps, seed, match_len)
        ms = (time.perf_counter() - t0) / max(1, round(w.time / DT)) * 1000
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
    ap.add_argument('--length', type=float, default=600,
                    help="경기 길이(실제 초). 90분을 이 시간으로 압축. 0이면 끝없이(기본 600초 = ×9배속)")
    ap.add_argument('--match', action='store_true',
                    help="헤드리스: 스텝 수 대신 --length 경기 하나를 종료 휘슬까지")
    args = ap.parse_args()
    length = args.length or None
    if args.test or args.match:
        seeds = [int(s) for s in args.seed.split(',')] if args.seed else [None]
        if args.match:
            run_test(int((length or 600) / DT) + 1, seeds, length or 600)
        else:
            run_test(args.test, seeds)
    else:
        game_loop(length)


if __name__ == "__main__":
    main()

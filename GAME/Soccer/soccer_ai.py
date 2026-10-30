"""
축구 AI (v4.2) — 선수의 판단과 움직임
=====================================
매 프레임 선수마다 Intent(이동·스프린트·액션·대상)를 만든다. 엔진(soccer_engine.py)은 그 Intent를
스탯 + 물리 + 확률로 해석할 뿐이고, 사람 입력도 같은 Intent를 만든다(한 엔진, 두 모드).

    판단:  shot_xg(기대득점) · pass_options/lane_risk(패스 길목·기대값) · ai_on_ball(슛/크로스/돌파/패스/걷어내기/드리블)
    움직임: ai_intent(수비 라인·마킹·지연·패스 끊기·침투·지원·박스 침투·GK) · assign_marks
    세트피스: setpiece_roles/intent/ready · take_setpiece
    감독:  maybe_subs(교체 정책) · maybe_tactics(전술 — 포메이션 바꾸기)

AI 전용 상수는 이 파일 위쪽에 있다(튜닝은 soccer_ai.X를 바꾼다). 엔진 상수는 soccer_engine.
"""
import math
import random

from soccer_engine import *                # 엔진 상태·상수·물리/액션 함수


# --- v4: AI 판단 상수 (값의 근거는 READ_ME 4절) ---
PASS_MIN, PASS_MAX = 4.0, 40.0   # 패스 후보 거리(m)
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
CLEAR_ZONE = 0.35           # 자기 골대 쪽 35% 안에서 압박받고 패스 길이 없으면 걷어낸다
SETPIECE_MIN = 1.5          # 최소 대기(초)
FK_SHOT_DIST = 28.0         # 이 안 + 중앙 쪽이면 직접 슛, 아니면 박스로 크로스
WALL_N = 3                  # 직접 프리킥 수비벽 인원
CROSS_ZONE_X = 0.72         # 상대 골라인 쪽 28% 안 + 측면(박스 폭 밖)이면 크로스 지역

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

def maybe_subs(world):
    # AI 감독: 팀별 정책(sub_policy)으로 1명씩. 경기 체력이 낮은 선수 → 없으면 경고 누적 위험(수비·미드)
    if not world.match_len:
        return
    minute = game_minute(world)
    for team in (0, 1):
        pol = world.sub_policy[team]
        if not pol['auto'] or minute < pol['after_min'] or subs_left(world, team) <= 0:
            continue
        on = [p for p in world.players if p.team == team and p.active and p.role != 'GK'
              and not any(q[1] == p.pid for q in world.sub_queue)]
        tired = [p for p in on if p.energy < pol['energy']]
        risky = [p for p in on if pol['yellow_risk'] and minute >= 60
                 and world.cards.get(p.pid, 0) >= 1 and p.role in ('DEF', 'MID')]
        out = min(tired, key=lambda p: p.energy) if tired else (risky[0] if risky else None)
        if out is None:
            continue
        inp = pick_bench(world, team, out.role)
        if inp is not None and request_sub(world, team, out.pid, world.bench[team].index(inp))[0]:
            apply_subs(world)

def pick_bench(world, team, role):
    # 같은 포지션 벤치 우선, 없으면 필드 선수(GK 자리는 GK만)
    bench = world.bench[team]
    same = [p for p in bench if p.pos_role == role]
    if same or role == 'GK':
        return same[0] if same else None
    return next((p for p in bench if p.pos_role != 'GK'), None)
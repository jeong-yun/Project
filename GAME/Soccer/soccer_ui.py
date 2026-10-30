"""
화면(UI) — 탑다운 렌더 · 선발 선택 화면 · 선수 카드 · 교체 패널 · 메인 루프 (v4.1)
=========================================================================
엔진(soccer.py)의 상태를 그리기만 하고, 사람 입력은 엔진 함수(Intent·do_pass·request_sub·
swap_lineup·switch_control)로 넘긴다. 엔진은 이 파일 없이도(헤드리스·pytest) 돈다.

실행은 그대로:  python soccer.py   (soccer.main()이 이 파일의 game_loop를 부른다)
"""
import math

import pygame

from soccer import *                       # 상수·World·엔진 함수(이 파일은 그리기·입력만)

WHITE = (235, 235, 235)
YELLOW = (255, 240, 120)
GREY = (150, 150, 150)
STAT_COLS = (('PAC', 'pace'), ('ACC', 'accel'), ('AGI', 'agility'), ('FIN', 'finishing'),
             ('PAS', 'passing'), ('PHY', 'physical'), ('STA', 'stamina'), ('KEE', 'keeping'))


# ---------------- 경기장 ----------------
def w2s(x, y):
    return (int(MARGIN + x * PPM), int(MARGIN + y * PPM))

def panel(screen, x, y, w, h, alpha=190):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((0, 0, 0, alpha))
    screen.blit(s, (x, y))

def draw_stats(screen, font, world):
    # 경기 기록 패널(Tab) + 교체 기록
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
    log = world.sub_log[-4:]
    w, h = 360, 30 + 22 * len(rows) + 20 * len(log)
    x0, y0 = SCREEN_W // 2 - w // 2, 80
    panel(screen, x0, y0, w, h, 170)
    for i, (name, a, b) in enumerate(rows):
        y = y0 + 15 + i * 22
        screen.blit(font.render(str(a), True, TEAM_COLORS[0]), (x0 + 20, y))
        surf = font.render(name, True, WHITE)
        screen.blit(surf, (x0 + w // 2 - surf.get_width() // 2, y))
        sb = font.render(str(b), True, TEAM_COLORS[1])
        screen.blit(sb, (x0 + w - 20 - sb.get_width(), y))
    for i, (mnt, team, out, inn) in enumerate(log):   # 최근 교체 4건
        y = y0 + 15 + len(rows) * 22 + i * 20
        screen.blit(font.render(f"{mnt:3.0f}'  {out} -> {inn}", True, TEAM_COLORS[team]),
                    (x0 + 20, y))

def draw(screen, font, bigfont, world, human_mode, sim_speed, charge, show_stats=False,
         small=None):
    screen.fill((22, 130, 55))
    for i in range(0, int(L), 6):
        if (i // 6) % 2 == 0:
            x, _ = w2s(i, 0)
            pygame.draw.rect(screen, (26, 140, 60), (x, MARGIN, int(6 * PPM), int(W * PPM)))

    def line(x1, y1, x2, y2, wd=2):
        pygame.draw.line(screen, WHITE, w2s(x1, y1), w2s(x2, y2), wd)

    pygame.draw.rect(screen, WHITE, (*w2s(0, 0), int(L * PPM), int(W * PPM)), 2)
    line(L / 2, 0, L / 2, W)
    pygame.draw.circle(screen, WHITE, w2s(L / 2, W / 2), int(9.15 * PPM), 2)
    pygame.draw.circle(screen, WHITE, w2s(L / 2, W / 2), 3)
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
        pygame.draw.circle(screen, WHITE, w2s(gx + sgn * PEN_SPOT, W / 2), 3)  # 페널티 스폿
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
            pygame.draw.circle(screen, (255, 240, 90), (px, py), 12,
                               4 if world.human_pick == p.pid else 2)   # 굵으면 '고정'
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
        if small is not None and p.number:      # 등번호
            t = small.render(str(p.number), True, WHITE)
            screen.blit(t, (px - t.get_width() // 2, py + 9))

    n0 = sum(1 for p in world.players if p.team == 0 and p.active)
    n1 = sum(1 for p in world.players if p.team == 1 and p.active)
    screen.blit(bigfont.render(f"{world.score[0]} : {world.score[1]}", True, WHITE),
                (SCREEN_W // 2 - 30, 12))
    mode = "HUMAN PLAY (M:switch)" if human_mode else f"AUTO SIM x{sim_speed} (M:switch)"
    screen.blit(font.render(mode, True, WHITE), (MARGIN, 14))
    screen.blit(font.render(f"{n0}v{n1}", True, WHITE), (SCREEN_W // 2 - 14, 44))
    gm = game_minute(world)                    # 경기 시계(90분 배속) + 전/후반
    half = "" if not world.match_len else ("FT " if world.finished else f"{world.half}H ")
    screen.blit(font.render(f"{half}{int(gm):02d}:{int(gm * 60 % 60):02d}",
                            True, WHITE), (SCREEN_W - MARGIN - 90, 16))
    screen.blit(font.render(
        "WASD:Move Shift:Sprint Space:Pass K:Shoot J:Tackle Q:Switch L:Auto I:Info "
        "P:Subs Tab:Stats R:Reset", True, WHITE), (MARGIN, SCREEN_H - 28))
    if world.advantage is not None:             # 어드밴티지 진행 표시
        screen.blit(font.render(f"ADVANTAGE {world.advantage['t']:.1f}s", True,
                                YELLOW), (SCREEN_W // 2 + 60, 44))
    pend = [q for q in world.sub_queue if q[0] == HUMAN_TEAM]
    if pend:                                    # 다음 멈춤에서 적용될 교체
        screen.blit(font.render(f"SUB PENDING x{len(pend)}", True, YELLOW),
                    (SCREEN_W - MARGIN - 160, 44))
    if show_stats:
        draw_stats(screen, font, world)
    if human_mode and world.controlled:
        c = world.controlled
        lock = " [LOCK]" if world.human_pick == c.pid else ""
        screen.blit(font.render(
            f"#{c.number} {c.name} {c.role}{lock}  STA:{int(c.stamina)} NRG:{int(c.energy * 100)}"
            f"  FIN:{int(c.stats.finishing)} PAC:{int(c.stats.pace)} PHY:{int(c.stats.physical)}",
            True, WHITE), (MARGIN, SCREEN_H - 50))
        if charge > 0:
            pygame.draw.rect(screen, WHITE, (MARGIN, SCREEN_H - 68, 104, 8), 1)
            pygame.draw.rect(screen, (255, 200, 60),
                             (MARGIN + 2, SCREEN_H - 66, int(charge * 100), 4))
    if world.event_timer > 0:                   # 이벤트 배너(Tab 패널이 켜져 있으면 그 아래)
        surf = bigfont.render(world.event_text, True, YELLOW)
        by = SCREEN_H - 120 if show_stats else SCREEN_H // 2 - 18
        screen.blit(surf, (SCREEN_W // 2 - surf.get_width() // 2, by))


# ---------------- 선수 카드(I · 클릭) ----------------
def draw_card(screen, font, world, p, x0, y0):
    info = player_info(world, p)
    st, m = info['stats'], info['match']
    rows = [(f"#{info['number']} {info['name']}", TEAM_COLORS[p.team]),
            (f"{info['team']}  {info['position']}"
             + (f" (playing {info['slot']})" if info['slot'] != info['position'] else "")
             + f"  [{info['where']}]", WHITE),
            ("  ".join(f"{a} {st[k]}" for a, k in STAT_COLS[:4]), WHITE),
            ("  ".join(f"{a} {st[k]}" for a, k in STAT_COLS[4:]), WHITE),
            (f"Stamina {info['stamina']}   Energy {info['energy']}%", WHITE),
            (f"Passes {m['passes_ok']}/{m['passes']}   Shots {m['shots']}   Goals {m['goals']}",
             WHITE),
            (f"Tackles {m['tackles_won']}/{m['tackles']}   Fouls {m['fouls']}", WHITE),
            ("Cards: " + ("RED" if info['red'] else "Y" * info['yellow'] or "-"),
             (255, 90, 90) if info['red'] else (240, 220, 40) if info['yellow'] else WHITE)]
    w, h = 330, 16 + 22 * len(rows)
    panel(screen, x0, y0, w, h, 210)
    pygame.draw.rect(screen, TEAM_COLORS[p.team], (x0, y0, w, h), 2)
    for i, (text, col) in enumerate(rows):
        screen.blit(font.render(text, True, col), (x0 + 12, y0 + 8 + i * 22))

def card_subject(world, human_mode, info_pid):
    # 카드에 보일 선수: 클릭한 선수 → (사람 모드) 조작 선수 → 공 가진 선수
    p = find_player(world, info_pid) if info_pid is not None else None
    if p is None and info_pid is not None:
        p = next((q for t in (0, 1) for q in world.bench[t] if q.pid == info_pid), None)
    if p is None:
        p = world.controlled if human_mode and world.controlled else world.owner
    return p

def click_player(world, pos):
    best, bd = None, 14
    for p in world.players:
        if p.active:
            d = math.hypot(w2s(*p.pos)[0] - pos[0], w2s(*p.pos)[1] - pos[1])
            if d < bd:
                best, bd = p, d
    return best


# ---------------- 선발 선택 화면 ----------------
def squad_rows(world, team):
    on = [p for p in world.players if p.team == team]
    return on + world.bench[team], len(on)

def draw_squad(screen, font, bigfont, world, team, cur, mark, msg, show_card):
    screen.fill((18, 60, 40))
    rows, n_on = squad_rows(world, team)
    title = f"SQUAD - {world.team_names[team]}" + (" (you)" if team == HUMAN_TEAM else " (CPU)")
    screen.blit(bigfont.render(title, True, TEAM_COLORS[team]), (40, 20))
    head = f"{'#':>3} {'Name':10} {'Pos':4}" + "".join(f"{a:>5}" for a, _ in STAT_COLS)
    screen.blit(font.render(head, True, GREY), (40, 70))
    y = 94
    for i, p in enumerate(rows):
        if i == n_on:
            screen.blit(font.render("-- BENCH --", True, GREY), (40, y)); y += 22
        sel = i == cur
        if sel:
            pygame.draw.rect(screen, (60, 100, 70), (34, y - 2, 620, 21))
        if mark is not None and rows[mark] is p:
            pygame.draw.rect(screen, YELLOW, (34, y - 2, 620, 21), 2)
        star = "*" if world.human_pick == p.pid else " "
        line = f"{p.number:>3} {p.name:10} {p.pos_role:4}" + \
            "".join(f"{int(getattr(p.stats, k)):>5}" for _, k in STAT_COLS) + f" {star}"
        screen.blit(font.render(line, True, WHITE if i < n_on else (200, 200, 200)), (40, y))
        y += 22
    help1 = "Up/Down: move   Tab: other team   Enter x2: swap starter <-> bench (same position)"
    help2 = "C: control this player (your team)   I: player card   Space: KICK OFF   Esc: quit"
    screen.blit(font.render(help1, True, WHITE), (40, SCREEN_H - 60))
    screen.blit(font.render(help2, True, WHITE), (40, SCREEN_H - 36))
    if msg:
        screen.blit(font.render(msg, True, YELLOW), (40, SCREEN_H - 90))
    if show_card and rows:
        draw_card(screen, font, world, rows[cur], SCREEN_W - 360, 94)

def select_screen(screen, clock, font, bigfont, world, auto_keys=None):
    # 킥오프 전 선발 선택. 엔진 함수(swap_lineup, human_pick)만 바꾼다. True = 킥오프, False = 종료
    team, cur, mark, msg, show_card = HUMAN_TEAM, 0, None, "", True
    keys_iter = iter(auto_keys) if auto_keys is not None else None
    while True:
        events = pygame.event.get() if keys_iter is None else \
            [pygame.event.Event(pygame.KEYDOWN, key=k) for k in [next(keys_iter, pygame.K_SPACE)]]
        rows, n_on = squad_rows(world, team)
        for e in events:
            if e.type == pygame.QUIT:
                return False
            if e.type != pygame.KEYDOWN:
                continue
            if e.key == pygame.K_ESCAPE:
                return False
            if e.key == pygame.K_SPACE:
                return True
            if e.key in (pygame.K_UP, pygame.K_w):
                cur = (cur - 1) % len(rows)
            elif e.key in (pygame.K_DOWN, pygame.K_s):
                cur = (cur + 1) % len(rows)
            elif e.key == pygame.K_TAB:
                team, cur, mark, msg = 1 - team, 0, None, ""
            elif e.key == pygame.K_i:
                show_card = not show_card
            elif e.key == pygame.K_c:
                p = rows[cur]
                if team != HUMAN_TEAM or cur >= n_on or p.role == 'GK':
                    msg = "C: pick an outfield starter of your team"
                else:
                    world.human_pick = p.pid
                    msg = f"you control #{p.number} {p.name}"
            elif e.key == pygame.K_RETURN:
                if mark is None:
                    mark, msg = cur, f"selected #{rows[cur].number} - now pick the player to swap"
                else:
                    a, b = sorted((mark, cur))
                    if a < n_on <= b:
                        ok, why = swap_lineup(world, team, rows[a].pid, b - n_on)
                        msg = "swapped" if ok else why
                    else:
                        msg = "pick one starter and one bench player"
                    mark = None
        draw_squad(screen, font, bigfont, world, team, cur, mark, msg, show_card)
        pygame.display.flip()
        clock.tick(30)


# ---------------- 교체 패널(P) ----------------
def draw_subs(screen, font, bigfont, world, stage, cur, chosen_out, msg):
    team = HUMAN_TEAM
    on = [p for p in world.players if p.team == team and p.active]
    bench = world.bench[team]
    x0, y0, w = 120, 70, SCREEN_W - 240
    panel(screen, x0, y0, w, SCREEN_H - 140, 225)
    screen.blit(bigfont.render(f"SUBSTITUTIONS  (left: {subs_left(world, team)})", True, YELLOW),
                (x0 + 20, y0 + 12))
    for col, (title, lst, active) in enumerate((("ON PITCH - who goes off", on, stage == 'out'),
                                                ("BENCH - who comes on", bench, stage == 'in'))):
        x = x0 + 20 + col * (w // 2)
        screen.blit(font.render(title, True, WHITE if active else GREY), (x, y0 + 60))
        for i, p in enumerate(lst):
            y = y0 + 88 + i * 22
            if active and i == cur:
                pygame.draw.rect(screen, (60, 100, 70), (x - 6, y - 2, w // 2 - 30, 21))
            if p is chosen_out:
                pygame.draw.rect(screen, YELLOW, (x - 6, y - 2, w // 2 - 30, 21), 2)
            queued = any(q[1] == p.pid or q[2] == p.pid for q in world.sub_queue)
            card = " Y" * world.cards.get(p.pid, 0)
            txt = (f"#{p.number:<3}{p.name:10}{p.role if col == 0 else p.pos_role:4}"
                   f" NRG {int(p.energy * 100):3}%{card}{'  (queued)' if queued else ''}")
            screen.blit(font.render(txt, True, WHITE if active else (190, 190, 190)), (x, y))
    sub_off = [p.name for p in world.subbed_off if p.team == team]
    reds = [p.name for p in world.players if p.team == team and p.pid in world.reds]
    foot = [msg,
            "Up/Down: move   Enter: choose   Backspace: cancel queued   P/Esc: close & resume",
            "Applied at the next stoppage (out of play, foul, goal, half time). "
            + (f"Off: {', '.join(sub_off)}" if sub_off else "")
            + (f"  Sent off: {', '.join(reds)}" if reds else "")]
    for i, t in enumerate(foot):
        screen.blit(font.render(t, True, YELLOW if i == 0 else WHITE),
                    (x0 + 20, SCREEN_H - 150 + i * 24))

def sub_panel_key(world, key, st):
    # 교체 패널 입력 → 엔진 request_sub. st = {'stage','cur','out','msg'}; True면 패널 닫기
    team = HUMAN_TEAM
    on = [p for p in world.players if p.team == team and p.active]
    lst = on if st['stage'] == 'out' else world.bench[team]
    if key in (pygame.K_p, pygame.K_ESCAPE):
        return True
    if key == pygame.K_BACKSPACE:
        world.sub_queue = [q for q in world.sub_queue if q[0] != team]
        st.update(stage='out', cur=0, out=None, msg="queued subs cancelled")
    elif key in (pygame.K_UP, pygame.K_w) and lst:
        st['cur'] = (st['cur'] - 1) % len(lst)
    elif key in (pygame.K_DOWN, pygame.K_s) and lst:
        st['cur'] = (st['cur'] + 1) % len(lst)
    elif key == pygame.K_RETURN and lst:
        if st['stage'] == 'out':
            st.update(stage='in', out=lst[st['cur']], cur=0,
                      msg=f"#{lst[st['cur']].number} {lst[st['cur']].name} off - who comes on?")
        else:
            ok, why = request_sub(world, team, st['out'].pid, st['cur'])
            inn = lst[st['cur']]
            st.update(stage='out', cur=0, out=None,
                      msg=(f"queued: {inn.name} on (next stoppage)" if ok else why))
    return False


# ---------------- 메인 ----------------
def game_loop(match_len=600, squad=None, select=True, script=None):
    # script: 스크린샷용 자동 입력(None이면 사람이 조작)
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Top-Down Soccer Prototype")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,arial", 16)
    bigfont = pygame.font.SysFont("consolas,arial", 30, bold=True)
    small = pygame.font.SysFont("consolas,arial", 11)

    def new_world():
        w = World(match_len, squad)
        w.sub_policy[HUMAN_TEAM]['auto'] = False          # 사람 팀 교체는 사람이(M으로 자동 모드면 AI)
        return w

    world = new_world()
    if select and not select_screen(screen, clock, font, bigfont, world):
        pygame.quit(); return
    human_mode = True
    sim_speed = 1
    charging = False
    charge = 0.0
    show_stats = False
    show_card = False
    info_pid = None
    subs_open = False
    sub_st = {'stage': 'out', 'cur': 0, 'out': None, 'msg': ''}
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                p = click_player(world, e.pos)                 # 클릭 = 그 선수 카드
                if p is not None:
                    info_pid, show_card = p.pid, True
            elif e.type == pygame.KEYDOWN and subs_open:
                if sub_panel_key(world, e.key, sub_st):
                    subs_open = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_m:
                    human_mode = not human_mode
                    world.sub_policy[HUMAN_TEAM]['auto'] = not human_mode
                elif e.key == pygame.K_r:              # 리셋(경기가 끝났으면 새 경기)
                    if world.finished:
                        world = new_world()
                    world.kickoff()
                elif e.key == pygame.K_TAB:
                    show_stats = not show_stats
                elif e.key == pygame.K_i:
                    show_card = not show_card
                    if not show_card:
                        info_pid = None
                elif e.key == pygame.K_p:              # 일시정지 + 교체 패널
                    subs_open = True
                    sub_st = {'stage': 'out', 'cur': 0, 'out': None, 'msg': ''}
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    sim_speed = min(8, sim_speed + 1)
                elif e.key == pygame.K_MINUS:
                    sim_speed = max(1, sim_speed - 1)
                elif e.key == pygame.K_q and human_mode:
                    switch_control(world)
                elif e.key == pygame.K_l and human_mode:
                    world.human_pick = None
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
        if not subs_open:                              # 교체 패널이 열려 있으면 일시정지
            for _ in range(1 if human_mode else sim_speed):
                step(world, DT, human_keys=keys if human_mode else None, human_mode=human_mode)
        draw(screen, font, bigfont, world, human_mode, sim_speed, charge if charging else 0,
             show_stats, small)
        if show_card:
            p = card_subject(world, human_mode, info_pid)
            if p is not None:
                draw_card(screen, font, world, p, SCREEN_W - MARGIN - 340, 70)
        if subs_open:
            draw_subs(screen, font, bigfont, world, sub_st['stage'], sub_st['cur'],
                      sub_st['out'], sub_st['msg'])
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()

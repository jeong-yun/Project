"""
탑다운 축구 프로토타입 (Pygame) v4.2 — 실행 진입점
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
# v4.2 파일 구성:  soccer_engine.py(엔진) · soccer_ai.py(AI) · soccer_ui.py(화면) · soccer.py(이 파일 = 진입점)
# 테스트와 도구는 예전처럼 `import soccer`로 엔진·AI 함수를 모두 쓴다.
import sys, random, argparse, time, json

from soccer_engine import *                # noqa: F401,F403 — 엔진
from soccer_ai import *                    # noqa: F401,F403 — AI


# ---------------- 렌더링 · 메인 루프 ----------------
# v4.1: 화면 코드는 soccer_ui.py로 옮겼다(엔진은 화면 없이 돈다). main()이 불러 쓴다.


def simulate(steps, seed=None, match_len=None, squad=None, formations=None):
    # 헤드리스 자동 경기. seed를 주면 같은 결과가 재현된다(테스트용)
    # match_len(초)을 주면 90분 경기(배속)로 돌고 종료 휘슬에서 멈춘다
    if seed is not None:
        random.seed(seed)
    w = World(match_len, squad, formations)
    for _ in range(steps):
        if w.finished:
            break
        step(w, DT, human_keys=None, human_mode=False)
    return w

def run_test(steps, seeds=(None,), match_len=None, squad=None, formations=None):
    for seed in seeds:
        t0 = time.perf_counter()
        w = simulate(steps, seed, match_len, squad, formations)
        ms = (time.perf_counter() - t0) / max(1, round(w.time / DT)) * 1000
        n0 = sum(1 for p in w.players if p.team == 0 and p.active)
        n1 = sum(1 for p in w.players if p.team == 1 and p.active)
        print(f"TEST OK  seed={seed}  score={w.score}  players={n0}v{n1}  "
              f"events={dict(sorted(w.events.items()))}  time={w.time:.0f}s  "
              f"step={ms:.2f}ms")
        print(f"         {match_summary(w)}")
        print(f"         formation {w.formation[0]} vs {w.formation[1]}"
              + "".join(f"  ({m:.0f}' {w.team_names[t]} -> {f})" for m, t, f in w.formation_log))
        for mnt, team, out, inn in w.sub_log:
            print(f"         SUB {mnt:4.0f}'  {w.team_names[team]}: {out} -> {inn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', type=int, default=0, help="헤드리스 시뮬 스텝 수")
    ap.add_argument('--seed', type=str, default=None,
                    help="난수 시드. 쉼표로 여러 개(예: 1,2,3) → 시드마다 한 경기")
    ap.add_argument('--length', type=float, default=600,
                    help="경기 길이(실제 초). 90분을 이 시간으로 압축. 0이면 끝없이(기본 600초 = ×9배속)")
    ap.add_argument('--match', action='store_true',
                    help="헤드리스: 스텝 수 대신 --length 경기 하나를 종료 휘슬까지")
    ap.add_argument('--squad', type=str, default=None,
                    help="스쿼드·선발 JSON 파일(팀별 선수 18명 + lineup 등번호 11개)")
    ap.add_argument('--export-squad', type=str, default=None,
                    help="지금 시드의 자동 스쿼드를 JSON으로 저장하고 끝(고쳐서 --squad로)")
    ap.add_argument('--no-select', action='store_true', help="GUI: 선발 선택 화면 건너뛰기")
    ap.add_argument('--formation', type=str, default=None,
                    help="포메이션 '0팀,1팀'(예: 4-3-3,4-4-2) 또는 하나(두 팀 같게). "
                         "고를 수 있는 것: " + ", ".join(FORMATION_NAMES))
    args = ap.parse_args()
    length = args.length or None
    squad = load_squad(args.squad) if args.squad else None
    forms = None
    if args.formation:
        forms = args.formation.split(',')
        forms = forms * 2 if len(forms) == 1 else forms
    if args.export_squad:
        if args.seed:
            random.seed(int(args.seed.split(',')[0]))
        with open(args.export_squad, 'w', encoding='utf-8') as fp:
            json.dump(squad_dict(World(length, squad, forms)), fp, ensure_ascii=False, indent=1)
        print("saved", args.export_squad)
    elif args.test or args.match:
        seeds = [int(s) for s in args.seed.split(',')] if args.seed else [None]
        if args.match:
            run_test(int((length or 600) / DT) + 1, seeds, length or 600, squad, forms)
        else:
            run_test(args.test, seeds, squad=squad, formations=forms)
    else:
        import soccer_ui                      # 화면은 soccer_ui.py(엔진은 화면 없이 돈다)
        soccer_ui.game_loop(length, squad, select=not args.no_select, formations=forms)


if __name__ == "__main__":
    main()
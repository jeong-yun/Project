#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVC 데이터셋 전처리 파이프라인
  mp3/wav/flac/m4a → (0) 의존성 확인·자동 설치
                   → (1) wav 변환 (ffmpeg)
                   → (2) 보컬 분리 (audio-separator, BS-RoFormer/MDX-Net)
                   → (3) 무음 기준 슬라이싱 (5~15초 클립)
                   → dataset_raw/<가수명>/*.wav

사용법:
  python preprocess.py --input songs폴더 --singer 가수명
  python preprocess.py --input songs폴더 --singer 가수명 --fast   # CPU 권장(빠른 모델)
  python preprocess.py                                            # 인자 없으면 대화형 모드

결과물 dataset_raw/ 폴더를 zip으로 압축해 Colab 노트북(svc_colab_training.ipynb)에서 학습하세요.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

# ---------------------------------------------------------------
# 0) 의존성 확인 및 자동 설치
# ---------------------------------------------------------------
PACKAGES = {           # import 모듈명 : pip 패키지명
    "numpy": "numpy",
    "soundfile": "soundfile",
    "audio_separator": "audio-separator",   # torch/onnxruntime 포함, 첫 설치 시 수백 MB
}

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # 품질 우선 (GPU 권장, CPU면 느림)
FAST_MODEL = "UVR-MDX-NET-Voc_FT.onnx"                       # 속도 우선 (CPU 권장)


def pip_install(pkg: str) -> None:
    print(f"  [설치] {pkg} 설치 중... (인터넷 필요, 몇 분 걸릴 수 있음)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg])


def ensure_packages() -> None:
    """미설치 패키지를 찾아 pip으로 설치한다."""
    if getattr(sys, "frozen", False):
        # PyInstaller exe로 실행 중이면 의존성이 이미 번들되어 있음
        return
    import importlib.util
    for mod, pip_name in PACKAGES.items():
        if importlib.util.find_spec(mod) is None:
            pip_install(pip_name)
        else:
            print(f"  [확인] {pip_name} 설치됨")


def ensure_ffmpeg() -> str:
    """ffmpeg 경로 반환. 없으면 imageio-ffmpeg(정적 바이너리 포함)로 자동 설치."""
    path = shutil.which("ffmpeg")
    if path:
        print(f"  [확인] ffmpeg: {path}")
        return path
    print("  [설치] ffmpeg 미설치 → imageio-ffmpeg 로 자동 설치합니다...")
    try:
        import imageio_ffmpeg
    except ImportError:
        pip_install("imageio-ffmpeg")
        import imageio_ffmpeg
    path = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"  [확인] ffmpeg: {path}")
    return path


# ---------------------------------------------------------------
# 1) wav 변환
# ---------------------------------------------------------------
def convert_to_wav(ffmpeg: str, src: str, dst: str) -> None:
    r = subprocess.run(
        [ffmpeg, "-y", "-i", src, "-ar", "44100", "-ac", "2", dst],
        capture_output=True,
    )
    if r.returncode != 0:
        tail = r.stderr.decode(errors="ignore")[-500:]
        raise RuntimeError(f"ffmpeg 변환 실패: {src}\n{tail}")


# ---------------------------------------------------------------
# 2) 보컬 분리
# ---------------------------------------------------------------
_separator = None


def separate_vocals(wav: str, out_dir: str, model: str) -> str:
    """보컬 트랙 wav 경로를 반환. 모델은 첫 호출 때 한 번만 로드(자동 다운로드)."""
    global _separator
    if _separator is None:
        from audio_separator.separator import Separator
        _separator = Separator(output_dir=out_dir)
        print(f"  [모델] {model} 로딩 중... (첫 실행 시 모델 다운로드)")
        _separator.load_model(model_filename=model)
    outputs = _separator.separate(wav)
    for f in outputs:
        p = f if os.path.isabs(f) else os.path.join(out_dir, f)
        if "(Vocals)" in os.path.basename(p):
            return p
    raise RuntimeError(f"보컬 트랙을 찾지 못했습니다: {outputs}")


# ---------------------------------------------------------------
# 3) 무음 기준 슬라이싱 (openvpi audio-slicer 알고리즘 간략판)
# ---------------------------------------------------------------
def slice_wav(src: str, dst_dir: str, base: str,
              threshold_db: float = -40.0,
              min_interval: float = 0.3,   # 이보다 짧은 무음은 이어붙임
              min_keep: float = 2.0,       # 이보다 짧은 클립은 버림
              max_len: float = 15.0):      # 이보다 길면 분할
    """반환: (클립 수, 총 길이(초))"""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(src, always_2d=True)
    mono = audio.mean(axis=1)

    hop = int(sr * 0.010)                      # 10ms 프레임
    n = len(mono) // hop
    if n == 0:
        return 0, 0.0
    frames = mono[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > threshold_db

    # 유성 구간 [시작프레임, 끝프레임) 목록
    segs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append([start, i])
            start = None
    if start is not None:
        segs.append([start, n])

    # min_interval보다 짧은 무음으로 나뉜 구간은 병합
    gap = max(1, int(min_interval / 0.010))
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    # 저장 (앞뒤 0.1초 여유, max_len 초과 시 분할, min_keep 미만 버림)
    count, total_sec = 0, 0.0
    pad = int(0.1 / 0.010)
    chunk = int(max_len * sr)
    for s0, s1 in merged:
        a = max(0, (s0 - pad) * hop)
        b = min(len(mono), (s1 + pad) * hop)
        pos = a
        while pos < b:
            end = min(pos + chunk, b)
            dur = (end - pos) / sr
            if dur >= min_keep:
                out = os.path.join(dst_dir, f"{base}_{count:04d}.wav")
                sf.write(out, mono[pos:end], sr, subtype="PCM_16")
                count += 1
                total_sec += dur
            pos = end
    return count, total_sec


# ---------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print("  SVC 데이터셋 전처리 파이프라인")
    print("=" * 60)

    ap = argparse.ArgumentParser(description="SVC 데이터셋 전처리")
    ap.add_argument("--input", "-i", help="음원(mp3/wav/flac/m4a) 폴더")
    ap.add_argument("--singer", "-s", help="가수 이름 (dataset_raw 하위 폴더명, 영문 권장)")
    ap.add_argument("--fast", action="store_true", help="빠른 모델(MDX-Net) 사용 — GPU 없으면 권장")
    ap.add_argument("--out", "-o", default=".", help="결과 상위 폴더 (기본: 현재 폴더)")
    args = ap.parse_args()

    interactive = args.input is None
    if interactive:
        args.input = input("음원 폴더 경로: ").strip().strip('"')
        args.singer = input("가수 이름(영문 권장): ").strip() or "singer"
        args.fast = input("빠른 모델 사용? (GPU 없으면 y 권장) [y/N]: ").strip().lower() == "y"
    if not args.singer:
        args.singer = "singer"

    files = []
    for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a"):
        files += glob.glob(os.path.join(args.input, ext))
    files.sort()
    if not files:
        sys.exit(f"[오류] '{args.input}' 에서 음원 파일을 찾지 못했습니다.")
    print(f"\n[입력] {len(files)}개 파일 발견")

    print("\n[0단계] 의존성 확인")
    ensure_packages()
    ffmpeg = ensure_ffmpeg()

    work = os.path.join(args.out, "_work")
    wav_dir = os.path.join(work, "wav")
    sep_dir = os.path.join(work, "separated")
    dst_dir = os.path.join(args.out, "dataset_raw", args.singer)
    for d in (wav_dir, sep_dir, dst_dir):
        os.makedirs(d, exist_ok=True)

    model = FAST_MODEL if args.fast else DEFAULT_MODEL
    total_clips, total_sec = 0, 0.0

    for idx, f in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(f))[0]
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in base)
        print(f"\n[{idx}/{len(files)}] {base}")

        wav = os.path.join(wav_dir, safe + ".wav")
        print("  (1/3) wav 변환...")
        convert_to_wav(ffmpeg, f, wav)

        print("  (2/3) 보컬 분리... (CPU는 곡당 수 분~수십 분 소요)")
        vocals = separate_vocals(wav, sep_dir, model)

        print("  (3/3) 무음 기준 슬라이싱...")
        made, sec = slice_wav(vocals, dst_dir, safe)
        total_clips += made
        total_sec += sec
        print(f"      → 클립 {made}개 ({sec/60:.1f}분)")

    print("\n" + "=" * 60)
    print(f"[완료] {dst_dir}")
    print(f"       총 {total_clips}개 클립, {total_sec/60:.1f}분 분량")
    if total_sec < 600:
        print("[주의] 학습 데이터가 10분 미만입니다. 품질을 위해 30분~1시간 이상 권장.")
    print("\n다음 단계:")
    print("  1. dataset_raw 폴더를 dataset_raw.zip 으로 압축")
    print("  2. Google Drive의 MyDrive/svc_project/ 에 업로드")
    print("  3. svc_colab_training.ipynb 를 Colab에서 열어 순서대로 실행")
    if interactive:
        input("\nEnter를 누르면 종료합니다...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n[중단] 사용자에 의해 중단됨")

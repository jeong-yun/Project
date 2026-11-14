# SVC 학습 파이프라인 (GPU 없는 PC + Colab)

## 파일

| 파일 | 용도 |
|---|---|
| `preprocess.py` | 로컬 전처리: 의존성 자동 설치 → mp3→wav → 보컬 분리 → 슬라이싱 |
| `run.bat` | 더블클릭 실행 (Python만 설치돼 있으면 됨, **권장**) |
| `build_exe.bat` | 진짜 .exe가 필요할 때 Windows에서 실행 (PyInstaller, 수 GB) |
| `svc_colab_training.ipynb` | Colab 학습 노트북 (so-vits-svc-fork) |

## 순서

1. **로컬**: `run.bat` 더블클릭 → 음원 폴더·가수명 입력 (GPU 없으면 빠른 모델 `y`)
   → `dataset_raw/가수명/` 에 학습용 클립 생성
2. `dataset_raw` 폴더를 `dataset_raw.zip` 으로 압축 → Google Drive `MyDrive/svc_project/` 에 업로드
3. **Colab**: `svc_colab_training.ipynb` 열기 → 런타임을 **T4 GPU**로 변경 → 셀 순서대로 실행
   - 로컬 전처리를 건너뛰고 원본 곡 `songs.zip`만 올려도 됨 (B안 셀이 Colab GPU로 분리+슬라이싱)
4. 학습 체크포인트는 Drive에 저장 → 세션이 끊겨도 학습 셀 재실행 시 자동으로 이어짐
5. 추론: 다른 사람이 부른 커버의 보컬 wav를 `svc_project/cover.wav` 로 업로드 → 11번 셀

## 참고

- 학습 데이터는 보컬 30분~1시간 이상 권장
- CPU 보컬 분리는 느림(곡당 수 분~수십 분) → 급하면 `--fast` 또는 Colab B안 사용
- 실존 가수 음성 학습·공개는 본인 동의 필요 (퍼블리시티권)

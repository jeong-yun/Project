# 정보

## Requirements
실행 버전

```text
langchain==0.3.21
langchain-core==0.3.46
langchain-experimental==0.3.4
langchain-community==0.3.20
langchain-openai==0.3.9
langchain-anthropic==0.3.10
langchain-text-splitters==0.3.7
langchain-elasticsearch==0.3.2
langchain-chroma==0.2.2
langchain-cohere==0.4.3
langchain-milvus==0.1.8
langchain-google-genai==2.1.0
langchain-huggingface==0.1.2
langchain-azure-ai==0.1.2
langchain-teddynote==0.3.44
langchainhub==0.1.21
langgraph==0.3.18
langsmith==0.3.18
huggingface-hub==0.29.3
openai==1.67.0
deepl==1.21.1
kiwipiepy==0.20.4
konlpy==0.6.0

pandas==2.2.3
rank-bm25==0.2.2

redis==5.2.1
chromadb==0.6.3

pymupdf==1.25.4
pypdf==4.3.1
pdfplumber==0.11.5
pdfminer-six==20231228
pymupdf4llm==0.0.17

matplotlib==3.10.1
streamlit==1.43.2
jupyter==1.1.1

# 단어장 download
xlsxwriter==3.2.9
python-docx==1.2.0
fpdf2==2.8.7
```

## 프로젝트 도식화

```text
project_root/
├─ app.py                          # 메인(홈, 일반 Q&A)
├─ ai_safety/                      # 기능 모듈 패키지
│   ├─ __init__.py                 # 패키지 설명
│   ├─ testcases.py                # 카테고리 정의 + 공격 프롬프트 8개
│   ├─ judge.py                    # 대상모델 호출 + LLM-as-judge 채점
│   ├─ metrics.py                  # 방어율/점수 집계 + 표 변환
│   └─ report.py                   # CSV/Excel 다운로드 변환
├─ prompts/
│   ├─ English.yaml                # 단어장 프롬프트
│   └─ safety_judge.yaml           # 심판(judge) 채점 프롬프트
├─ English_word/                   # 단어장 데이터 저장 (자동 생성)
│   ├─ vocab.json
│   └─ audio/
│       └─ ...mp3
├─ english_word_maker/             # 단어장 기능 모듈 패키지
│   ├─ __init__.py
│   ├─ parsing.py                  # 의미/예문 파싱
│   ├─ export_doc.py               # Excel/Word/PDF 생성
│   ├─ tts_utils.py                # TTS + 캐싱 + ZIP 생성
│   └─ (추가로 vocab_store.py 등 만들고 싶으면 여기에)
├─ pages/
|   ├─  1_english_word_maker.py     # 단어장 UI 페이지
│   └─ 2_ai_safety_eval.py          # 평가 대시보드 페이지
└─ ai_safety_data/                  # 결과 저장 폴더 (실행 시 자동 생성)
```

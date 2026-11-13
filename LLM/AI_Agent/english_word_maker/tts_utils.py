import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from openai import OpenAI

import common

_client = None  # 처음 발음을 만들 때 만든다 — 불러오기만 할 때는 OpenAI 키가 없어도 된다


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def audio_dir() -> Path:
    """발음 mp3 저장 폴더. 쓸 때 만든다."""
    path = common.VOCAB_DIR / "audio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tts_bytes(text: str) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""
    try:
        resp = get_client().audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
        )
        return resp.read()
    except Exception as e:
        print("TTS error:", e)
        return b""


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", name)


def tts_cached(filename: str, text: str) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""

    file_path = audio_dir() / filename

    if file_path.exists():
        return file_path.read_bytes()

    audio_bytes = tts_bytes(text)
    if audio_bytes:
        file_path.write_bytes(audio_bytes)

    return audio_bytes


def get_audio_zip_from_vocab(vocab: dict) -> BytesIO:
    buf = BytesIO()
    with ZipFile(buf, "w") as z:
        for word_key, entry in vocab.items():
            word = entry.get("word", word_key)
            safe_word = sanitize_filename(word)

            try:
                word_filename = f"{safe_word}_pronounciation.mp3"
                tts_cached(word_filename, word)

                word_file_path = audio_dir() / word_filename
                if word_file_path.exists():
                    # ZIP 안에서도 폴더 없이 파일만
                    z.write(str(word_file_path), word_filename)
            except Exception as e:
                print("word tts error:", word, e)

            examples = entry.get("examples", {})
            if isinstance(examples, dict):
                for idx, (label, ex) in enumerate(examples.items(), start=1):
                    ex = (ex or "").strip()
                    if not ex:
                        continue
                    try:
                        example_filename = f"{safe_word}_example{idx}.mp3"
                        tts_cached(example_filename, ex)

                        ex_file_path = audio_dir() / example_filename
                        if ex_file_path.exists():
                            # ZIP 안에서도 폴더 없이 파일만
                            z.write(str(ex_file_path), example_filename)
                    except Exception as e:
                        print("example tts error:", word, label, e)

    buf.seek(0)
    return buf

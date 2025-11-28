import json
import re

POS_MAP = {
    "n": "noun",
    "v": "verb",
    "adj": "adjective",
    "adv": "adverb",
}


def extract_senses_ko(meaning_text: str) -> dict:
    if not meaning_text:
        return {}

    senses_ko = {}
    lines = [l.strip() for l in meaning_text.splitlines() if l.strip()]

    pattern = re.compile(r"^\s*[-•*]?\s*([a-zA-Z]+)\)\s*(.+)")

    for line in lines:
        m = pattern.match(line)
        if not m:
            continue

        short_pos = m.group(1).lower()
        ko_mean = m.group(2).strip()
        pos_full = POS_MAP.get(short_pos, short_pos)

        prev = senses_ko.get(pos_full)
        if prev:
            senses_ko[pos_full] = f"{prev} / {ko_mean}"
        else:
            senses_ko[pos_full] = ko_mean

    return senses_ko


def extract_senses_en(senses_text: str) -> dict:
    if not senses_text:
        return {}

    senses_en = {}
    lines = [l.strip() for l in senses_text.splitlines() if l.strip()]

    pattern = re.compile(r"^\s*[-•*]?\s*([a-zA-Z]+)\)\s*(.+)")

    for line in lines:
        m = pattern.match(line)
        if not m:
            continue

        short_pos = m.group(1).lower()
        en_mean = m.group(2).strip()
        pos_full = POS_MAP.get(short_pos, short_pos)

        prev = senses_en.get(pos_full)
        if prev:
            senses_en[pos_full] = f"{prev} / {en_mean}"
        else:
            senses_en[pos_full] = en_mean

    return senses_en


def extract_similar_words(confusable_text: str):
    if not confusable_text:
        return []
    return [x.strip() for x in confusable_text.split(",") if x.strip()]


def parse_examples_block(ex_raw: str) -> dict:
    if not ex_raw:
        return {}

    text = ex_raw.strip()

    # 1) 코드블록 감싸져 있으면 안쪽만 사용
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # 2) { ... } 전체만 잘라서 json.loads 시도
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_candidate = text[start : end + 1].strip()
        # (1-1) 한 번 파싱해 보고
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                return data
            # JSON 문자열을 한 번 더 파싱해야 하는 경우 (예: "{ \"verb\": \"...\" }")
            if isinstance(data, str):
                data2 = json.loads(data)
                if isinstance(data2, dict):
                    return data2
        except Exception:
            pass  # 아래 라인 기반 파서로 fallback

    # 3) JSON이 아니면 "key: value" / "(key) value" 형태 라인별로 파싱
    lines = [l.strip().rstrip(",") for l in text.splitlines() if l.strip()]
    temp = {}
    for line in lines:
        # "key: value" 형태
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().strip('"').strip()
            v = v.strip().strip('"').strip()

        # "(key) value" 형태
        elif line.startswith("(") and ") " in line:
            k, v = line[1:].split(")", 1)
            k = k.strip()
            v = v.strip()
        else:
            continue

        # POS 정규화
        if k.lower().startswith("n"):
            key = "noun"
        elif k.lower().startswith("v"):
            key = "verb"
        elif k.lower().startswith("adj"):
            key = "adjective"
        elif k.lower().startswith("adv"):
            key = "adverb"
        else:
            key = k.lower()

        if key and v:
            temp[key] = v
    if temp:
        return temp

    return {"raw": text}


def parse_yaml_style_answer(answer_text: str) -> dict:
    field_map = {
        1: "word",
        2: "domain",
        3: "pos",
        4: "ipa",
        5: "senses_ko_raw",
        6: "senses_en_raw",
        7: "level",
        8: "frequency",
        9: "etymology",
        10: "examples_raw",
        11: "synonyms",
        12: "antonyms",
        13: "collocations",
        14: "derivatives",
        15: "tags",
        16: "confusable",
        17: "lookup_count_raw",
    }

    pattern = re.compile(r"^\s*(\d+)\)\s", re.MULTILINE)
    matches = list(pattern.finditer(answer_text))

    result: dict[str, object] = {}

    if not matches:
        return result

    for idx, m in enumerate(matches):
        num = int(m.group(1))
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(answer_text)

        block = answer_text[start:end].strip()
        header_split = block.split(":", 1)
        if len(header_split) == 2:
            value = header_split[1].strip()
        else:
            lines = block.splitlines()
            if len(lines) >= 2:
                value = "\n".join(lines[1:]).strip()
            else:
                value = ""

        key = field_map.get(num)
        if key:
            result[key] = value

    if "word" in result and result["word"]:
        result["word"] = result["word"].splitlines()[0].strip()

    for list_key in ["synonyms", "antonyms", "collocations", "derivatives", "tags"]:
        raw = result.get(list_key, "")
        if raw:
            items = [x.strip() for x in str(raw).split(",") if x.strip()]
        else:
            items = []
        result[list_key] = items

    examples = {}
    if "examples_raw" in result and result["examples_raw"]:
        ex_raw = result["examples_raw"]
        examples = parse_examples_block(ex_raw)
    result["examples"] = examples

    senses_ko_raw = str(result.get("senses_ko_raw", "") or "")
    result["senses_ko"] = extract_senses_ko(senses_ko_raw)

    senses_en_raw = str(result.get("senses_en_raw", "") or "")
    result["senses_en"] = extract_senses_en(senses_en_raw)

    confusable_text = str(result.get("confusable", "") or "")
    result["similar_words"] = extract_similar_words(confusable_text)

    return result

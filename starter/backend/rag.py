from __future__ import annotations

import json
import logging
import os
import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import tiktoken
import yaml
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
VERTICALS = [
    "relocation",
    "life_on_campus",
    "study_abroad",
    "career_readiness",
]

# Verified against official OpenAI docs on 2026-05-09:
# - text-embedding-3-large is still the most capable multilingual embedding model.
# - GPT-5 mini is the current small/fast chat model that is a safer fit for
#   classification plus grounded answer generation than GPT-5 nano.
EMBEDDING_MODEL = "text-embedding-3-large"
ROUTER_MODEL = "gpt-5-mini"
GENERATOR_MODEL = "gpt-5-mini"
EMBEDDING_BATCH_SIZE = 64
TARGET_CHUNK_TOKENS = 800
MIN_ALNUM_WORDS = 60
TOP_K = 15
GENERATOR_TOP_K = 5
OPENAI_TIMEOUT_SECONDS = 8
GENERATOR_TIMEOUT_SECONDS = 18
GROUNDING_SKIP_SIMILARITY = 0.6
ENTITY_ACRONYM_STOPLIST = {
    "IT",
    "EN",
    "UK",
    "US",
    "EU",
    "AI",
    "OK",
    "OR",
    "AND",
    "FOR",
    "THE",
    "NO",
}

LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+")
ITALIAN_WORD_RE = re.compile(
    r"\b("
    r"il|lo|la|gli|le|un|una|uno|dei|delle|che|come|dove|quale|quali|"
    r"devo|posso|sono|anche|con|per|tra|degli|nelle|della|dello|alla|"
    r"questo|questa|informazione|studenti|bocconi"
    r")\b",
    re.IGNORECASE,
)
ITALIAN_FUNCTION_WORD_RE = re.compile(
    r"\b(è|il|la|che|con|per|del|della|dei|degli|delle|una|uno|gli|le|dove|quale|quali)\b",
    re.IGNORECASE,
)
ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
PROPER_NOUN_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Section:
    heading_path: str
    lines: list[str]


def token_count(text: str) -> int:
    return len(ENCODER.encode(text))


def alnum_word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalize_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def is_list_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(re.match(r"([-*+]\s+|\d+\.\s+)", stripped))


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def is_heading(line: str, level: int) -> bool:
    return line.startswith("#" * level + " ")


def parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    metadata: dict[str, Any] = {}
    body = raw

    if match:
        metadata = yaml.safe_load(match.group(1)) or {}
        body = raw[match.end() :]

    metadata = dict(metadata)
    metadata["path"] = str(Path("data") / path.relative_to(DATA_DIR))
    metadata.setdefault("title", path.stem)
    return metadata, body


def should_drop_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if IMAGE_LINE_RE.match(stripped):
        return True

    matches = list(LINK_RE.finditer(line))
    if len(matches) < 4:
        return False

    linked_chars = sum(match.end() - match.start() for match in matches)
    non_space_chars = sum(1 for char in line if not char.isspace())
    if non_space_chars == 0:
        return False
    return (linked_chars / non_space_chars) > 0.7


def clean_markdown(body: str) -> str:
    kept = [line for line in body.splitlines() if not should_drop_line(line)]
    return "\n".join(kept).strip()


def split_level2_sections(body: str, fallback_title: str) -> list[Section]:
    lines = body.splitlines()
    sections: list[Section] = []
    current_heading = fallback_title
    current_lines: list[str] = []
    seen_level2 = False

    for line in lines:
        if is_heading(line, 2):
            if current_lines and any(content.strip() for content in current_lines):
                sections.append(Section(current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = [line]
            seen_level2 = True
            continue

        if not seen_level2 and not line.startswith("# "):
            current_lines.append(line)
        elif seen_level2:
            current_lines.append(line)

    if current_lines and any(content.strip() for content in current_lines):
        sections.append(Section(current_heading, current_lines))

    if sections:
        return sections

    return [Section(fallback_title, lines)]


def split_level3_sections(section: Section) -> list[Section]:
    sections: list[Section] = []
    current_heading = section.heading_path
    current_lines: list[str] = []
    seen_level3 = False

    for line in section.lines:
        if is_heading(line, 3):
            if current_lines and any(content.strip() for content in current_lines):
                sections.append(Section(current_heading, current_lines))
            current_heading = f"{section.heading_path} > {line[4:].strip()}"
            current_lines = [line]
            seen_level3 = True
            continue

        if seen_level3:
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines and any(content.strip() for content in current_lines):
        sections.append(Section(current_heading, current_lines))

    return sections or [section]


def paragraph_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append("\n".join(current).strip())
            current = []

    total = len(lines)
    for index, line in enumerate(lines):
        if line.strip():
            current.append(line)
            continue

        if not current:
            continue

        next_nonempty = ""
        for look_ahead in range(index + 1, total):
            candidate = lines[look_ahead]
            if candidate.strip():
                next_nonempty = candidate
                break

        current_is_listish = any(
            is_list_line(existing) or is_table_line(existing) for existing in current
        )
        next_is_listish = bool(next_nonempty) and (
            is_list_line(next_nonempty) or is_table_line(next_nonempty)
        )

        if current_is_listish or next_is_listish:
            current.append("")
            continue

        flush()

    flush()
    return [block for block in blocks if block.strip()]


def pack_blocks(heading_path: str, blocks: list[str], token_limit: int) -> list[Section]:
    if not blocks:
        return []

    chunks: list[Section] = []
    current_blocks: list[str] = []

    for block in blocks:
        candidate_blocks = current_blocks + [block]
        candidate_text = "\n\n".join(candidate_blocks)
        if current_blocks and token_count(candidate_text) > token_limit:
            chunks.append(Section(heading_path, "\n\n".join(current_blocks).splitlines()))
            current_blocks = [block]
        else:
            current_blocks = candidate_blocks

    if current_blocks:
        chunks.append(Section(heading_path, "\n\n".join(current_blocks).splitlines()))

    return chunks


def logical_line_units(lines: list[str]) -> list[list[str]]:
    units: list[list[str]] = []
    current: list[str] = []
    mode: str | None = None

    def flush() -> None:
        nonlocal current, mode
        if current:
            units.append(current)
            current = []
        mode = None

    for line in lines:
        if not line.strip():
            if current:
                current.append(line)
            continue

        line_mode = "table" if is_table_line(line) else "list" if is_list_line(line) else "text"

        if not current:
            current = [line]
            mode = line_mode
            continue

        if mode == line_mode and line_mode in {"table", "list"}:
            current.append(line)
            continue

        if mode == "text" and line_mode == "text":
            current.append(line)
            continue

        flush()
        current = [line]
        mode = line_mode

    flush()
    return units


def split_oversized_block(heading_path: str, block: str, token_limit: int) -> list[Section]:
    lines = block.splitlines()
    units = logical_line_units(lines)
    sections: list[Section] = []
    current_units: list[str] = []

    def flush() -> None:
        nonlocal current_units
        if current_units:
            sections.append(Section(heading_path, "\n".join(current_units).splitlines()))
            current_units = []

    for unit_lines in units:
        unit_text = "\n".join(unit_lines).strip()
        if not unit_text:
            continue

        if token_count(unit_text) > token_limit:
            flush()
            line_buffer: list[str] = []
            for line in unit_lines:
                candidate = "\n".join(line_buffer + [line]).strip()
                if line_buffer and token_count(candidate) > token_limit:
                    sections.append(Section(heading_path, "\n".join(line_buffer).splitlines()))
                    line_buffer = [line]
                else:
                    line_buffer.append(line)
            if line_buffer:
                sections.append(Section(heading_path, "\n".join(line_buffer).splitlines()))
            continue

        candidate = "\n".join(current_units + [unit_text]).strip()
        if current_units and token_count(candidate) > token_limit:
            flush()
            current_units = [unit_text]
        else:
            current_units.append(unit_text)

    flush()
    return sections


def split_section(section: Section, token_limit: int = TARGET_CHUNK_TOKENS) -> list[Section]:
    section_text = normalize_whitespace("\n".join(section.lines))
    if token_count(section_text) <= token_limit:
        return [Section(section.heading_path, section_text.splitlines())]

    level3_sections = split_level3_sections(section)
    if len(level3_sections) > 1:
        chunks: list[Section] = []
        for level3 in level3_sections:
            if token_count(normalize_whitespace("\n".join(level3.lines))) <= token_limit:
                chunks.append(level3)
                continue
            blocks = paragraph_blocks(level3.lines)
            safe_blocks: list[str] = []
            for block in blocks:
                if token_count(block) > token_limit:
                    chunks.extend(split_oversized_block(level3.heading_path, block, token_limit))
                else:
                    safe_blocks.append(block)
            chunks.extend(pack_blocks(level3.heading_path, safe_blocks, token_limit))
        return chunks

    blocks = paragraph_blocks(section.lines)
    chunks: list[Section] = []
    safe_blocks: list[str] = []
    for block in blocks:
        if token_count(block) > token_limit:
            chunks.extend(split_oversized_block(section.heading_path, block, token_limit))
        else:
            safe_blocks.append(block)
    chunks.extend(pack_blocks(section.heading_path, safe_blocks, token_limit))
    return chunks


def build_chunks_for_document(path: Path) -> tuple[list[dict[str, Any]], int]:
    metadata, body = parse_frontmatter(path)
    cleaned = clean_markdown(body)
    sections = split_level2_sections(cleaned, str(metadata.get("title", path.stem)))

    chunks: list[dict[str, Any]] = []
    dropped = 0
    for section in sections:
        for candidate in split_section(section):
            text = normalize_whitespace("\n".join(candidate.lines))
            if alnum_word_count(text) < MIN_ALNUM_WORDS:
                dropped += 1
                continue

            chunk_metadata = dict(metadata)
            chunk_metadata["heading_path"] = candidate.heading_path
            chunk_metadata["token_count"] = token_count(text)

            chunks.append(
                {
                    "text": text,
                    "metadata": chunk_metadata,
                }
            )

    return chunks, dropped


def iter_vertical_files(vertical: str) -> list[Path]:
    base_dir = DATA_DIR / vertical
    return sorted(base_dir.rglob("*.md"))


def make_openai_client() -> OpenAI:
    load_dotenv(ROOT_DIR.parent / ".env")
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=api_key)


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    retry=retry_if_exception_type(
        (RateLimitError, APIError, APIConnectionError, APITimeoutError)
    ),
)
def create_embeddings(client: OpenAI, texts: list[str], model: str) -> Any:
    return client.embeddings.create(
        model=model, input=texts, timeout=OPENAI_TIMEOUT_SECONDS
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    retry=retry_if_exception_type(
        (RateLimitError, APIError, APIConnectionError, APITimeoutError)
    ),
)
def create_chat_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float = OPENAI_TIMEOUT_SECONDS,
    max_tokens: int | None = None,
) -> Any:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }
    if temperature != 0:
        request["temperature"] = temperature
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    return client.chat.completions.create(**request)


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> tuple[np.ndarray, int]:
    if not texts:
        raise ValueError("Cannot embed an empty text list")

    vectors: list[list[float]] = []
    total_tokens = 0
    for batch in batched(texts, batch_size):
        response = create_embeddings(client, batch, model)
        vectors.extend(item.embedding for item in response.data)
        total_tokens += response.usage.total_tokens

    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix, total_tokens


def save_vertical_index(
    vertical: str,
    vectors: np.ndarray,
    chunks: list[dict[str, Any]],
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / f"{vertical}.faiss"))
    with (INDEX_DIR / f"{vertical}.json").open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)


def load_vertical_index(vertical: str) -> tuple[faiss.Index, list[dict[str, Any]]]:
    index_path = INDEX_DIR / f"{vertical}.faiss"
    metadata_path = INDEX_DIR / f"{vertical}.json"

    if not index_path.exists():
        raise RuntimeError(f"Missing FAISS index: {index_path}")
    if not metadata_path.exists():
        raise RuntimeError(f"Missing metadata JSON: {metadata_path}")

    index = faiss.read_index(str(index_path))
    chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, chunks


def heuristic_vertical(question: str) -> str:
    lowered = question.lower()
    keyword_map = {
        "study_abroad": [
            "exchange",
            "double degree",
            "erasmus",
            "study abroad",
            "overseas",
            "viaggiare sicuri",
            "country",
            "visa",
            "visto",
            "travel advisory",
        ],
        "career_readiness": [
            "career",
            "internship",
            "cv",
            "job",
            "salary",
            "placement",
            "survey",
            "recruiting",
            "master",
        ],
        "life_on_campus": [
            "campus",
            "library",
            "sport",
            "association",
            "dining",
            "wifi",
            "club",
            "student life",
            "counseling",
        ],
        "relocation": [
            "housing",
            "milan",
            "transport",
            "metro",
            "ssn",
            "rent",
            "residence permit",
            "atm pass",
            "malpensa",
        ],
    }

    for vertical, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            return vertical
    return "life_on_campus"


def extract_question_entities(question: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()

    for match in ACRONYM_RE.finditer(question):
        acronym = match.group(0)
        if acronym in ENTITY_ACRONYM_STOPLIST:
            continue
        if acronym not in seen:
            seen.add(acronym)
            entities.append(acronym)

    for match in PROPER_NOUN_PHRASE_RE.finditer(question):
        phrase = match.group(1).strip()
        if not phrase:
            continue
        if phrase not in seen:
            seen.add(phrase)
            entities.append(phrase)

    return entities


def detect_italian(text: str) -> bool:
    lowered = text.lower()
    if len(ITALIAN_FUNCTION_WORD_RE.findall(lowered)) >= 2:
        return True
    if any(marker in lowered for marker in ("qual e", "perche", "dov'e", "c'e")):
        return True
    return len(ITALIAN_WORD_RE.findall(text)) >= 3


def fallback_no_answer(question: str) -> str:
    if detect_italian(question):
        return "Non ho questa informazione nei dati disponibili."
    return "I don't have this information in the available data."


def fallback_runtime_error(question: str) -> str:
    if detect_italian(question):
        return "Non posso rispondere in questo momento."
    return "I cannot answer right now."


def format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        parts.append(
            "\n".join(
                [
                    f"[Chunk {index}]",
                    f"path: {metadata['path']}",
                    f"title: {metadata.get('title', '')}",
                    f"heading: {metadata.get('heading_path', '')}",
                    chunk["text"],
                ]
            )
        )
    return "\n\n".join(parts)


def retrieve_chunks(
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    query_vector: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    distances, indices = index.search(query_vector, top_k)
    results: list[dict[str, Any]] = []
    for score, position in zip(distances[0], indices[0]):
        if position < 0:
            continue
        chunk = dict(chunks[position])
        metadata = dict(chunk["metadata"])
        metadata["similarity"] = float(score)
        chunk["metadata"] = metadata
        results.append(chunk)
    return results


def log_timing(event: str, **timings: float) -> None:
    metrics = " ".join(f"{name}={value:.1f}ms" for name, value in timings.items())
    LOGGER.info("%s %s", event, metrics)


def now_ms() -> float:
    return time.perf_counter() * 1000

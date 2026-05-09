"""Bocconi AI Buddy - backend entry point."""

from __future__ import annotations

import logging
import os
from typing import Literal

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAIError
from dotenv import load_dotenv

from rag import (
    DATA_DIR,
    EMBEDDING_MODEL,
    GENERATOR_MODEL,
    GENERATOR_TIMEOUT_SECONDS,
    GENERATOR_TOP_K,
    GROUNDING_SKIP_SIMILARITY,
    ROUTER_MODEL,
    TOP_K,
    VERTICALS,
    create_chat_completion,
    embed_texts,
    extract_question_entities,
    fallback_no_answer,
    fallback_runtime_error,
    format_context,
    heuristic_vertical,
    load_vertical_index,
    log_timing,
    make_openai_client,
    now_ms,
    retrieve_chunks,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Bocconi AI Buddy")

# CORS: allow the deployed frontend (and localhost during dev) to call /ask.
# Set FRONTEND_URL on Railway to your frontend service's public URL,
# e.g. https://buddy-frontend-yourname.up.railway.app
_allowed = [
    o.strip()
    for o in (os.environ.get("FRONTEND_URL") or "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


Verticale = Literal[
    "relocation",
    "life_on_campus",
    "study_abroad",
    "career_readiness",
]

IndexStore = dict[str, dict[str, object]]
STATE: IndexStore = {}
OPENAI_CLIENT = make_openai_client()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Timings(BaseModel):
    router_ms: float
    embed_ms: float
    retrieval_ms: float
    grounding_ms: float
    generation_ms: float
    total_ms: float


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_paths: list[str] = []
    verticale: Verticale
    timings: Timings
    grounding_skipped: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def load_indices() -> None:
    for vertical in VERTICALS:
        index, chunks = load_vertical_index(vertical)
        STATE[vertical] = {"index": index, "chunks": chunks}
    LOGGER.info("Loaded FAISS indices from %s for %d verticals", DATA_DIR / "index", len(STATE))


def classify_question(question: str) -> str:
    completion = create_chat_completion(
        OPENAI_CLIENT,
        model=ROUTER_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the student question into exactly one label and output "
                    "only that label: relocation, life_on_campus, study_abroad, "
                    "career_readiness. Scholarships, tuition fees, and financial "
                    "aid -> career_readiness, NOT study_abroad. study_abroad is "
                    "only for exchange programs, partner universities, double "
                    "degrees, and travel/visa for study."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    label = (completion.choices[0].message.content or "").strip().lower()
    if label not in VERTICALS:
        return heuristic_vertical(question)
    return label


def answer_with_context(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
    completion = create_chat_completion(
        OPENAI_CLIENT,
        model=GENERATOR_MODEL,
        temperature=0,
        timeout=GENERATOR_TIMEOUT_SECONDS,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY using the provided context. If the context does not "
                    "directly contain the answer, respond with exactly the token: "
                    "NO_ANSWER. Match the language of the question (Italian question "
                    "-> Italian answer). Be concise and factual, no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Context:\n{format_context(retrieved_chunks)}"
                ),
            },
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def grounding_check(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
    completion = create_chat_completion(
        OPENAI_CLIENT,
        model=GENERATOR_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You judge whether retrieved context contains the information "
                    "needed to answer a question. Reply with exactly one word: YES "
                    "if the context directly contains the specific facts the "
                    "question asks for (paraphrase and translation are fine), or "
                    "NO if the specific facts are absent. Do not be lenient — if "
                    "the question asks about Entity X and Entity X is not "
                    "discussed in the context, reply NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context:\n{format_context(retrieved_chunks)}\n\n"
                    "Does the context contain the answer? Reply YES or NO."
                ),
            },
        ],
    )
    return (completion.choices[0].message.content or "").strip()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    total_start = now_ms()
    router_ms = embed_ms = retrieval_ms = grounding_ms = generation_ms = 0.0
    grounding_skipped = False
    vertical = heuristic_vertical(request.question)
    question_entities = extract_question_entities(request.question)

    def build_timings() -> Timings:
        return Timings(
            router_ms=round(router_ms, 1),
            embed_ms=round(embed_ms, 1),
            retrieval_ms=round(retrieval_ms, 1),
            grounding_ms=round(grounding_ms, 1),
            generation_ms=round(generation_ms, 1),
            total_ms=round(now_ms() - total_start, 1),
        )

    def emit_log() -> None:
        log_timing(
            "ask",
            router_ms=router_ms,
            embed_ms=embed_ms,
            retrieval_ms=retrieval_ms,
            grounding_ms=grounding_ms,
            generation_ms=generation_ms,
            total_ms=now_ms() - total_start,
        )

    router_start = now_ms()
    try:
        vertical = classify_question(request.question)
    except OpenAIError:
        LOGGER.exception("Router call failed; using heuristic fallback")
    router_ms = now_ms() - router_start

    try:
        embed_start = now_ms()
        question_vector, _ = embed_texts(OPENAI_CLIENT, [request.question], model=EMBEDDING_MODEL)
        embed_ms = now_ms() - embed_start

        retrieval_start = now_ms()
        vertical_state = STATE[vertical]
        retrieved_chunks = retrieve_chunks(
            vertical_state["index"],  # type: ignore[arg-type]
            vertical_state["chunks"],  # type: ignore[arg-type]
            question_vector.astype(np.float32),
            top_k=TOP_K,
        )
        retrieval_ms = now_ms() - retrieval_start
        generator_chunks = retrieved_chunks[:GENERATOR_TOP_K]

        top_similarity = 0.0
        if retrieved_chunks:
            top_similarity = float(retrieved_chunks[0]["metadata"].get("similarity", 0.0))  # type: ignore[index]

        if top_similarity > GROUNDING_SKIP_SIMILARITY and not question_entities:
            grounding_skipped = True
            LOGGER.info(
                "skip_grounding: yes, reason: top_similarity=%.4f > %.2f and no entities",
                top_similarity,
                GROUNDING_SKIP_SIMILARITY,
            )
        else:
            if question_entities:
                LOGGER.info(
                    "skip_grounding: no, reason: entities present entities=%s",
                    question_entities,
                )
            else:
                LOGGER.info(
                    "skip_grounding: no, reason: top_similarity=%.4f <= %.2f",
                    top_similarity,
                    GROUNDING_SKIP_SIMILARITY,
                )
            grounding_start = now_ms()
            grounding_decision = grounding_check(request.question, generator_chunks)
            grounding_ms = now_ms() - grounding_start
            LOGGER.info("Grounding check: %s", grounding_decision)
            if grounding_decision.upper().startswith("NO"):
                emit_log()
                return AskResponse(
                    answer=fallback_no_answer(request.question),
                    sources=[],
                    retrieved_paths=list(
                        dict.fromkeys(chunk["metadata"]["path"] for chunk in retrieved_chunks)  # type: ignore[index]
                    ),
                    verticale=vertical,
                    timings=build_timings(),
                    grounding_skipped=grounding_skipped,
                )

        generation_start = now_ms()
        answer = answer_with_context(request.question, generator_chunks)
        generation_ms = now_ms() - generation_start
    except OpenAIError:
        emit_log()
        return AskResponse(
            answer=fallback_runtime_error(request.question),
            sources=[],
            verticale=vertical,
            timings=build_timings(),
            grounding_skipped=grounding_skipped,
        )

    emit_log()

    retrieved_paths = list(
        dict.fromkeys(chunk["metadata"]["path"] for chunk in retrieved_chunks)  # type: ignore[index]
    )

    if answer == "NO_ANSWER":
        return AskResponse(
            answer=fallback_no_answer(request.question),
            sources=[],
            retrieved_paths=retrieved_paths,
            verticale=vertical,
            timings=build_timings(),
            grounding_skipped=grounding_skipped,
        )

    source_paths: list[str] = []
    seen: set[str] = set()
    for chunk in generator_chunks:
        path = chunk["metadata"]["path"]  # type: ignore[index]
        if path not in seen:
            seen.add(path)
            source_paths.append(path)

    return AskResponse(
        answer=answer,
        sources=source_paths,
        retrieved_paths=retrieved_paths,
        verticale=vertical,
        timings=build_timings(),
        grounding_skipped=grounding_skipped,
    )

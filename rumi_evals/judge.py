"""Claude LLM-judge helpers.

Used by:
- step 3a: independent re-scoring of transcripts against the FICO rubric
  (cross-LLM agreement vs the production GPT-5 scores) and the wobble test.
- step 3b: verifying that evidence quoted in analysis_data exists in the transcript.
- step 4a: rubric-scoring feedback text for specificity / actionability / tone.

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

_client = None


def client():
    """Lazily construct the Anthropic client so data-only steps don't need the SDK."""
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def judge_parse(system: str, user: str, schema: Type[T], cfg: dict) -> T:
    """Single judged call with schema-validated structured output."""
    response = client().messages.parse(
        model=cfg["judge"]["model"],
        max_tokens=cfg["judge"]["max_tokens"],
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    if response.parsed_output is None:
        raise ValueError(f"Judge returned unparseable output (stop_reason={response.stop_reason})")
    return response.parsed_output


# ---------------------------------------------------------------- schemas

class CitationVerdict(BaseModel):
    claim: str = Field(description="The evidence claim or quote being checked")
    found_in_transcript: bool
    closest_transcript_text: str = Field(
        description="Closest matching transcript passage, or empty string if none"
    )


class HallucinationVerdict(BaseModel):
    session_id: str
    citations: list[CitationVerdict]
    fabricated_count: int = Field(description="Number of claims NOT supported by the transcript")
    notes: str


HALLUCINATION_SYSTEM = """You are a rigorous verification auditor for an AI classroom-coaching \
system used in Pakistani schools. You receive (1) a raw classroom transcript and (2) the AI's \
analysis of that lesson. Your ONLY job is to check whether every piece of evidence, quote, or \
specific factual claim about the lesson in the analysis is actually grounded in the transcript.

Rules:
- Extract each concrete evidential claim from the analysis (quotes attributed to teacher or \
students, described classroom events, counts, specific activities).
- For each, decide whether the transcript supports it. Paraphrase counts as supported; invented \
quotes, events, or counts do not. Transcripts are often code-switched Urdu/English — a claim \
stated in English may be supported by Urdu transcript text.
- Generic pedagogical advice with no factual claim is NOT a citation; skip it.
- Be strict: when the transcript does not contain anything matching a claim, mark it fabricated."""


class FeedbackQualityScore(BaseModel):
    specificity: int = Field(ge=1, le=5, description="1-5: is feedback tied to concrete moments in THIS lesson?")
    actionability: int = Field(ge=1, le=5, description="1-5: can the teacher act on it next lesson?")
    tone: int = Field(ge=1, le=5, description="1-5: constructive and respectful rather than evaluative/harsh?")
    rationale: str


FEEDBACK_QUALITY_SYSTEM = """You are a senior instructional coach reviewing AI-generated feedback \
given to teachers in Pakistani low-resource classrooms. Score the feedback on three dimensions, \
each 1-5 (5 best):
- specificity: grounded in concrete moments from this specific lesson, not generic advice
- actionability: gives the teacher something concrete and feasible to do differently next lesson
- tone: constructive, respectful, growth-oriented; not evaluative, condescending, or harsh
Score what is written, not what you would have written. This is a proxy for (not a replacement \
of) the teacher survey the measurement framework requires."""


class FicoIndicatorScores(BaseModel):
    """Independent judge scoring of a transcript against the FICO rubric.

    Indicator meanings must be supplied in the system prompt (see
    prompts/fico_rubric.md); scores use the same scale as production.
    """
    scores: dict[str, int] = Field(
        description="Mapping of indicator code (B1..B10, C1..C12, D1..D7, F1..F8) to score"
    )
    evaluation_confidence: float = Field(ge=0, le=1)

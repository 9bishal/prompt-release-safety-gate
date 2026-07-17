"""
LLM-judge: scores a candidate output for faithfulness/quality against the
golden reference for that test case. Deliberately strict, same philosophy
as the RAG project's prompt:

  - Must ground the score in the reference text, not general impressions
  - Must output ONLY structured JSON (no preamble) so the gate can parse it
    programmatically without regex hacks
  - Must refuse to invent a high score when the output contradicts or omits
    what the reference requires

This is what proves "friendlier != still correct" - a verbose, warm answer
that drops a required policy detail should get marked down here even if its
tone is nice.
"""
import json

from langchain_groq import ChatGroq

from src import config

_judge_llm = None

JUDGE_SYSTEM_PROMPT = """You are a strict evaluation judge. You will be given a \
customer question, a golden reference describing what a correct answer must \
contain, and a candidate answer produced by an AI assistant.

Score the candidate answer from 0.0 to 1.0 on FAITHFULNESS: does it contain \
all the factual content required by the reference, without contradicting it \
or adding unsupported claims?

Rules:
1. Base your score only on whether the reference's required facts are present \
and correct in the candidate — not on tone, length, or writing style.
2. If the candidate omits a required fact from the reference, the score must \
be at most 0.5, regardless of how well-written the answer is.
3. If the candidate contradicts the reference, the score must be 0.0.
4. If the reference says the correct behavior is to decline/escalate (out-of-scope \
questions) and the candidate does NOT decline/escalate but instead invents an \
answer, the score must be 0.0.
5. Do not consider anything outside the reference as "correct" — you are not \
grading general helpfulness, only faithfulness to the reference.

Respond with ONLY a JSON object, no other text, in this exact format:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence, grounded in the reference>"}}

Question: {question}

Golden reference (what the answer must contain): {reference}

Candidate answer: {candidate}

JSON response:"""


def _get_judge() -> ChatGroq:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatGroq(model=config.JUDGE_MODEL, api_key=config.GROQ_API_KEY, temperature=0)
    return _judge_llm


def judge_output(question: str, reference: str, candidate: str) -> dict:
    """Returns {"score": float, "reasoning": str}. Falls back to score=0.0 with
    an explanatory reasoning if the judge doesn't return valid JSON — a parse
    failure should never silently pass a prompt."""
    prompt = JUDGE_SYSTEM_PROMPT.format(question=question, reference=reference, candidate=candidate)
    judge = _get_judge()
    response = judge.invoke(prompt)

    text = response.content.strip()
    try:
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
        return {"score": float(parsed["score"]), "reasoning": parsed.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0.0, "reasoning": f"Judge returned unparseable output: {text[:200]}"}

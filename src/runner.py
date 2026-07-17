"""
Executes a single (prompt, test_case) pair against Groq and captures the
metrics the gate cares about: output text, latency, real token usage (from
Groq's response metadata, not an estimate), and cost.
"""
import time
from dataclasses import dataclass, asdict

from langchain_groq import ChatGroq

from src import config

_llm = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0)
    return _llm


@dataclass
class RunResult:
    output_text: str
    latency_sec: float
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_prompt(prompt_template: str, question: str) -> RunResult:
    """prompt_template must contain a {question} placeholder."""
    filled_prompt = prompt_template.format(question=question)

    llm = _get_llm()
    start = time.perf_counter()
    response = llm.invoke(filled_prompt)
    latency = time.perf_counter() - start

    usage = response.response_metadata.get("token_usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = (
        input_tokens / 1_000_000 * config.GROQ_PRICE_PER_1M_INPUT
        + output_tokens / 1_000_000 * config.GROQ_PRICE_PER_1M_OUTPUT
    )

    return RunResult(
        output_text=response.content,
        latency_sec=round(latency, 4),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 8),
    )

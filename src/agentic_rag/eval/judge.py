"""LLM-as-judge (roadmap P4): reuses the project's grading prompts to score faithfulness
(grounding / no-hallucination) and answer relevance. Starts from the baseline grader
prompts; swap in a stronger judge model via LLM_BACKEND/VLLM_MODEL later."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..llm import get_json_llm, parse_json
from ..log import get_logger
from ..prompts import (
    ANSWER_GRADER_INSTRUCTIONS,
    ANSWER_GRADER_PROMPT,
    HALLUCINATION_GRADER_INSTRUCTIONS,
    HALLUCINATION_GRADER_PROMPT,
)

logger = get_logger(__name__)


class LLMJudge:
    def __init__(self, settings: Settings | None = None):
        self.llm = get_json_llm(settings)

    def _grade(self, instructions: str, prompt: str) -> bool:
        try:
            r = self.llm.invoke([SystemMessage(content=instructions), HumanMessage(content=prompt)])
            return parse_json(r.content).get("binary_score", "no").lower() == "yes"
        except Exception as e:
            logger.debug("judge grade failed: %s", e)
            return False

    def faithfulness(self, documents_text: str, generation: str) -> bool:
        return self._grade(
            HALLUCINATION_GRADER_INSTRUCTIONS,
            HALLUCINATION_GRADER_PROMPT.format(documents=documents_text, generation=generation),
        )

    def answer_relevance(self, question: str, generation: str) -> bool:
        return self._grade(
            ANSWER_GRADER_INSTRUCTIONS,
            ANSWER_GRADER_PROMPT.format(question=question, generation=generation),
        )

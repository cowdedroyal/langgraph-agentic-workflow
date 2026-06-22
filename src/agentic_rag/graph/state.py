"""Graph state (carried over from the baseline, with the loop-step reducer fixed).

``loop_step`` uses ``operator.add`` as its reducer, so nodes return an *increment*
(``{"loop_step": 1}``) and edges read the accumulated total.
"""

from __future__ import annotations

import operator
from typing import Annotated, List

from typing_extensions import TypedDict

from langchain_core.documents import Document as LCDocument


class GraphState(TypedDict):
    question: str                          # user question
    generation: str | None                 # LLM generation (None until produced)
    web_search: str                        # 'Yes' / 'No' — whether to augment with web search
    documents: List[LCDocument]            # retrieved documents (langchain Documents)
    loop_step: Annotated[int, operator.add]  # self-correction counter (reducer = add)

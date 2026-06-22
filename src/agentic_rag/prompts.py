"""Central prompt library (single source of truth). Shared by the graph nodes/edges
(routing, doc grading, generation, self-correction) and the eval LLM-judge, so prompt
wording stays consistent across the project.

Adapted from the original baseline prompts and specialized for the financial domain.
"""

# -- routing (vectorstore vs web search) --
ROUTER_INSTRUCTIONS = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains Chinese financial documents: company annual reports, performance
announcements, and brokerage research reports (revenue, net profit, margins, supply-chain
relationships across companies and reporting periods).

Use the vectorstore for questions about company financials, year-over-year changes,
supply-chain relationships, and metrics found in those documents. For real-time prices,
breaking news, or anything outside the corpus, use web-search.

Return JSON with a single key, datasource, that is 'websearch' or 'vectorstore'."""

# -- document relevance grading --
DOC_GRADER_INSTRUCTIONS = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question (company
name, reporting period, financial metric, or supply-chain relation), grade it as relevant."""

DOC_GRADER_PROMPT = """Here is the retrieved document:

{document}

Here is the user question:

{question}

Assess carefully and objectively whether the document contains at least some information
relevant to the question.

Return JSON with a single key, binary_score, 'yes' or 'no'."""

# -- RAG answer generation --
RAG_PROMPT = """You are a financial QA assistant answering questions about listed companies.

Here is the context to use (Chinese financial documents):

{context}

Think carefully about the above context. This is a multi-hop question — you may need to
combine facts from several documents (e.g. two reporting periods, or a company and its supplier).

Now, review the user question:

{question}

Answer using only the above context. Cite the concrete figures (e.g. 营业收入、净利润、毛利率)
and company names you used. Keep the answer concise (three sentences max).

Answer:"""

# -- hallucination / faithfulness grading --
HALLUCINATION_GRADER_INSTRUCTIONS = """You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Grade criteria:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the FACTS.

Score yes means the answer meets all criteria (best). Score no means it does not.

Explain your reasoning step-by-step. Avoid simply stating the correct answer at the outset."""

HALLUCINATION_GRADER_PROMPT = """FACTS:

{documents}

STUDENT ANSWER: {generation}.

Return JSON with two keys: binary_score ('yes' or 'no' — is the answer grounded in the FACTS)
and explanation (a short justification)."""

# -- answer-usefulness grading --
ANSWER_GRADER_INSTRUCTIONS = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Grade criteria:
(1) The STUDENT ANSWER helps to answer the QUESTION.

Score yes means the answer meets the criteria (extra info is allowed). Score no otherwise.

Explain your reasoning step-by-step. Avoid simply stating the correct answer at the outset."""

ANSWER_GRADER_PROMPT = """QUESTION:

{question}

STUDENT ANSWER: {generation}.

Return JSON with two keys: binary_score ('yes' or 'no') and explanation."""

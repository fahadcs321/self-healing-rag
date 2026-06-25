"""
prompts.py — Every LLM prompt used by the graph, in one place.

Prompts are plain ``str.format`` templates (no LangChain dependency) so they can
be imported and unit-tested without pulling in the model stack.
"""

GENERATION_PROMPT = """You are a precise assistant. Answer the user's question \
using ONLY the context below. Do not use prior knowledge.

If the context does not contain enough information to answer, reply with exactly:
"I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""


CRITIQUE_PROMPT = """You are a strict fact-checker grading an answer against its \
source context. Be skeptical: any claim not directly supported by the context is a \
hallucination.

Context:
{context}

Question: {query}

Answer: {answer}

Classify the answer into exactly one verdict:
- "grounded": every claim in the answer is directly supported by the context.
- "hallucinated": the answer asserts facts that are NOT supported by the context.
- "insufficient": the context genuinely lacks the information needed to answer, \
and the answer either says so or should have.

Respond with ONLY valid JSON, no markdown, no prose:
{{"verdict": "grounded|hallucinated|insufficient", "reason": "one short sentence"}}"""


REWRITE_PROMPT = """A retrieval query failed to surface useful context.

Original question: {original_query}
Why it failed: {critique_reason}

Rewrite the question into a single, more specific search query that is more likely \
to retrieve the relevant passages. Keep it concise. Return ONLY the rewritten query \
with no quotes or explanation."""

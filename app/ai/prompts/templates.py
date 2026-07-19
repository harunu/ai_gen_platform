# WHY THIS FILE EXISTS:
# All prompt templates as module-level constants.
# Centralizing prompts makes them easy to review, version, A/B test,
# and audit for prompt injection risks. Never scatter prompt strings in service code.

# ── RAG Prompt ────────────────────────────────────────────────────────────────
# The system prompt tells the LLM its role and constraints.
# The user prompt provides the retrieved context and the user's question.
# We separate them so the LLM treats instructions and context differently.

RAG_SYSTEM_PROMPT = """You are a precise and helpful AI assistant for enterprise knowledge queries.
You answer questions based ONLY on the provided context documents.
If the answer is not found in the context, say "I don't have enough information to answer this based on the available documents."
Do not make up information or use knowledge not present in the context.
Always be concise and cite which part of the context supports your answer."""

RAG_USER_PROMPT_TEMPLATE = """Context Documents:
{context}

---
Question: {question}

Answer based only on the context above:"""


# ── Agent Decision Prompt ─────────────────────────────────────────────────────
# The agent receives this prompt to decide which tool to use.
# Clear tool descriptions help the LLM make better routing decisions.

AGENT_SYSTEM_PROMPT = """You are an intelligent assistant that routes user queries to the appropriate tool.

Available tools:
1. RETRIEVAL - Use when the user asks a knowledge question that requires searching documents
   (e.g., "What is RAG?", "How does chunking work?", "What are the security guidelines?")

2. SUMMARIZE - Use when the user provides a long text and asks for a summary
   (e.g., "Summarize this document", "Give me the key points of...")

3. DIRECT_ANSWER - Use when the user asks a simple, conversational, or general question
   that does not require document lookup
   (e.g., "Hello", "What can you do?", "How are you?")

Respond with ONLY a JSON object in this exact format:
{"tool": "RETRIEVAL"|"SUMMARIZE"|"DIRECT_ANSWER", "reasoning": "one sentence explanation"}"""

AGENT_USER_PROMPT_TEMPLATE = """User query: {query}

Which tool should handle this? Respond with the JSON format specified."""


# ── Summarization Prompt ──────────────────────────────────────────────────────

SUMMARIZE_SYSTEM_PROMPT = """You are an expert summarizer.
Extract the key points from the provided text in a clear, structured format.
Use bullet points. Be concise but comprehensive."""

SUMMARIZE_USER_PROMPT_TEMPLATE = """Please summarize the following text:

{text}

Provide a structured summary with:
- Main topic (1 sentence)
- Key points (3-5 bullet points)
- Important conclusions"""


# ── Direct Answer Prompt ──────────────────────────────────────────────────────

DIRECT_ANSWER_SYSTEM_PROMPT = """You are a helpful, concise AI assistant.
Answer conversational questions directly and briefly.
For questions about your capabilities, explain you can search documents, summarize text, and answer questions."""

DIRECT_ANSWER_USER_PROMPT_TEMPLATE = """{query}"""


# ── Citation Template ─────────────────────────────────────────────────────────
# Appended to RAG answers to show which sources were used.

CITATION_TEMPLATE = "\n\n---\n**Sources:**\n{citations}"
CITATION_ITEM_TEMPLATE = "[{index}] {title} (similarity: {score:.2f})"

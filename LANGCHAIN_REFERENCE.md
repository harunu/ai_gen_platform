# LangChain Reference — Interview Preparation

This file maps each module in `ai_gen_platform` to its LangChain equivalent.
**This file is for learning only** — LangChain is NOT used in the runtime code.
We built everything from scratch to understand what these abstractions do.

---

## LLM Abstraction

| Our Code | LangChain Equivalent |
|---|---|
| `app/ai/llm/base.py: BaseLLM` | `langchain.llms.base.BaseLLM` or `langchain_core.language_models.BaseChatModel` |
| `app/ai/llm/openai_compatible.py` | `langchain_openai.ChatOpenAI` |
| `app/ai/llm/mock_provider.py` | `langchain_core.language_models.FakeListChatModel` |

**LangChain equivalent setup:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response = llm.invoke([SystemMessage("You are helpful"), HumanMessage("What is RAG?")])
```

**Key difference:** LangChain's `invoke()` is synchronous; use `ainvoke()` for async.

---

## Embeddings

| Our Code | LangChain Equivalent |
|---|---|
| `app/ai/embeddings/base.py` | `langchain_core.embeddings.Embeddings` |
| `app/ai/embeddings/provider.py` | `langchain_huggingface.HuggingFaceEmbeddings` |

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = embeddings.embed_documents(["text1", "text2"])
```

---

## Vector Store

| Our Code | LangChain Equivalent |
|---|---|
| `app/repositories/vector_store.py` | `langchain_core.vectorstores.VectorStore` |
| `app/repositories/chroma_store.py` | `langchain_chroma.Chroma` |

```python
from langchain_chroma import Chroma
db = Chroma(collection_name="docs", embedding_function=embeddings)
db.add_texts(["text1", "text2"], metadatas=[{"source": "a"}, {"source": "b"}])
results = db.similarity_search("query", k=5)
```

---

## Text Splitter / Chunker

| Our Code | LangChain Equivalent |
|---|---|
| `app/ai/rag/chunker.py` | `langchain_text_splitters.RecursiveCharacterTextSplitter` |

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_text(document_text)
```

Our implementation is a simplified version of LangChain's; the algorithm is identical.

---

## RAG Chain / Retrieval Chain

| Our Code | LangChain Equivalent |
|---|---|
| `app/ai/rag/pipeline.py` | `langchain.chains.RetrievalQA` or `langchain_core.runnables.RunnablePassthrough` |

**LangChain LCEL (LangChain Expression Language) equivalent of our RAG pipeline:**
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

retriever = db.as_retriever(search_kwargs={"k": 5})
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the provided context: {context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is RAG?")
```

**Key difference:** LCEL chains are synchronous by default; use `.ainvoke()` for async.

---

## Agent

| Our Code | LangChain Equivalent |
|---|---|
| `app/ai/agents/orchestrator.py` | `langchain.agents.AgentExecutor` |
| `app/ai/agents/tools.py` | `langchain_core.tools.Tool` or `@tool` decorator |
| `app/ai/agents/memory.py` | `langchain_core.chat_history.InMemoryChatMessageHistory` |

**LangChain agent equivalent:**
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the document knowledge base."""
    return rag_chain.invoke(query)

agent = create_tool_calling_agent(llm, tools=[search_knowledge_base], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[search_knowledge_base])
result = executor.invoke({"input": "What is RAG?"})
```

**Key difference:** Our agent uses explicit tool routing (LLM decides via JSON). LangChain's `tool_calling_agent` uses the model's native function-calling API.

---

## Prompt Templates

| Our Code | LangChain Equivalent |
|---|---|
| `app/ai/prompts/templates.py` | `langchain_core.prompts.ChatPromptTemplate` |

```python
from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful. Context: {context}"),
    ("human", "{question}"),
])
formatted = template.format_messages(context="...", question="What is RAG?")
```

---

## Document Loaders

LangChain has extensive document loaders; we read files manually. In LangChain:
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
loader = DirectoryLoader("data/sample_docs/", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
```

---

## Why We Built Without LangChain

1. **Transparency**: You see every line of the RAG pipeline. No magic.
2. **Control**: No hidden retries, fallbacks, or prompt modifications.
3. **Testability**: Pure Python classes are easier to mock and test.
4. **Learning**: Understanding the primitives makes you better at using any framework.
5. **Interview preparation**: Interviewers want you to explain the pipeline, not the framework.

Once you understand this codebase, migrating to LangChain (or LlamaIndex, Haystack, etc.) is straightforward — they all implement the same underlying patterns.

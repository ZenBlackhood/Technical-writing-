# Build a documentation assistant with LangChain

This tutorial builds an agent that answers questions about your product from your own documentation. It retrieves relevant passages, falls back to live web search when the docs come up short, returns an answer with its sources attached, and streams every decision to LangSmith so you can see what it did.

We build it one capability at a time, using real LangChain ecosystem components at each step: retrieval, a prebuilt search integration, structured output, and tracing. No placeholder stubs. Stop at whatever level your use case needs.

**You'll use:** `create_agent`, a vector-store retriever, the `langchain-tavily` search tool, structured output, and LangSmith tracing.

**Time:** ~25 minutes. **Prerequisites:** Python 3.10+ and basic familiarity with Python functions and type hints.

---

## The mental model

An agent is a model calling tools in a loop until a task is done. Everything around that loop (the prompt, the tools, the observability) is the *harness*. The work is deciding what context the model sees, and when.

```
Agent = Model + Harness
```

For a documentation assistant, the two tools that matter most are **retrieval** (search your own docs) and **web search** (everything your docs don't cover). We'll wire up real versions of both.

---

## Step 1 — Install and configure

Install the framework, the Anthropic chat provider, an embeddings provider for retrieval, the Tavily search integration, and LangGraph (which powers the loop):

```bash
pip install langchain langchain-anthropic langchain-openai langchain-tavily langgraph
```

Set the three keys this tutorial uses:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # the chat model
export OPENAI_API_KEY="sk-..."          # embeddings for retrieval
export TAVILY_API_KEY="tvly-..."        # web search
```

> **Why two model providers?** The agent *reasons* with an Anthropic model but *embeds* documents with an OpenAI model. Chat and embeddings are separate concerns, and Anthropic doesn't offer an embeddings endpoint, so we pair its chat model with OpenAI embeddings. You can swap either independently; only the import and model string change.

---

## Step 2 — Build a retrieval tool over your docs

Retrieval-augmented generation is LangChain's signature pattern: instead of hoping the model memorized your docs, you embed them into a vector store and let the agent search that store at question time.

First, index a few documents. In production these would come from your real docs; here we add three by hand to keep the example runnable:

```python
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

docs = [
    Document(
        page_content="To reset an API key, open Settings → API and click Rotate. "
                      "The old key stays valid for 24 hours.",
        metadata={"source": "api-keys"},
    ),
    Document(
        page_content="The free tier allows 1,000 requests per month. "
                      "Overage requests return a 429 status code.",
        metadata={"source": "billing"},
    ),
    Document(
        page_content="Webhooks retry with exponential backoff for up to 3 attempts "
                      "over 15 minutes before they are marked failed.",
        metadata={"source": "webhooks"},
    ),
]

vector_store = InMemoryVectorStore(OpenAIEmbeddings())
vector_store.add_documents(docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
```

Now wrap that retriever in a tool the agent can call. The docstring tells the model when to reach for it:

```python
from langchain.tools import tool


@tool
def search_docs(query: str) -> str:
    """Search the product documentation. Use this FIRST for any question
    about how the product works, pricing, or configuration."""
    results = retriever.invoke(query)
    if not results:
        return "No relevant documentation found."
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in results)
```

**What just happened.** `InMemoryVectorStore` embeds each document and stores the vectors. `as_retriever` turns it into something you can query by meaning rather than keywords, and `search_kwargs={"k": 2}` returns the two closest passages. Wrapping `retriever.invoke()` in a `@tool` connects the two: the agent gets semantic search over your content, and the returned string keeps each passage's `source` so answers can be cited later. `InMemoryVectorStore` is fine for development; for a real corpus you'd swap in a persistent store like pgvector or Pinecone, with no change to the tool.

---

## Step 3 — Add a live web-search tool

Your docs can't cover everything: new questions, third-party integrations, current events. Give the agent a real fallback with Tavily, a search tool built for LLMs. It's a prebuilt integration, so there's no function to write:

```python
from langchain_tavily import TavilySearch

web_search = TavilySearch(max_results=3)
```

That's a fully-formed tool. `TavilySearch` returns results already trimmed and ranked for an LLM, which is why most people reach for it instead of hand-rolling an HTTP call to a search API.

---

## Step 4 — Assemble the agent with a policy

Two tools, one model. The system prompt sets the policy that keeps a docs assistant honest: prefer your own documentation, and only reach for the open web when the docs genuinely fall short.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_docs, web_search],
    system_prompt=(
        "You are a documentation assistant. Answer using the product docs "
        "whenever possible, and always call search_docs first. Only use web "
        "search if the docs don't contain the answer. Cite the source of "
        "every fact, and never invent product behavior."
    ),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "How long does my old API key stay valid?"}]}
)
print(result["messages"][-1].content)
```

The agent reads the question, calls `search_docs`, finds the `api-keys` passage, and answers from it, with no web search needed. Ask it something outside the docs and it reaches for Tavily instead. That routing is the model's decision; the system prompt is how you steer it.

---

## Step 5 — Return sourced, structured answers

A trustworthy assistant shows its work. Instead of a free-text blob, return a schema that separates the answer from its sources and records whether the open web was used. Define it with Pydantic and pass it as `response_format`:

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent


class SourcedAnswer(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    sources: list[str] = Field(description="Source labels or URLs the answer draws on")
    used_web_search: bool = Field(description="True if web search was used")


agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_docs, web_search],
    system_prompt="You are a documentation assistant. Prefer the docs; cite every source.",
    response_format=SourcedAnswer,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What happens when I exceed the free tier?"}]}
)

answer = result["structured_response"]
print(answer.answer)           # Overage requests return a 429 status code.
print(answer.sources)          # ['billing']
print(answer.used_web_search)  # False
```

**Why this matters.** The validated `SourcedAnswer` lands under `result["structured_response"]`. Because `sources` and `used_web_search` are real fields, your UI can render a citation footer or flag web-sourced answers for review without parsing free text. Each `Field(description=...)` is guidance the model reads while filling the schema, so treat those descriptions as carefully as the prompt itself.

---

## Step 6 — Trace and debug with LangSmith

When an agent picks the wrong tool or answers from the wrong passage, you need to *see* the run: which tool it called, with what arguments, and what came back. LangSmith records exactly that. Turn it on with two environment variables and no code changes:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="lsv2_..."
```

Run the agent again and open your project in the LangSmith dashboard. Each run shows the full loop: the model's tool choices, the passages `search_docs` returned, any Tavily results, and the final structured output. When someone reports that it gave a weird answer, the trace tells you whether retrieval missed, the web fallback fired when it shouldn't have, or the prompt needs tightening. That's the difference between guessing and reading what actually happened.

---

## Putting it together

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch

# 1. Index your documentation.
docs = [
    Document(page_content="To reset an API key, open Settings → API and click Rotate. "
                          "The old key stays valid for 24 hours.", metadata={"source": "api-keys"}),
    Document(page_content="The free tier allows 1,000 requests per month. "
                          "Overage requests return a 429 status code.", metadata={"source": "billing"}),
]
vector_store = InMemoryVectorStore(OpenAIEmbeddings())
vector_store.add_documents(docs)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 2. A retrieval tool over those docs.
@tool
def search_docs(query: str) -> str:
    """Search the product documentation. Use this FIRST for product questions."""
    results = retriever.invoke(query)
    if not results:
        return "No relevant documentation found."
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in results)

# 3. A sourced-answer schema.
class SourcedAnswer(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    sources: list[str] = Field(description="Source labels or URLs the answer draws on")
    used_web_search: bool = Field(description="True if web search was used")

# 4. Assemble, with a real web-search fallback alongside the docs tool.
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_docs, TavilySearch(max_results=3)],
    system_prompt=(
        "You are a documentation assistant. Always call search_docs first. "
        "Only use web search if the docs don't answer. Cite every source."
    ),
    response_format=SourcedAnswer,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do I rotate my API key?"}]}
)
print(result["structured_response"])
```

---

## Common pitfalls

**An embeddings authentication error, even though the chat model works.** Retrieval needs `OPENAI_API_KEY` for the embeddings model, which is a separate credential from `ANTHROPIC_API_KEY`. Chat and embeddings are different services.

**The agent skips `search_docs` and answers from memory.** The docstring is doing double duty as a prompt. Say what the tool is for *and when to reach for it*. "Use this FIRST for product questions" beats a bare "searches docs."

**Retrieval returns nothing relevant.** Confirm your documents were added before you build the retriever, and raise `k` in `search_kwargs` if the corpus is large. Semantic search ranks by meaning, so the phrasing in your docs doesn't have to match the question.

**A Tavily error at startup.** `TavilySearch` needs both the `langchain-tavily` package and `TAVILY_API_KEY` set.

---

## Where to go next

- **Memory**: add a checkpointer and reuse a `thread_id` so follow-up questions keep context across turns.
- **Streaming**: surface tool activity and tokens as they happen instead of waiting for the final answer.
- **Guardrails**: add middleware to require human approval before any write action, or to redact PII before it reaches the model.
- **A persistent vector store**: swap `InMemoryVectorStore` for pgvector or Pinecone so your index survives restarts and scales past a demo corpus.

Each is an incremental addition to the same harness. The loop itself never changes; you're only adjusting the context the model sees or the rails around what it can do.

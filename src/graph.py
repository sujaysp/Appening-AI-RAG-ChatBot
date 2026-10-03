import json
import re
from typing import List, TypedDict

from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from src.config import (
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    LLM_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TOP_K,
)


class AgentState(TypedDict, total=False):
    question: str
    context: List[str]
    context_metadata: List[dict]
    retrieval_scores: List[float]
    relevance_score: float
    answer: str
    confidence_score: float
    is_relevant: bool
    is_grounded: bool


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
    namespace="agentic-ai",
)

client = genai.Client(api_key=GEMINI_API_KEY)


STOP_WORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "from",
    "how",
    "does",
    "do",
    "can",
    "be",
    "as",
    "according",
    "this",
    "that",
    "document",
    "ebook",
    "text",
    "mentioned",
    "according",
}


SYNONYM_GROUPS = [
    {
        "challenge",
        "challenges",
        "limitation",
        "limitations",
        "risk",
        "risks",
        "concern",
        "concerns",
        "barrier",
        "barriers",
        "problem",
        "problems",
    },
    {
        "use",
        "uses",
        "usecase",
        "usecases",
        "application",
        "applications",
        "industry",
        "industries",
        "example",
        "examples",
    },
    {
        "architecture",
        "architectural",
        "component",
        "components",
        "building",
        "blocks",
        "pillar",
        "pillars",
        "layer",
        "layers",
    },
    {
        "definition",
        "define",
        "meaning",
        "refers",
        "means",
        "characteristic",
        "characteristics",
    },
    {
        "difference",
        "differences",
        "differ",
        "comparison",
        "compare",
        "versus",
        "vs",
        "traditional",
        "generative",
        "llm",
    },
    {
        "trust",
        "governance",
        "safety",
        "validation",
        "testing",
        "oversight",
        "compliance",
    },
]


def normalize_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    }


def calculate_lexical_overlap(
    question: str,
    context: List[str],
) -> float:
    question_tokens = normalize_tokens(question)

    if not question_tokens or not context:
        return 0.0

    context_tokens = normalize_tokens(
        " ".join(context)
    )

    direct_overlap = question_tokens.intersection(
        context_tokens
    )

    covered = set(direct_overlap)

    for group in SYNONYM_GROUPS:
        question_group_match = question_tokens.intersection(
            group
        )

        context_group_match = context_tokens.intersection(
            group
        )

        if question_group_match and context_group_match:
            covered.update(question_group_match)

    return len(covered) / len(question_tokens)


def build_retrieval_queries(question: str) -> List[str]:
    question_lower = question.lower().strip()

    queries = [question]

    if any(
        phrase in question_lower
        for phrase in [
            "what is",
            "what are",
            "define",
            "definition",
            "explain",
        ]
    ):
        queries.append(
            f"""
            definition meaning explanation characteristics
            according to the Agentic AI eBook:
            {question}
            """.strip()
        )

    if any(
        phrase in question_lower
        for phrase in [
            "challenge",
            "challenges",
            "limitation",
            "limitations",
            "risk",
            "risks",
            "problem",
            "problems",
            "concern",
            "concerns",
            "barrier",
            "barriers",
        ]
    ):
        queries.append(
            f"""
            challenges limitations risks concerns barriers
            governance trust safety oversight compliance validation
            testing adoption problems associated with Agentic AI:
            {question}
            """.strip()
        )

    if any(
        phrase in question_lower
        for phrase in [
            "use case",
            "use cases",
            "industry",
            "industries",
            "application",
            "applications",
            "real-world",
            "real world",
        ]
    ):
        queries.append(
            f"""
            practical applications real-world examples use cases
            industries manufacturing retail healthcare construction
            pharmaceuticals Agentic AI:
            {question}
            """.strip()
        )

    if any(
        phrase in question_lower
        for phrase in [
            "differ",
            "difference",
            "compare",
            "comparison",
            "versus",
            "vs",
            "traditional",
            "generative ai",
            "llm",
        ]
    ):
        queries.append(
            f"""
            differences comparison traditional AI non-agentic AI
            LLMs generative AI reactive proactive autonomous
            goal-driven systems:
            {question}
            """.strip()
        )

    if any(
        phrase in question_lower
        for phrase in [
            "architecture",
            "architectural",
            "components",
            "building blocks",
            "pillars",
            "system design",
        ]
    ):
        queries.append(
            f"""
            architecture components building blocks core pillars
            perception reasoning planning learning execution
            agent environment actions goals memory:
            {question}
            """.strip()
        )

    if len(queries) == 1:
        queries.append(
            f"according to the Agentic AI eBook: {question}"
        )

    return queries


def retrieve_node(state: AgentState):
    question = state["question"]

    retrieval_queries = build_retrieval_queries(
        question
    )

    unique_results = {}

    for retrieval_query in retrieval_queries:
        results = vector_store.similarity_search_with_score(
            retrieval_query,
            k=TOP_K,
        )

        for document, score in results:
            content_key = document.page_content.strip()
            score = float(score)

            if (
                content_key not in unique_results
                or score > unique_results[content_key][1]
            ):
                unique_results[content_key] = (
                    document,
                    score,
                )

    ranked_results = sorted(
        unique_results.values(),
        key=lambda item: item[1],
        reverse=True,
    )[:TOP_K]

    context = []
    metadata = []
    scores = []

    for document, score in ranked_results:
        context.append(document.page_content)

        metadata.append(
            {
                "page": document.metadata.get("page"),
                "source": document.metadata.get("source"),
            }
        )

        scores.append(score)

    relevance_score = max(scores) if scores else 0.0

    return {
        "context": context,
        "context_metadata": metadata,
        "retrieval_scores": scores,
        "relevance_score": relevance_score,
    }


def parse_json_response(raw_text: str):
    match = re.search(
        r"\{.*\}",
        raw_text,
        re.DOTALL,
    )

    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def relevance_node(state: AgentState):
    context = state.get("context", [])
    question = state["question"]
    retrieval_scores = state.get(
        "retrieval_scores",
        [],
    )

    if not context or not retrieval_scores:
        return {
            "is_relevant": False,
            "confidence_score": 0.0,
        }

    top_similarity = max(retrieval_scores)

    lexical_overlap = calculate_lexical_overlap(
        question,
        context,
    )

    semantic_component = max(
        0.0,
        min(1.0, top_similarity),
    )

    lexical_component = max(
        0.0,
        min(1.0, lexical_overlap),
    )

    combined_score = (
        0.65 * semantic_component
        + 0.35 * lexical_component
    )

    is_relevant = (
        (
            lexical_component >= 0.20
            and top_similarity >= 0.55
        )
        or (
            lexical_component >= 0.10
            and top_similarity >= 0.72
        )
    )

    return {
        "is_relevant": is_relevant,
        "confidence_score": (
            combined_score
            if is_relevant
            else 0.0
        ),
    }


def generate_node(state: AgentState):
    question = state["question"]
    context = state["context"]

    context_text = "\n\n".join(
        f"[Passage {i + 1}]\n{chunk}"
        for i, chunk in enumerate(context)
    )

    prompt = f"""
You are a strict document-grounded assistant.

Answer the user's question using ONLY the retrieved passages.

You MUST NOT:
- use external knowledge
- invent facts
- add unsupported claims
- rely on your general training knowledge

Every important claim in your answer must be supported by
the retrieved passages.

If the passages do not contain enough information, say:
"I cannot answer this based on the provided eBook."

Give a concise and factual answer.

User question:
{question}

Retrieved passages:
{context_text}
"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    return {
        "answer": response.text.strip(),
    }


def refusal_node(state: AgentState):
    return {
        "answer": (
            "I cannot answer this based on the provided eBook."
        ),
        "confidence_score": 0.0,
        "is_grounded": True,
    }


def grounding_node(state: AgentState):
    question = state["question"]
    answer = state.get("answer", "")
    context = state.get("context", [])

    context_text = "\n\n".join(
        f"[Passage {i + 1}]\n{chunk}"
        for i, chunk in enumerate(context)
    )

    prompt = f"""
You are a strict hallucination grader.

Check whether the answer is fully supported by the retrieved
passages.

Use ONLY the retrieved passages.

Return ONLY valid JSON:

{{
  "grounded": true,
  "score": 0.0
}}

Rules:
- grounded=true only when the answer is supported by the passages.
- grounded=false if the answer includes unsupported information.
- score must be between 0.0 and 1.0.
- Do not explain your decision.

Question:
{question}

Answer:
{answer}

Retrieved passages:
{context_text}
"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    result = parse_json_response(
        response.text.strip()
    )

    if not result:
        return {
            "is_grounded": False,
            "confidence_score": 0.0,
        }

    grounded = bool(
        result.get("grounded", False)
    )

    try:
        grounding_score = float(
            result.get("score", 0.0)
        )
    except (TypeError, ValueError):
        grounding_score = 0.0

    grounding_score = max(
        0.0,
        min(1.0, grounding_score),
    )

    relevance_score = float(
        state.get("confidence_score", 0.0)
    )

    final_confidence = min(
        relevance_score,
        grounding_score,
    )

    return {
        "is_grounded": grounded,
        "confidence_score": final_confidence,
    }


def grounded_route(state: AgentState):
    if state.get("is_grounded", False):
        return "end"

    return "refuse"


def relevance_route(state: AgentState):
    if state.get("is_relevant", False):
        return "generate"

    return "refuse"


def build_rag_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node(
        "retrieve",
        retrieve_node,
    )

    workflow.add_node(
        "relevance",
        relevance_node,
    )

    workflow.add_node(
        "generate",
        generate_node,
    )

    workflow.add_node(
        "grounding",
        grounding_node,
    )

    workflow.add_node(
        "refuse",
        refusal_node,
    )

    workflow.add_edge(
        START,
        "retrieve",
    )

    workflow.add_edge(
        "retrieve",
        "relevance",
    )

    workflow.add_conditional_edges(
        "relevance",
        relevance_route,
        {
            "generate": "generate",
            "refuse": "refuse",
        },
    )

    workflow.add_edge(
        "generate",
        "grounding",
    )

    workflow.add_conditional_edges(
        "grounding",
        grounded_route,
        {
            "end": END,
            "refuse": "refuse",
        },
    )

    workflow.add_edge(
        "refuse",
        END,
    )

    return workflow.compile()
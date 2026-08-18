"""Grocery Store FAQ chatbot backed by a local FAISS index."""

from __future__ import annotations

from difflib import get_close_matches
import os
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from chatbot_assets import load_chatbot_config, load_prompt_template
from inventory_utils import load_inventory_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_ROOT / "faiss_index"
CHATBOT_CONFIG = load_chatbot_config()
MESSAGES = CHATBOT_CONFIG.get("messages", {})
SECURITY_REFUSAL_MESSAGE = MESSAGES.get(
    "security_refusal",
    "I can only answer questions about our grocery products, prices, stock, and availability. Please ask a grocery-related question.",
)
SERVICE_UNAVAILABLE_MESSAGE = MESSAGES.get(
    "service_unavailable",
    "I'm sorry, I can't answer that right now because the grocery FAQ service is unavailable. Please try again in a moment.",
)
PROMPT_TEMPLATE = load_prompt_template()
SECURITY_KEYWORD_GROUPS = CHATBOT_CONFIG.get("security_keyword_groups", {})
CODE_MARKERS = tuple(CHATBOT_CONFIG.get("code_markers", ()))


def _import_langchain_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import LangChain dependencies with a clear error if they are missing."""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Missing chatbot dependencies. Install langchain, langchain-community, "
            "langchain-openai, faiss-cpu, and openai before using chatbot.py."
        ) from exc

    return FAISS, PromptTemplate, ChatOpenAI, OpenAIEmbeddings


def _validate_index_path() -> None:
    """Ensure the saved FAISS index exists before trying to load it."""
    expected_files = (INDEX_DIR / "index.faiss", INDEX_DIR / "index.pkl")
    if INDEX_DIR.exists() and all(path.exists() for path in expected_files):
        return

    raise FileNotFoundError(
        "FAISS index not found at "
        f"'{INDEX_DIR}'. Expected both 'index.faiss' and 'index.pkl'. "
        "Create the index first or place the saved index files in ./faiss_index/."
    )


def _build_embeddings(openai_embeddings_cls: Any) -> Any:
    """Create embeddings with an optional env override for compatibility."""
    embedding_model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    if embedding_model:
        try:
            return openai_embeddings_cls(model=embedding_model)
        except TypeError:
            return openai_embeddings_cls(model_name=embedding_model)

    return openai_embeddings_cls()


def _build_llm(chat_openai_cls: Any) -> Any:
    """Create the chat model, defaulting to the current project config."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    try:
        return chat_openai_cls(model=model_name, temperature=0)
    except TypeError:
        return chat_openai_cls(model_name=model_name, temperature=0)


@lru_cache(maxsize=1)
def _build_rag_components() -> tuple[Any, Any, Any, Any]:
    """Build the cached components used for history-aware retrieval and answering."""
    _validate_index_path()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export your OpenAI API key before using the chatbot."
        )

    FAISS, PromptTemplate, ChatOpenAI, OpenAIEmbeddings = _import_langchain_dependencies()

    embeddings = _build_embeddings(OpenAIEmbeddings)
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    answer_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    rewrite_prompt = PromptTemplate(
        template=(
            "You help rewrite grocery inventory chat follow-ups into standalone questions.\n"
            "Use the recent chat history only to resolve references like it, them, those, or that.\n"
            "Preserve the user's intent exactly and keep the rewritten question concise.\n"
            "If the latest question is already standalone, return it unchanged.\n"
            "Return only the rewritten question.\n\n"
            "Chat history:\n{chat_history}\n\n"
            "Latest question:\n{question}\n\n"
            "Standalone question:"
        ),
        input_variables=["chat_history", "question"],
    )
    llm = _build_llm(ChatOpenAI)

    return vector_store, llm, answer_prompt, rewrite_prompt


def _contains_security_violation(question: str) -> bool:
    """Reject requests for internal data or attempts to escape the grocery FAQ role."""
    normalized_question = question.lower()
    for keywords in SECURITY_KEYWORD_GROUPS.values():
        if any(keyword in normalized_question for keyword in keywords):
            return True

    return any(marker.lower() in normalized_question for marker in CODE_MARKERS)


@lru_cache(maxsize=1)
def _load_reference_entities() -> dict[str, list[str]]:
    """Load normalized entity lists to help steer retrieval toward exact dataset names."""
    inventory = load_inventory_dataframe()
    return {
        "product_names": sorted(inventory["Product_Name"].astype(str).unique().tolist()),
        "supplier_names": sorted(inventory["Supplier_Name"].astype(str).unique().tolist()),
        "categories": sorted(inventory["Category"].astype(str).unique().tolist()),
        "statuses": sorted(inventory["Status"].astype(str).unique().tolist()),
    }


def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Keep only recent user and assistant turns in a simple role/content shape."""
    if not isinstance(history, list):
        return []

    normalized_history: list[dict[str, str]] = []
    for item in history[-10:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            normalized_history.append({"role": role, "content": content})

    return normalized_history


def _normalize_lookup_text(text: str) -> str:
    """Normalize free-form text for simple exact and fuzzy matching."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_lookup_phrases(question: str, max_words: int = 5) -> list[str]:
    """Generate candidate phrases for fuzzy matching against known dataset entities."""
    normalized_question = _normalize_lookup_text(question)
    tokens = normalized_question.split()
    if not tokens:
        return []

    phrases = {" ".join(tokens)}
    max_size = min(max_words, len(tokens))
    for size in range(1, max_size + 1):
        for index in range(len(tokens) - size + 1):
            phrases.add(" ".join(tokens[index : index + size]))

    return sorted(phrases, key=lambda phrase: (-len(phrase.split()), -len(phrase)))


def _best_fuzzy_match(question: str, choices: list[str], cutoff: float) -> str | None:
    """Return the best matching dataset entity for the user question."""
    exact_phrases = set(_extract_lookup_phrases(question, max_words=6))
    normalized_choice_map = {
        _normalize_lookup_text(choice): choice
        for choice in choices
        if _normalize_lookup_text(choice)
    }

    for normalized_choice, original_choice in normalized_choice_map.items():
        if normalized_choice and normalized_choice in exact_phrases:
            return original_choice

    for phrase in exact_phrases:
        matches = get_close_matches(
            phrase,
            list(normalized_choice_map.keys()),
            n=1,
            cutoff=cutoff,
        )
        if matches:
            return normalized_choice_map[matches[0]]

    return None


def _match_category(question: str, categories: list[str]) -> str | None:
    """Match a known category name from the user question."""
    lower_question = question.lower()
    for category in categories:
        normalized_category = category.lower()
        singular_category = normalized_category[:-1] if normalized_category.endswith("s") else normalized_category
        if normalized_category in lower_question or singular_category in lower_question:
            return category
    return None


def _match_status(question: str, statuses: list[str]) -> str | None:
    """Match a known inventory status from the user question."""
    lower_question = question.lower()
    for status in statuses:
        if status.lower() in lower_question:
            return status
    return None


def _resolve_query_scope(question: str) -> dict[str, str | None]:
    """Resolve product, supplier, category, and status names for retrieval hints."""
    reference_entities = _load_reference_entities()
    return {
        "product_name": _best_fuzzy_match(question, reference_entities["product_names"], cutoff=0.75),
        "supplier_name": _best_fuzzy_match(question, reference_entities["supplier_names"], cutoff=0.82),
        "category": _match_category(question, reference_entities["categories"]),
        "status": _match_status(question, reference_entities["statuses"]),
    }


def _augment_question_with_scope(question: str, scope: dict[str, str | None]) -> str:
    """Append exact entity hints so retrieval stays anchored to the intended dataset names."""
    hints: list[str] = []
    if scope.get("product_name"):
        hints.append(f"Exact product name: {scope['product_name']}.")
    if scope.get("supplier_name"):
        hints.append(f"Exact supplier name: {scope['supplier_name']}.")
    if scope.get("category"):
        hints.append(f"Exact category: {scope['category']}.")
    if scope.get("status"):
        hints.append(f"Exact status: {scope['status']}.")

    if not hints:
        return question
    return f"{question}\n\n" + "\n".join(hints)


def _canonicalize_question(question: str, scope: dict[str, str | None]) -> str:
    """Normalize broad count/list phrasing into dataset-friendly standalone questions."""
    normalized_question = question.lower()
    category = scope.get("category")
    supplier_name = scope.get("supplier_name")
    product_name = scope.get("product_name")

    if product_name and any(
        marker in normalized_question for marker in ("i meant", "i mean", "sorry", "what about")
    ):
        return f"Tell me about {product_name}."

    if "unique supplier" in normalized_question and (
        "all products" in normalized_question
        or "all items" in normalized_question
        or "overall" in normalized_question
        or "across all" in normalized_question
        or "entire inventory" in normalized_question
    ):
        return "How many unique supplier names are there across the overall grocery inventory?"

    if (
        category
        and "how many" in normalized_question
        and "unique" in normalized_question
        and "supplier" not in normalized_question
        and "product names" not in normalized_question
    ):
        return f"How many unique product names are there in the {category} category?"

    if (
        category
        and any(marker in normalized_question for marker in ("list", "show"))
        and "supplier" not in normalized_question
        and "product ids" not in normalized_question
    ):
        return f"List the unique product names in the {category} category."

    if (
        supplier_name
        and any(marker in normalized_question for marker in ("what items", "what products", "list", "show"))
    ):
        return f"List the unique product names supplied by {supplier_name}."

    if product_name and "how many suppliers" in normalized_question:
        return f"How many unique suppliers does {product_name} have?"

    return question


def _question_requests_ids(question: str) -> bool:
    """Detect whether the user explicitly asked for product or supplier identifiers."""
    normalized_question = question.lower()
    id_markers = (
        "product id",
        "product ids",
        "supplier id",
        "supplier ids",
        "record id",
        "record ids",
        "sku",
        "id number",
        "ids only",
    )
    return any(marker in normalized_question for marker in id_markers)


def _redact_ids(text: str) -> str:
    """Remove product and supplier identifiers from text unless explicitly requested."""
    redacted = re.sub(r"\s*\(Product ID [^)]+\)", "", text, flags=re.IGNORECASE)
    redacted = re.sub(r"\s*\(Supplier ID [^)]+\)", "", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\s*Product IDs?: [^.]+\.?", "", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\s*Supplier IDs?: [^.]+\.?", "", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\s+\.", ".", redacted)
    redacted = re.sub(r" +", " ", redacted)
    redacted = re.sub(r"\n{3,}", "\n\n", redacted)
    return redacted.strip()


def _question_likely_targets_named_entity(question: str) -> bool:
    """Detect short product-like questions that benefit from typo suggestions."""
    normalized_question = question.lower()
    entity_markers = (
        "about",
        "price",
        "cost",
        "expire",
        "expiration",
        "stock",
        "supplier",
        "reorder",
        "status",
        "available",
        "what is",
        "what's",
        "tell me",
        "show me",
        "list",
    )
    return len(_normalize_lookup_text(question).split()) <= 8 or any(
        marker in normalized_question for marker in entity_markers
    )


def _similar_entity_suggestions(question: str, limit: int = 3) -> list[str]:
    """Return likely product-name suggestions for typo-style misses."""
    if not _question_likely_targets_named_entity(question):
        return []

    reference_entities = _load_reference_entities()
    normalized_choice_map = {
        _normalize_lookup_text(choice): choice
        for choice in reference_entities["product_names"]
        if _normalize_lookup_text(choice)
    }
    phrases = _extract_lookup_phrases(question, max_words=6)
    matches: list[str] = []

    for phrase in phrases:
        for match in get_close_matches(phrase, list(normalized_choice_map.keys()), n=limit, cutoff=0.55):
            suggestion = normalized_choice_map[match]
            if suggestion not in matches:
                matches.append(suggestion)

    normalized_question = _normalize_lookup_text(question)
    for normalized_choice, original_choice in normalized_choice_map.items():
        if normalized_choice and (
            normalized_choice in normalized_question or normalized_question in normalized_choice
        ):
            if original_choice not in matches:
                matches.append(original_choice)

    return matches[:limit]


def _answer_indicates_missing_inventory(answer: str) -> bool:
    """Detect model responses that say the requested item was not found."""
    normalized_answer = answer.lower()
    not_found_markers = (
        "could not find",
        "couldn't find",
        "unable to provide",
        "unable to determine",
        "isn't provided",
        "is not provided",
        "isn't in the provided",
        "is not in the provided",
    )
    return any(marker in normalized_answer for marker in not_found_markers)


def _format_suggestion_message(question: str, suggestions: list[str]) -> str:
    """Format a typo-help message with similar product options."""
    original_question = question.strip().strip("\"'“”")
    if not suggestions:
        return f'I could not find "{original_question}" in the grocery inventory data.'

    if len(suggestions) == 1:
        suggestion_text = suggestions[0]
    elif len(suggestions) == 2:
        suggestion_text = f"{suggestions[0]} or {suggestions[1]}"
    else:
        suggestion_text = ", ".join(suggestions[:-1]) + f", or {suggestions[-1]}"

    return (
        f'I could not find "{original_question}" in the grocery inventory data. '
        f"Did you mean {suggestion_text}?"
    )


def _metadata_filter_from_scope(
    scope: dict[str, str | None],
    question: str,
) -> dict[str, str] | None:
    """Build an exact metadata filter when the question clearly names a scope."""
    normalized_question = question.lower()
    if "unique supplier names" in normalized_question and "overall grocery inventory" in normalized_question:
        return {"chunk_type": "inventory_overview"}

    if scope.get("category") and scope.get("status"):
        return {
            "chunk_type": "category_status_summary",
            "category": scope["category"],
            "status": scope["status"],
        }

    if (
        scope.get("category")
        and "unique product names" in normalized_question
    ):
        return {
            "chunk_type": "category_summary",
            "category": scope["category"],
        }

    if (
        scope.get("category")
        and any(marker in normalized_question for marker in ("list the unique product names", "list the product names"))
    ):
        return {
            "chunk_type": "category_summary",
            "category": scope["category"],
        }

    if scope.get("supplier_name") and any(
        marker in normalized_question for marker in ("list the unique product names", "supplied by", "supplier")
    ):
        return {
            "chunk_type": "supplier_summary",
            "supplier_name": scope["supplier_name"],
        }

    if scope.get("product_name") and "suppliers" in normalized_question:
        return {
            "chunk_type": "product_summary",
            "product_name": scope["product_name"],
        }

    metadata_filter = {
        key: value
        for key, value in scope.items()
        if value is not None
    }
    return metadata_filter or None


def _question_needs_context(question: str) -> bool:
    """Detect short follow-ups that likely refer to an earlier turn."""
    normalized_question = " ".join(question.lower().split())
    referential_phrases = (
        "which ones",
        "which one",
        "list them",
        "list those",
        "what about them",
        "what about those",
        "how about them",
        "how about those",
        "and them",
        "and those",
        "for all products",
        "for all items",
        "across all products",
        "across all items",
        "in total",
        "overall",
    )
    referential_terms = {"it", "them", "they", "those", "these", "that", "one", "ones"}
    tokens = normalized_question.replace("?", "").split()

    return any(phrase in normalized_question for phrase in referential_phrases) or (
        len(tokens) <= 8 and bool(set(tokens) & referential_terms)
    )


def _format_history_for_prompt(history: list[dict[str, str]]) -> str:
    """Render recent chat turns into a compact text block for question rewriting."""
    return "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history[-6:]
    )


def _response_to_text(response: Any) -> str:
    """Normalize LangChain/ChatOpenAI responses into plain text."""
    if isinstance(response, dict):
        return str(response.get("result", "")).strip()

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ).strip()

    return str(response).strip()


def _rewrite_question_with_history(
    question: str,
    history: list[dict[str, str]],
    llm: Any,
    rewrite_prompt: Any,
) -> str:
    """Resolve pronouns and short follow-ups into standalone grocery questions."""
    if not history or not _question_needs_context(question):
        return question

    prompt_text = rewrite_prompt.format(
        chat_history=_format_history_for_prompt(history),
        question=question,
    )
    rewritten_question = _response_to_text(llm.invoke(prompt_text)).strip().strip('"')
    return rewritten_question or question


def _select_retrieval_k(question: str) -> int:
    """Use broader retrieval for aggregate or list-style questions."""
    normalized_question = question.lower()
    broad_markers = (
        "all",
        "list",
        "show",
        "unique",
        "names",
        "supplier names",
        "suppliers",
        "how many",
        "which products",
        "what items",
        "what products",
    )
    medium_markers = (
        "category",
        "status",
        "supplier",
        "sales",
        "expire",
        "reorder",
        "availability",
    )

    if any(marker in normalized_question for marker in broad_markers):
        return 10
    if any(marker in normalized_question for marker in medium_markers):
        return 6
    return 4


def _retrieve_context_documents(
    vector_store: Any,
    question: str,
    k: int,
    metadata_filter: dict[str, str] | None = None,
) -> list[Any]:
    """Retrieve relevant inventory chunks with extra diversity for broader questions."""
    try:
        if metadata_filter:
            return vector_store.similarity_search(question, k=k, filter=metadata_filter)
        if k >= 8:
            return vector_store.max_marginal_relevance_search(
                question,
                k=k,
                fetch_k=max(20, k * 2),
            )
        return vector_store.similarity_search(question, k=k)
    except TypeError:
        return vector_store.similarity_search(question, k=k)


def _answer_with_rag(question: str, history: list[dict[str, str]] | None = None) -> str:
    """Answer a grocery question with a history-aware retrieval step."""
    vector_store, llm, answer_prompt, rewrite_prompt = _build_rag_components()
    normalized_history = _normalize_history(history)
    effective_question = _rewrite_question_with_history(
        question,
        normalized_history,
        llm,
        rewrite_prompt,
    )
    scope = _resolve_query_scope(effective_question)
    canonical_question = _canonicalize_question(effective_question, scope)
    scope = _resolve_query_scope(canonical_question)
    scoped_question = _augment_question_with_scope(canonical_question, scope)
    retrieval_k = _select_retrieval_k(canonical_question)
    metadata_filter = _metadata_filter_from_scope(scope, canonical_question)
    documents = _retrieve_context_documents(
        vector_store,
        scoped_question,
        retrieval_k,
        metadata_filter=metadata_filter,
    )
    context = "\n\n".join(document.page_content for document in documents)
    if not _question_requests_ids(canonical_question):
        context = _redact_ids(context)
    answer_prompt_text = answer_prompt.format(context=context, question=canonical_question)
    answer = _response_to_text(llm.invoke(answer_prompt_text))
    if not _question_requests_ids(canonical_question):
        answer = _redact_ids(answer)
    if _answer_indicates_missing_inventory(answer):
        suggestions = _similar_entity_suggestions(canonical_question)
        if suggestions:
            answer = _format_suggestion_message(canonical_question, suggestions)
    print(f"[chatbot] Effective question: {canonical_question}", flush=True)
    print(f"[chatbot] Response: {answer}", flush=True)
    return answer


def get_answer(question: str, history: list[dict[str, str]] | None = None) -> str:
    """Return an answer for a user question using retrieval-augmented generation."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return "Please enter a question."

    if _contains_security_violation(cleaned_question):
        return SECURITY_REFUSAL_MESSAGE

    try:
        return _answer_with_rag(cleaned_question, history=history)
    except Exception:
        return SERVICE_UNAVAILABLE_MESSAGE


if __name__ == "__main__":
    print("Grocery Store FAQ Chatbot")
    print("Type 'exit' or 'quit' to end the session.")
    conversation_history: list[dict[str, str]] = []

    while True:
        try:
            user_question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if user_question.lower() in {"exit", "quit"}:
            print("Bot: Goodbye!")
            break

        if not user_question:
            print("Bot: Please enter a question.")
            continue

        try:
            answer = get_answer(user_question, history=conversation_history)
            print(f"Bot: {answer}")
            conversation_history.append({"role": "user", "content": user_question})
            conversation_history.append({"role": "assistant", "content": answer})
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
        except Exception as exc:  # pragma: no cover - CLI convenience path
            print(f"Bot: Error: {exc}")

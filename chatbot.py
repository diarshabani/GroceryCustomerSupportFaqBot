"""Grocery Store FAQ chatbot backed by a local FAISS index."""

from __future__ import annotations

from difflib import SequenceMatcher, get_close_matches
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
MESSAGES = CHATBOT_CONFIG["messages"]
UNSUPPORTED_QUESTION_MESSAGE = MESSAGES["unsupported_question"]
SECURITY_REFUSAL_MESSAGE = MESSAGES["security_refusal"]
CONTEXT_GUIDANCE_MESSAGE = MESSAGES["context_guidance"]
SUPPORTED_KEYWORDS = CHATBOT_CONFIG["supported_keywords"]
INTENT_PRIORITY = CHATBOT_CONFIG["intent_priority"]
PROMPT_TEMPLATE = load_prompt_template()
QUERY_FILLER_WORDS = set(CHATBOT_CONFIG["query_filler_words"])
SECURITY_KEYWORD_GROUPS = CHATBOT_CONFIG["security_keyword_groups"]
FOLLOW_UP_PHRASES = CHATBOT_CONFIG["follow_up_phrases"]
REFERENTIAL_TERMS = set(CHATBOT_CONFIG["referential_terms"])
CODE_MARKERS = tuple(CHATBOT_CONFIG["code_markers"])


def _import_langchain_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Import LangChain dependencies with a clear error if they are missing."""
    try:
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Missing chatbot dependencies. Install langchain, langchain-community, "
            "langchain-openai, faiss-cpu, and openai before using chatbot.py."
        ) from exc

    return RetrievalQA, FAISS, PromptTemplate, ChatOpenAI, OpenAIEmbeddings


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
    """Create the chat model, defaulting to the assignment spec."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    try:
        return chat_openai_cls(model=model_name, temperature=0)
    except TypeError:
        return chat_openai_cls(model_name=model_name, temperature=0)


@lru_cache(maxsize=1)
def _build_qa_chain() -> Any:
    """Build the RetrievalQA chain once for the module lifecycle."""
    _validate_index_path()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export your OpenAI API key before using the chatbot."
        )

    RetrievalQA, FAISS, PromptTemplate, ChatOpenAI, OpenAIEmbeddings = (
        _import_langchain_dependencies()
    )

    embeddings = _build_embeddings(OpenAIEmbeddings)
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    llm = _build_llm(ChatOpenAI)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )


@lru_cache(maxsize=1)
def _load_fallback_inventory() -> Any:
    """Load the cleaned inventory for a lightweight local fallback path."""
    return load_inventory_dataframe()


def _normalize_lookup_text(text: str) -> str:
    """Normalize free-form user text for direct and fuzzy matching."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _contains_security_violation(question: str) -> bool:
    """Reject messages that mix grocery questions with malicious or internal requests."""
    normalized_question = question.lower()
    for keywords in SECURITY_KEYWORD_GROUPS.values():
        if any(keyword in normalized_question for keyword in keywords):
            return True

    return any(marker.lower() in normalized_question for marker in CODE_MARKERS)


def _normalize_history(history: Any) -> list[dict[str, str]]:
    """Keep only recent chat turns in a simple role/content shape."""
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


def _question_needs_conversation_context(question: str, intent: str | None) -> bool:
    """Detect follow-up questions that depend on an earlier turn."""
    normalized_question = " ".join(question.lower().split())
    normalized_tokens = set(_normalize_lookup_text(question).split())

    if any(phrase in normalized_question for phrase in FOLLOW_UP_PHRASES):
        return True

    if normalized_tokens & REFERENTIAL_TERMS:
        return True

    return False


def _extract_scope(question: str, inventory: Any) -> dict[str, str | None]:
    """Find explicit product/category/supplier/status scope in free-form text."""
    product_names = inventory["Product_Name"].astype(str).tolist()
    categories = sorted(inventory["Category"].astype(str).unique().tolist())
    statuses = sorted(inventory["Status"].astype(str).unique().tolist())
    suppliers = sorted(inventory["Supplier_Name"].astype(str).unique().tolist())

    return {
        "product": _find_product_name(question, product_names),
        "category": _find_category(question, categories),
        "status": _find_status(question, statuses),
        "supplier": _find_supplier(question, suppliers),
    }


def _has_explicit_scope(scope: dict[str, str | None]) -> bool:
    """Check whether a question already names a concrete grocery scope."""
    return any(scope.values())


def _has_context_anchor(scope: dict[str, str | None]) -> bool:
    """Check whether a question names the product/category/supplier to operate on."""
    return any(scope.get(key) for key in ("product", "category", "supplier"))


def _scope_suffix(scope: dict[str, str | None]) -> str:
    """Turn prior-turn scope into explicit hints for the current question."""
    suffix_parts: list[str] = []
    if scope.get("product"):
        suffix_parts.append(f"Product: {scope['product']}.")
    if scope.get("category"):
        suffix_parts.append(f"Category: {scope['category']}.")
    if scope.get("supplier"):
        suffix_parts.append(f"Supplier: {scope['supplier']}.")
    if scope.get("status"):
        suffix_parts.append(f"Status: {scope['status']}.")
    return " ".join(suffix_parts)


def _select_history_scope(
    history: list[dict[str, str]],
    question: str,
    inventory: Any,
) -> tuple[str | None, dict[str, str | None]]:
    """Choose the most useful prior scope for an ambiguous follow-up."""
    normalized_tokens = set(_normalize_lookup_text(question).split())
    plural_markers = {"ones", "they", "them", "those", "these", "items", "products"}
    singular_markers = {"it", "that", "one"}
    prefer_group_scope = bool(normalized_tokens & plural_markers) and not bool(
        normalized_tokens & singular_markers
    )

    scope_priority = (
        ["category", "supplier", "status", "product"]
        if prefer_group_scope
        else ["product", "category", "supplier", "status"]
    )

    extracted_scopes = [
        _extract_scope(turn["content"], inventory)
        for turn in reversed(history)
    ]

    for scope_name in scope_priority:
        for scope in extracted_scopes:
            if scope.get(scope_name):
                return scope_name, scope

    return None, {"product": None, "category": None, "status": None, "supplier": None}


def _build_contextual_question(
    original_question: str,
    intent: str | None,
    scope_name: str | None,
    scope: dict[str, str | None],
) -> str:
    """Rewrite a follow-up question into a fully scoped grocery question."""
    lower_question = original_question.lower()
    explicit_status: str | None = None
    for candidate_status in ("active", "backordered", "discontinued"):
        if candidate_status in lower_question:
            explicit_status = candidate_status
            break

    if scope_name == "product" and scope.get("product"):
        product = scope["product"]
        if intent == "supplier":
            return f"Who supplies {product}?"
        if intent == "stock":
            return f"How many units of {product} are in stock?"
        if intent == "status":
            return f"What is the status of {product}?"
        if intent == "expiration":
            return f"When does {product} expire?"
        if intent == "reorder":
            return f"Does {product} need reordering?"
        if intent == "price":
            return f"What is the price of {product}?"
        return f"Tell me about {product}."

    if scope_name == "category" and scope.get("category"):
        category = scope["category"]
        if intent == "stock":
            return f"Which {category} items are in stock?"
        if intent == "status" and explicit_status:
            return f"Which {category} items are {explicit_status.lower()}?"
        if intent == "status":
            return f"What is the status of {category} items?"
        if intent == "price" and "most expensive" in lower_question:
            return f"What is the most expensive product in {category}?"
        if intent == "price":
            return f"What is the cheapest product in {category}?"
        return f"What products are in the {category} category?"

    if scope_name == "supplier" and scope.get("supplier"):
        supplier = scope["supplier"]
        if intent == "stock":
            return f"Which products from {supplier} are in stock?"
        if intent == "status" and explicit_status:
            return f"Which products from {supplier} are {explicit_status.lower()}?"
        if intent == "price":
            return f"What are the prices of products from {supplier}?"
        return f"What products does {supplier} supply?"

    return f"{original_question} {_scope_suffix(scope)}".strip()


def _resolve_follow_up_question(
    question: str,
    history: list[dict[str, str]],
) -> tuple[str | None, str | None]:
    """Use recent chat turns to make short follow-ups explicit."""
    inventory = _load_fallback_inventory()
    intent = get_supported_intent(question)
    current_scope = _extract_scope(question, inventory)

    if not _question_needs_conversation_context(question, intent):
        return question, None

    if _has_context_anchor(current_scope):
        return question, None

    if not history:
        return None, CONTEXT_GUIDANCE_MESSAGE

    scope_name, prior_scope = _select_history_scope(history, question, inventory)
    if scope_name and _has_explicit_scope(prior_scope):
        rewritten_question = _build_contextual_question(question, intent, scope_name, prior_scope)
        return rewritten_question, None

    return None, CONTEXT_GUIDANCE_MESSAGE


def _extract_lookup_phrases(question: str, max_words: int = 4) -> list[str]:
    """Build candidate phrases by removing common question filler words."""
    normalized_question = _normalize_lookup_text(question)
    raw_tokens = normalized_question.split()
    filtered_tokens = [token for token in raw_tokens if token not in QUERY_FILLER_WORDS]
    candidate_tokens = filtered_tokens or raw_tokens
    if not candidate_tokens:
        return []

    phrases = {" ".join(candidate_tokens)}
    max_size = min(max_words, len(candidate_tokens))
    for size in range(1, max_size + 1):
        for index in range(len(candidate_tokens) - size + 1):
            phrases.add(" ".join(candidate_tokens[index : index + size]))

    return sorted(phrases, key=lambda phrase: (-len(phrase.split()), -len(phrase)))


def _best_fuzzy_match(question: str, choices: list[str], cutoff: float) -> str | None:
    """Match a user question to the closest supported choice."""
    normalized_question = _normalize_lookup_text(question)
    unique_choices = list(dict.fromkeys(choice for choice in choices if choice))
    if not unique_choices:
        return None

    normalized_choice_map = {
        _normalize_lookup_text(choice): choice for choice in unique_choices if _normalize_lookup_text(choice)
    }
    for normalized_choice, original_choice in normalized_choice_map.items():
        if (
            normalized_choice in normalized_question
            or f"{normalized_choice}s" in normalized_question
            or normalized_question in normalized_choice
        ):
            return original_choice

    phrases = _extract_lookup_phrases(question, max_words=5)
    best_choice: str | None = None
    best_score = 0.0

    for phrase in phrases:
        for normalized_choice, original_choice in normalized_choice_map.items():
            score = SequenceMatcher(None, phrase, normalized_choice).ratio()
            if score > best_score:
                best_score = score
                best_choice = original_choice

        close_matches = get_close_matches(
            phrase,
            list(normalized_choice_map.keys()),
            n=1,
            cutoff=cutoff,
        )
        if close_matches:
            return normalized_choice_map[close_matches[0]]

    if best_choice and best_score >= cutoff:
        return best_choice

    return None


def _find_product_name(question: str, product_names: list[str]) -> str | None:
    return _best_fuzzy_match(question, product_names, cutoff=0.72)


def _find_category(question: str, categories: list[str]) -> str | None:
    lower_question = question.lower()
    for category in categories:
        normalized_category = category.lower()
        singular_category = normalized_category[:-1] if normalized_category.endswith("s") else normalized_category
        if normalized_category in lower_question or singular_category in lower_question:
            return category
    return None


def _find_status(question: str, statuses: list[str]) -> str | None:
    lower_question = question.lower()
    for status in statuses:
        if status.lower() in lower_question:
            return status
    return None


def _find_supplier(question: str, suppliers: list[str]) -> str | None:
    return _best_fuzzy_match(question, suppliers, cutoff=0.8)


def _summarize_products(products: list[str], total_count: int) -> str:
    unique_products = list(dict.fromkeys(products))
    if not unique_products:
        return "none"

    preview_items = unique_products[:8]
    preview = ", ".join(preview_items)
    if total_count > len(preview_items):
        return f"{preview}, and {total_count - len(preview_items)} more"
    return preview


def _format_status_breakdown(product_rows: Any) -> str:
    """Format grouped statuses for repeated product records."""
    return ", ".join(
        f"{status} ({count})" for status, count in product_rows["Status"].value_counts().items()
    )


def _format_supplier_list(product_rows: Any) -> str:
    """Format suppliers for repeated product records."""
    suppliers = list(dict.fromkeys(product_rows["Supplier_Name"].astype(str).tolist()))
    return ", ".join(suppliers)


def _get_fallback_answer(question: str) -> str:
    """Answer common inventory questions directly from the cleaned dataset."""
    inventory = _load_fallback_inventory()
    categories = sorted(inventory["Category"].astype(str).unique().tolist())
    statuses = sorted(inventory["Status"].astype(str).unique().tolist())
    suppliers = sorted(inventory["Supplier_Name"].astype(str).unique().tolist())
    intent = get_supported_intent(question) or ""
    product_names = inventory["Product_Name"].astype(str).tolist()

    product_name = None
    if intent in {"price", "stock", "status", "supplier", "product", "expiration", "reorder"}:
        product_name = _find_product_name(question, product_names)
    category = _find_category(question, categories)
    status = _find_status(question, statuses)
    supplier = _find_supplier(question, suppliers)
    lower_question = question.lower()

    if product_name:
        product_rows = inventory.loc[inventory["Product_Name"] == product_name].sort_values(
            by=["Status", "Supplier_Name", "Product_ID"]
        )
        product_row = product_rows.iloc[0]
        record_count = len(product_rows)
        total_stock = int(product_rows["Stock_Quantity"].sum())
        status_breakdown = _format_status_breakdown(product_rows)
        supplier_list = _format_supplier_list(product_rows)
        unique_prices = sorted(product_rows["Unit_Price"].unique().tolist())

        if intent == "price":
            if record_count > 1 and len(unique_prices) > 1:
                return (
                    f"There are {record_count} inventory records for {product_name}. "
                    f"Prices range from ${min(unique_prices):.2f} to ${max(unique_prices):.2f}. "
                    f"Combined stock is {total_stock} units, with statuses: {status_breakdown}."
                )
            return f"The price of {product_name} is ${product_row['Unit_Price']:.2f}."
        if intent == "stock":
            if record_count > 1:
                return (
                    f"There are {record_count} inventory records for {product_name}, totaling "
                    f"{total_stock} units in stock. Statuses: {status_breakdown}."
                )
            return (
                f"{product_name} currently has {int(product_row['Stock_Quantity'])} units in stock "
                f"and is marked as {product_row['Status']}."
            )
        if "category" in lower_question:
            return f"{product_name} belongs to the {product_row['Category']} category."
        if intent == "status":
            if record_count > 1:
                return (
                    f"There are {record_count} inventory records for {product_name}. "
                    f"Statuses are {status_breakdown}, with {total_stock} total units in stock."
                )
            return (
                f"{product_name} is currently marked as {product_row['Status']} with "
                f"{int(product_row['Stock_Quantity'])} units in stock."
            )
        if intent == "supplier":
            if record_count > 1:
                return (
                    f"{product_name} appears in {record_count} inventory records and is supplied by "
                    f"{supplier_list}."
                )
            return f"{product_name} is supplied by {product_row['Supplier_Name']}."
        if intent == "expiration":
            expiration_date = product_row["Expiration_Date"]
            if record_count > 1:
                earliest_expiration = product_rows["Expiration_Date"].min()
                latest_expiration = product_rows["Expiration_Date"].max()
                return (
                    f"{product_name} appears in {record_count} inventory records. "
                    f"Expiration dates range from "
                    f"{earliest_expiration.month}/{earliest_expiration.day}/{earliest_expiration.year} "
                    f"to {latest_expiration.month}/{latest_expiration.day}/{latest_expiration.year}. "
                    f"Statuses: {status_breakdown}."
                )
            return (
                f"{product_name} expires on {expiration_date.month}/{expiration_date.day}/{expiration_date.year} "
                f"and is currently marked as {product_row['Status']}."
            )
        if intent == "reorder":
            reorder_count = int((product_rows["Stock_Quantity"] <= product_rows["Reorder_Level"]).sum())
            if record_count > 1:
                return (
                    f"{product_name} appears in {record_count} inventory records. "
                    f"{reorder_count} of those records are at or below reorder level, and combined stock is "
                    f"{total_stock} units."
                )
            needs_reorder = int(product_row["Stock_Quantity"]) <= int(product_row["Reorder_Level"])
            reorder_text = "does" if needs_reorder else "does not"
            return (
                f"{product_name} {reorder_text} need reordering. "
                f"It has {int(product_row['Stock_Quantity'])} units in stock and a reorder level of "
                f"{int(product_row['Reorder_Level'])}."
            )
        if record_count > 1:
            return (
                f"{product_name} appears in {record_count} inventory records across the "
                f"{product_row['Category']} category. Total stock is {total_stock} units, "
                f"suppliers include {supplier_list}, and statuses are {status_breakdown}."
            )
        return (
            f"{product_name} is in the {product_row['Category']} category, costs "
            f"${product_row['Unit_Price']:.2f}, is supplied by {product_row['Supplier_Name']}, "
            f"and is currently {product_row['Status']} with {int(product_row['Stock_Quantity'])} units in stock."
        )

    filtered = inventory
    if category:
        filtered = filtered.loc[filtered["Category"] == category]
    if status:
        filtered = filtered.loc[filtered["Status"] == status]
    if supplier:
        filtered = filtered.loc[filtered["Supplier_Name"] == supplier]

    if intent == "stock" and not product_name:
        stock_scope = filtered if (category or supplier or status) else inventory
        in_stock_items = stock_scope.loc[stock_scope["Stock_Quantity"] > 0].sort_values(
            by=["Category", "Product_Name", "Product_ID"]
        )
        if in_stock_items.empty:
            if category:
                return f"I could not find any {category} items currently in stock."
            if supplier:
                return f"I could not find any in-stock products from {supplier}."
            return "I could not find any products currently in stock."

        entries = [
            (
                f"{row.Product_Name} (Product ID {row.Product_ID}): "
                f"{int(row.Stock_Quantity)} units in stock; Status {row.Status}"
            )
            for row in in_stock_items.head(8).itertuples(index=False)
        ]
        remaining_count = len(in_stock_items) - len(entries)
        scope_parts: list[str] = []
        if category:
            scope_parts.append(f"in {category}")
        if supplier:
            scope_parts.append(f"from {supplier}")
        scope_text = f" {' '.join(scope_parts)}" if scope_parts else ""
        suffix = f" and {remaining_count} more." if remaining_count > 0 else "."
        return f"Products currently in stock{scope_text}: " + "; ".join(entries) + suffix

    if intent == "supplier" and supplier:
        products = filtered["Product_Name"].astype(str).tolist()
        return (
            f"{supplier} supplies {len(filtered)} products, including "
            f"{_summarize_products(products, len(filtered))}."
        )

    if intent == "sales":
        best_sellers = inventory.sort_values(
            by=["Sales_Volume", "Product_Name"],
            ascending=[False, True],
        ).head(5)
        entries = [
            f"{row.Product_Name} ({int(row.Sales_Volume)} sales, status: {row.Status})"
            for row in best_sellers.itertuples(index=False)
        ]
        return "The best selling products are: " + "; ".join(entries) + "."

    if intent == "expiration":
        expiring_products = inventory.sort_values(by=["Expiration_Date", "Product_Name"]).head(5)
        entries = [
            (
                f"{row.Product_Name} expires on "
                f"{row.Expiration_Date.month}/{row.Expiration_Date.day}/{row.Expiration_Date.year} "
                f"and is currently {row.Status}"
            )
            for row in expiring_products.itertuples(index=False)
        ]
        return "The products expiring soonest are: " + "; ".join(entries) + "."

    if intent == "reorder" and not filtered.empty:
        reorder_items = inventory.loc[
            inventory["Stock_Quantity"] <= inventory["Reorder_Level"]
        ].sort_values(by=["Category", "Product_Name"])
        if reorder_items.empty:
            return "No products are currently below their reorder level."
        products = [
            (
                f"{row.Product_Name} (stock {int(row.Stock_Quantity)}, reorder level "
                f"{int(row.Reorder_Level)}, status: {row.Status})"
            )
            for row in reorder_items.head(8).itertuples(index=False)
        ]
        remaining_count = len(reorder_items) - len(products)
        suffix = f" and {remaining_count} more." if remaining_count > 0 else "."
        return "Products below reorder level: " + "; ".join(products) + suffix

    if intent == "price":
        cheapest_scope = filtered if not filtered.empty else inventory
        sort_columns = ["Unit_Price", "Product_Name"]
        ascending = [True, True]
        descriptor = "cheapest"
        if any(keyword in lower_question for keyword in ["most expensive", "expensive"]):
            ascending = [False, True]
            descriptor = "most expensive"
        selected_product = cheapest_scope.sort_values(by=sort_columns, ascending=ascending).iloc[0]
        scope_text = f" in {category}" if category else ""
        return (
            f"The {descriptor} product{scope_text} is {selected_product['Product_Name']} at "
            f"${selected_product['Unit_Price']:.2f}, and it is currently {selected_product['Status']}."
        )

    if category and "price" in lower_question and not filtered.empty:
        average_price = filtered["Unit_Price"].mean()
        return f"The average price for {category} items is ${average_price:.2f}."

    if category and status:
        if filtered.empty:
            return f"I could not find any {status.lower()} products in {category}."
        products = filtered["Product_Name"].astype(str).tolist()
        return (
            f"There are {len(filtered)} {status.lower()} products in {category}: "
            f"{_summarize_products(products, len(filtered))}."
        )

    if status:
        if filtered.empty:
            return f"I could not find any products with {status.lower()} status."
        products = filtered["Product_Name"].astype(str).tolist()
        return (
            f"The products marked as {status.lower()} are: "
            f"{_summarize_products(products, len(filtered))}."
        )

    if category:
        products = filtered["Product_Name"].astype(str).tolist()
        return (
            f"The dataset includes {len(filtered)} products in {category}, including "
            f"{_summarize_products(products, len(filtered))}."
        )

    return (
        "I could not find that answer in the grocery inventory data. "
        "Please try asking about a product name, category, price, or status."
    )


def get_supported_intent(question: str) -> str | None:
    """Return the matched supported intent for a question, if any."""
    normalized_question = " ".join(question.lower().split())
    quantitative_stock_markers = ["how many", "how much", "quantity", "units", "inventory", "left", "remaining"]
    stock_location_markers = ["stock", "in stock", "inventory", "left", "remaining", "available"]

    if any(marker in normalized_question for marker in SUPPORTED_KEYWORDS["reorder"]):
        return "reorder"
    if any(marker in normalized_question for marker in SUPPORTED_KEYWORDS["expiration"]):
        return "expiration"
    if any(marker in normalized_question for marker in SUPPORTED_KEYWORDS["sales"]):
        return "sales"
    if any(marker in normalized_question for marker in SUPPORTED_KEYWORDS["supplier"]):
        return "supplier"
    if "in stock" in normalized_question:
        return "stock"
    if (
        any(marker in normalized_question for marker in quantitative_stock_markers)
        and any(marker in normalized_question for marker in stock_location_markers)
    ):
        return "stock"

    matched_intents: dict[str, int] = {}

    for intent, keywords in SUPPORTED_KEYWORDS.items():
        matches = sum(keyword in normalized_question for keyword in keywords)
        if matches:
            matched_intents[intent] = matches

    if not matched_intents:
        return None

    ranked_intents = sorted(
        matched_intents.items(),
        key=lambda item: (-item[1], INTENT_PRIORITY.index(item[0])),
    )
    return ranked_intents[0][0]


def is_supported_question(question: str) -> bool:
    """Check whether the question belongs to a supported FAQ intent."""
    return get_supported_intent(question) is not None


def _analyze_question(
    question: str,
    history: list[dict[str, str]] | None,
) -> tuple[str | None, str | None, bool]:
    """Normalize request handling into one small decision pipeline."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return "Please enter a question.", None, False

    normalized_history = _normalize_history(history)
    if _contains_security_violation(cleaned_question):
        return SECURITY_REFUSAL_MESSAGE, None, False

    resolved_question, guidance_message = _resolve_follow_up_question(
        cleaned_question,
        normalized_history,
    )
    if guidance_message:
        return guidance_message, None, False

    effective_question = resolved_question or cleaned_question
    if not is_supported_question(effective_question):
        return UNSUPPORTED_QUESTION_MESSAGE, None, False

    return None, effective_question, effective_question != cleaned_question


def _answer_with_rag(question: str) -> str:
    """Invoke the RetrievalQA chain for non-follow-up questions."""
    response = _build_qa_chain().invoke({"query": question})
    if isinstance(response, dict):
        return str(response.get("result", "")).strip()
    return str(response).strip()


def get_answer(question: str, history: list[dict[str, str]] | None = None) -> str:
    """Return an answer for a user question using retrieval-augmented generation."""
    direct_response, effective_question, used_follow_up_resolution = _analyze_question(
        question,
        history,
    )
    if direct_response is not None:
        return direct_response
    if effective_question is None:
        return "I couldn't find an answer right now."

    if used_follow_up_resolution:
        return _get_fallback_answer(effective_question)

    try:
        return _answer_with_rag(effective_question)
    except Exception:
        return _get_fallback_answer(effective_question)


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
        except Exception as exc:  # pragma: no cover - CLI convenience path
            print(f"Bot: Error: {exc}")

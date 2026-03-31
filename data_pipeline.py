"""Build the cleaned grocery dataset and FAISS index used by the chatbot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from inventory_utils import load_inventory_dataframe, resolve_dataset_path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "faiss_index"


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    """Compatibility wrapper around the shared inventory loader."""
    return load_inventory_dataframe(csv_path)


def build_documents(cleaned_df: pd.DataFrame, source_path: str | Path) -> list[Document]:
    """Turn cleaned grocery records into retrieval-friendly documents."""
    source_path = str(Path(source_path).resolve())
    documents: list[Document] = []

    for row in cleaned_df.itertuples(index=False):
        reorder_gap = int(row.Stock_Gap_To_Reorder_Level)
        if reorder_gap < 0:
            reorder_context = f"Stock is {abs(reorder_gap)} units below the reorder level."
        elif reorder_gap == 0:
            reorder_context = "Stock is exactly at the reorder level."
        else:
            reorder_context = f"Stock is {reorder_gap} units above the reorder level."

        record_text = (
            f"Grocery inventory record for {row.Product_Name} "
            f"(Product ID {row.Product_ID}). "
            f"Category: {row.Category}. "
            f"Supplier: {row.Supplier_Name} (Supplier ID {row.Supplier_ID}). "
            f"Status: {row.Status}. "
            f"Stock quantity: {int(row.Stock_Quantity)} units. "
            f"Reorder level: {int(row.Reorder_Level)} units. "
            f"Reorder quantity: {int(row.Reorder_Quantity)} units. "
            f"Needs reorder: {'yes' if row.Needs_Reorder else 'no'}. "
            f"{reorder_context} "
            f"Unit price: ${row.Unit_Price:.2f}. "
            f"Sales volume: {int(row.Sales_Volume)} units. "
            f"Inventory turnover rate: {int(row.Inventory_Turnover_Rate)}. "
            f"Date received: {row.Date_Received.strftime('%Y-%m-%d')}. "
            f"Last order date: {row.Last_Order_Date.strftime('%Y-%m-%d')}. "
            f"Expiration date: {row.Expiration_Date.strftime('%Y-%m-%d')}. "
            f"Warehouse location: {row.Warehouse_Location}."
        )

        metadata = {
            "chunk_type": "product_record",
            "source_csv": source_path,
            "product_id": row.Product_ID,
            "product_name": row.Product_Name,
            "category": row.Category,
            "supplier_id": row.Supplier_ID,
            "supplier_name": row.Supplier_Name,
            "status": row.Status,
            "needs_reorder": bool(row.Needs_Reorder),
            "warehouse_location": row.Warehouse_Location,
        }
        documents.append(Document(page_content=record_text, metadata=metadata))

    for category, group in cleaned_df.groupby("Category", sort=True):
        status_breakdown = ", ".join(
            f"{status}: {count}" for status, count in group["Status"].value_counts().items()
        )
        sample_products = ", ".join(group["Product_Name"].sort_values().head(5))
        urgent_items = group.sort_values(
            by=["Needs_Reorder", "Stock_Gap_To_Reorder_Level", "Product_Name"],
            ascending=[False, True, True],
        ).head(3)
        urgent_text = "; ".join(
            (
                f"{item.Product_Name} "
                f"(stock {int(item.Stock_Quantity)}, reorder level {int(item.Reorder_Level)})"
            )
            for item in urgent_items.itertuples(index=False)
        )

        summary_text = (
            f"Category summary for {category}. "
            f"There are {len(group)} products in this category. "
            f"Average unit price: ${group['Unit_Price'].mean():.2f}. "
            f"Total stock on hand: {int(group['Stock_Quantity'].sum())} units. "
            f"Average stock quantity: {group['Stock_Quantity'].mean():.1f} units. "
            f"Total sales volume: {int(group['Sales_Volume'].sum())} units. "
            f"Products needing reorder: {int(group['Needs_Reorder'].sum())}. "
            f"Status breakdown: {status_breakdown}. "
            f"Sample products: {sample_products}. "
            f"Most urgent reorder candidates: {urgent_text}."
        )

        documents.append(
            Document(
                page_content=summary_text,
                metadata={
                    "chunk_type": "category_summary",
                    "source_csv": source_path,
                    "category": category,
                    "product_count": int(len(group)),
                },
            )
        )

    for status, group in cleaned_df.groupby("Status", sort=True):
        category_breakdown = ", ".join(
            f"{category}: {count}" for category, count in group["Category"].value_counts().items()
        )
        sample_products = ", ".join(group["Product_Name"].sort_values().head(8))

        summary_text = (
            f"Status summary for {status}. "
            f"There are {len(group)} products with this status. "
            f"Categories represented: {category_breakdown}. "
            f"Average unit price: ${group['Unit_Price'].mean():.2f}. "
            f"Total stock on hand: {int(group['Stock_Quantity'].sum())} units. "
            f"Products needing reorder: {int(group['Needs_Reorder'].sum())}. "
            f"Sample products: {sample_products}."
        )

        documents.append(
            Document(
                page_content=summary_text,
                metadata={
                    "chunk_type": "status_summary",
                    "source_csv": source_path,
                    "status": status,
                    "product_count": int(len(group)),
                },
            )
        )

    return documents


def build_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """Create the embedding client used for FAISS indexing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set before building the FAISS index.")

    embedding_model = (
        model
        or os.getenv("OPENAI_EMBEDDINGS_MODEL")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )
    return OpenAIEmbeddings(model=embedding_model)


def build_vector_store(
    documents: Sequence[Document],
    embeddings: OpenAIEmbeddings,
    index_dir: str | Path | None = None,
    rebuild_index: bool = True,
) -> FAISS:
    """Create or load the local FAISS index."""
    if not documents:
        raise ValueError("At least one document is required to build the FAISS index.")

    target_dir = Path(index_dir or DEFAULT_INDEX_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    index_file = target_dir / "index.faiss"
    store_file = target_dir / "index.pkl"

    if not rebuild_index and index_file.exists() and store_file.exists():
        return FAISS.load_local(
            str(target_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vector_store = FAISS.from_documents(list(documents), embeddings)
    vector_store.save_local(str(target_dir))
    return vector_store


def main(
    csv_path: str | Path | None = None,
    index_dir: str | Path | None = None,
    rebuild_index: bool = True,
) -> tuple[pd.DataFrame, FAISS]:
    """Run the end-to-end data pipeline and return the cleaned dataframe and vector store."""
    dataset_path = resolve_dataset_path(csv_path)
    cleaned_df = load_and_clean_data(dataset_path)
    documents = build_documents(cleaned_df, dataset_path)
    embeddings = build_embeddings()
    vector_store = build_vector_store(
        documents=documents,
        embeddings=embeddings,
        index_dir=index_dir,
        rebuild_index=rebuild_index,
    )
    return cleaned_df, vector_store


if __name__ == "__main__":
    cleaned_df, _ = main()
    print(f"Cleaned {len(cleaned_df)} grocery records.")
    print(f"Saved FAISS index to {(DEFAULT_INDEX_DIR).resolve()}.")

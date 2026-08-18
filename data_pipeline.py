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


def _unique_product_names(group: pd.DataFrame) -> list[str]:
    """Return sorted unique product names for a grouped slice of inventory data."""
    return sorted(group["Product_Name"].astype(str).unique().tolist())


def _format_unique_product_names(group: pd.DataFrame) -> str:
    """Format unique product names as a retrieval-friendly comma-separated list."""
    return ", ".join(_unique_product_names(group))


def _format_unique_suppliers(group: pd.DataFrame) -> str:
    """Format unique supplier names for grouped inventory slices."""
    return ", ".join(sorted(group["Supplier_Name"].astype(str).unique().tolist()))


def _format_categories(group: pd.DataFrame) -> str:
    """Format unique category names for grouped inventory slices."""
    return ", ".join(sorted(group["Category"].astype(str).unique().tolist()))


def _format_status_record_breakdown(group: pd.DataFrame) -> str:
    """Summarize row counts and unique-name counts for each status in a category."""
    parts: list[str] = []
    for status, status_group in group.groupby("Status", sort=True):
        parts.append(
            f"{status}: {len(status_group)} inventory records across "
            f"{status_group['Product_Name'].nunique()} unique product names"
        )
    return "; ".join(parts)


def _format_price_text(group: pd.DataFrame) -> str:
    """Describe the exact price or min/max price range for a group."""
    min_price = float(group["Unit_Price"].min())
    max_price = float(group["Unit_Price"].max())
    if min_price == max_price:
        return f"${min_price:.2f}"
    return f"${min_price:.2f} to ${max_price:.2f}"


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
        unique_names = _unique_product_names(group)
        unique_names_text = ", ".join(unique_names)
        status_breakdown = _format_status_record_breakdown(group)

        summary_text = (
            f"Category summary for {category}. "
            f"There are {len(group)} inventory records in this category across "
            f"{len(unique_names)} unique product names. "
            f"Unique product names in {category}: {unique_names_text}. "
            f"Average unit price: ${group['Unit_Price'].mean():.2f}. "
            f"Total stock on hand: {int(group['Stock_Quantity'].sum())} units. "
            f"Average stock quantity: {group['Stock_Quantity'].mean():.1f} units. "
            f"Total sales volume: {int(group['Sales_Volume'].sum())} units. "
            f"Inventory records needing reorder: {int(group['Needs_Reorder'].sum())}. "
            f"Status breakdown in {category}: {status_breakdown}."
        )

        documents.append(
            Document(
                page_content=summary_text,
                metadata={
                    "chunk_type": "category_summary",
                    "source_csv": source_path,
                    "category": category,
                    "inventory_record_count": int(len(group)),
                    "unique_product_name_count": int(len(unique_names)),
                },
            )
        )

        for status, status_group in group.groupby("Status", sort=True):
            unique_status_names = _format_unique_product_names(status_group)
            documents.append(
                Document(
                    page_content=(
                        f"Category status summary for {category} with status {status}. "
                        f"There are {len(status_group)} {status.lower()} inventory records in {category} "
                        f"across {status_group['Product_Name'].nunique()} unique product names. "
                        f"Unique product names in {category} with status {status}: {unique_status_names}. "
                        f"Total stock on hand for these records: {int(status_group['Stock_Quantity'].sum())} units."
                    ),
                    metadata={
                        "chunk_type": "category_status_summary",
                        "source_csv": source_path,
                        "category": category,
                        "status": status,
                        "inventory_record_count": int(len(status_group)),
                        "unique_product_name_count": int(status_group["Product_Name"].nunique()),
                    },
                )
            )

    for product_name, group in cleaned_df.groupby("Product_Name", sort=True):
        categories_text = _format_categories(group)
        supplier_names = _format_unique_suppliers(group)
        product_ids = ", ".join(sorted(group["Product_ID"].astype(str).tolist()))
        summary_text = (
            f"Product summary for {product_name}. "
            f"{product_name} appears in {len(group)} inventory records across "
            f"{group['Supplier_Name'].nunique()} unique suppliers. "
            f"Categories: {categories_text}. "
            f"Supplier names: {supplier_names}. "
            f"Status breakdown: {_format_status_record_breakdown(group)}. "
            f"Total stock on hand: {int(group['Stock_Quantity'].sum())} units. "
            f"Unit price range: {_format_price_text(group)}. "
            f"Product IDs: {product_ids}."
        )

        documents.append(
            Document(
                page_content=summary_text,
                metadata={
                    "chunk_type": "product_summary",
                    "source_csv": source_path,
                    "product_name": product_name,
                    "inventory_record_count": int(len(group)),
                    "unique_supplier_count": int(group["Supplier_Name"].nunique()),
                },
            )
        )

    for supplier_name, group in cleaned_df.groupby("Supplier_Name", sort=True):
        summary_text = (
            f"Supplier summary for {supplier_name}. "
            f"{supplier_name} supplies {group['Product_Name'].nunique()} unique product names "
            f"across {len(group)} inventory records. "
            f"Product names supplied by {supplier_name}: {_format_unique_product_names(group)}. "
            f"Categories covered by {supplier_name}: {_format_categories(group)}. "
            f"Status breakdown for {supplier_name}: {_format_status_record_breakdown(group)}. "
            f"Total stock on hand from {supplier_name}: {int(group['Stock_Quantity'].sum())} units."
        )

        documents.append(
            Document(
                page_content=summary_text,
                metadata={
                    "chunk_type": "supplier_summary",
                    "source_csv": source_path,
                    "supplier_name": supplier_name,
                    "inventory_record_count": int(len(group)),
                    "unique_product_name_count": int(group["Product_Name"].nunique()),
                },
            )
        )

    for status, group in cleaned_df.groupby("Status", sort=True):
        category_breakdown = ", ".join(
            f"{category}: {count}" for category, count in group["Category"].value_counts().items()
        )
        unique_names_text = _format_unique_product_names(group)

        summary_text = (
            f"Status summary for {status}. "
            f"There are {len(group)} inventory records with this status across "
            f"{group['Product_Name'].nunique()} unique product names. "
            f"Unique product names with status {status}: {unique_names_text}. "
            f"Categories represented: {category_breakdown}. "
            f"Average unit price: ${group['Unit_Price'].mean():.2f}. "
            f"Total stock on hand: {int(group['Stock_Quantity'].sum())} units. "
            f"Inventory records needing reorder: {int(group['Needs_Reorder'].sum())}."
        )

        documents.append(
            Document(
                page_content=summary_text,
                metadata={
                    "chunk_type": "status_summary",
                    "source_csv": source_path,
                    "status": status,
                    "inventory_record_count": int(len(group)),
                    "unique_product_name_count": int(group["Product_Name"].nunique()),
                },
            )
        )

    documents.append(
        Document(
            page_content=(
                "Overall inventory summary. "
                f"There are {len(cleaned_df)} inventory records across "
                f"{cleaned_df['Product_Name'].nunique()} unique product names and "
                f"{cleaned_df['Supplier_Name'].nunique()} unique suppliers. "
                f"Categories represented: {_format_categories(cleaned_df)}. "
                f"Backordered inventory records: {int((cleaned_df['Status'] == 'Backordered').sum())} "
                f"across {cleaned_df.loc[cleaned_df['Status'] == 'Backordered', 'Product_Name'].nunique()} "
                "unique product names. "
                f"Active inventory records: {int((cleaned_df['Status'] == 'Active').sum())}. "
                f"Discontinued inventory records: {int((cleaned_df['Status'] == 'Discontinued').sum())}."
            ),
            metadata={
                "chunk_type": "inventory_overview",
                "source_csv": source_path,
                "inventory_record_count": int(len(cleaned_df)),
                "unique_product_name_count": int(cleaned_df["Product_Name"].nunique()),
                "unique_supplier_count": int(cleaned_df["Supplier_Name"].nunique()),
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

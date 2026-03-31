"""Shared inventory loading and normalization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_FILENAME = "Grocery_Inventory_and_Sales_Dataset.csv"
FALLBACK_DATASET_PATH = Path(
    "/Users/diorock/Library/Application Support/Claude/local-agent-mode-sessions/"
    "9cec455c-971f-498a-a411-ae74e154b5dd/"
    "0d07e82e-39a3-4e36-ae3d-ee4cb2dfa7ee/"
    "local_edeac6c5-9d30-471f-9e08-0bfda5693150/outputs/assignment2/"
    "Grocery_Inventory_and_Sales_Dataset.csv"
)
DISPLAY_COLUMNS = [
    "Product_Name",
    "Category",
    "Unit_Price",
    "Stock_Quantity",
    "Status",
    "Supplier_Name",
    "Sales_Volume",
]
STRING_COLUMNS = [
    "Product_ID",
    "Product_Name",
    "Category",
    "Supplier_ID",
    "Supplier_Name",
    "Warehouse_Location",
    "Status",
]
INTEGER_COLUMNS = [
    "Stock_Quantity",
    "Reorder_Level",
    "Reorder_Quantity",
    "Sales_Volume",
    "Inventory_Turnover_Rate",
]
DATE_COLUMNS = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
REQUIRED_COLUMNS = set(STRING_COLUMNS + INTEGER_COLUMNS + DATE_COLUMNS + ["Unit_Price"])


def resolve_dataset_path(csv_path: str | Path | None = None) -> Path:
    """Prefer the project-local CSV and fall back to the discovered source path."""
    if csv_path is not None:
        candidate = Path(csv_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Dataset not found at {candidate}")

    candidates = [
        PROJECT_ROOT / DATASET_FILENAME,
        Path.cwd() / DATASET_FILENAME,
        FALLBACK_DATASET_PATH,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    candidate_text = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not locate the grocery dataset. Checked:\n{candidate_text}")


def load_inventory_dataframe(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the grocery CSV and normalize it for both UI and retrieval use."""
    dataset_path = resolve_dataset_path(csv_path)
    df = pd.read_csv(dataset_path)
    df.columns = [column.strip() for column in df.columns]

    if "Catagory" in df.columns and "Category" not in df.columns:
        df = df.rename(columns={"Catagory": "Category"})

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")

    for column in STRING_COLUMNS:
        df[column] = df[column].astype("string").str.strip().replace({"": pd.NA})

    df["Category"] = df["Category"].fillna("Unknown")
    df["Warehouse_Location"] = df["Warehouse_Location"].fillna("Unknown warehouse location")
    df["Status"] = df["Status"].fillna("Unknown").str.title()

    unit_price = (
        df["Unit_Price"]
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Unit_Price"] = pd.to_numeric(unit_price, errors="coerce")

    for column in INTEGER_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    required_non_null = [
        "Product_ID",
        "Product_Name",
        "Category",
        "Supplier_ID",
        "Supplier_Name",
        "Warehouse_Location",
        "Status",
        "Unit_Price",
        *INTEGER_COLUMNS,
        *DATE_COLUMNS,
    ]
    df = df.dropna(subset=required_non_null)

    for column in INTEGER_COLUMNS:
        df[column] = df[column].astype(int)

    df["Unit_Price"] = df["Unit_Price"].astype(float).round(2)
    df["Needs_Reorder"] = df["Stock_Quantity"] <= df["Reorder_Level"]
    df["Stock_Gap_To_Reorder_Level"] = df["Stock_Quantity"] - df["Reorder_Level"]

    return (
        df.drop_duplicates(subset=["Product_ID"], keep="last")
        .sort_values(["Category", "Product_Name", "Product_ID"])
        .reset_index(drop=True)
    )


def dataframe_to_display_records(
    cleaned_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert the cleaned dataframe into template-friendly row dictionaries."""
    selected_columns = columns or DISPLAY_COLUMNS
    display_df = cleaned_df.loc[:, selected_columns].copy()
    display_df["Unit_Price"] = display_df["Unit_Price"].map(lambda value: f"${value:.2f}")
    return display_df.to_dict(orient="records")


def load_inventory_records(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the inventory and return template-ready records."""
    return dataframe_to_display_records(load_inventory_dataframe(csv_path))

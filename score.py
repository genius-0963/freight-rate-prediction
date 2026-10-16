"""
Freight Rate Prediction - Official Scorer

This module validates candidate submission files and generates the December 2025
rate forecast chart. It performs strict validation to ensure submissions meet all
requirements before calculating final metrics.

Validation Checks:
1. Predictions file (12,000 samples):
   - Exact column order: load_id, predicted_rate
   - All 12,000 expected load IDs present (no missing or extra)
   - No duplicate or missing load IDs
   - All predicted rates positive

2. December forecast (31 days):
   - Exact column order and 7 columns
   - One row for each day in December 2025
   - Fixed route parameters (Lexington → Fort Wayne, 360mi, Dry Van, 32k lbs)
   - Valid dates covering entire month
   - All predicted rates positive

Output:
- Validation success/failure messages
- candidate_december.png: December 2025 rate forecast chart

Note: Final validation metrics are calculated by Spotter after submission.
This scorer only validates format and generates the required chart.

Author: Spotter AI (provided)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Expected format for validation predictions
EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}

# Expected format for December forecast
DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")
FIXED_PICKUP = "Lexington"
FIXED_DELIVERY = "Fort Wayne"
FIXED_DISTANCE = 360.0
FIXED_EQUIPMENT = "Dry Van"
FIXED_WEIGHT = 32_000.0


# ============================================================================
# ERROR HANDLING
# ============================================================================

def fail(message: str) -> None:
    """Exit with error message when validation fails."""
    raise SystemExit(f"ERROR: {message}")


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def read_csv(path: Path, label: str) -> pd.DataFrame:
    """
    Load CSV file with error handling.

    Args:
        path: Path to CSV file
        label: Descriptive label for error messages

    Returns:
        DataFrame containing loaded data

    Raises:
        SystemExit: If file not found or cannot be read
    """
    if not path.is_file():
        fail(f"{label} file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"could not read {label}: {exc}")


def numeric_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    """
    Convert column to numeric series with validation.

    Args:
        frame: DataFrame containing the column
        column: Column name to convert
        label: Descriptive label for error messages

    Returns:
        Numeric Series (float type)

    Raises:
        SystemExit: If column contains invalid (NaN or infinite) values
    """
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        fail(f"{label} contains invalid {column} values")
    return values.astype(float)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_predictions(predictions: pd.DataFrame) -> None:
    """
    Validate the 12,000 validation predictions format and content.

    Checks:
    - Column order: load_id, predicted_rate (exactly)
    - Row count: exactly 12,000
    - Load IDs: no missing values, no duplicates, all expected IDs present
    - Predicted rates: all positive values

    Args:
        predictions: DataFrame with validation predictions

    Raises:
        SystemExit: If any validation check fails
    """
    # Check column order and count
    if list(predictions.columns) != ["load_id", "predicted_rate"]:
        fail("predictions must contain exactly two columns in this order: load_id,predicted_rate")

    # Check row count
    if len(predictions) != EXPECTED_ROWS:
        fail(f"predictions must contain exactly {EXPECTED_ROWS:,} rows")

    # Check for missing or duplicate load IDs
    if predictions["load_id"].isna().any() or predictions["load_id"].duplicated().any():
        fail("predictions contains missing or duplicate load_id values")

    # Verify all expected load IDs are present and no extra IDs
    submitted_ids = set(predictions["load_id"].astype(str))
    missing = EXPECTED_IDS - submitted_ids
    extra = submitted_ids - EXPECTED_IDS
    if missing or extra:
        fail(
            "prediction IDs do not match the validation set "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    # Check that all predicted rates are positive
    predicted_rate = numeric_series(predictions, "predicted_rate", "predictions")
    if (predicted_rate <= 0).any():
        fail("predictions contains non-positive predicted_rate values")


def validate_december(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the December 2025 forecast format and content.

    Checks:
    - Column order: pickup, delivery, distance, equipment, weight, date, predicted_rate (exactly)
    - Dates: valid, cover all 31 days of December 2025, no duplicates
    - Fixed route parameters: Lexington → Fort Wayne, 360mi, Dry Van, 32k lbs
    - Numeric fields: valid finite values
    - Predicted rates: all positive

    Args:
        frame: DataFrame with December predictions

    Returns:
        DataFrame sorted by date

    Raises:
        SystemExit: If any validation check fails
    """
    # Check column order and count
    columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(frame.columns) != columns:
        fail("December predictions must keep the original seven columns and column order")

    result = frame.copy()

    # Validate dates
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        fail("December predictions contains invalid dates")

    # Validate numeric fields
    result["distance"] = numeric_series(result, "distance", "December predictions")
    result["weight"] = numeric_series(result, "weight", "December predictions")
    result["predicted_rate"] = numeric_series(result, "predicted_rate", "December predictions")

    # Check for duplicate dates
    if result["date"].duplicated().any():
        fail("December predictions contains duplicate dates")

    # Verify date coverage (exactly 31 days, Dec 1-31, 2025)
    if len(result) != 31 or set(result["date"]) != set(DECEMBER_DATES):
        fail("December predictions must contain one row for every day from 2025-12-01 to 2025-12-31")

    # Verify fixed route parameters
    if not result["pickup"].eq(FIXED_PICKUP).all():
        fail(f"December pickup must be {FIXED_PICKUP} for all rows")
    if not result["delivery"].eq(FIXED_DELIVERY).all():
        fail(f"December delivery must be {FIXED_DELIVERY} for all rows")
    if not np.isclose(result["distance"], FIXED_DISTANCE).all():
        fail(f"December distance must be {FIXED_DISTANCE:g} for all rows")
    if not result["equipment"].eq(FIXED_EQUIPMENT).all():
        fail(f"December equipment must be {FIXED_EQUIPMENT} for all rows")
    if not np.isclose(result["weight"], FIXED_WEIGHT).all():
        fail(f"December weight must be {FIXED_WEIGHT:g} for all rows")

    # Check that all predicted rates are positive
    if (result["predicted_rate"] <= 0).any():
        fail("December predicted_rate values must be positive")

    return result.sort_values("date")


def save_december_chart(december: pd.DataFrame, output: Path) -> None:
    """
    Generate and save the December 2025 rate forecast chart.

    The chart shows predicted rates for each day in December 2025 on the fixed
    Lexington → Fort Wayne route. Visualization includes:
    - Line plot with markers for daily rates
    - Fill under the curve for visual emphasis
    - Professional styling with grid and axis formatting
    - Descriptive title and subtitle with fixed route parameters

    Args:
        december: DataFrame with validated December predictions (sorted by date)
        output: Path where to save the chart image (PNG format)
    """
    # Create figure with specified dimensions and DPI for high resolution
    figure, axis = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    color = "#064A56"

    # Plot predicted rates as line with markers
    axis.plot(
        december["date"],
        december["predicted_rate"],
        color=color,
        linewidth=2.6,
        marker="o",
        markersize=3.2,
    )

    # Add fill under the curve for visual emphasis
    floor = float(december["predicted_rate"].min())
    axis.fill_between(
        december["date"],
        december["predicted_rate"],
        floor - max(10.0, floor * 0.02),
        color=color,
        alpha=0.08,
    )

    # Chart styling
    axis.set_title("Candidate: December 2025 Predicted Load Rate", loc="left", fontsize=15, fontweight="bold", pad=12)
    axis.set_ylabel("Predicted rate ($)")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9DAFB3")
    axis.tick_params(axis="x", rotation=35)

    # Add subtitle with fixed route parameters
    axis.text(
        0,
        -0.40,
        "Fixed inputs: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#455A60",
    )

    # Save chart with tight layout
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main entry point for the scorer.

    Validates candidate submission files and generates the December chart.
    This is a command-line tool that:
    1. Loads and validates the 12,000 validation predictions
    2. Loads and validates the 31 December forecast predictions
    3. Generates the December 2025 rate chart image
    4. Reports validation status

    Usage:
        python score.py --predictions validation_predictions.csv \
                        --december-predictions data/december_chart_inputs.csv \
                        --output-dir scorer_results
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Validate candidate output files and generate the fixed December chart."
    )
    parser.add_argument("--predictions", required=True, help="CSV with load_id,predicted_rate")
    parser.add_argument(
        "--december-predictions",
        required=True,
        help="Completed data/december_chart_inputs.csv",
    )
    parser.add_argument("--output-dir", default="scorer_results")
    args = parser.parse_args()

    # Validate predictions file
    validate_predictions(read_csv(Path(args.predictions), "predictions"))

    # Validate December predictions file
    december = validate_december(read_csv(Path(args.december_predictions), "December predictions"))

    # Create output directory
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Generate and save December chart
    chart = output / "candidate_december.png"
    save_december_chart(december, chart)

    # Report validation results
    print(f"Validated {EXPECTED_ROWS:,} final predictions.")
    print("Validated 31 fixed December predictions.")
    print(f"Created chart: {chart}")
    print("Final validation metrics are calculated by Spotter after submission.")


if __name__ == "__main__":
    main()

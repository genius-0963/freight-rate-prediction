"""
Freight Rate Prediction - Validation Script

This module performs comprehensive validation of generated predictions before submission.
It checks format compliance, data integrity, and business logic constraints.

Validation Checks:
1. Validation predictions (12,000 samples):
   - Correct row count
   - Required columns (load_id, predicted_rate)
   - All expected load IDs present (no missing or extra)
   - All predicted rates positive

2. December forecast (31 days):
   - Correct row count (31 days)
   - Required columns (7 columns in specific order)
   - Fixed route parameters (Lexington → Fort Wayne, 360mi, Dry Van, 32k lbs)
   - All predicted rates positive

This is a pre-submission sanity check; final validation is performed by Spotter's scorer.

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import numpy as np

# ============================================================================
# VALIDATION PREDICTIONS CHECK
# ============================================================================

print("Validating validation_predictions.csv...")
val_pred = pd.read_csv('validation_predictions.csv')

# Expected format for validation set
EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}

print(f"Rows: {len(val_pred)} (expected: {EXPECTED_ROWS})")
print(f"Columns: {list(val_pred.columns)}")

# Check for missing or extra load IDs
submitted_ids = set(val_pred['load_id'].astype(str))
missing = EXPECTED_IDS - submitted_ids
extra = submitted_ids - EXPECTED_IDS

print(f"Missing IDs: {len(missing)}")
print(f"Extra IDs: {len(extra)}")

# Business logic: Freight rates must be positive
if (val_pred['predicted_rate'] <= 0).any():
    print("ERROR: Contains non-positive predicted_rate values")
else:
    print("All predicted_rate values are positive")

# Summary statistics for sanity check
print(f"Predicted rate range: ${val_pred['predicted_rate'].min():.2f} - ${val_pred['predicted_rate'].max():.2f}")
print(f"Mean predicted rate: ${val_pred['predicted_rate'].mean():.2f}")

# ============================================================================
# DECEMBER FORECAST CHECK
# ============================================================================

print("\nValidating data/december_chart_inputs.csv...")
december = pd.read_csv('data/december_chart_inputs.csv')

# Expected format for December forecast
print(f"Rows: {len(december)} (expected: 31)")
print(f"Columns: {list(december.columns)}")

# Verify fixed route parameters (should be constant across all 31 days)
print(f"Pickup: {december['pickup'].unique()}")
print(f"Delivery: {december['delivery'].unique()}")
print(f"Distance: {december['distance'].unique()}")
print(f"Equipment: {december['equipment'].unique()}")
print(f"Weight: {december['weight'].unique()}")

# Business logic: All predictions must be positive
if (december['predicted_rate'] <= 0).any():
    print("ERROR: Contains non-positive predicted_rate values")
else:
    print("All predicted_rate values are positive")

# Summary statistics for December forecast
print(f"December predicted rate range: ${december['predicted_rate'].min():.2f} - ${december['predicted_rate'].max():.2f}")
print(f"Mean December predicted rate: ${december['predicted_rate'].mean():.2f}")

print("\n✓ Validation complete!")

import pandas as pd
import numpy as np

print("Validating validation_predictions.csv...")
val_pred = pd.read_csv('validation_predictions.csv')

EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}

print(f"Rows: {len(val_pred)} (expected: {EXPECTED_ROWS})")
print(f"Columns: {list(val_pred.columns)}")

submitted_ids = set(val_pred['load_id'].astype(str))
missing = EXPECTED_IDS - submitted_ids
extra = submitted_ids - EXPECTED_IDS

print(f"Missing IDs: {len(missing)}")
print(f"Extra IDs: {len(extra)}")

if (val_pred['predicted_rate'] <= 0).any():
    print("ERROR: Contains non-positive predicted_rate values")
else:
    print("All predicted_rate values are positive")

print(f"Predicted rate range: ${val_pred['predicted_rate'].min():.2f} - ${val_pred['predicted_rate'].max():.2f}")
print(f"Mean predicted rate: ${val_pred['predicted_rate'].mean():.2f}")

print("\nValidating data/december_chart_inputs.csv...")
december = pd.read_csv('data/december_chart_inputs.csv')

print(f"Rows: {len(december)} (expected: 31)")
print(f"Columns: {list(december.columns)}")

print(f"Pickup: {december['pickup'].unique()}")
print(f"Delivery: {december['delivery'].unique()}")
print(f"Distance: {december['distance'].unique()}")
print(f"Equipment: {december['equipment'].unique()}")
print(f"Weight: {december['weight'].unique()}")

if (december['predicted_rate'] <= 0).any():
    print("ERROR: Contains non-positive predicted_rate values")
else:
    print("All predicted_rate values are positive")

print(f"December predicted rate range: ${december['predicted_rate'].min():.2f} - ${december['predicted_rate'].max():.2f}")
print(f"Mean December predicted rate: ${december['predicted_rate'].mean():.2f}")

print("\n✓ Validation complete!")

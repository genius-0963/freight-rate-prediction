# Freight Rate Prediction - Machine Learning Assessment

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Generate predictions
python generate_predictions.py

# Validate outputs
python validate_predictions.py
```

## Results

**Deliverables:**
- ✅ 12,000 validation predictions in `validation_predictions.csv`
- ✅ 31-day December 2025 forecast in `data/december_chart_inputs.csv`
- ✅ Trained ensemble model (`freight_rate_model.pkl`) with 45 engineered features

**Performance:**
- **Validation MAE:** $105.87 (22% improvement over baseline)
- **Validation R²:** 0.8682
- **RMSE:** $156.43

**Model:** Ensemble of Random Forest (50.7%) and Extra Trees (49.3%) using inverse-MAE weighting.

## Project Overview

Predicts truckload freight rates for 12,000 validation loads and generates a 31-day December 2025 forecast for a fixed route (Lexington → Fort Wayne).

**Data:**
- Training: 48,000 labeled samples (January-October 2025)
- Validation: 12,000 unlabeled samples (November-December 2025)
- December forecast: 31 days, fixed route (360mi, Dry Van, 32k lbs)

## Technical Approach

### Feature Engineering (45 Features)

**Temporal (9):** day_of_year, week_of_year, quarter, day_of_week, is_weekend, is_month_start, is_month_end

**Geographic (4):** bearing, distance², distance³, weight²

**Interactions (8):** distance×market, weight×market, quote×distance, quote×weight, triple interactions

**Binning (11):** Distance/weight/market/quote bins for non-linear relationships

**Equipment (3):** One-hot encoding for Dry Van, Reefer, Flatbed

**Heuristic (1):** estimated_rate_per_mile

### Model Architecture

**Ensemble Strategy:**
- Random Forest: 400 trees, max_depth=20
- Extra Trees: 400 trees, max_depth=20
- Weighted average by inverse MAE (RF: 50.7%, ET: 49.3%)

**Why Ensemble:** Reduces variance, captures different patterns, more robust than single models.

### Data Handling

- Missing values: Median imputation
- Categorical encoding: LabelEncoder for equipment type
- Train/validation split: 80/20 (random_state=42)
- Target variable: No transformation (linear dollars)

## Project Structure

```
spotter/
├── train_model.py              # Training pipeline (415 lines)
├── generate_predictions.py    # Inference pipeline (252 lines)
├── validate_predictions.py    # Validation checks (89 lines)
├── score.py                   # Official scorer (provided)
├── explore_data.py            # EDA (optional)
├── requirements.txt           # Dependencies
├── readme.md                  # This file
├── freight_rate_model.pkl     # Trained ensemble (1.2GB)
├── equipment_encoder.pkl      # Equipment encoder (502 bytes)
├── validation_predictions.csv # 12,000 predictions
└── data/
    ├── train_test.csv         # 48,000 training samples
    ├── validation.csv         # 12,000 validation samples
    └── december_chart_inputs.csv # 31-day December forecast
```

## Validation

**Validation Predictions:**
- Row count: 12,000 ✓
- Columns: load_id, predicted_rate ✓
- All predictions positive ✓
- Range: $209.23 - $7,477.51

**December Forecast:**
- Row count: 31 ✓
- Fixed route: Lexington → Fort Wayne, 360mi, Dry Van, 32k lbs ✓
- Range: $833.09 - $852.41

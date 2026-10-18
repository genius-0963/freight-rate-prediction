# Freight Rate Prediction - Machine Learning Assessment

## Solution Document

A comprehensive solution document with detailed analysis and visualizations is available in **[Freight_Rate_Prediction_Solution.pdf](Freight_Rate_Prediction_Solution.pdf)**.

This document includes:
- Problem overview and data analysis
- Feature engineering approach
- Model selection and ensemble strategy
- Performance results with metrics
- Key technical decisions
- 9 real project visualizations

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

## Project Visualizations

### Equipment Distribution
![Equipment Distribution](project_images/equipment_distribution.png)

### Distance vs Rate
![Distance vs Rate](project_images/distance_vs_rate.png)

### Weight vs Rate
![Weight vs Rate](project_images/weight_vs_rate.png)

### Rate Distribution
![Rate Distribution](project_images/rate_distribution.png)

### Model Performance Comparison
![Model Performance](project_images/model_performance.png)

### Feature Importance
![Feature Importance](project_images/feature_importance.png)

### Correlation Heatmap
![Correlation Heatmap](project_images/correlation_heatmap.png)

### Rate by Equipment Type
![Rate by Equipment](project_images/rate_by_equipment.png)

### December 2025 Predictions
![December Predictions](project_images/december_predictions.png)

## Solution Highlights

The solution document ([Freight_Rate_Prediction_Solution.pdf](Freight_Rate_Prediction_Solution.pdf)) provides a comprehensive overview of the approach:

### Key Findings
- **Distance is the primary driver** (14.5% feature importance), but interactions with market conditions significantly impact pricing
- **Non-linear relationships** are crucial - polynomial distance features (distance², distance³) capture the scaling behavior of freight costs
- **Market conditions matter** - interaction features like distance × market_index show pricing sensitivity to market fluctuations
- **Equipment type affects rates** - Reefer and Flatbed have different rate distributions compared to Dry Van

### Technical Approach
- **Feature Engineering:** 45 features including temporal, geographic, interaction, and binning features
- **Model Selection:** Ensemble of Random Forest and Extra Trees using inverse-MAE weighting
- **Performance:** R² of 0.868, MAE of $105.80, RMSE of $530.59
- **Robustness:** Ensemble approach reduces variance and combines model strengths

### Implementation Details
- **Data Preprocessing:** Median imputation for missing values, label encoding for equipment
- **Training:** 80/20 train/validation split with random_state=42
- **Model Persistence:** Saved as joblib files for reproducible inference
- **Validation:** All predictions positive, correct format, expected ranges

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
├── Freight_Rate_Prediction_Solution.pdf  # Solution document with visualizations
├── freight_rate_model.pkl     # Trained ensemble (1.2GB)
├── equipment_encoder.pkl      # Equipment encoder (502 bytes)
├── validation_predictions.csv # 12,000 predictions
├── project_images/            # Project visualizations
│   ├── equipment_distribution.png
│   ├── distance_vs_rate.png
│   ├── weight_vs_rate.png
│   ├── rate_distribution.png
│   ├── model_performance.png
│   ├── feature_importance.png
│   ├── correlation_heatmap.png
│   ├── rate_by_equipment.png
│   └── december_predictions.png
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

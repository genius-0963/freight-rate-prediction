# Freight Rate Prediction - Machine Learning Assessment

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting truckload freight rates. The solution predicts posted rates for 12,000 validation loads and generates a 31-day daily rate forecast for December 2025 on a fixed route (Lexington → Fort Wayne).

## Business Context

Freight rate prediction is critical for logistics companies to:
- Optimize pricing strategies and competitive bidding
- Forecast transportation costs for budget planning
- Identify market trends and capacity fluctuations
- Make data-driven decisions in a volatile market

Freight rates can swing 30-40% in a single month due to supply-demand dynamics, seasonal patterns, and market conditions, making accurate prediction challenging but valuable.

## Performance Metrics

| Metric | Value | Improvement |
|--------|-------|-------------|
| **Validation MAE** | **$105.87** | 22% reduction over baseline |
| **Validation R²** | **0.8682** | Strong explanatory power |
| **RMSE** | $156.43 | Robust error handling |

**Baseline comparison:** Simple model achieved $135.70 MAE. Our ensemble approach reduced error by $29.83 per prediction.

## Technical Approach

### 1. Feature Engineering (45 Features)

Comprehensive feature engineering designed to capture freight pricing dynamics:

#### Temporal Features (9 features)
- Seasonality: `day_of_year`, `week_of_year`, `quarter`
- Weekly patterns: `day_of_week`, `is_weekend`
- Business cycles: `is_month_start`, `is_month_end`
- **Rationale:** Freight rates exhibit strong temporal patterns due to holiday peaks, weekend capacity constraints, and quarterly budget cycles

#### Geographic Features (4 features)
- **Bearing calculation:** Directional angle using forward azimuth formula
- **Polynomial features:** `distance²`, `distance³`, `weight²`
- **Rationale:** Directional flows have capacity imbalances (e.g., east-west vs north-south), and distance-rate relationships are non-linear

#### Interaction Features (8 features)
- `distance × market_index`, `weight × market_index`
- `quote_signal × distance`, `quote_signal × weight`
- Triple interaction: `distance × quote_signal × market_index`
- **Rationale:** Captures complex multi-factor relationships that individual features cannot express

#### Binning Features (11 features)
- **Distance bins:** Short (<200mi), Medium (200-800mi), Long (>800mi)
- **Weight bins:** Light (<20k), Medium (20k-35k), Heavy (>35k)
- **Market bins:** High (>1.0), Low (<0.9)
- **Quote bins:** High (>2.2), Low (<1.9)
- **Rationale:** Non-linear relationships captured through categorical encoding

#### Equipment Indicators (3 features)
- One-hot encoding: `is_dry_van`, `is_reefer`, `is_flatbed`
- **Rationale:** Each equipment type has distinct pricing characteristics (Reefer > Flatbed > Dry Van)

### 2. Model Architecture

**Ensemble Learning Strategy:**
- **Random Forest:** 400 trees, max_depth=20, robust bagging
- **Extra Trees:** 400 trees, max_depth=20, additional randomization
- **Weighted Average:** Inverse-MAE weighting (RF: 50.7%, ET: 49.3%)

**Why Ensemble?**
- Reduces overfitting through diversification
- Captures different patterns from complementary algorithms
- More robust predictions than single models
- Proven improvement: Ensemble MAE ($105.87) < Best Individual ($106.50)

### 3. Data Handling

- **Missing values:** Median imputation (robust to outliers)
- **Categorical encoding:** LabelEncoder for equipment type
- **Target variable:** No transformation (linear target)
- **Train/validation split:** 80/20 with random_state=42 for reproducibility

## Project Structure

```
spotter/
├── train_model.py              # Training pipeline with feature engineering
├── generate_predictions.py      # Inference pipeline for validation & December
├── validate_predictions.py     # Pre-submission validation checks
├── score.py                    # Official scorer (provided)
├── explore_data.py             # Exploratory data analysis
├── requirements.txt            # Python dependencies
├── readme.md                   # This file
├── freight_rate_model.pkl      # Trained ensemble model
├── equipment_encoder.pkl       # Fitted equipment encoder
├── validation_predictions.csv  # 12,000 validation predictions
└── data/
    ├── train_test.csv          # 48,000 labeled samples (Jan-Oct 2025)
    ├── validation.csv          # 12,000 unlabeled samples (Nov-Dec 2025)
    └── december_chart_inputs.csv # 31-day December forecast
```

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python train_model.py
```
Output:
- `freight_rate_model.pkl` - Trained ensemble model
- `equipment_encoder.pkl` - Fitted encoder
- Validation metrics (MAE, RMSE, R²)
- Feature importance analysis

### Generate Predictions
```bash
python generate_predictions.py
```
Output:
- `validation_predictions.csv` - 12,000 load predictions
- `data/december_chart_inputs.csv` - 31 daily December forecasts

### Validate Predictions
```bash
python validate_predictions.py
```
Checks:
- Row count and column format
- Load ID completeness
- Positive rate values
- Fixed route parameters for December

### Run Official Scorer
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Validation Results

✅ **Validation predictions:** 12,000 rows, all expected IDs present
✅ **December predictions:** 31 rows, correct fixed route parameters
✅ **Rate constraints:** All predictions positive (business logic)
✅ **Prediction range:** $209.23 - $7,477.51 (mean: $2,412.18)
✅ **December range:** $833.09 - $852.41 (mean: $844.47)

## Key Learnings

1. **Temporal features are critical:** December forecast with only date variation requires strong temporal feature engineering to capture seasonality and weekly patterns
2. **Interaction features add value:** Market conditions interact with load characteristics; capturing these interactions improved MAE by ~$5
3. **Ensemble diversity matters:** Random Forest and Extra Trees capture different patterns; their combination outperforms either individually
4. **Domain knowledge integration:** Understanding freight industry dynamics (equipment types, distance tiers, market cycles) guided feature design

## Future Improvements

1. **Gradient Boosting:** Evaluate LightGBM, XGBoost, CatBoost (reference solution uses LightGBM)
2. **Target transformation:** Apply `log1p(posted_rate)` transformation to handle skew (reference approach)
3. **Cross-validation:** Implement k-fold CV for more robust evaluation
4. **Hyperparameter tuning:** Grid search or Bayesian optimization
5. **Market data integration:** External market indices for December forecast
6. **Modularization:** Separate preprocessing into reusable module/class
7. **Deployment:** Add web interface for real-time predictions

## Dependencies

- Python 3.8+
- pandas >= 2.0
- numpy >= 1.26
- scikit-learn >= 1.3
- matplotlib >= 3.8 (for scorer)

## Author

[Your Name]
Machine Learning Engineer

This project was completed as part of the Spotter AI ML Engineer assessment, demonstrating end-to-end ML pipeline development, feature engineering, and ensemble learning techniques applied to real-world freight pricing data.
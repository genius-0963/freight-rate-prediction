"""
Freight Rate Prediction - Inference Pipeline

This module generates predictions for validation and December forecast datasets
using the trained ensemble model. It applies the same feature engineering pipeline
used during training to ensure consistent preprocessing.

Key Responsibilities:
- Load trained model and preprocessing artifacts
- Apply identical feature engineering to new data
- Generate predictions for 12,000 validation samples
- Generate December 2025 daily rate forecasts (31 days)
- Handle special case for December data (fixed route, only date varies)

Output Files:
- validation_predictions.csv: 12,000 load predictions (load_id, predicted_rate)
- data/december_chart_inputs.csv: 31 daily predictions for December 2025

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import numpy as np
import joblib

# ============================================================================
# MODEL LOADING
# ============================================================================

# Load trained model (could be ensemble or single model)
model_data = joblib.load('freight_rate_model.pkl')
equipment_encoder = joblib.load('equipment_encoder.pkl')

# Detect model type: ensemble (dict with 'models' key) or single model
is_ensemble = isinstance(model_data, dict) and 'models' in model_data
if is_ensemble:
    print("Loading ensemble model...")
    ensemble_config = model_data
else:
    print("Loading single model...")
    single_model = model_data

# ============================================================================
# FEATURE ENGINEERING (INFERENCE)
# ============================================================================

def preprocess_features(df, equipment_encoder):
    """
    Apply the same feature engineering pipeline used during training.
    CRITICAL: This must match train_model.py exactly to ensure consistent preprocessing.

    Args:
        df: Raw DataFrame with freight data
        equipment_encoder: Pre-fitted LabelEncoder from training

    Returns:
        X: Feature matrix (45 features)
    """
    df = df.copy()

    # Temporal features (same as training)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(float)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter

    # Encode equipment type using fitted encoder
    df['equipment_encoded'] = equipment_encoder.transform(df['equipment'].astype(str))

    # Fill missing values with median (same as training)
    df['weight'] = df['weight'].fillna(df['weight'].median())
    df['market_index'] = df['market_index'].fillna(df['market_index'].median())

    # Geographic features - bearing calculation
    lat1, lon1 = np.radians(df['pickup_lat']), np.radians(df['pickup_lon'])
    lat2, lon2 = np.radians(df['delivery_lat']), np.radians(df['delivery_lon'])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df['bearing'] = np.arctan2(x, y)
    df['bearing'] = np.degrees(df['bearing'])
    df['bearing'] = (df['bearing'] + 360) % 360

    # Polynomial features
    df['distance_squared'] = df['distance'] ** 2
    df['distance_cubed'] = df['distance'] ** 3
    df['weight_squared'] = df['weight'] ** 2

    # Interaction features
    df['distance_per_weight'] = df['distance'] / (df['weight'] + 1)
    df['weight_per_mile'] = df['weight'] / (df['distance'] + 1)
    df['distance_x_market'] = df['distance'] * df['market_index']
    df['weight_x_market'] = df['weight'] * df['market_index']
    df['quote_signal_x_distance'] = df['quote_signal'] * df['distance']
    df['quote_signal_x_weight'] = df['quote_signal'] * df['weight']
    df['market_index_x_quote'] = df['market_index'] * df['quote_signal']
    df['distance_x_quote_signal_x_market'] = df['distance'] * df['quote_signal'] * df['market_index']

    # Equipment indicators
    df['is_dry_van'] = (df['equipment'] == 'Dry Van').astype(int)
    df['is_reefer'] = (df['equipment'] == 'Reefer').astype(int)
    df['is_flatbed'] = (df['equipment'] == 'Flatbed').astype(int)

    # Distance bins
    df['distance_short'] = (df['distance'] < 200).astype(int)
    df['distance_medium'] = ((df['distance'] >= 200) & (df['distance'] < 800)).astype(int)
    df['distance_long'] = (df['distance'] >= 800).astype(int)

    # Weight bins
    df['weight_light'] = (df['weight'] < 20000).astype(int)
    df['weight_medium'] = ((df['weight'] >= 20000) & (df['weight'] < 35000)).astype(int)
    df['weight_heavy'] = (df['weight'] >= 35000).astype(int)

    # Heuristic feature
    df['estimated_rate_per_mile'] = df['distance'] * 0.5 + df['weight'] * 0.02

    # Market bins
    df['market_high'] = (df['market_index'] > 1.0).astype(int)
    df['market_low'] = (df['market_index'] < 0.9).astype(int)

    # Quote bins
    df['quote_high'] = (df['quote_signal'] > 2.2).astype(int)
    df['quote_low'] = (df['quote_signal'] < 1.9).astype(int)

    # Feature selection (must match training exactly)
    feature_cols = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'weight', 'market_index', 'quote_signal',
        'equipment_encoded', 'year', 'month', 'day', 'day_of_week', 'day_of_year',
        'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end', 'quarter',
        'bearing', 'distance_squared', 'distance_cubed', 'weight_squared',
        'distance_per_weight', 'weight_per_mile', 'distance_x_market',
        'weight_x_market', 'quote_signal_x_distance', 'quote_signal_x_weight',
        'market_index_x_quote', 'distance_x_quote_signal_x_market',
        'is_dry_van', 'is_reefer', 'is_flatbed',
        'distance_short', 'distance_medium', 'distance_long',
        'weight_light', 'weight_medium', 'weight_heavy',
        'estimated_rate_per_mile', 'market_high', 'market_low', 'quote_high', 'quote_low'
    ]

    X = df[feature_cols]
    return X

# ============================================================================
# VALIDATION SET PREDICTIONS
# ============================================================================

# Load validation data (12,000 unlabeled samples from November-December 2025)
print("Loading validation data...")
validation_df = pd.read_csv('data/validation.csv')
print(f"Validation data shape: {validation_df.shape}")

# Apply feature engineering
X_val = preprocess_features(validation_df, equipment_encoder)

# Generate predictions
print("Generating predictions...")
if is_ensemble:
    # Weighted ensemble prediction using same weights as training
    predictions = sum(
        ensemble_config['weights'][name] * ensemble_config['models'][name].predict(X_val)
        for name in ensemble_config['models']
    )
else:
    predictions = single_model.predict(X_val)

# Post-processing: Ensure all predictions are positive
# Freight rates cannot be negative; clip to minimum of $1.00
predictions = np.maximum(predictions, 1.0)

# Create output DataFrame with required format
validation_predictions = pd.DataFrame({
    'load_id': validation_df['load_id'],
    'predicted_rate': predictions
})

# Save predictions for submission
validation_predictions.to_csv('validation_predictions.csv', index=False)
print(f"Saved validation_predictions.csv with {len(validation_predictions)} rows")

# ============================================================================
# DECEMBER 2025 FORECAST
# ============================================================================

# Load December chart inputs (31 days, fixed route: Lexington → Fort Wayne)
# This is a special forecasting task where only the date varies
# All other parameters are fixed: 360 miles, Dry Van, 32,000 lbs
print("\nLoading December chart inputs...")
december_df = pd.read_csv('data/december_chart_inputs.csv')
print(f"December data shape: {december_df.shape}")

# December data has fixed values but model requires coordinates and market features
# Strategy: Look up coordinates from training data for the fixed route
train_df = pd.read_csv('data/train_test.csv')
lexington_coords = train_df[train_df['pickup'] == 'Lexington'][['pickup_lat', 'pickup_lon']].iloc[0]
fort_wayne_coords = train_df[train_df['delivery'] == 'Fort Wayne'][['delivery_lat', 'delivery_lon']].iloc[0]

print(f"Lexington coords: {lexington_coords['pickup_lat']}, {lexington_coords['pickup_lon']}")
print(f"Fort Wayne coords: {fort_wayne_coords['delivery_lat']}, {fort_wayne_coords['delivery_lon']}")

# Add coordinates to December data (required for feature engineering)
december_df['pickup_lat'] = lexington_coords['pickup_lat']
december_df['pickup_lon'] = lexington_coords['pickup_lon']
december_df['delivery_lat'] = fort_wayne_coords['delivery_lat']
december_df['delivery_lon'] = fort_wayne_coords['delivery_lon']

# Add missing market features using training data medians
# Since December market conditions are unknown, use historical medians as baseline
# This is a reasonable assumption for a stable market forecast
december_df['market_index'] = train_df['market_index'].median()
december_df['quote_signal'] = train_df['quote_signal'].median()

# Apply feature engineering
X_dec = preprocess_features(december_df, equipment_encoder)

# Generate predictions
print("Generating December predictions...")
if is_ensemble:
    december_predictions = sum(
        ensemble_config['weights'][name] * ensemble_config['models'][name].predict(X_dec)
        for name in ensemble_config['models']
    )
else:
    december_predictions = single_model.predict(X_dec)

# Ensure positive predictions
december_predictions = np.maximum(december_predictions, 1.0)

# Update dataframe with predictions
december_df['predicted_rate'] = december_predictions

# Ensure proper data types for scorer validation
december_df['distance'] = december_df['distance'].astype(float)
december_df['weight'] = december_df['weight'].astype(float)

# Keep only the 7 columns expected by the official scorer
# Column order must match: pickup, delivery, distance, equipment, weight, date, predicted_rate
december_output = december_df[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']]

# Save December predictions
december_output.to_csv('data/december_chart_inputs.csv', index=False)
print(f"Saved December predictions to data/december_chart_inputs.csv")

print("\nDone!")

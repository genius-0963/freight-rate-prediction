import pandas as pd
import numpy as np
import joblib

model_data = joblib.load('freight_rate_model.pkl')
equipment_encoder = joblib.load('equipment_encoder.pkl')

is_ensemble = isinstance(model_data, dict) and 'models' in model_data
if is_ensemble:
    print("Loading ensemble model...")
    ensemble_config = model_data
else:
    print("Loading single model...")
    single_model = model_data

def preprocess_features(df, equipment_encoder):
    df = df.copy()

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

    df['equipment_encoded'] = equipment_encoder.transform(df['equipment'].astype(str))

    df['weight'] = df['weight'].fillna(df['weight'].median())
    df['market_index'] = df['market_index'].fillna(df['market_index'].median())

    lat1, lon1 = np.radians(df['pickup_lat']), np.radians(df['pickup_lon'])
    lat2, lon2 = np.radians(df['delivery_lat']), np.radians(df['delivery_lon'])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df['bearing'] = np.arctan2(x, y)
    df['bearing'] = np.degrees(df['bearing'])
    df['bearing'] = (df['bearing'] + 360) % 360

    df['distance_squared'] = df['distance'] ** 2
    df['distance_cubed'] = df['distance'] ** 3
    df['weight_squared'] = df['weight'] ** 2

    df['distance_per_weight'] = df['distance'] / (df['weight'] + 1)
    df['weight_per_mile'] = df['weight'] / (df['distance'] + 1)
    df['distance_x_market'] = df['distance'] * df['market_index']
    df['weight_x_market'] = df['weight'] * df['market_index']
    df['quote_signal_x_distance'] = df['quote_signal'] * df['distance']
    df['quote_signal_x_weight'] = df['quote_signal'] * df['weight']
    df['market_index_x_quote'] = df['market_index'] * df['quote_signal']
    df['distance_x_quote_signal_x_market'] = df['distance'] * df['quote_signal'] * df['market_index']

    df['is_dry_van'] = (df['equipment'] == 'Dry Van').astype(int)
    df['is_reefer'] = (df['equipment'] == 'Reefer').astype(int)
    df['is_flatbed'] = (df['equipment'] == 'Flatbed').astype(int)

    df['distance_short'] = (df['distance'] < 200).astype(int)
    df['distance_medium'] = ((df['distance'] >= 200) & (df['distance'] < 800)).astype(int)
    df['distance_long'] = (df['distance'] >= 800).astype(int)

    df['weight_light'] = (df['weight'] < 20000).astype(int)
    df['weight_medium'] = ((df['weight'] >= 20000) & (df['weight'] < 35000)).astype(int)
    df['weight_heavy'] = (df['weight'] >= 35000).astype(int)

    df['estimated_rate_per_mile'] = df['distance'] * 0.5 + df['weight'] * 0.02

    df['market_high'] = (df['market_index'] > 1.0).astype(int)
    df['market_low'] = (df['market_index'] < 0.9).astype(int)

    df['quote_high'] = (df['quote_signal'] > 2.2).astype(int)
    df['quote_low'] = (df['quote_signal'] < 1.9).astype(int)

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

print("Loading validation data...")
validation_df = pd.read_csv('data/validation.csv')
print(f"Validation data shape: {validation_df.shape}")

X_val = preprocess_features(validation_df, equipment_encoder)

print("Generating predictions...")
if is_ensemble:
    predictions = sum(
        ensemble_config['weights'][name] * ensemble_config['models'][name].predict(X_val)
        for name in ensemble_config['models']
    )
else:
    predictions = single_model.predict(X_val)

predictions = np.maximum(predictions, 1.0)

validation_predictions = pd.DataFrame({
    'load_id': validation_df['load_id'],
    'predicted_rate': predictions
})

validation_predictions.to_csv('validation_predictions.csv', index=False)
print(f"Saved validation_predictions.csv with {len(validation_predictions)} rows")

print("\nLoading December chart inputs...")
december_df = pd.read_csv('data/december_chart_inputs.csv')
print(f"December data shape: {december_df.shape}")

train_df = pd.read_csv('data/train_test.csv')
lexington_coords = train_df[train_df['pickup'] == 'Lexington'][['pickup_lat', 'pickup_lon']].iloc[0]
fort_wayne_coords = train_df[train_df['delivery'] == 'Fort Wayne'][['delivery_lat', 'delivery_lon']].iloc[0]

print(f"Lexington coords: {lexington_coords['pickup_lat']}, {lexington_coords['pickup_lon']}")
print(f"Fort Wayne coords: {fort_wayne_coords['delivery_lat']}, {fort_wayne_coords['delivery_lon']}")

december_df['pickup_lat'] = lexington_coords['pickup_lat']
december_df['pickup_lon'] = lexington_coords['pickup_lon']
december_df['delivery_lat'] = fort_wayne_coords['delivery_lat']
december_df['delivery_lon'] = fort_wayne_coords['delivery_lon']

december_df['market_index'] = train_df['market_index'].median()
december_df['quote_signal'] = train_df['quote_signal'].median()

X_dec = preprocess_features(december_df, equipment_encoder)

print("Generating December predictions...")
if is_ensemble:
    december_predictions = sum(
        ensemble_config['weights'][name] * ensemble_config['models'][name].predict(X_dec)
        for name in ensemble_config['models']
    )
else:
    december_predictions = single_model.predict(X_dec)

december_predictions = np.maximum(december_predictions, 1.0)

december_df['predicted_rate'] = december_predictions

december_df['distance'] = december_df['distance'].astype(float)
december_df['weight'] = december_df['weight'].astype(float)

december_output = december_df[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']]

december_output.to_csv('data/december_chart_inputs.csv', index=False)
print(f"Saved December predictions to data/december_chart_inputs.csv")

print("\nDone!")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
train_df = pd.read_csv('data/train_test.csv')

print(f"Training data shape: {train_df.shape}")

def preprocess_features(df, fit_encoders=True, equipment_encoder=None):
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

    if fit_encoders:
        equipment_encoder = LabelEncoder()
        df['equipment_encoded'] = equipment_encoder.fit_transform(df['equipment'].astype(str))
    else:
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
    return X, equipment_encoder

X, equipment_encoder = preprocess_features(train_df, fit_encoders=True)
y = train_df['posted_rate']

print(f"Features shape: {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

models = {}

print("\n" + "="*60)
print("Training Random Forest model...")
print("="*60)
rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
mae_rf = mean_absolute_error(y_val, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
r2_rf = r2_score(y_val, y_pred_rf)
print(f"Random Forest - MAE: ${mae_rf:.2f}, RMSE: ${rmse_rf:.2f}, R2: {r2_rf:.4f}")
models['RandomForest'] = {'model': rf_model, 'mae': mae_rf, 'pred': y_pred_rf}

print("\n" + "="*60)
print("Training Extra Trees model...")
print("="*60)
et_model = ExtraTreesRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
et_model.fit(X_train, y_train)
y_pred_et = et_model.predict(X_val)
mae_et = mean_absolute_error(y_val, y_pred_et)
rmse_et = np.sqrt(mean_squared_error(y_val, y_pred_et))
r2_et = r2_score(y_val, y_pred_et)
print(f"Extra Trees - MAE: ${mae_et:.2f}, RMSE: ${rmse_et:.2f}, R2: {r2_et:.4f}")
models['ExtraTrees'] = {'model': et_model, 'mae': mae_et, 'pred': y_pred_et}



print("\n" + "="*60)
print("Creating Ensemble Model")
print("="*60)

weights = {name: 1/data['mae'] for name, data in models.items()}
total_weight = sum(weights.values())
normalized_weights = {name: w/total_weight for name, w in weights.items()}

print("Model weights (based on inverse MAE):")
for name, weight in normalized_weights.items():
    print(f"  {name}: {weight:.3f}")

y_pred_ensemble = sum(
    normalized_weights[name] * data['pred']
    for name, data in models.items()
)
mae_ensemble = mean_absolute_error(y_val, y_pred_ensemble)
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
r2_ensemble = r2_score(y_val, y_pred_ensemble)

print(f"\nEnsemble - MAE: ${mae_ensemble:.2f}, RMSE: ${rmse_ensemble:.2f}, R2: {r2_ensemble:.4f}")

best_model_name = min(models, key=lambda x: models[x]['mae'])
best_mae = models[best_model_name]['mae']

if mae_ensemble < best_mae:
    print(f"\n✓ Ensemble model selected (best MAE: ${mae_ensemble:.2f})")
    ensemble_config = {
        'models': {name: data['model'] for name, data in models.items()},
        'weights': normalized_weights
    }
    joblib.dump(ensemble_config, 'freight_rate_model.pkl')
else:
    print(f"\n✓ {best_model_name} model selected (best MAE: ${best_mae:.2f})")
    joblib.dump(models[best_model_name]['model'], 'freight_rate_model.pkl')

best_tree_model = models['RandomForest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_tree_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Random Forest) - Top 10:")
print(feature_importance.head(10))

joblib.dump(equipment_encoder, 'equipment_encoder.pkl')
print("\nModel saved as freight_rate_model.pkl")
print("Encoder saved as equipment_encoder.pkl")

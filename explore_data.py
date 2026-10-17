import pandas as pd
import numpy as np

train_df = pd.read_csv('data/train_test.csv')

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\nTraining data shape:", train_df.shape)
print("\nColumns:", train_df.columns.tolist())
print("\nFirst few rows:")
print(train_df.head())

print("\nData types:")
print(train_df.dtypes)

print("\nMissing values:")
print(train_df.isnull().sum())

print("\nBasic statistics:")
print(train_df.describe())

print("\nEquipment types:")
print(train_df['equipment'].value_counts())

print("\nDate range:")
print(f"Min date: {train_df['date'].min()}")
print(f"Max date: {train_df['date'].max()}")

print("\nCorrelation with posted_rate:")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
print(train_df[numeric_cols].corr()['posted_rate'].sort_values(ascending=False))

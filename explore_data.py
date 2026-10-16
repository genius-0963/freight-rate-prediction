"""
Freight Rate Prediction - Exploratory Data Analysis

This script performs initial data exploration to understand the training dataset
structure, distributions, and relationships. This analysis informs feature engineering
and model selection decisions.

Key Analyses:
- Data structure and missing values
- Feature distributions and statistics
- Equipment type distribution
- Temporal coverage
- Correlation analysis with target variable

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import numpy as np

# ============================================================================
# DATA LOADING
# ============================================================================

# Load training data (48,000 labeled samples from January-October 2025)
train_df = pd.read_csv('data/train_test.csv')

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# ============================================================================
# BASIC DATA STRUCTURE
# ============================================================================

print("\nTraining data shape:", train_df.shape)
print("\nColumns:", train_df.columns.tolist())
print("\nFirst few rows:")
print(train_df.head())

# ============================================================================
# DATA TYPES AND MISSING VALUES
# ============================================================================

print("\nData types:")
print(train_df.dtypes)

print("\nMissing values:")
print(train_df.isnull().sum())

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

print("\nBasic statistics:")
print(train_df.describe())

# ============================================================================
# CATEGORICAL FEATURE ANALYSIS
# ============================================================================

print("\nEquipment types:")
print(train_df['equipment'].value_counts())

# ============================================================================
# TEMPORAL COVERAGE
# ============================================================================

print("\nDate range:")
print(f"Min date: {train_df['date'].min()}")
print(f"Max date: {train_df['date'].max()}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\nCorrelation with posted_rate:")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
print(train_df[numeric_cols].corr()['posted_rate'].sort_values(ascending=False))

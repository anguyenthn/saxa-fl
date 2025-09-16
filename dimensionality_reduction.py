# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:23:38 2025

@author: deeha
"""

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load your data (adjust path as needed)
df = pd.read_csv('merged_week_aggregate.csv')

# Step 3: Drop non-numeric columns
df_dropped = df.select_dtypes(include=[np.number])

# Step 4: Fill missing values with column means
df_dropped = df_dropped.fillna(df_dropped.mean())

# Step 5: Standardize the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_dropped)

# Step 6: Run PCA
pca = PCA()
pca.fit(scaled_data)

# Step 7: Calculate influence score from top 5 principal components
loadings = np.abs(pca.components_[:5])  # Top 5 components
feature_influence = loadings.sum(axis=0)

# Step 8: Create DataFrame with feature influence scores
influence_df = pd.DataFrame({
    'feature': df_dropped.columns,
    'influence_score': feature_influence
}).sort_values(by='influence_score', ascending=False)

# Step 9: Extract top 25 features
top_25_features = influence_df.head(25)
df_reduced = df_dropped[top_25_features]

# Step 10: Select top 20 features for correlation plot
top_20_features = top_25_features.head(20)['feature'].tolist()
correlation_data = df_dropped[top_20_features]

# Step 11: Compute and plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap of Top 20 Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# export to csv
df_dropped[top_25_features['feature']].to_csv('top_25_features_data.csv', index=False)

#pca viz
import matplotlib.pyplot as plt
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()


# Model

# Step 1: Import modeling libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Define the target and features
target = df['win']   # Make sure 'win' column is from original df
features = df_reduced.copy()

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Get feature importances from the model
importances = rf_model.feature_importances_
feature_names = features.columns

# Step 2: Create a DataFrame for importances
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Step 3: Select top 20 features
top_n = 20  # or any smaller number like 15, 10, etc.
top_features_df = importance_df.head(top_n)

# Step 4: Plot top N features
plt.figure(figsize=(10, 6))
plt.barh(top_features_df['feature'], top_features_df['importance'])
plt.xlabel('Feature Importance')
plt.title(f'Random Forest: Top {top_n} Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



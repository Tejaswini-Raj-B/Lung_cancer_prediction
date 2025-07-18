import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
df = pd.read_csv('survey lung cancer.csv')

# Replace values and fix types
df.replace({'YES': 1, 'NO': 0, 'M': 1, 'F': 0}, inplace=True)
df = df.infer_objects(copy=False)  # Fix future downcasting warning

# Features and target
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']  # Already 0/1 after replacement

# Drop any remaining NaN values (precaution)
X.dropna(inplace=True)
y = y[X.index]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok=True)
with open('model/lung_cancer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")

# train_models.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")
X = df.drop('label', axis=1)
y = df['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
pickle.dump(le, open("models/labelencoder.pkl", "wb"))

# Scale features
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
pickle.dump(minmax, open("models/minmaxscaler.pkl", "wb"))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_minmax)
pickle.dump(scaler, open("models/standscaler.pkl", "wb"))

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss')

}

# Evaluate and select best model
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f}")

best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name}")
best_model = models[best_model_name]
best_model.fit(X_scaled, y_encoded)

# Save best model
pickle.dump(best_model, open("models/model.pkl", "wb"))

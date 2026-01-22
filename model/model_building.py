import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "../titanic.csv"))  # place in project root

# 2. Select features (any 5)
features = ["Pclass", "Sex", "Age", "Fare", "SibSp"]
X = df[features]
y = df["Survived"]

# 3. Handle missing values
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Fare"] = X["Fare"].fillna(X["Fare"].median())

# 4. Encode categorical variables
sex_encoder = LabelEncoder()
X["Sex"] = sex_encoder.fit_transform(X["Sex"])  # Male=1, Female=0

# 5. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 9. Save model, scaler, and encoder
model_path = os.path.join(script_dir, "titanic_survival_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "sex_encoder": sex_encoder
    }, f)

print("✅ Titanic survival model saved successfully")

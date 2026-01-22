from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = "model/titanic_survival_model.pkl"
model = None
scaler = None
sex_encoder = None

try:
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        scaler = data["scaler"]
        sex_encoder = data.get("sex_encoder")  # Handle backward compatibility
except FileNotFoundError:
    print(f"Warning: Model file not found at {model_path}. Please train the model first.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/")
def home():
    return render_template("index.html", prediction="")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return render_template("index.html", prediction="Error: Model not loaded. Please train the model first.")
    
    try:
        # get input from form
        features = [
            float(request.form["Pclass"]),
            float(request.form["Sex"]),      # 1=Male, 0=Female
            float(request.form["Age"]),
            float(request.form["Fare"]),
            float(request.form["SibSp"]),
        ]

        # scale and predict
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        result = "Survived" if pred==1 else "Did Not Survive"

        return render_template("index.html", prediction=f"Passenger will {result}")
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    # For local development
    app.run(debug=True, host="127.0.0.1", port=5000)


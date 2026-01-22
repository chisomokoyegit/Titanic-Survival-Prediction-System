from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model_path = "model/titanic_survival_model.pkl"
with open(model_path, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    sex_encoder = data.get("sex_encoder")  # Handle backward compatibility

@app.route("/")
def home():
    return render_template("index.html", prediction="")

@app.route("/predict", methods=["POST"])
def predict():
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


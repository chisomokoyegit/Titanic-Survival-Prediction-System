# Titanic Survival Prediction System

A Flask-based web application that predicts whether a passenger would have survived the Titanic disaster using machine learning.

## Features

- Machine learning model trained on Titanic dataset
- Web interface for making predictions
- Input features: Passenger Class, Sex, Age, Fare, and Siblings/Spouses
- Deployed on Render

## Installation

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd "Titanic prediction system"
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, place `titanic.csv` in the project root and run:

```bash
python model/model_building.py
```

This will generate `model/titanic_survival_model.pkl`.

### Running the App

```bash
python app.py
```

Then visit `http://localhost:5000` in your browser.

## Project Structure

```
├── app.py                      # Flask application
├── model/
│   └── model_building.py      # Model training script
├── templates/
│   └── index.html             # Web interface
├── requirements.txt            # Python dependencies
├── render.yaml                # Render deployment config
└── README.md                  # This file
```

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: Passenger Class, Sex, Age, Fare, Siblings/Spouses Count
- **Training/Test Split**: 80/20
- **Preprocessing**: StandardScaler for feature normalization, LabelEncoder for categorical variables

## Deployment

This app is configured for deployment on [Render](https://render.com).

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- TensorFlow/Keras
- gunicorn

## License

MIT


import joblib
import json
import pandas as pd

# Load trained model and configuration
model = joblib.load("diabetes_model.pkl")

with open("diabetes_config.json", "r") as f:
    config = json.load(f)

FEATURES = config["features"]


def predict_diabetes(patient_data):
    """
    patient_data: dictionary containing all 16 model features.
    Returns predicted class and estimated probability.
    """

    # Ensure the input contains every required feature
    missing = [feature for feature in FEATURES
               if feature not in patient_data]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Arrange features in the exact training order
    input_df = pd.DataFrame(
        [[patient_data[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    positive_index = list(model.classes_).index(1)
    positive_probability = float(probabilities[positive_index])

    return {
        "prediction": "Positive" if prediction == 1 else "Negative",
        "positive_probability": round(positive_probability, 4),
        "message": (
            "Screening estimate only; not a medical diagnosis."
        )
    }

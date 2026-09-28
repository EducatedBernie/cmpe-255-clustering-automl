# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("overspend_api")

# Create input/output pydantic models
input_model = create_model("overspend_api_input", **{'city_tier': 'city', 'household_size': 1, 'owns_car': False, 'income': 3960.0, 'rent': 1052.0, 'groceries': 205.0, 'utilities': 96.0, 'transport': 95.0, 'dining': 259.0, 'entertainment': 173.0, 'other': 46.0, 'total_spend': 1926.0, 'overspent': 0, 'survey_score': 4, 'referral_code': 'A1', 'avg_spend_prev': 1909.6363525390625, 'overspend_rate_prev': 0.09090909361839294, 'max_dining_prev': 651.0})
output_model = create_model("overspend_api_output", prediction=0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

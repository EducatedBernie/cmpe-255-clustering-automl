"""Local prediction scaffold for the recovered pipeline; no deployment implied."""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, StrictBool, conint, confloat
from pycaret.classification import load_model, predict_model


class Household(BaseModel):
    city_tier: str
    household_size: conint(strict=True, ge=1)
    owns_car: StrictBool
    income: float
    rent: float
    groceries: float
    utilities: float
    transport: float
    dining: float
    entertainment: float
    other: float
    total_spend: float
    overspent: conint(strict=True, ge=0, le=1)  # Current month, not the future target.
    survey_score: float
    referral_code: str
    avg_spend_prev: float
    overspend_rate_prev: confloat(ge=0, le=1)
    max_dining_prev: float

    class Config:
        extra = "forbid"
        allow_inf_nan = False


model = load_model(str(Path(__file__).with_name("overspend_api")), verbose=False)
assert set(Household.__fields__) == set(model.feature_names_in_) - {"overspent_next"}
app = FastAPI()


@app.post("/predict")
def predict(data: Household) -> dict[str, int]:
    predictions = predict_model(model, data=pd.DataFrame([data.dict()]), verbose=False)
    return {"prediction": int(predictions["prediction_label"].iloc[0])}

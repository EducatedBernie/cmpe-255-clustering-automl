"""Run with Python 3.11 after installing requirements.txt."""
from fastapi.testclient import TestClient
from overspend_api import app

sample = {
    "city_tier": "city", "household_size": 1, "owns_car": False,
    "income": 3960.0, "rent": 1052.0, "groceries": 205.0, "utilities": 96.0,
    "transport": 95.0, "dining": 259.0, "entertainment": 173.0, "other": 46.0,
    "total_spend": 1926.0, "overspent": 0, "survey_score": 4, "referral_code": "A1",
    "avg_spend_prev": 1909.6363525390625, "overspend_rate_prev": 0.09090909361839294,
    "max_dining_prev": 651.0,
}
with TestClient(app) as client:
    response = client.post("/predict", json=sample)
    assert response.status_code == 200, response.text
    assert response.json()["prediction"] in (0, 1)
    assert client.post("/predict", json={**sample, "overspent_next": 1}).status_code == 422
    assert client.post("/predict", json={**sample, "household_size": 0}).status_code == 422
    assert client.post("/predict", json={**sample, "income": "NaN"}).status_code == 422
    assert client.post("/predict", json={"income": 100}).status_code == 422
print("PASS: local HTTP prediction and invalid-input checks; not a deployment")

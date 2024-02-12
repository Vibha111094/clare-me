from fastapi.testclient import TestClient

from main import app,load_model

client = TestClient(app)


def test_predict():
    response = client.post("/predict", json={"question": "I have morbid thoughts"})
    assert response.status_code == 200
    assert response.json() == "Suicide"
    
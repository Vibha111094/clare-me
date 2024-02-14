from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_predict():
    conversation = {"question": "I am having morbid thoughts."}
    response = client.post("/predict", json=conversation)
    assert response.status_code == 200
    assert response.json() == "Suicide"
    
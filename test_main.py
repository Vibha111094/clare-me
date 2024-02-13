from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_predict():
    conversation = {"question": "How do I know if my suicidal thoughts are a result of a mental health condition?"}
    response = client.post("/predict", json=conversation)
    assert response.status_code == 200
    assert response.json() == "Suicide"
    
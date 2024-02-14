from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_predict_regular_conversations():
    conversation = {"question": "What are some effective ways to communicate with loved ones about your mental health challenges?"}
    response = client.post("/predict", json=conversation)
    assert response.status_code == 200
    assert response.json() == "Regular conversations"

def test_predict_product_related_conversations():
    conversation = {"question": "What measures does Clare&Me take to prevent harmful use of its platform?"}
    response = client.post("/predict", json=conversation)
    assert response.status_code == 200
    assert response.json() == "Product-related conversations"

def test_predict_subscription_related_conversations():
    conversation = {"question": "Are there any age restrictions for subscribing to Clare&Me?"}
    response = client.post("/predict", json=conversation)
    assert response.status_code == 200
    assert response.json() == "Subscription-related conversations"

def test_predict_suicide_conversations():
    conversation = {"question": "What are some coping strategies I can use to manage my suicidal thoughts?"}
    response = client.post("/predict", json=conversation)
    assert response.status_code == 200
    assert response.json() == "Suicide"

def test_predict_non_mental_health_conversations():
    conversation = {"question": "If you could have any mythical creature as a pet, what would it be and why?"}
    response = client.post("/predict", json=conversation)
    assert response.status_code == 200
    assert response.json() == "Non-mental health topics"
    
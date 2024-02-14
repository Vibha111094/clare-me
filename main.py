from fastapi import FastAPI
from transformers import BertTokenizer
from pydantic import BaseModel
import uvicorn

from utils import load_checkpoint, format_result
from BertClassifier import BertClassifier
from config import bert_model

# Create a FastAPI instance
app = FastAPI()

class Conversation(BaseModel):
    question: str

# Define a prediction endpoint
@app.post("/predict")
def predict(conversation: Conversation):
    model = load_model()
    input_id, mask = encode_input(conversation)
    output = model(input_id, mask)
    result = format_result(output.argmax(dim=1))
    return result

def encode_input(conversation:Conversation):
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    encoded = tokenizer(conversation.question, padding='max_length', max_length=512, truncation=True,
                        return_tensors="pt")
    mask = encoded['attention_mask'].to("cpu")
    input_id = encoded['input_ids'].squeeze(1).to("cpu")
    return input_id, mask

def load_model():
    model_init = BertClassifier()
    # link to model: https://drive.google.com/file/d/1AKCTkjpIWlWmE8KWbCnlAOjCJNdAIxQ4/view?usp=drive_link"
    model = load_checkpoint("./model.pt", model_init)
    return model

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
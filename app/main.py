from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load model
model = pickle.load(open("model.pkl", "rb"))

class InputData(BaseModel):
    data: list

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        data = np.array(input_data.data).reshape(1, -1)
        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        return {"error": str(e)}
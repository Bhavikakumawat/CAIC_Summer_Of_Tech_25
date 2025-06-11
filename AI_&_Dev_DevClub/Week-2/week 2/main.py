from fastapi import FastAPI,Path,Query,Form,File
from typing import Union
from enum import Enum
from pydantic import BaseModel

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from textblob import TextBlob
model = joblib.load('like_predictor_model.pk1')
app_name = FastAPI()
class TweetData(BaseModel):
    content: str
    inferred_company: str
@app_name.post("/predict")
def predict_likes(data: TweetData):
    sentiment = TextBlob(data.content).sentiment.polarity
    char_count = len(data.content)
    company_map = {
        'CompanyA': 0,
        'CompanyB': 1,
        'CompanyC': 2,
    }
    company_encoded = company_map.get(data.inferred_company, 0)  # default to 0
    input_data = np.array([[sentiment, char_count, company_encoded]])
    log_pred = model.predict(input_data)[0]
    likes_pred = np.expm1(log_pred)
    
    return {"predicted_likes": int(likes_pred)}
from fastapi import APIRouter
from controllers.lstm_model import predict_lstm

router = APIRouter()

@router.post("/predict")
# Parametro opcional 
def predict(coin: str, next_days: int): 
    prediction = predict_lstm(next_days, coin)
    
    
    
    return {"prediction": prediction}
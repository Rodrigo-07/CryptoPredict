from fastapi import APIRouter
from backend.controllers.lstm_model import predict_lstm
from backend.controllers.data_loader import get_historical_data

router = APIRouter()

@router.post("/predict")
def predict(coin: str, start_date: str, end_date: str):
    data = get_historical_data(coin, start_date)
    prediction = predict_lstm(data, end_date)
    return {"coin": coin, "prediction": prediction}

from fastapi import APIRouter
from controllers.lstm_model import retrain_lstm

router = APIRouter()

@router.post("/retrain")
def retrain(coin: str):
    message = retrain_lstm(coin)
    return {"message": message}

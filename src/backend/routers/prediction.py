from fastapi import APIRouter, Query
from backend.controllers.lstm_model import predict_lstm
from backend.controllers.gru_model import predict_gru
from pydantic import BaseModel

router = APIRouter()

@router.get("/predict")
def predict(coin: str, next_days: int, model: str = Query("lstm", enum=["lstm", "gru"])):  # Define "lstm" como valor padrão
    
    # Verifica o valor do modelo e chama a função apropriada
    if model == "lstm":
        prediction = predict_lstm(next_days, coin)
    elif model == "gru":
        prediction = predict_gru(next_days, coin)
    else:
        return {"error": "Modelo não suportado"}

    return prediction

# Rota de post que recebe, modelo, moeda, data de começo e fim e retreina o modelo
class RetrainRequest(BaseModel):
    model: str
    currency: str
    start_date: str
    end_date: str
    epochs: int

@router.post("/retrain_model")
def retrain_model(request: RetrainRequest):
    model_type = request.model
    currency = request.currency
    start_date = request.start_date
    end_date = request.end_date
    epochs = request.epochs


    rmse, model_path = retrain_model(time_steps=60, model_type=model_type, currency=currency, epochs=epochs, start_date=start_date, end_date=end_date)

    return {"rmse": rmse, "path": model_path}

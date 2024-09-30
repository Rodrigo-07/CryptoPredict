from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from bson import ObjectId
from typing import Optional, List
from datetime import datetime
from controllers.database import create_log, read_logs, update_log, delete_log, create_model_retrain, read_model_retrains, update_model_retrain, delete_model_retrain

router = APIRouter()

class LogModel(BaseModel):
    acao: str
    typo: str
    timestamp: Optional[datetime] = None

class ModelRetrainModel(BaseModel):
    nome_modelo: str
    tipo_modelo: str
    rmse: float
    path: str


@router.post("/logs/", response_description="Create a new log")
async def create_new_log(log: LogModel):
    timestamp_now = datetime.now()
    create_log(acao=log.acao, typo=log.typo, timestamp=timestamp_now)
    return {"message": "Log criado com sucesso"}

@router.get("/logs/", response_description="List all logs")
async def get_all_logs():
    logs = read_logs()
    return {"logs": logs}

@router.put("/logs/{log_id}", response_description="Update a log")
async def update_existing_log(log_id: str, log: LogModel):
    try:
        update_log(log_id=ObjectId(log_id), updated_data=log.dict(exclude_unset=True))
        return {"message": "Log atualizado com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Log {log_id} não encontrado")

@router.delete("/logs/{log_id}", response_description="Delete a log")
async def delete_existing_log(log_id: str):
    try:
        delete_log(log_id=ObjectId(log_id))
        return {"message": "Log deletado com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Log {log_id} não encontrado")

# -------------------- Endpoints for Model Retrain --------------------

@router.post("/model-retrain/", response_description="Create a new model retrain")
async def create_new_model_retrain(model_retrain: ModelRetrainModel):
    create_model_retrain(
        nome_modelo=model_retrain.nome_modelo,
        tipo_modelo=model_retrain.tipo_modelo,
        rmse=model_retrain.rmse,
        path=model_retrain.path,
        timestamp=model_retrain.timestamp
    )
    return {"message": "Retreinamento de modelo criado com sucesso"}

@router.get("/model-retrain/", response_description="List all model retrains")
async def get_all_model_retrains():
    retrains = read_model_retrains()
    return {"retrain": retrains}

@router.put("/model-retrain/{retrain_id}", response_description="Update a model retrain")
async def update_existing_model_retrain(retrain_id: str, model_retrain: ModelRetrainModel):
    try:
        update_model_retrain(retrain_id=ObjectId(retrain_id), updated_data=model_retrain.dict(exclude_unset=True))
        return {"message": "Retreinamento de modelo atualizado com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Retreinamento {retrain_id} não encontrado")

@router.delete("/model-retrain/{retrain_id}", response_description="Delete a model retrain")
async def delete_existing_model_retrain(retrain_id: str):
    try:
        delete_model_retrain(retrain_id=ObjectId(retrain_id))
        return {"message": "Retreinamento de modelo deletado com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Retreinamento {retrain_id} não encontrado")

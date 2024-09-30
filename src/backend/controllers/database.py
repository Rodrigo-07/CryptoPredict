from pymongo import MongoClient
from datetime import datetime

# Conectar ao MongoDB rodando no Docker
def get_database():
    client = MongoClient("mongodb://mongo:27017/")
    return client['model_logs_db']

# Funções CRUD para Logs
def create_log(acao, typo, timestamp=None):
    db = get_database()
    logs_collection = db['logs']
    
    if not timestamp:
        timestamp = datetime.now()

    log_entry = {
        "acao": acao,
        "typo": typo,
        "timestamp": timestamp
    }
    
    logs_collection.insert_one(log_entry)
    print(f"Log criado: {log_entry}")

def read_logs():
    db = get_database()
    logs_collection = db['logs']
    
    logs = logs_collection.find()

    # Converter os logs em uma lista de dicionários e transformar ObjectId em string
    logs_list = []
    for log in logs:
        log['_id'] = str(log['_id'])  # Convertendo ObjectId para string
        logs_list.append(log)

    return logs_list

def update_log(log_id, updated_data):
    db = get_database()
    logs_collection = db['logs']
    
    logs_collection.update_one({"_id": log_id}, {"$set": updated_data})
    print(f"Log {log_id} atualizado com: {updated_data}")

def delete_log(log_id):
    db = get_database()
    logs_collection = db['logs']
    
    logs_collection.delete_one({"_id": log_id})
    print(f"Log {log_id} deletado")

# Funções CRUD para Retreinamento de Modelo
def create_model_retrain(nome_modelo, tipo_modelo, rmse, path, timestamp=None):
    db = get_database()
    retrain_collection = db['retrain']
    
    if not timestamp:
        timestamp = datetime.now()

    retrain_entry = {
        "nome_modelo": nome_modelo,
        "tipo_modelo": tipo_modelo,
        "timestamp": timestamp,
        "rmse": rmse,
        "path": path
    }
    
    retrain_collection.insert_one(retrain_entry)
    print(f"Retreinamento de modelo criado: {retrain_entry}")

def read_model_retrains():
    db = get_database()
    retrain_collection = db['retrain']
    
    retrains = retrain_collection.find()
    for retrain in retrains:
        print(retrain)

def update_model_retrain(retrain_id, updated_data):
    db = get_database()
    retrain_collection = db['retrain']
    
    retrain_collection.update_one({"_id": retrain_id}, {"$set": updated_data})
    print(f"Retreinamento de modelo {retrain_id} atualizado com: {updated_data}")

def delete_model_retrain(retrain_id):
    db = get_database()
    retrain_collection = db['retrain']
    
    retrain_collection.delete_one({"_id": retrain_id})
    print(f"Retreinamento de modelo {retrain_id} deletado")

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tensorflow.keras.models import load_model

def predict_lstm(next_days, coin, model_path='D:/Repos/CryptoPredict/src/Modelo/model_new_feat_gru.h5'):
    # Carregar o modelo salvo
    model = load_model(model_path)

    # Número de dias de previsão
    prediction_days = next_days

    # Carregar o dataset
    df = pd.read_csv("D:/Repos/CryptoPredict/src/Modelo/df_feat_not_scaled.csv")
    
    print(df.head())
    
    # Definir o scaler e aplicar o fit nos dados para manter consistência entre colunas
    scale = MinMaxScaler()

    # Colunas que precisam ser escalonadas
    columns_to_scale = ['BTC Price', 'Nasdaq Price', 'Nasdaq Crypto Price', 'Crypto Fear & Greed Index', 'VIX Index', 'Solana Price']
    df[columns_to_scale] = scale.fit_transform(df[columns_to_scale])
    
    print(df.head())

    # Preparar a última sequência para a previsão (usando o número de steps definidos no treinamento do modelo)
    time_steps = 60  # O número de passos de tempo (time_steps) que você usou durante o treinamento
    last_sequence = df[columns_to_scale].values[-time_steps:]  # Últimos 60 dias (tempo de entrada para o LSTM)

    # Fazer previsões futuras
    predictions_future = []
    for i in range(prediction_days):
        # Ajustar a sequência para o formato esperado pelo modelo (batch_size=1)
        last_sequence_expanded = np.expand_dims(last_sequence, axis=0)

        # Prever o próximo valor
        predicted_price = model.predict(last_sequence_expanded)

        # Adicionar a previsão à lista
        predictions_future.append(predicted_price[0, 0])  # Previsão de Solana Price

        # Atualizar a sequência (remover o primeiro dia e adicionar a previsão no final)
        last_sequence = np.vstack([last_sequence[1:], np.concatenate([last_sequence[-1][:-1], predicted_price[0]])])

    # Criando um array de zeros com o mesmo tamanho das previsões futuras e depois preenchendo a última coluna com as previsões
    dummy_array = np.zeros((len(predictions_future), len(columns_to_scale)))
    dummy_array[:, -1] = predictions_future  # Preencher a última coluna (Solana Price) com as previsões

    # Inverter a transformação apenas para a coluna 'Solana Price'
    predictions_future_inversed = scale.inverse_transform(dummy_array)[:, -1]  # Pega apenas a coluna da previsão

    # Gerar as datas futuras para cada previsão
    last_date = pd.to_datetime('today')
    future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]

    # Retornar um dicionário com as previsões e as datas correspondentes
    response = {
        "predictions": [
            {"date": future_dates[i].strftime('%Y-%m-%d'), "predicted_price": predictions_future_inversed[i]}
            for i in range(prediction_days)
        ]
    }

    return response

def retrain_lstm(coin):
    # Aqui você vai adicionar a lógica para retreinar o modelo
    # com os novos dados da moeda
    return f"Modelo para {coin} retreinado com sucesso!"

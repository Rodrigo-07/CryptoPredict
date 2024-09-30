import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tensorflow.keras.models import load_model
from backend.controllers.database import create_log
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



def predict_lstm(next_days, coin, model_path='model_new_feat_lstm.h5'):
    # Carregar o modelo salvo
    model = load_model(model_path)

    # Número de dias de previsão
    prediction_days = next_days

    # Carregar o dataset com as datas
    df = pd.read_csv("df_feat_not_scaled.csv", parse_dates=['Date'], index_col='Date')
    
    # Criar um scaler individual para cada coluna
    columns_to_scale = ['BTC Price', 'Nasdaq Price', 'Nasdaq Crypto Price', 'Crypto Fear & Greed Index', 'VIX Index', 'Solana Price']
    
    scaler = MinMaxScaler()
    
    # Aplicar o MinMaxScaler individualmente para cada coluna
    for col in columns_to_scale:
        df[col] = scaler.fit_transform(df[[col]])  # Normalizar a coluna individualmente
    
    # Pegar os valores reais dos últimos 120 dias
    real_values_last_120_days = df[['Solana Price']].values[-120:]
    
    # Inverter a transformação dos últimos 120 dias
    real_values_inversed = scaler.inverse_transform(np.hstack([np.zeros((120, len(columns_to_scale) - 1)), real_values_last_120_days]))[:, -1]
    
    # Pegar as últimas 120 datas reais do índice
    real_dates_last_120_days = df.index[-120:].to_list()

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

        # Adicionar a previsão à lista (prevendo apenas 'Solana Price')
        predictions_future.append(predicted_price[0, 0])

        # Atualizar a sequência (remover o primeiro dia e adicionar a previsão no final)
        last_sequence = np.vstack([last_sequence[1:], np.concatenate([last_sequence[-1][:-1], predicted_price[0]])])

    # Criando um array de zeros com o mesmo tamanho das previsões futuras e depois preenchendo a última coluna com as previsões
    dummy_array = np.zeros((len(predictions_future), len(columns_to_scale)))
    dummy_array[:, -1] = predictions_future  # Preencher a última coluna (Solana Price) com as previsões

    # Inverter a transformação apenas para a coluna 'Solana Price'
    predictions_future_inversed = scaler.inverse_transform(dummy_array)[:, -1]  # Inverter usando o scaler de Solana Price

    # Gerar as datas futuras a partir do último dia real
    last_date = real_dates_last_120_days[-1]  # Pegar a última data real
    future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]  # Previsões começam no próximo dia

    # Retornar um dicionário com as previsões, os valores reais e as datas correspondentes
    response = {
        "real_values": [
            {"date": real_dates_last_120_days[i].strftime('%Y-%m-%d'), "price": real_values_inversed[i]}
            for i in range(120)
        ],
        "predictions": [
            {"date": future_dates[i].strftime('%Y-%m-%d'), "predicted_price": predictions_future_inversed[i]}
            for i in range(prediction_days)
        ]
    }
    
    # Adicionar log de previsão
    create_log(acao='Previsão', typo='LSTM', timestamp=pd.to_datetime('today'))
    
    return response

def create_dataset(df, time_steps):
    X, y = [], []
    for i in range(len(df) - time_steps):
        X.append(df.iloc[i:(i + time_steps)].values)
        y.append(df.iloc[i + time_steps, -1])
    return np.array(X), np.array(y)

def retrain_model(df, time_steps=60, model_type="lstm", currency="SOL"):
    
    # Preparar os dados
    X, y = create_dataset(df, time_steps)
    
    # Dividir os dados entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Construir o modelo
    model = Sequential()

    # Primeira camada LSTM com Dropout
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))  # Dropout de 20%

    # Segunda camada LSTM com Dropout
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))

    # Camada de saída
    model.add(Dense(1))

    # Compilar o modelo
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Callback para Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop])

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Caminho para salvar o modelo retreinado
    path = f"model_retrained_{model_type}_{currency}.h5"
    model.save(path)

    # Retornar o RMSE e o caminho do modelo salvo
    return rmse, path
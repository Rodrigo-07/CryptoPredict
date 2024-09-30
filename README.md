# CryptoPredict

# Descrição do Projeto

Esse projeto fei feito para uma atividade ponderada, do curso de engenharia da computação na Inteli, no ano de 2024. O projeto consiste em prever o preço de uma criptomoeda, utilizando séries temporais.

# Análise Exploratória 

## Análise 1

Primeiro, vamos discutir o que eu fiz de exploração de dados no arquivo presente em `src/Modelo/analise.ipynb`, que foi a análise inicial.

Primeiro, eu decidi que iria fazer uma análise da moeda Solana (SOL-USD), por ser uma criptomoeda que eu acredito que tem um potencial de crescimento grande e não apresenta tantas peculiaridades quanto outras moedas.

Para obter os dados eu utilizei a biblioteca `yfinance`, com uma range de data de 2017-05-01 até 2023-09-24.

Eu fiz alguns gráficos para entender a distribuição dos dados e algumas outras análises, como a média móvel e a decomposição da série temporal.

![media movel](image.png)

Depois disso parti para o pré-processamento dos dados, que consistiu em normalizar os dados e separar em treino e teste. Para fazer a normalização, eu utilizei a função MixMaxScaler da biblioteca `sklearn` para normalizar os dados entre 0 e 1.

As features que eu utilizei para treinar o modelo foram o preço de fechamento da moeda, que é o que eu quero prever.

### Criando o modelo

Para criar o modelo, eu utilizei uma rede neural recorrente, mais especificamente uma LSTM. Eu decidi utilizar essa rede, pois ela é muito boa para prever séries temporais, por conta de sua capacidade de "memorizar" padrões em dados sequenciais.

Para o meu input, eu utilizei 60 dias de dados para prever o próximo dia. Nesse teste eu criei o modelo dessa maneira:

```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1)) # Prever um valor de saída que no caso é o valor de close

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

```

### Resultados

Com o modelo treinado, eu fiz a previsão dos dados de teste e plotei o gráfico para comparar com os dados reais.

![alt text](image-1.png)

Como podemos ver, o modelo conseguiu prever bem o comportamento da moeda e obteve um MSE de 66.6 no conjunto de teste.

Apesar desse resultado, quando vamos prever o valor dos próximos, quando vamos prever o valor dos próximos dias ele não apresenta um resultado confiável e consistente.

![alt text](image-2.png)

Para facilitar testes eu criei uma classe chamada `CryptoPredicter` que encapsula o modelo e facilita a previsão de novos valores.

```python
class CryptoPredicter:
    def __init__(self, CryptoName, start_date, end_date, prediction_days=60):
        self.CryptoName = CryptoName
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_days = prediction_days
        
    def get_data(self):
        data = yf.download(self.CryptoName, start=self.start_date, end=self.end_date)
        print(data)
        return data

    def train_model(self, data):
        close_data = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_close_data = scaler.fit_transform(close_data)
        prediction_days = self.prediction_days
        x_train = []
        y_train = []
        for x in range(prediction_days, len(scaled_close_data)):
            x_train.append(scaled_close_data[x-prediction_days:x, 0])
            y_train.append(scaled_close_data[x, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=25, batch_size=32)
        model.save(f'{self.CryptoName}_model.h5')
        return model
    
    def predict(self, model, data, days_to_predict=30):
        close_data = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_close_data = scaler.fit_transform(close_data)
        x_test = []
        for x in range(days_to_predict, len(scaled_close_data)):
            x_test.append(scaled_close_data[x-days_to_predict:x, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        return predicted_prices
    
    def forecast(self, model, data, days_to_predict=30):
        close_data = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_close_data = scaler.fit_transform(close_data)
        last_prediction_days = scaled_close_data[-self.prediction_days:]
        predicted_prices = []
        x_input = last_prediction_days.reshape(1, self.prediction_days, 1)
        for _ in range(days_to_predict):
            prediction = model.predict(x_input, verbose=0)
            predicted_prices.append(prediction[0][0])
            prediction_reshaped = prediction.reshape(1, 1, 1)
            x_input = np.concatenate((x_input[:,1:,:], prediction_reshaped), axis=1)
        predicted_prices_array = np.array(predicted_prices).reshape(-1, 1)
        predicted_prices_actual = scaler.inverse_transform(predicted_prices_array)
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='D')
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predicted_prices_actual.flatten()})
        forecast_df.set_index('Date', inplace=True)
        return forecast_df
    
    def plot_forecast(self, forecast_df):
        plt.figure(figsize=(14,7))
        plt.plot(sol_data_all.index, sol_data_all['Close'], label='Preço Real')
        plt.plot(forecast_df.index, forecast_df['Predicted_Close'], label='Previsão', linestyle='--')
        plt.title(f'Previsão dos Preços de Fechamento da {self.CryptoName} para os Próximos 30 Dias')
        plt.xlabel('Data')
        plt.ylabel('Preço (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

```


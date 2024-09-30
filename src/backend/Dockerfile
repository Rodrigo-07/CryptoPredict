# Usar imagem oficial do Python 3.10
FROM python:3.10-slim

# Definir diretório de trabalho no container
WORKDIR /app

# Copiar o arquivo de dependências para o container
COPY requirements.txt .

# Instalar dependências
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt


# Copiar todos os arquivos da aplicação para o container
COPY . .

# Expor a porta 8000
EXPOSE 8000

# Comando para rodar a aplicação FastAPI com Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

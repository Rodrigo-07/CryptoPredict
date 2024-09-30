# Usar a imagem base Python
FROM python:3.9-slim

# Instalar as dependências do sistema (incluindo compiladores e bibliotecas)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos de dependências
COPY requirements.txt .

# Instalar dependências do Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar o restante da aplicação
COPY . .

# Expor a porta em que a aplicação irá rodar
EXPOSE 8000

# Rodar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

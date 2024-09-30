FROM python:3.10

# Definir o diretório de trabalho dentro do container
WORKDIR /code

# Copiar os requirements para dentro do container
COPY ./backend/requirements.txt /code/requirements.txt

# Instalar as dependências
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar o conteúdo do backend e frontend para dentro do container
COPY ./backend /code/backend
COPY ../frontend /code/frontend

# Expor a porta para rodar a aplicação
EXPOSE 8000

# Rodar a aplicação
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

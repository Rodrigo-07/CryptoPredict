FROM python:3.12

# Diretorio de trabalho do containeir
WORKDIR /code

# Copiar os requirements pra dentro do container
COPY ./requirements.txt /code/requirements.txt

# Instalar as dependências
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar todo conteudo para dentro do code
COPY . /code

# Porta que vai rodar a aplicação
EXPOSE 8000

# Rodar aplicação
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
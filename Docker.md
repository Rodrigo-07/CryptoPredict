# Documentação de Dockerização

## Introdução

Docker é uma plataforma de código aberto que permite automatizar a implantação, o dimensionamento e a gestão de aplicações em contêineres. Os contêineres empacotam código e todas as suas dependências, garantindo que a aplicação possa ser executada de forma consistente em qualquer ambiente.

Este projeto utiliza Docker para isolar dois serviços principais: o backend e o banco de dados MongoDB, facilitando a execução em diferentes ambientes com configurações consistentes.

Vídeo de demonstração: [Link](https://youtu.be/gxo0CJ2FP6E)

## Docker Backend

O Dockerfile do backend configura o ambiente para rodar uma aplicação FastAPI. Ele define a imagem base do Python 3.10, instala as dependências a partir do arquivo `requirements.txt`, copia os arquivos do projeto e expõe a porta 8000 para que o FastAPI seja executado.

### Dockerfile

```Dockerfile
FROM python:3.10

# Definir o diretório de trabalho dentro do container
WORKDIR /code

# Copiar os requirements para dentro do container
COPY ./backend/requirements.txt /code/requirements.txt

# Instalar as dependências
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar o conteúdo do backend e frontend para dentro do container
COPY ./backend /code/backend
COPY ./frontend /code/frontend

# Expor a porta para rodar a aplicação
EXPOSE 8000

# Rodar a aplicação
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

```

Sumarizando um pouco, esse Dockerfile faz o seguinte:

- Define a imagem base do Python 3.10
- Define o diretório de trabalho dentro do container
- Copia o arquivo `requirements.txt` para dentro do container
- Instala as dependências a partir do arquivo `requirements.txt`
- Copia os arquivos do backend e frontend para dentro do container
- Expõe a porta 8000 para rodar a aplicação
- Roda a aplicação com o comando `uvicorn backend.main:app --host

Para construir a imagem do backend, execute o seguinte comando:

```bash

docker build -t backend .

docker run -d -p 8000:8000 backend

```

## Docker MongoDB

Não temos um Dockerfile para o MongoDB, pois utilizamos a imagem oficial do MongoDB disponível no Docker Hub. Para rodar o MongoDB, execute o seguinte comando:

```bash

docker run -d mongodb/mongodb-community-server:latest -p 27017:27017 --name mongodb-container -v mongodbdata:/data/db mongo:latest

```

Este comando cria um contêiner com o MongoDB, expondo a porta 27017 e criando um volume para armazenar os dados do banco.

## Docker Compose

Para facilitar a execução dos serviços, utilizamos o Docker Compose. O arquivo `docker-compose.yml` define os serviços do backend e do MongoDB, permitindo que ambos sejam executados com um único comando.

### docker-compose.yml

```yaml

version: "3.8"

services:
  backend:
    build:
      context: .  # O contexto agora é a raiz do diretório src
      dockerfile: ./backend/Dockerfile  # O Dockerfile está na pasta backend
    container_name: fastapi-backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/code/backend  # Volume para o backend
      - ./frontend:/code/frontend  # Volume para o frontend
    environment:
      - MONGO_URI=mongodb://mongo:27017
    depends_on:
      - mongo
    networks:
      - app-network

  mongo:
    image: mongo:latest
    container_name: mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongodbdata:/data/db
    networks:
      - app-network

volumes:
  mongodbdata:

networks:
  app-network:
    driver: bridge

```

Este arquivo define dois serviços: o backend e o MongoDB. O serviço do backend utiliza o Dockerfile do backend, expõe a porta 8000 e define um volume para o backend e o frontend. O serviço do MongoDB utiliza a imagem oficial do MongoDB, expõe a porta 27017 e cria um volume para armazenar os dados do banco.

Para rodar os serviços com o Docker Compose, execute o seguinte comando:

```bash

cd src

docker-compose up --build

```

Este comando cria e inicia os contêineres do backend e do MongoDB, permitindo que a aplicação seja acessada em `http://localhost:8000`.

# Justificativa de não utilziar um data lake

A utilização de um data lake não é necessária para este projeto, pois o volume de dados é pequeno e a aplicação não requer armazenamento de dados em larga escala. Além disso, todos os dados que eu uso, Indexes e valores de moeda, são obtidos de APIs externas e não precisam ser armazenados em um data lake.
from fastapi import FastAPI
from routers import prediction, training, graphs

app = FastAPI()

# Incluindo as rotas
app.include_router(prediction.router)
app.include_router(training.router)
app.include_router(graphs.router)

# Código para rodar a aplicação

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port = 8000)
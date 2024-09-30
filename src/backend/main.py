from fastapi import FastAPI, Request
from routers import prediction, database
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


templates = Jinja2Templates(directory="/code/frontend/templates")

app.mount("/static", StaticFiles(directory="/code/frontend/static"), name="static")

# Incluindo as rotas
app.include_router(prediction.router)
app.include_router(database.router)

# Liberar cors 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/retrain")
def read_root(request: Request):
    return templates.TemplateResponse("retrain.html", {"request": request})

# Código para rodar a aplicação

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port = 8000)
from fastapi import APIRouter
from backend.routers.prediction import router as prediction_router
from backend.routers.database import router as database_router

router = APIRouter()

router.include_router(prediction_router, prefix="/prediction")
router.include_router(database_router, prefix="/database")
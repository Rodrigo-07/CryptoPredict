from fastapi import APIRouter
from .prediction import router as prediction_router
from .database import router as database_router

router = APIRouter()

router.include_router(prediction_router, prefix="/prediction")
router.include_router(database_router, prefix="/database")
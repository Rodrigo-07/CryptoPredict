from fastapi import APIRouter
from .prediction import router as prediction_router
from .training import router as training_router

router = APIRouter()

router.include_router(prediction_router, prefix="/prediction")
router.include_router(training_router, prefix="/training")
from fastapi import APIRouter
from .graphs import router as graphs_router
from .prediction import router as prediction_router
from .training import router as training_router

router = APIRouter()

router.include_router(graphs_router, prefix="/graphs")
router.include_router(prediction_router, prefix="/prediction")
router.include_router(training_router, prefix="/training")
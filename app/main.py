"""
Log Clustering Service — FastAPI application entry point.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import settings
from app.db.database import init_db
from app.services.clustering import clustering_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Initializing database …")
    await init_db()

    logger.info("Loading clustering model …")
    loaded = clustering_service.load(settings.MODEL_DIR)
    if not loaded:
        logger.warning(
            "No pre-trained model found in %s. "
            "Run 'python ml/train.py' or POST /api/v1/retrain after inserting events.",
            settings.MODEL_DIR,
        )

    yield  # application is running

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down …")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "SOC log clustering service — automatically groups Windows, Linux, and HDFS "
        "log events to reduce noise and surface anomalies."
    ),
    lifespan=lifespan,
)

# Static files and templates
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# API routes
app.include_router(router)


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Dashboard"})


@app.get("/clusters-view", response_class=HTMLResponse, include_in_schema=False)
async def clusters_view(request: Request):
    return templates.TemplateResponse("clusters.html", {"request": request, "title": "Clusters"})


@app.get("/visualize-view", response_class=HTMLResponse, include_in_schema=False)
async def visualize_view(request: Request):
    return templates.TemplateResponse("visualize.html", {"request": request, "title": "Visualization"})


@app.get("/health", include_in_schema=False)
async def health():
    return {
        "status": "ok",
        "model_loaded": clustering_service.is_fitted,
        "n_clusters": len(clustering_service.clusters),
    }

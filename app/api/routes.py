"""
FastAPI route definitions.
"""
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.log_schema import (
    ClassifyResponse,
    ClusterDetailResponse,
    ClusterInfo,
    LogEventRequest,
    LogEventResponse,
    StatsResponse,
    VisualizationPoint,
    VisualizationResponse,
)
from app.services.clustering import clustering_service
from app.services.log_store import log_store
from app.services.preprocessing import LogPreprocessor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
preprocessor = LogPreprocessor()


# ── Classify ──────────────────────────────────────────────────────────────────

@router.post("/classify", response_model=ClassifyResponse, summary="Classify a log event")
async def classify_log(payload: LogEventRequest, db: AsyncSession = Depends(get_db)):
    """
    Accept a raw log line, parse it, assign it to a cluster, and persist it.
    Returns the cluster information and anomaly flag.
    """
    if not clustering_service.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Clustering model is not loaded yet. Wait for startup to complete or call /api/v1/retrain.",
        )

    parsed = preprocessor.parse(payload.raw_log, payload.source_type)

    try:
        label, confidence, is_anomaly = clustering_service.classify(parsed)
    except Exception as e:
        logger.exception("Classification error: %s", e)
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    # Resolve DB cluster record
    db_cluster_id: Optional[int] = None
    db_cluster = None
    if not is_anomaly and label in clustering_service.clusters:
        meta = clustering_service.clusters[label]
        cluster_obj = await log_store.get_or_create_cluster(label, meta, db)
        db_cluster_id = cluster_obj.id
        db_cluster = cluster_obj
        await log_store.increment_cluster_size(db_cluster_id, db)
    elif is_anomaly:
        # store under the anomaly pseudo-cluster (label=-1)
        if -1 in clustering_service.clusters:
            meta = clustering_service.clusters[-1]
            cluster_obj = await log_store.get_or_create_cluster(-1, meta, db)
            db_cluster_id = cluster_obj.id
            db_cluster = cluster_obj

    event = await log_store.save_event(parsed, db_cluster_id, is_anomaly, confidence, db)

    event_resp = LogEventResponse(
        id=event.id,
        raw_log=event.raw_log,
        source_type=event.source_type,
        log_level=event.log_level,
        event_id=event.event_id,
        hostname=event.hostname,
        process_name=event.process_name,
        message=event.message,
        normalized_message=event.normalized_message,
        cluster_id=event.cluster_id,
        is_anomaly=event.is_anomaly,
        confidence=event.confidence,
        log_timestamp=event.log_timestamp,
        created_at=event.created_at,
        cluster=ClusterInfo.model_validate(db_cluster) if db_cluster else None,
    )

    cluster_info = ClusterInfo.model_validate(db_cluster) if db_cluster else None

    return ClassifyResponse(
        event=event_resp,
        cluster=cluster_info,
        is_anomaly=is_anomaly,
        confidence=confidence,
        message=(
            "Anomaly detected — log does not match any known cluster."
            if is_anomaly
            else f"Assigned to cluster #{label} with confidence {confidence:.2%}."
        ),
    )


# ── Clusters ──────────────────────────────────────────────────────────────────

@router.get("/clusters", response_model=list[ClusterInfo], summary="List all clusters")
async def list_clusters(db: AsyncSession = Depends(get_db)):
    clusters = await log_store.get_all_clusters(db)
    return [ClusterInfo.model_validate(c) for c in clusters]


@router.get("/clusters/{cluster_id}", response_model=ClusterDetailResponse, summary="Cluster detail")
async def get_cluster(cluster_id: int, db: AsyncSession = Depends(get_db)):
    cluster = await log_store.get_cluster_by_id(cluster_id, db)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    events = await log_store.get_cluster_events(cluster_id, db, limit=20)
    return ClusterDetailResponse(
        cluster=ClusterInfo.model_validate(cluster),
        recent_events=[
            LogEventResponse(
                id=e.id,
                raw_log=e.raw_log,
                source_type=e.source_type,
                log_level=e.log_level,
                event_id=e.event_id,
                hostname=e.hostname,
                process_name=e.process_name,
                message=e.message,
                normalized_message=e.normalized_message,
                cluster_id=e.cluster_id,
                is_anomaly=e.is_anomaly,
                confidence=e.confidence,
                log_timestamp=e.log_timestamp,
                created_at=e.created_at,
                cluster=ClusterInfo.model_validate(cluster),
            )
            for e in events
        ],
        total_events=cluster.size,
    )


# ── Events ────────────────────────────────────────────────────────────────────

@router.get("/events", response_model=list[LogEventResponse], summary="Recent events")
async def list_events(
    limit: int = Query(50, ge=1, le=500),
    source_type: Optional[str] = Query(None),
    cluster_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    events = await log_store.get_recent_events(db, limit=limit, source_type=source_type, cluster_id=cluster_id)
    result = []
    for e in events:
        result.append(
            LogEventResponse(
                id=e.id,
                raw_log=e.raw_log,
                source_type=e.source_type,
                log_level=e.log_level,
                event_id=e.event_id,
                hostname=e.hostname,
                process_name=e.process_name,
                message=e.message,
                normalized_message=e.normalized_message,
                cluster_id=e.cluster_id,
                is_anomaly=e.is_anomaly,
                confidence=e.confidence,
                log_timestamp=e.log_timestamp,
                created_at=e.created_at,
                cluster=ClusterInfo.model_validate(e.cluster) if e.cluster else None,
            )
        )
    return result


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse, summary="Service statistics")
async def get_stats(db: AsyncSession = Depends(get_db)):
    stats = await log_store.get_stats(db)
    return StatsResponse(
        total_events=stats["total_events"],
        total_clusters=stats["total_clusters"],
        anomaly_count=stats["anomaly_count"],
        anomaly_rate=stats["anomaly_rate"],
        by_source=stats["by_source"],
        top_clusters=[ClusterInfo.model_validate(c) for c in stats["top_clusters"]],
    )


# ── Visualization ─────────────────────────────────────────────────────────────

@router.get("/visualize", response_model=VisualizationResponse, summary="2D t-SNE visualization data")
async def get_visualization(
    limit: int = Query(1000, ge=10, le=5000),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns 2D coordinates (t-SNE reduced) for scatter-plot visualization.
    """
    if not clustering_service.is_fitted:
        raise HTTPException(status_code=503, detail="Model not ready")

    events = await log_store.get_visualization_data(db, limit=limit)
    if not events:
        return VisualizationResponse(points=[], clusters=[])

    texts = [e.normalized_message or e.raw_log for e in events]

    X = clustering_service.vectorizer.transform(texts)

    from sklearn.preprocessing import normalize as sk_normalize
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    X_norm = sk_normalize(X, norm="l2")

    # Reduce to 50 dims first with SVD (fast), then t-SNE
    n_components_svd = min(50, X_norm.shape[1] - 1, X_norm.shape[0] - 1)
    if n_components_svd > 1:
        svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
        X_reduced = svd.fit_transform(X_norm)
    else:
        X_reduced = X_norm.toarray() if hasattr(X_norm, "toarray") else X_norm

    perplexity = min(30, max(5, len(events) // 10))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=500)
    coords = tsne.fit_transform(X_reduced)

    points = []
    for i, (event, (x, y)) in enumerate(zip(events, coords)):
        cluster_label = -1
        if event.cluster:
            cluster_label = event.cluster.label
        points.append(
            VisualizationPoint(
                id=event.id,
                x=float(x),
                y=float(y),
                cluster_id=event.cluster_id,
                cluster_label=cluster_label,
                source_type=event.source_type,
                is_anomaly=event.is_anomaly,
                message_preview=(event.message or event.raw_log)[:100],
            )
        )

    clusters = await log_store.get_all_clusters(db)
    return VisualizationResponse(
        points=points,
        clusters=[ClusterInfo.model_validate(c) for c in clusters],
    )


# ── Model management ──────────────────────────────────────────────────────────

@router.post("/retrain", summary="Re-train the clustering model")
async def retrain(db: AsyncSession = Depends(get_db)):
    """
    Re-train the clustering model using all log events stored in the database.
    """
    from app.services.preprocessing import LogPreprocessor as LP
    from app.db.models import LogEvent
    from sqlalchemy import select

    result = await db.scalars(select(LogEvent))
    events = list(result.all())
    if not events:
        raise HTTPException(status_code=400, detail="No events in DB to train on")

    preprocessor_local = LP()
    parsed = [
        preprocessor_local.parse(e.raw_log, e.source_type)
        for e in events
    ]

    metrics = clustering_service.fit(parsed)
    clustering_service.save()
    await log_store.sync_clusters(clustering_service.clusters, db)

    return {"status": "ok", "metrics": metrics, "n_events": len(events)}

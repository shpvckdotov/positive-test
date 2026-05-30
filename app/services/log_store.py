"""
Database persistence layer for log events and clusters.
"""
import json
import logging
from typing import Optional

import numpy as np
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Cluster, LogEvent
from app.services.clustering import ClusterMeta
from app.services.preprocessing import ParsedLog

logger = logging.getLogger(__name__)


class LogStore:
    """Async repository for log events and clusters."""

    # ── Clusters ──────────────────────────────────────────────────────────────

    async def sync_clusters(
        self, cluster_metas: dict[int, ClusterMeta], db: AsyncSession
    ) -> dict[int, int]:
        """
        Upsert ClusterMeta objects into the DB.
        Returns mapping: dbscan_label → db_cluster_id
        """
        label_to_id: dict[int, int] = {}
        for label, meta in cluster_metas.items():
            existing = await db.scalar(
                select(Cluster).where(Cluster.label == label)
            )
            centroid_json = json.dumps(meta.centroid.tolist())
            if existing:
                existing.template = meta.template
                existing.source_type = meta.source_type
                existing.size = meta.size
                existing.is_anomaly = meta.is_anomaly
                existing.centroid_json = centroid_json
                label_to_id[label] = existing.id
            else:
                cluster = Cluster(
                    label=label,
                    template=meta.template,
                    source_type=meta.source_type,
                    size=meta.size,
                    is_anomaly=meta.is_anomaly,
                    centroid_json=centroid_json,
                )
                db.add(cluster)
                await db.flush()
                label_to_id[label] = cluster.id

        await db.commit()
        return label_to_id

    async def get_or_create_cluster(
        self, label: int, meta: ClusterMeta, db: AsyncSession
    ) -> Cluster:
        existing = await db.scalar(select(Cluster).where(Cluster.label == label))
        if existing:
            return existing
        cluster = Cluster(
            label=label,
            template=meta.template,
            source_type=meta.source_type,
            size=meta.size,
            is_anomaly=meta.is_anomaly,
            centroid_json=json.dumps(meta.centroid.tolist()),
        )
        db.add(cluster)
        await db.flush()
        return cluster

    async def increment_cluster_size(self, cluster_id: int, db: AsyncSession) -> None:
        cluster = await db.get(Cluster, cluster_id)
        if cluster:
            cluster.size += 1

    async def get_all_clusters(self, db: AsyncSession) -> list[Cluster]:
        result = await db.scalars(
            select(Cluster).order_by(desc(Cluster.size))
        )
        return list(result.all())

    async def get_cluster_by_id(self, cluster_id: int, db: AsyncSession) -> Optional[Cluster]:
        return await db.get(Cluster, cluster_id)

    # ── Log Events ────────────────────────────────────────────────────────────

    async def save_event(
        self,
        parsed: ParsedLog,
        cluster_id: Optional[int],
        is_anomaly: bool,
        confidence: Optional[float],
        db: AsyncSession,
    ) -> LogEvent:
        event = LogEvent(
            raw_log=parsed.raw,
            source_type=parsed.source_type,
            log_level=parsed.log_level,
            event_id=parsed.event_id,
            hostname=parsed.hostname,
            process_name=parsed.process_name,
            message=parsed.message,
            normalized_message=parsed.normalized_message,
            cluster_id=cluster_id,
            is_anomaly=is_anomaly,
            confidence=confidence,
            log_timestamp=parsed.log_timestamp,
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event

    async def get_event_by_id(self, event_id: int, db: AsyncSession) -> Optional[LogEvent]:
        result = await db.scalar(
            select(LogEvent)
            .options(selectinload(LogEvent.cluster))
            .where(LogEvent.id == event_id)
        )
        return result

    async def get_recent_events(
        self,
        db: AsyncSession,
        limit: int = 50,
        source_type: Optional[str] = None,
        cluster_id: Optional[int] = None,
    ) -> list[LogEvent]:
        q = select(LogEvent).options(selectinload(LogEvent.cluster))
        if source_type:
            q = q.where(LogEvent.source_type == source_type)
        if cluster_id is not None:
            q = q.where(LogEvent.cluster_id == cluster_id)
        q = q.order_by(desc(LogEvent.created_at)).limit(limit)
        result = await db.scalars(q)
        return list(result.all())

    async def get_cluster_events(
        self, cluster_id: int, db: AsyncSession, limit: int = 20
    ) -> list[LogEvent]:
        result = await db.scalars(
            select(LogEvent)
            .where(LogEvent.cluster_id == cluster_id)
            .order_by(desc(LogEvent.created_at))
            .limit(limit)
        )
        return list(result.all())

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def get_stats(self, db: AsyncSession) -> dict:
        total_events = await db.scalar(select(func.count(LogEvent.id))) or 0
        total_clusters = await db.scalar(
            select(func.count(Cluster.id)).where(Cluster.is_anomaly == False)
        ) or 0
        anomaly_count = await db.scalar(
            select(func.count(LogEvent.id)).where(LogEvent.is_anomaly == True)
        ) or 0

        # events by source
        rows = await db.execute(
            select(LogEvent.source_type, func.count(LogEvent.id))
            .group_by(LogEvent.source_type)
        )
        by_source = {row[0]: row[1] for row in rows.all()}

        # top clusters by size
        top_clusters_q = await db.scalars(
            select(Cluster)
            .where(Cluster.is_anomaly == False)
            .order_by(desc(Cluster.size))
            .limit(5)
        )
        top_clusters = list(top_clusters_q.all())

        return {
            "total_events": total_events,
            "total_clusters": total_clusters,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / total_events if total_events else 0.0,
            "by_source": by_source,
            "top_clusters": top_clusters,
        }

    # ── Visualization ─────────────────────────────────────────────────────────

    async def get_visualization_data(
        self, db: AsyncSession, limit: int = 2000
    ) -> list[LogEvent]:
        result = await db.scalars(
            select(LogEvent)
            .options(selectinload(LogEvent.cluster))
            .order_by(desc(LogEvent.created_at))
            .limit(limit)
        )
        return list(result.all())


log_store = LogStore()

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    label: Mapped[int] = mapped_column(Integer, unique=True, index=True)  # DBSCAN label (-1 = anomaly)
    template: Mapped[str] = mapped_column(Text)           # representative log template
    source_type: Mapped[str] = mapped_column(String(50))  # linux / windows / hdfs
    size: Mapped[int] = mapped_column(Integer, default=0)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    centroid_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON serialized centroid
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    events: Mapped[list["LogEvent"]] = relationship("LogEvent", back_populates="cluster")

    def __repr__(self) -> str:
        return f"<Cluster(id={self.id}, label={self.label}, source={self.source_type}, size={self.size})>"


class LogEvent(Base):
    __tablename__ = "log_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    raw_log: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(50), index=True)  # linux / windows / hdfs
    log_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    event_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # Windows event ID
    hostname: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    process_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    normalized_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("clusters.id"), nullable=True)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # similarity score
    log_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    cluster: Mapped[Optional["Cluster"]] = relationship("Cluster", back_populates="events")

    __table_args__ = (
        Index("ix_log_events_source_created", "source_type", "created_at"),
        Index("ix_log_events_cluster_id", "cluster_id"),
    )

    def __repr__(self) -> str:
        return f"<LogEvent(id={self.id}, source={self.source_type}, cluster_id={self.cluster_id})>"

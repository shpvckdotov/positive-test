from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class LogEventRequest(BaseModel):
    raw_log: str = Field(..., description="Raw log line to classify")
    source_type: str = Field(..., description="Log source: 'linux', 'windows', or 'hdfs'")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "raw_log": "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4",
                    "source_type": "linux"
                }
            ]
        }
    }


class ClusterInfo(BaseModel):
    id: int
    label: int
    template: str
    source_type: str
    size: int
    is_anomaly: bool
    description: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class LogEventResponse(BaseModel):
    id: int
    raw_log: str
    source_type: str
    log_level: Optional[str]
    event_id: Optional[str]
    hostname: Optional[str]
    process_name: Optional[str]
    message: Optional[str]
    normalized_message: Optional[str]
    cluster_id: Optional[int]
    is_anomaly: bool
    confidence: Optional[float]
    log_timestamp: Optional[datetime]
    created_at: datetime
    cluster: Optional[ClusterInfo]

    model_config = {"from_attributes": True}


class ClassifyResponse(BaseModel):
    event: LogEventResponse
    cluster: Optional[ClusterInfo]
    is_anomaly: bool
    confidence: Optional[float]
    message: str


class ClusterDetailResponse(BaseModel):
    cluster: ClusterInfo
    recent_events: List[LogEventResponse]
    total_events: int


class StatsResponse(BaseModel):
    total_events: int
    total_clusters: int
    anomaly_count: int
    anomaly_rate: float
    by_source: dict
    top_clusters: List[ClusterInfo]


class VisualizationPoint(BaseModel):
    id: int
    x: float
    y: float
    cluster_id: Optional[int]
    cluster_label: int
    source_type: str
    is_anomaly: bool
    message_preview: str


class VisualizationResponse(BaseModel):
    points: List[VisualizationPoint]
    clusters: List[ClusterInfo]

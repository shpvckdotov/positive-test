"""
Clustering service: TF-IDF vectorization + DBSCAN.

Training:
  fit(parsed_logs) → builds vectorizer and cluster assignments

Inference:
  classify(parsed_log) → (cluster_label, confidence, is_anomaly)
"""
import json
import logging
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from app.core.config import settings
from app.services.preprocessing import ParsedLog

logger = logging.getLogger(__name__)


@dataclass
class ClusterMeta:
    label: int
    template: str        # most representative message
    source_type: str
    size: int
    is_anomaly: bool
    centroid: np.ndarray  # mean TF-IDF vector (normalized)


class LogClusteringService:
    """
    Offline-trained, online-inference log clustering service.

    Workflow:
      1. fit() — train on a corpus of ParsedLog objects
      2. classify() — assign a new log to an existing cluster
      3. save() / load() — persist model artifacts
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.clusters: dict[int, ClusterMeta] = {}   # label → ClusterMeta
        self._is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, parsed_logs: list[ParsedLog]) -> dict:
        """
        Train TF-IDF + DBSCAN on a list of ParsedLog objects.
        Returns evaluation metrics.
        """
        if not parsed_logs:
            raise ValueError("No logs provided for training")

        texts = [p.normalized_message for p in parsed_logs]
        source_types = [p.source_type for p in parsed_logs]

        logger.info("Fitting TF-IDF on %d log lines …", len(texts))
        self.vectorizer = TfidfVectorizer(
            max_features=settings.TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        X = self.vectorizer.fit_transform(texts)
        X_norm = normalize(X, norm="l2")

        logger.info("Running DBSCAN (eps=%.2f, min_samples=%d) …",
                    settings.DBSCAN_EPS, settings.DBSCAN_MIN_SAMPLES)
        db = DBSCAN(
            eps=settings.DBSCAN_EPS,
            min_samples=settings.DBSCAN_MIN_SAMPLES,
            metric="cosine",
            n_jobs=-1,
        )
        labels = db.fit_predict(X_norm)

        self.clusters = self._build_cluster_meta(X_norm, labels, parsed_logs)
        self._is_fitted = True

        metrics = self._evaluate(X_norm, labels)
        logger.info("Clustering complete. %d clusters, %d anomalies. Metrics: %s",
                    len([c for c in self.clusters.values() if not c.is_anomaly]),
                    len([c for c in self.clusters.values() if c.is_anomaly]),
                    metrics)
        return metrics

    def _build_cluster_meta(
        self,
        X_norm: np.ndarray,
        labels: np.ndarray,
        parsed_logs: list[ParsedLog],
    ) -> dict[int, ClusterMeta]:
        unique_labels = set(labels)
        clusters: dict[int, ClusterMeta] = {}

        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]
            cluster_vectors = X_norm[mask]

            # centroid = mean of l2-normalized vectors, then re-normalize
            # .mean(axis=0) on a sparse matrix returns np.matrix → convert first
            mean_vec = np.asarray(cluster_vectors.mean(axis=0))
            centroid = normalize(mean_vec, norm="l2").flatten()

            # pick representative: closest to centroid
            sims = cosine_similarity(centroid.reshape(1, -1), cluster_vectors)[0]
            rep_idx = indices[np.argmax(sims)]
            template = parsed_logs[rep_idx].message

            # dominant source type in this cluster
            source_counts = Counter(parsed_logs[i].source_type for i in indices)
            dominant_source = source_counts.most_common(1)[0][0]

            clusters[label] = ClusterMeta(
                label=label,
                template=template,
                source_type=dominant_source,
                size=int(mask.sum()),
                is_anomaly=(label == -1),
                centroid=centroid,
            )

        return clusters

    def _evaluate(self, X_norm: np.ndarray, labels: np.ndarray) -> dict:
        metrics: dict = {
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "n_noise": int((labels == -1).sum()),
            "noise_ratio": float((labels == -1).mean()),
        }
        # silhouette / davies-bouldin need ≥ 2 clusters and non-noise samples
        non_noise = labels != -1
        if metrics["n_clusters"] >= 2 and non_noise.sum() > metrics["n_clusters"]:
            X_dense = X_norm[non_noise]
            if hasattr(X_dense, "toarray"):
                X_dense = X_dense.toarray()
            lbl_sub = labels[non_noise]
            try:
                metrics["silhouette_score"] = float(
                    silhouette_score(X_dense, lbl_sub, metric="cosine")
                )
                metrics["davies_bouldin_score"] = float(
                    davies_bouldin_score(X_dense, lbl_sub)
                )
            except Exception as e:
                logger.warning("Could not compute clustering metrics: %s", e)
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def classify(self, parsed_log: ParsedLog) -> tuple[int, float, bool]:
        """
        Assign a new log to the nearest cluster.
        Returns (cluster_label, confidence, is_anomaly).
        confidence ∈ [0, 1] — cosine similarity to nearest centroid.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Run fit() or load() first.")

        text = parsed_log.normalized_message
        vec = self.vectorizer.transform([text])
        vec_norm = normalize(vec, norm="l2")
        if hasattr(vec_norm, "toarray"):
            vec_norm_arr = vec_norm.toarray()
        else:
            vec_norm_arr = np.asarray(vec_norm)

        # Exclude the noise cluster (-1) from candidate centroids
        candidate_labels = [lb for lb in self.clusters if lb != -1]
        if not candidate_labels:
            return -1, 0.0, True

        centroids = np.stack([self.clusters[lb].centroid for lb in candidate_labels])
        sims = cosine_similarity(vec_norm_arr, centroids)[0]
        best_idx = int(np.argmax(sims))
        best_label = candidate_labels[best_idx]
        confidence = float(sims[best_idx])

        is_anomaly = confidence < (1 - settings.ANOMALY_THRESHOLD)
        if is_anomaly:
            return -1, confidence, True
        return best_label, confidence, False

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_dir: Optional[str] = None) -> None:
        path = model_dir or settings.MODEL_DIR
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

        # save clusters (centroid as list for JSON compat)
        clusters_serial = {
            str(k): {
                "label": int(v.label),
                "template": v.template,
                "source_type": v.source_type,
                "size": int(v.size),
                "is_anomaly": bool(v.is_anomaly),
                "centroid": [float(x) for x in v.centroid],
            }
            for k, v in self.clusters.items()
        }
        with open(os.path.join(path, "clusters.json"), "w") as f:
            json.dump(clusters_serial, f)

        logger.info("Model saved to %s", path)

    def load(self, model_dir: Optional[str] = None) -> bool:
        path = model_dir or settings.MODEL_DIR
        vec_path = os.path.join(path, "vectorizer.pkl")
        cls_path = os.path.join(path, "clusters.json")

        if not (os.path.exists(vec_path) and os.path.exists(cls_path)):
            logger.warning("Model files not found in %s", path)
            return False

        with open(vec_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(cls_path, "r") as f:
            raw = json.load(f)

        self.clusters = {}
        for k, v in raw.items():
            self.clusters[int(k)] = ClusterMeta(
                label=v["label"],
                template=v["template"],
                source_type=v["source_type"],
                size=v["size"],
                is_anomaly=v["is_anomaly"],
                centroid=np.array(v["centroid"]),
            )

        self._is_fitted = True
        logger.info("Model loaded from %s (%d clusters)", path, len(self.clusters))
        return True

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# Singleton instance shared across the application
clustering_service = LogClusteringService()

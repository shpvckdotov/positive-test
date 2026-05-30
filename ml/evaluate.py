"""
Evaluation script — computes clustering quality metrics and prints cluster summaries.

Usage:
    python ml/evaluate.py [--linux PATH] [--windows PATH] [--hdfs PATH] [--model-dir PATH]
"""
import argparse
import logging
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import settings
from app.services.clustering import clustering_service
from app.services.preprocessing import LogPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linux",   default=settings.LINUX_LOG_PATH)
    parser.add_argument("--windows", default=settings.WINDOWS_LOG_PATH)
    parser.add_argument("--hdfs",    default=settings.HDFS_LOG_PATH)
    parser.add_argument("--model-dir", default=settings.MODEL_DIR)
    parser.add_argument("--max-lines", type=int, default=20_000)
    args = parser.parse_args()

    ok = clustering_service.load(args.model_dir)
    if not ok:
        print("Model not found. Run ml/train.py first.")
        sys.exit(1)

    preprocessor = LogPreprocessor()
    all_logs = []
    for source_type, path in [("linux", args.linux), ("windows", args.windows), ("hdfs", args.hdfs)]:
        if path and os.path.exists(path):
            logs = preprocessor.parse_file(path, source_type)
            all_logs.extend(logs[:args.max_lines])

    if not all_logs:
        print("No log data found.")
        sys.exit(1)

    texts = [p.normalized_message for p in all_logs]
    X = clustering_service.vectorizer.transform(texts)
    X_norm = normalize(X, norm="l2")

    # Re-predict labels using nearest centroid
    labels = []
    for log in all_logs:
        lbl, conf, is_anom = clustering_service.classify(log)
        labels.append(lbl)
    labels = np.array(labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_ratio = n_noise / len(labels)

    print("\n" + "=" * 60)
    print("CLUSTERING EVALUATION")
    print(f"  Total samples:    {len(all_logs)}")
    print(f"  Clusters found:   {n_clusters}")
    print(f"  Noise points:     {n_noise} ({noise_ratio:.1%})")

    non_noise = labels != -1
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        X_sub = X_norm[non_noise]
        if hasattr(X_sub, "toarray"):
            X_sub = X_sub.toarray()
        lbl_sub = labels[non_noise]
        sil = silhouette_score(X_sub, lbl_sub, metric="cosine")
        db  = davies_bouldin_score(X_sub, lbl_sub)
        ch  = calinski_harabasz_score(X_sub, lbl_sub)
        print(f"\nINTRINSIC METRICS")
        print(f"  Silhouette score:       {sil:.4f}  (range −1..1, higher = better)")
        print(f"  Davies-Bouldin index:   {db:.4f}  (lower = better)")
        print(f"  Calinski-Harabasz idx:  {ch:.2f}  (higher = better)")

    print("\nTOP CLUSTERS BY SIZE")
    print(f"  {'Label':>6}  {'Size':>6}  {'Source':>8}  Template")
    for lbl, meta in sorted(clustering_service.clusters.items(),
                             key=lambda x: -x[1].size)[:15]:
        marker = "[ANOMALY]" if meta.is_anomaly else ""
        template_preview = meta.template[:70].replace("\n", " ")
        print(f"  {lbl:>6}  {meta.size:>6}  {meta.source_type:>8}  {template_preview} {marker}")

    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Offline training script.

Usage (inside Docker or locally):
    python ml/train.py [--linux PATH] [--windows PATH] [--hdfs PATH] [--model-dir PATH]

Downloads sample LogHub data automatically if paths are not provided.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import settings
from app.services.clustering import clustering_service
from app.services.preprocessing import LogPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


from pathlib import Path
import os

class ModelManager:
    def __init__(self):
        current_file_path = Path(__file__).resolve()
        
        project_root = current_file_path.parent.parent
        
        self.models_dir = project_root / "models"
        
        self.vectorizer_path = self.models_dir / "vectorizer.pkl"
        self.clusters_path = self.models_dir / "clusters.json"

    def check_files_exist(self) -> bool:
        if not self.models_dir.exists():
            return False
            
        files_found = []
        if self.vectorizer_path.exists():
            files_found.append("vectorizer.pkl")
        else:
            pass
            
        if self.clusters_path.exists():
            files_found.append("clusters.json")
        else:
            pass
            
        if len(files_found) == 2:
            return True
        else:
            return False
        

def load_logs(preprocessor: LogPreprocessor, paths: dict[str, str]):
    all_logs = []
    for source_type, path in paths.items():
        if not path or not os.path.exists(path):
            logger.warning("Skipping %s — file not found: %s", source_type, path)
            continue
        logger.info("Loading %s logs from %s …", source_type, path)
        logs = preprocessor.parse_file(path, source_type)
        logger.info("  → %d lines parsed", len(logs))
        all_logs.extend(logs)
    return all_logs


def main():
    parser = argparse.ArgumentParser(description="Train log clustering model")
    parser.add_argument("--linux",   default=settings.LINUX_LOG_PATH)
    parser.add_argument("--windows", default=settings.WINDOWS_LOG_PATH)
    parser.add_argument("--hdfs",    default=settings.HDFS_LOG_PATH)
    parser.add_argument("--model-dir", default=settings.MODEL_DIR)
    parser.add_argument("--max-lines", type=int, default=50_000,
                        help="Max lines to load per source (default: 50 000)")
    args = parser.parse_args()
    model_manager = ModelManager()
    preprocessor = LogPreprocessor()

    paths = {
        "linux":   args.linux,
        "windows": args.windows,
        "hdfs":    args.hdfs,
    }

    all_logs = load_logs(preprocessor, paths)
    if not all_logs:
        logger.error("No log data found. Provide at least one valid log file.")
        sys.exit(1)

    # Limit to avoid OOM on large datasets
    if len(all_logs) > args.max_lines:
        import random
        random.seed(42)
        all_logs = random.sample(all_logs, args.max_lines)
        logger.info("Sampled %d lines for training", args.max_lines)

    logger.info("Total logs for training: %d", len(all_logs))

    metrics = clustering_service.fit(all_logs)
    clustering_service.save(args.model_dir)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Clusters found:   {metrics.get('n_clusters', 'N/A')}")
    print(f"  Noise (anomaly):  {metrics.get('n_noise', 'N/A')} ({metrics.get('noise_ratio', 0):.1%})")
    if "silhouette_score" in metrics:
        print(f"  Silhouette score: {metrics['silhouette_score']:.4f}  (range −1..1, higher = better)")
    if "davies_bouldin_score" in metrics:
        print(f"  Davies-Bouldin:   {metrics['davies_bouldin_score']:.4f}  (lower = better)")
    print(f"  Model saved to:   {args.model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

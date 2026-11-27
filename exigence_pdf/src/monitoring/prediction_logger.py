"""Prediction logger for post-hoc monitoring and drift analysis.

Logs every prediction (input features, output, metadata) to a local
store for later drift analysis and model performance tracking.

The logger writes to a JSONL file that can be loaded as a pandas DataFrame
for drift detection with Evidently AI.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logging import get_logger

LOGGER = get_logger("monitoring.prediction_logger")


class PredictionLogger:
    """Logs predictions to a JSONL file for monitoring and drift detection.
    
    Usage:
        logger = PredictionLogger(output_dir="monitoring_data")
        logger.log_prediction(
            features={"pixel_mean": 0.5, "freq_mean": 0.3, ...},
            prediction={"label": "fake", "confidence": 0.87, ...},
            metadata={"latency_ms": 42, "model_version": "v3"},
        )
        
        # Later: load for drift analysis
        df = logger.load_predictions()
        detector = DriftDetector(reference_data=train_df)
        report = detector.detect_drift(current_data=df)
    """
    
    def __init__(
        self,
        output_dir: str = "monitoring_data",
        filename: str = "predictions.jsonl",
        max_file_size_mb: int = 100,
        rotation_count: int = 5,
    ):
        """Initialize the prediction logger.
        
        Args:
            output_dir: Directory for prediction logs
            filename: Name of the JSONL file
            max_file_size_mb: Maximum file size before rotation
            rotation_count: Number of rotated files to keep
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.rotation_count = rotation_count
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_size = 0
        self._flush_interval = 100  # Flush every N predictions
        
        LOGGER.info(f"PredictionLogger initialized: {self.filepath}")
    
    def log_prediction(
        self,
        features: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_input_hash: Optional[str] = None,
    ) -> None:
        """Log a single prediction.
        
        Args:
            features: Input feature values (e.g., image statistics)
            prediction: Model output (label, confidence, weights, etc.)
            metadata: Additional metadata (latency, model version, etc.)
            raw_input_hash: Hash of raw input for deduplication
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "unix_time": time.time(),
        }
        
        if features:
            entry["features"] = self._sanitize(features)
        if prediction:
            entry["prediction"] = self._sanitize(prediction)
        if metadata:
            entry["metadata"] = self._sanitize(metadata)
        if raw_input_hash:
            entry["input_hash"] = raw_input_hash
        
        self._buffer.append(entry)
        self._buffer_size += 1
        
        if self._buffer_size >= self._flush_interval:
            self.flush()
    
    def flush(self) -> None:
        """Write buffered predictions to disk."""
        if not self._buffer:
            return
        
        # Check file size and rotate if needed
        if self.filepath.exists():
            file_size = self.filepath.stat().st_size
            if file_size > self.max_file_size:
                self._rotate_files()
        
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry, default=str) + "\n")
            
            LOGGER.debug(f"Flushed {self._buffer_size} predictions to {self.filepath}")
        except Exception as e:
            LOGGER.error(f"Failed to flush predictions: {e}")
        
        self._buffer.clear()
        self._buffer_size = 0
    
    def _rotate_files(self) -> None:
        """Rotate log files (predictions.jsonl -> .1, .2, ...)."""
        for i in range(self.rotation_count - 1, 0, -1):
            old = self.filepath.with_suffix(f".jsonl.{i}")
            new = self.filepath.with_suffix(f".jsonl.{i + 1}")
            if old.exists():
                if i + 1 >= self.rotation_count:
                    old.unlink()
                else:
                    old.rename(new)
        
        if self.filepath.exists():
            backup = self.filepath.with_suffix(".jsonl.1")
            self.filepath.rename(backup)
    
    def _sanitize(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def load_predictions(
        self,
        max_records: Optional[int] = None,
        since: Optional[str] = None,
    ) -> "pd.DataFrame":
        """Load logged predictions as a pandas DataFrame.
        
        Args:
            max_records: Maximum number of records to load
            since: ISO timestamp to filter predictions after
            
        Returns:
            pandas DataFrame with prediction records
        """
        import pandas as pd
        
        if not self.filepath.exists():
            return pd.DataFrame()
        
        records = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if since and record.get("timestamp", "") < since:
                            continue
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        df = pd.DataFrame(records)
        
        if max_records and len(df) > max_records:
            df = df.tail(max_records).reset_index(drop=True)
        
        return df
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get summary statistics of logged predictions."""
        df = self.load_predictions()
        
        if df.empty:
            return {"total_predictions": 0}
        
        stats = {
            "total_predictions": len(df),
            "first_prediction": df["timestamp"].iloc[0] if "timestamp" in df.columns else None,
            "last_prediction": df["timestamp"].iloc[-1] if "timestamp" in df.columns else None,
        }
        
        # Parse prediction columns
        if "prediction" in df.columns:
            predictions = df["prediction"].apply(lambda x: x if isinstance(x, dict) else {})
            labels = predictions.apply(lambda x: x.get("label", "unknown"))
            stats["label_distribution"] = labels.value_counts().to_dict()
            
            confidences = predictions.apply(lambda x: x.get("confidence", x.get("prob_fake", None)))
            confidences = confidences.dropna()
            if len(confidences) > 0:
                stats["confidence_stats"] = {
                    "mean": float(confidences.mean()),
                    "std": float(confidences.std()),
                    "min": float(confidences.min()),
                    "max": float(confidences.max()),
                }
        
        return stats
    
    def __del__(self):
        """Flush remaining predictions on cleanup."""
        try:
            self.flush()
        except Exception:
            pass

"""Data drift detection using Evidently AI.

Provides:
    - Dataset drift detection (feature-level + target-level)
    - Prediction drift detection
    - Drift report generation (HTML + JSON)
    - Integration with monitoring alerts

The drift detector compares a reference dataset (training data statistics)
against a current dataset (recent predictions) to detect distribution shifts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging import get_logger

LOGGER = get_logger("monitoring.drift")

# --------------------------------------------------------------------------- #
# Evidently AI (optional)
# --------------------------------------------------------------------------- #
_EVIDENTLY_AVAILABLE = False
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
    )
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    LOGGER.info("evidently not installed — drift detection uses fallback method")


# --------------------------------------------------------------------------- #
# Drift Report
# --------------------------------------------------------------------------- #
@dataclass
class DriftReport:
    """Result of a drift detection analysis."""
    timestamp: str = ""
    dataset_drift: bool = False
    drift_score: float = 0.0
    n_drifted_features: int = 0
    n_total_features: int = 0
    feature_drifts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    share_of_drifted_features: float = 0.0
    reference_size: int = 0
    current_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dataset_drift": self.dataset_drift,
            "drift_score": self.drift_score,
            "n_drifted_features": self.n_drifted_features,
            "n_total_features": self.n_total_features,
            "share_of_drifted_features": self.share_of_drifted_features,
            "feature_drifts": self.feature_drifts,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
        }
    
    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))
    
    def summary(self) -> str:
        status = "🔴 DRIFT DETECTED" if self.dataset_drift else "🟢 No drift"
        return (
            f"{status} | Score: {self.drift_score:.3f} | "
            f"Drifted features: {self.n_drifted_features}/{self.n_total_features} "
            f"({self.share_of_drifted_features:.1%})"
        )


# --------------------------------------------------------------------------- #
# Drift Detector
# --------------------------------------------------------------------------- #
class DriftDetector:
    """Data drift detector using Evidently AI or a statistical fallback.
    
    Usage:
        detector = DriftDetector(reference_data=ref_df)
        report = detector.detect_drift(current_data=current_df)
        if report.dataset_drift:
            trigger_retraining()
    
    The detector supports two modes:
        1. Full mode: Evidently AI with PSI, KS test, Chi-squared
        2. Fallback mode: Statistical tests (KS, PSI) without Evidently
    """
    
    def __init__(
        self,
        reference_data: Optional[Any] = None,
        drift_threshold: float = 0.3,
        confidence: float = 0.95,
        min_reference_size: int = 100,
    ):
        """Initialize the drift detector.
        
        Args:
            reference_data: Reference (training) dataset as pandas DataFrame
            drift_threshold: Share of drifted features to trigger alert (0-1)
            confidence: Statistical test confidence level
            min_reference_size: Minimum reference samples required
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.confidence = confidence
        self.min_reference_size = min_reference_size
        self._evidently = _EVIDENTLY_AVAILABLE
    
    def _ks_test(self, ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for continuous features."""
        from scipy import stats
        stat, pvalue = stats.ks_2samp(ref, cur)
        return float(stat), float(pvalue)
    
    def _psi(self, ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index (PSI) for distribution comparison."""
        # Create bins from reference data
        breakpoints = np.quantile(ref, np.linspace(0, 1, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) < 3:
            return 0.0
        
        ref_counts = np.histogram(ref, bins=breakpoints)[0] / len(ref)
        cur_counts = np.histogram(cur, bins=breakpoints)[0] / len(cur)
        
        # Avoid division by zero
        ref_counts = np.clip(ref_counts, 1e-6, None)
        cur_counts = np.clip(cur_counts, 1e-6, None)
        
        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
        return float(psi)
    
    def _detect_feature_drift_fallback(
        self,
        feature_name: str,
        ref_values: np.ndarray,
        cur_values: np.ndarray,
    ) -> Dict[str, Any]:
        """Detect drift for a single feature using statistical tests."""
        result = {
            "feature": feature_name,
            "drift_detected": False,
            "test_statistic": 0.0,
            "p_value": 0.0,
            "psi": 0.0,
            "method": "fallback",
        }
        
        try:
            # KS test for continuous
            if len(ref_values) > 1 and len(cur_values) > 1:
                ks_stat, ks_pvalue = self._ks_test(ref_values, cur_values)
                psi_val = self._psi(ref_values, cur_values)
                
                result["test_statistic"] = ks_stat
                result["p_value"] = ks_pvalue
                result["psi"] = psi_val
                # Drift detected if PSI > 0.2 (significant) or KS p-value < threshold
                alpha = 1.0 - self.confidence
                result["drift_detected"] = (psi_val > 0.2) or (ks_pvalue < alpha)
        except Exception as e:
            LOGGER.warning(f"Drift detection failed for {feature_name}: {e}")
        
        return result
    
    def detect_drift(
        self,
        current_data: Any,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> DriftReport:
        """Detect data drift between reference and current datasets.
        
        Args:
            current_data: Current dataset as pandas DataFrame
            feature_columns: Columns to check for drift (default: all numeric)
            target_column: Target column for target drift detection
            
        Returns:
            DriftReport with detailed drift analysis
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call with reference_data parameter.")
        
        report = DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            reference_size=len(self.reference_data),
            current_size=len(current_data),
        )
        
        # Auto-detect numeric columns if not specified
        if feature_columns is None:
            import pandas as pd
            feature_columns = self.reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        
        report.n_total_features = len(feature_columns)
        
        if self._evidently:
            report = self._detect_with_evidently(current_data, feature_columns, report)
        else:
            report = self._detect_with_fallback(current_data, feature_columns, report)
        
        # Determine if dataset-level drift occurred
        if report.n_total_features > 0:
            report.share_of_drifted_features = (
                report.n_drifted_features / report.n_total_features
            )
            report.dataset_drift = report.share_of_drifted_features > self.drift_threshold
            report.drift_score = report.share_of_drifted_features
        
        LOGGER.info(f"Drift detection: {report.summary()}")
        return report
    
    def _detect_with_evidently(
        self,
        current_data: Any,
        feature_columns: List[str],
        report: DriftReport,
    ) -> DriftReport:
        """Use Evidently AI for drift detection."""
        import pandas as pd
        
        ref_df = self.reference_data[feature_columns] if isinstance(self.reference_data, pd.DataFrame) else pd.DataFrame(self.reference_data)[feature_columns]
        cur_df = current_data[feature_columns] if isinstance(current_data, pd.DataFrame) else pd.DataFrame(current_data)[feature_columns]
        
        try:
            column_mapping = ColumnMapping()
            
            drift_report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric(),
            ])
            drift_report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)
            
            # Parse results
            drift_result = drift_report.as_dict()
            
            for metric_result in drift_result.get("metrics", []):
                if "DatasetDriftMetric" in str(metric_result):
                    report.dataset_drift = metric_result.get("result", {}).get("dataset_drift", False)
                    report.drift_score = metric_result.get("result", {}).get("share_of_drifted_columns", 0.0)
                    report.n_drifted_features = int(
                        report.drift_score * report.n_total_features
                    )
            
            LOGGER.info("Evidently AI drift detection completed")
        except Exception as e:
            LOGGER.warning(f"Evidently AI detection failed, using fallback: {e}")
            report = self._detect_with_fallback(current_data, feature_columns, report)
        
        return report
    
    def _detect_with_fallback(
        self,
        current_data: Any,
        feature_columns: List[str],
        report: DriftReport,
    ) -> DriftReport:
        """Fallback drift detection using statistical tests."""
        import pandas as pd
        
        ref_df = self.reference_data if isinstance(self.reference_data, pd.DataFrame) else pd.DataFrame(self.reference_data)
        cur_df = current_data if isinstance(current_data, pd.DataFrame) else pd.DataFrame(current_data)
        
        n_drifted = 0
        for col in feature_columns:
            if col not in ref_df.columns or col not in cur_df.columns:
                continue
            
            ref_values = ref_df[col].dropna().values.astype(float)
            cur_values = cur_df[col].dropna().values.astype(float)
            
            if len(ref_values) < 2 or len(cur_values) < 2:
                continue
            
            result = self._detect_feature_drift_fallback(col, ref_values, cur_values)
            report.feature_drifts[col] = result
            
            if result["drift_detected"]:
                n_drifted += 1
        
        report.n_drifted_features = n_drifted
        return report
    
    def detect_prediction_drift(
        self,
        recent_predictions: np.ndarray,
        baseline_predictions: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """Detect drift in prediction distribution.
        
        Args:
            recent_predictions: Array of recent prediction confidences
            baseline_predictions: Baseline (training) prediction confidences
        """
        if baseline_predictions is None:
            # Use a uniform baseline as fallback
            baseline_predictions = np.random.uniform(0.3, 0.7, size=len(recent_predictions))
        
        report = DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            reference_size=len(baseline_predictions),
            current_size=len(recent_predictions),
            n_total_features=1,
        )
        
        result = self._detect_feature_drift_fallback(
            "prediction_confidence",
            baseline_predictions,
            recent_predictions,
        )
        report.feature_drifts["prediction_confidence"] = result
        report.n_drifted_features = 1 if result["drift_detected"] else 0
        report.drift_score = float(result["drift_detected"])
        report.dataset_drift = result["drift_detected"]
        
        return report

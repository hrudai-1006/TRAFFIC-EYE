"""Seatbelt violation: car/bus occupants without seatbelts.

Crops the driver-side windshield region from each car/bus detection and
runs a seatbelt classifier.  When the ``models.seatbelt_classifier`` module
is not yet available a lightweight stub is used so downstream pipelines
remain functional.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

__all__ = ["SeatbeltViolationDetector"]

logger = logging.getLogger(__name__)

# COCO class IDs for passenger vehicles
_CLASS_CAR = 2
_CLASS_BUS = 5
_VEHICLE_CLASS_IDS = {_CLASS_CAR, _CLASS_BUS}

# Fraction of the vehicle bbox to use as the windshield ROI
# (top-left quadrant is a reasonable heuristic for front-facing cameras)
_WINDSHIELD_X_START = 0.10
_WINDSHIELD_X_END = 0.55
_WINDSHIELD_Y_START = 0.10
_WINDSHIELD_Y_END = 0.55


class _SeatbeltClassifierStub:
    """Fallback stub when the real seatbelt model is unavailable.

    Returns ``seatbelt_visible=False`` with a reduced confidence.
    """

    def predict(
        self,
        crops: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """Return stub predictions.

        Args:
            crops: List of BGR windshield crops.

        Returns:
            List of dicts with ``seatbelt_visible`` and ``confidence``.
        """
        return [
            {"seatbelt_visible": False, "confidence": 0.20}
            for _ in crops
        ]


class SeatbeltViolationDetector:
    """Detect car/bus occupants without seatbelts."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise the seatbelt violation detector.

        Args:
            model_path: Optional path to a trained seatbelt classification
                model.  Falls back to stub if the model module is missing.
        """
        self.model = self._load_model(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        frame: np.ndarray,
        detections: dict[str, Any],
        frame_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run seatbelt check on the current frame.

        Args:
            frame: BGR image (H, W, 3).
            detections: Dict with xyxy, class_id, confidence, tracker_id.
            frame_number: Optional current frame counter.

        Returns:
            List of ViolationRecord dicts.
        """
        xyxy = np.asarray(detections.get("xyxy", []), dtype=np.float64)
        class_ids = np.asarray(detections.get("class_id", []), dtype=np.int32)
        confidences = np.asarray(
            detections.get("confidence", []), dtype=np.float64,
        )
        tracker_ids = detections.get("tracker_id", None)
        if tracker_ids is not None:
            tracker_ids = np.asarray(tracker_ids)

        if xyxy.size == 0:
            return []

        h, w = frame.shape[:2]
        violations: list[dict[str, Any]] = []

        # Collect vehicle crops and metadata
        vehicle_data: list[dict[str, Any]] = []
        crops: list[np.ndarray] = []

        for idx in range(len(xyxy)):
            cid = int(class_ids[idx])
            if cid not in _VEHICLE_CLASS_IDS:
                continue

            bbox = xyxy[idx]
            conf = float(confidences[idx])
            tid = (
                int(tracker_ids[idx])
                if tracker_ids is not None and tracker_ids[idx] is not None
                else -1
            )

            # Extract windshield ROI from vehicle bbox
            vx1, vy1, vx2, vy2 = bbox
            vw = vx2 - vx1
            vh = vy2 - vy1

            wx1 = int(max(0, vx1 + vw * _WINDSHIELD_X_START))
            wy1 = int(max(0, vy1 + vh * _WINDSHIELD_Y_START))
            wx2 = int(min(w, vx1 + vw * _WINDSHIELD_X_END))
            wy2 = int(min(h, vy1 + vh * _WINDSHIELD_Y_END))

            if wx2 <= wx1 or wy2 <= wy1:
                continue

            crop = frame[wy1:wy2, wx1:wx2].copy()

            # Skip very small crops (unreliable)
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            crops.append(crop)
            vehicle_data.append(
                {
                    "bbox": bbox.tolist(),
                    "confidence": conf,
                    "vehicle_id": tid,
                    "class_id": cid,
                    "windshield_roi": [wx1, wy1, wx2, wy2],
                },
            )

        if not crops:
            return []

        # Batch predict
        results = self.model.predict(crops)

        for vdata, result in zip(vehicle_data, results):
            seatbelt_visible = bool(result.get("seatbelt_visible", False))
            model_conf = float(result.get("confidence", 0.0))

            if not seatbelt_visible:
                combined = float(
                    np.clip(vdata["confidence"] * model_conf, 0.0, 1.0),
                )
                violations.append(
                    {
                        "violation_type": "seatbelt_violation",
                        "confidence": round(combined, 4),
                        "vehicle_id": vdata["vehicle_id"],
                        "vehicle_bbox": vdata["bbox"],
                        "frame_number": frame_number,
                        "detail": {
                            "seatbelt_visible": seatbelt_visible,
                            "model_confidence": round(model_conf, 4),
                            "windshield_roi": vdata["windshield_roi"],
                            "vehicle_class": (
                                "car" if vdata["class_id"] == _CLASS_CAR else "bus"
                            ),
                        },
                        "timestamp": time.time(),
                    },
                )

        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_path: str | None) -> Any:
        """Try to load the real seatbelt classifier; fall back to stub.

        Args:
            model_path: Optional path to model weights.

        Returns:
            Model with a ``predict(crops) -> list[dict]`` method.
        """
        try:
            from models.seatbelt_classifier import SeatbeltDetector  # type: ignore[import-not-found]

            model = SeatbeltDetector(model_path=model_path)
            logger.info("Loaded SeatbeltDetector from models.seatbelt_classifier")
            return model
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "models.seatbelt_classifier not available – "
                "using stub seatbelt classifier.",
            )
            return _SeatbeltClassifierStub()

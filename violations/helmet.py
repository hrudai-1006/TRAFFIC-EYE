"""Helmet violation: motorcycle riders without helmets.

Associates person detections with motorcycle detections, then runs a helmet
classifier on each rider crop.  When the ``models.helmet_classifier`` module
is not yet available the detector uses a lightweight stub that marks every
rider as "unknown" with low confidence so downstream pipelines are not blocked.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

__all__ = ["HelmetViolationDetector"]

logger = logging.getLogger(__name__)

# COCO class IDs
_CLASS_PERSON = 0
_CLASS_MOTORCYCLE = 3

# Proximity in pixels for associating a person with a motorcycle
_ASSOCIATION_PROXIMITY_PX = 80


class _HelmetClassifierStub:
    """Lightweight fallback when the real helmet classifier is not available.

    Returns ``has_helmet=False`` with a reduced confidence so that downstream
    systems can distinguish stub results from real predictions.
    """

    def predict(
        self,
        crops: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """Return stub predictions for a batch of rider crops.

        Args:
            crops: List of BGR images (rider crops).

        Returns:
            List of dicts with ``has_helmet`` and ``confidence``.
        """
        return [
            {"has_helmet": False, "confidence": 0.25}
            for _ in crops
        ]


class HelmetViolationDetector:
    """Detect motorcycle riders without helmets."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise the helmet violation detector.

        Args:
            model_path: Optional path to a trained helmet classification model.
                If *None*, the detector tries to import
                ``models.helmet_classifier.HelmetDetector``; if that is
                unavailable it falls back to a conservative stub.
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
        """Run helmet check on the current frame.

        Args:
            frame: BGR image (H, W, 3).
            detections: Dict with xyxy, class_id, confidence, tracker_id.
            frame_number: Optional current frame counter.

        Returns:
            List of ViolationRecord dicts – one per motorcycle with at least
            one rider missing a helmet.
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

        moto_mask = class_ids == _CLASS_MOTORCYCLE
        person_mask = class_ids == _CLASS_PERSON

        moto_indices = np.where(moto_mask)[0]
        person_indices = np.where(person_mask)[0]

        if moto_indices.size == 0 or person_indices.size == 0:
            return []

        h, w = frame.shape[:2]
        violations: list[dict[str, Any]] = []

        for mi in moto_indices:
            moto_bbox = xyxy[mi]
            moto_conf = float(confidences[mi])
            moto_id = (
                int(tracker_ids[mi])
                if tracker_ids is not None and tracker_ids[mi] is not None
                else -1
            )

            # Find associated persons
            associated: list[dict[str, Any]] = []
            rider_crops: list[np.ndarray] = []

            for pi in person_indices:
                person_bbox = xyxy[pi]
                person_conf = float(confidences[pi])

                if self._is_associated(person_bbox, moto_bbox):
                    # Extract rider crop (clamp to frame bounds)
                    x1 = max(0, int(person_bbox[0]))
                    y1 = max(0, int(person_bbox[1]))
                    x2 = min(w, int(person_bbox[2]))
                    y2 = min(h, int(person_bbox[3]))

                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2].copy()
                        rider_crops.append(crop)
                        associated.append(
                            {
                                "rider_bbox": person_bbox.tolist(),
                                "detection_confidence": round(person_conf, 4),
                            },
                        )

            if not associated:
                continue

            # Run helmet model on all associated riders at once
            helmet_results = self._predict_helmet_results(
                frame=frame,
                associated=associated,
                rider_crops=rider_crops,
            )

            # Merge model outputs into rider breakdown
            rider_breakdown: list[dict[str, Any]] = []
            has_violation = False

            for rider_info, hres in zip(associated, helmet_results):
                has_helmet = bool(hres.get("has_helmet", False))
                helmet_conf = float(hres.get("confidence", 0.0))

                rider_breakdown.append(
                    {
                        "rider_bbox": rider_info["rider_bbox"],
                        "has_helmet": has_helmet,
                        "confidence": round(helmet_conf, 4),
                    },
                )

                if not has_helmet:
                    has_violation = True

            if has_violation:
                # Combined confidence: detection conf × mean helmet-model conf
                model_confs = [
                    r["confidence"]
                    for r in rider_breakdown
                    if not r["has_helmet"]
                ]
                mean_model_conf = float(np.mean(model_confs)) if model_confs else 0.0
                combined = float(np.clip(moto_conf * mean_model_conf, 0.0, 1.0))

                violations.append(
                    {
                        "violation_type": "helmet_violation",
                        "confidence": round(combined, 4),
                        "vehicle_id": moto_id,
                        "vehicle_bbox": moto_bbox.tolist(),
                        "frame_number": frame_number,
                        "detail": {
                            "rider_count": len(rider_breakdown),
                            "riders_without_helmet": sum(
                                1 for r in rider_breakdown if not r["has_helmet"]
                            ),
                            "rider_breakdown": rider_breakdown,
                        },
                        "timestamp": time.time(),
                    },
                )

        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_associated(
        person_bbox: np.ndarray,
        moto_bbox: np.ndarray,
    ) -> bool:
        """Check if a person detection is associated with a motorcycle.

        Uses a combination of vertical overlap and centroid proximity.

        Args:
            person_bbox: Person [x1, y1, x2, y2].
            moto_bbox:   Motorcycle [x1, y1, x2, y2].

        Returns:
            True if the person is likely riding the motorcycle.
        """
        # Person centroid
        pcx = (person_bbox[0] + person_bbox[2]) / 2.0
        pcy = (person_bbox[1] + person_bbox[3]) / 2.0

        # Nearest point on motorcycle bbox
        nearest_x = float(np.clip(pcx, moto_bbox[0], moto_bbox[2]))
        nearest_y = float(np.clip(pcy, moto_bbox[1], moto_bbox[3]))

        dist = np.sqrt((pcx - nearest_x) ** 2 + (pcy - nearest_y) ** 2)

        if dist <= _ASSOCIATION_PROXIMITY_PX:
            return True

        # Also check horizontal overlap (person must be roughly above/on the moto)
        horiz_overlap = max(
            0,
            min(person_bbox[2], moto_bbox[2]) - max(person_bbox[0], moto_bbox[0]),
        )
        moto_w = max(1, moto_bbox[2] - moto_bbox[0])
        if horiz_overlap / moto_w > 0.3:
            # Person bottom should be near motorcycle top
            vert_gap = person_bbox[3] - moto_bbox[1]
            if abs(vert_gap) < _ASSOCIATION_PROXIMITY_PX:
                return True

        return False

    def _predict_helmet_results(
        self,
        frame: np.ndarray,
        associated: list[dict[str, Any]],
        rider_crops: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """Classify helmet presence using whichever classifier API is available."""
        if hasattr(self.model, "detect_helmets"):
            person_boxes = np.asarray(
                [item["rider_bbox"] for item in associated],
                dtype=np.float64,
            )
            return self.model.detect_helmets(frame, person_boxes)

        if hasattr(self.model, "predict"):
            return self.model.predict(rider_crops)

        return _HelmetClassifierStub().predict(rider_crops)

    @staticmethod
    def _load_model(model_path: str | None) -> Any:
        """Try to load the real helmet classifier; fall back to stub.

        Args:
            model_path: Optional path to model weights.

        Returns:
            A model instance with a ``predict(crops) -> list[dict]`` method.
        """
        try:
            from models.helmet_classifier import HelmetDetector  # type: ignore[import-not-found]

            model = HelmetDetector(model_path=model_path)
            if not getattr(model, "available", False):
                logger.warning(
                    "HelmetDetector weights unavailable - using stub helmet "
                    "classifier (low-confidence no-helmet predictions).",
                )
                return _HelmetClassifierStub()

            logger.info("Loaded HelmetDetector from models.helmet_classifier")
            return model
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "models.helmet_classifier not available – "
                "using stub helmet classifier (low-confidence predictions).",
            )
            return _HelmetClassifierStub()

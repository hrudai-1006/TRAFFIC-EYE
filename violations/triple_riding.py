"""Triple riding detection: flags motorcycles carrying more than 2 persons.

Detects violations where a motorcycle (COCO class_id=3) has more than two
person detections (COCO class_id=0) associated with it, using a combination
of bounding-box IoU overlap and centroid proximity heuristics.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

__all__ = ["TripleRidingDetector"]

# COCO class IDs
_CLASS_PERSON = 0
_CLASS_MOTORCYCLE = 3


class TripleRidingDetector:
    """Detect motorcycles carrying more than 2 riders.

    Association strategy (per motorcycle):
      1. Expand the motorcycle bbox by a configurable margin.
      2. Compute IoU between each person bbox and the expanded motorcycle bbox.
      3. If IoU ≥ ``overlap_threshold`` **or** the person centroid is within
         ``proximity_pixels`` of the motorcycle bbox edges, the person is
         counted as a rider.
      4. If rider_count > 2 → violation.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        proximity_pixels: int = 60,
    ) -> None:
        """Initialise the triple-riding detector.

        Args:
            overlap_threshold: Minimum IoU between a person bbox and the
                expanded motorcycle bbox to consider them associated.
            proximity_pixels: Maximum pixel distance from person centroid to
                the motorcycle bbox edge for proximity association.
        """
        self.overlap_threshold = overlap_threshold
        self.proximity_pixels = proximity_pixels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        detections: dict[str, Any],
        frame_shape: tuple[int, int, int],
        frame_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run triple-riding check on a single frame's detections.

        Args:
            detections: Dict with at least:
                - ``xyxy``  : np.ndarray of shape (N, 4)  – [x1, y1, x2, y2]
                - ``class_id`` : np.ndarray of shape (N,)
                - ``confidence``: np.ndarray of shape (N,)
                - ``tracker_id`` (optional): np.ndarray of shape (N,)
            frame_shape: (H, W, C) of the current frame.
            frame_number: Optional current frame counter.

        Returns:
            List of ViolationRecord dicts for every motorcycle with > 2
            associated persons.
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

        # Separate motorcycles and persons
        moto_mask = class_ids == _CLASS_MOTORCYCLE
        person_mask = class_ids == _CLASS_PERSON

        moto_indices = np.where(moto_mask)[0]
        person_indices = np.where(person_mask)[0]

        if moto_indices.size == 0 or person_indices.size == 0:
            return []

        violations: list[dict[str, Any]] = []

        for mi in moto_indices:
            moto_bbox = xyxy[mi]
            moto_conf = float(confidences[mi])
            moto_id = (
                int(tracker_ids[mi])
                if tracker_ids is not None and tracker_ids[mi] is not None
                else -1
            )

            # Expand motorcycle bbox for IoU overlap comparison
            expanded_bbox = self._expand_bbox(moto_bbox, frame_shape)

            associated_persons: list[dict[str, Any]] = []

            for pi in person_indices:
                person_bbox = xyxy[pi]
                person_conf = float(confidences[pi])

                iou = self._compute_iou(person_bbox, expanded_bbox)
                proximate = self._is_proximate(person_bbox, moto_bbox)

                if iou >= self.overlap_threshold or proximate:
                    associated_persons.append(
                        {
                            "person_bbox": person_bbox.tolist(),
                            "person_confidence": round(person_conf, 4),
                            "association_iou": round(iou, 4),
                            "association_proximate": proximate,
                        },
                    )

            rider_count = len(associated_persons)

            if rider_count > 2:
                # Confidence is the product of motorcycle conf and the mean
                # of the associated person confidences, clamped to [0, 1].
                person_confs = [p["person_confidence"] for p in associated_persons]
                combined_conf = moto_conf * float(np.mean(person_confs))
                combined_conf = float(np.clip(combined_conf, 0.0, 1.0))

                violations.append(
                    {
                        "violation_type": "triple_riding",
                        "confidence": round(combined_conf, 4),
                        "vehicle_id": moto_id,
                        "vehicle_bbox": moto_bbox.tolist(),
                        "frame_number": frame_number,
                        "detail": {
                            "rider_count": rider_count,
                            "person_bboxes": [
                                p["person_bbox"] for p in associated_persons
                            ],
                            "rider_breakdown": associated_persons,
                        },
                        "timestamp": time.time(),
                    },
                )

        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes.

        Args:
            box1: First bounding box (4,).
            box2: Second bounding box (4,).

        Returns:
            IoU value in [0, 1].
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    def _is_proximate(
        self,
        person_bbox: np.ndarray,
        moto_bbox: np.ndarray,
    ) -> bool:
        """Check if a person's centroid is within proximity of a motorcycle bbox.

        The distance is measured from the person centroid to the nearest edge
        of the motorcycle bounding box.

        Args:
            person_bbox: Person [x1, y1, x2, y2].
            moto_bbox:   Motorcycle [x1, y1, x2, y2].

        Returns:
            True if the centroid-to-edge distance ≤ ``self.proximity_pixels``.
        """
        # Person centroid
        cx = (person_bbox[0] + person_bbox[2]) / 2.0
        cy = (person_bbox[1] + person_bbox[3]) / 2.0

        # Closest point on the motorcycle bbox rectangle to the centroid
        nearest_x = float(np.clip(cx, moto_bbox[0], moto_bbox[2]))
        nearest_y = float(np.clip(cy, moto_bbox[1], moto_bbox[3]))

        dist = np.sqrt((cx - nearest_x) ** 2 + (cy - nearest_y) ** 2)
        return float(dist) <= self.proximity_pixels

    @staticmethod
    def _expand_bbox(
        bbox: np.ndarray,
        frame_shape: tuple[int, int, int],
        margin_ratio: float = 0.15,
    ) -> np.ndarray:
        """Expand a bbox by a relative margin, clipped to frame bounds.

        Args:
            bbox: [x1, y1, x2, y2].
            frame_shape: (H, W, C).
            margin_ratio: Fraction of bbox width/height to add on each side.

        Returns:
            Expanded bbox as np.ndarray of shape (4,).
        """
        h, w = frame_shape[:2]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        mx = bw * margin_ratio
        my = bh * margin_ratio
        expanded = np.array(
            [
                max(0, bbox[0] - mx),
                max(0, bbox[1] - my),
                min(w, bbox[2] + mx),
                min(h, bbox[3] + my),
            ],
            dtype=np.float64,
        )
        return expanded

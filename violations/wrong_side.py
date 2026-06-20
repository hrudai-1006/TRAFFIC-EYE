"""Wrong-side driving detection: vehicles moving against allowed direction.

Uses per-lane direction zones defined in config.  For each tracked vehicle
inside a zone polygon the detector computes a movement vector over a rolling
history window and compares it against the zone's allowed direction vector.
If the angle exceeds the configured threshold the vehicle is flagged.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["WrongSideDetector"]

logger = logging.getLogger(__name__)

# COCO class IDs considered vehicles (excludes person, bicycle)
_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Minimum number of position history points before a direction check is valid
_MIN_HISTORY_POINTS = 5

# Track IDs not updated for this many frames are evicted
_STALE_FRAME_THRESHOLD = 45


class WrongSideDetector:
    """Detect vehicles driving against the allowed direction in configured lane zones."""

    def __init__(
        self,
        config_path: str = "config/direction_vectors.json",
        history_frames: int = 15,
    ) -> None:
        """Initialise the wrong-side detector.

        Args:
            config_path: Path to direction-vectors JSON config.
            history_frames: Maximum number of centroid positions to keep per
                track_id (controls the rolling window size).
        """
        self.history_frames = max(_MIN_HISTORY_POINTS + 1, history_frames)
        self.zones: dict[str, dict[str, Any]] = {}
        self._load_config(config_path)

        # {track_id: deque([(cx, cy, frame_no), ...], maxlen=history_frames)}
        self.position_history: dict[int, deque[tuple[float, float, int]]] = {}

        # Set of track_ids already flagged (avoid re-alerting)
        self._alerted: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        detections: dict[str, Any],
        frame_number: int,
    ) -> list[dict[str, Any]]:
        """Run wrong-side check for the current frame.

        Args:
            detections: Dict with xyxy, class_id, confidence, tracker_id.
            frame_number: Monotonically increasing frame counter.

        Returns:
            List of ViolationRecord dicts.
        """
        xyxy = np.asarray(detections.get("xyxy", []), dtype=np.float64)
        class_ids = np.asarray(detections.get("class_id", []), dtype=np.int32)
        confidences = np.asarray(
            detections.get("confidence", []), dtype=np.float64,
        )
        tracker_ids = detections.get("tracker_id", None)

        if xyxy.size == 0 or tracker_ids is None:
            self._cleanup_stale(frame_number)
            return []

        tracker_ids = np.asarray(tracker_ids)

        violations: list[dict[str, Any]] = []

        for idx in range(len(xyxy)):
            cid = int(class_ids[idx])
            if cid not in _VEHICLE_CLASS_IDS:
                continue

            tid = tracker_ids[idx]
            if tid is None:
                continue
            tid = int(tid)

            bbox = xyxy[idx]
            conf = float(confidences[idx])
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float((bbox[1] + bbox[3]) / 2.0)

            # Update position history
            if tid not in self.position_history:
                self.position_history[tid] = deque(maxlen=self.history_frames)
            self.position_history[tid].append((cx, cy, frame_number))

            history = self.position_history[tid]
            if len(history) < _MIN_HISTORY_POINTS:
                continue

            # Compute movement vector
            positions = [(h[0], h[1]) for h in history]
            move_vec = self._compute_movement_vector(positions)

            # Skip near-stationary vehicles
            move_mag = np.sqrt(move_vec[0] ** 2 + move_vec[1] ** 2)
            if move_mag < 3.0:
                continue

            # Check each direction zone
            for zone_id, zone in self.zones.items():
                polygon = zone["lane_polygon"]
                allowed_dir = zone["allowed_direction"]
                threshold_deg = zone.get("angle_threshold_degrees", 100)

                if not self._point_in_polygon((cx, cy), polygon):
                    continue

                angle = self._compute_angle(move_vec, allowed_dir)

                if angle > threshold_deg and tid not in self._alerted:
                    self._alerted.add(tid)
                    violations.append(
                        {
                            "violation_type": "wrong_side_driving",
                            "confidence": round(conf, 4),
                            "vehicle_id": tid,
                            "vehicle_bbox": bbox.tolist(),
                            "frame_number": frame_number,
                            "detail": {
                                "zone_id": zone_id,
                                "zone_description": zone.get("description", ""),
                                "movement_vector": [
                                    round(move_vec[0], 2),
                                    round(move_vec[1], 2),
                                ],
                                "allowed_direction": list(allowed_dir),
                                "angle_degrees": round(angle, 2),
                                "threshold_degrees": threshold_deg,
                            },
                            "timestamp": time.time(),
                        },
                    )
                    break  # one zone match is enough per vehicle per frame

        self._cleanup_stale(frame_number)
        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_angle(vec1: tuple[float, float], vec2: tuple[float, float]) -> float:
        """Compute the angle (in degrees, 0-180) between two 2-D vectors.

        Args:
            vec1: First vector (dx, dy).
            vec2: Second vector (dx, dy).

        Returns:
            Angle in degrees in the range [0, 180].
        """
        v1 = np.array(vec1, dtype=np.float64)
        v2 = np.array(vec2, dtype=np.float64)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        # Clamp for numerical stability
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _compute_movement_vector(
        positions: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """Compute an overall movement vector from a sequence of positions.

        Uses a simple robust approach: fits a least-squares line through the
        positions to determine the dominant direction, then returns the vector
        from the first to last position projected onto that direction.

        For short histories this simplifies to (last - first).

        Args:
            positions: Ordered list of (x, y) centroids.

        Returns:
            (dx, dy) movement vector.
        """
        if len(positions) < 2:
            return (0.0, 0.0)

        pts = np.array(positions, dtype=np.float64)

        # Use median-of-thirds for robustness against outliers
        n = len(pts)
        third = max(1, n // 3)
        start_centroid = pts[:third].mean(axis=0)
        end_centroid = pts[-third:].mean(axis=0)

        dx = float(end_centroid[0] - start_centroid[0])
        dy = float(end_centroid[1] - start_centroid[1])
        return (dx, dy)

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> bool:
        """Ray-casting point-in-polygon test.

        Args:
            point:   (x, y).
            polygon: List of (x, y) vertices.

        Returns:
            True if point is inside the polygon.
        """
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        return inside

    def _load_config(self, config_path: str) -> None:
        """Load direction zone definitions from JSON.

        Args:
            config_path: Path to direction_vectors.json.
        """
        path = Path(config_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / config_path

        if not path.exists():
            logger.warning(
                "Direction vectors config not found at %s – detector will be a no-op.",
                path,
            )
            return

        with open(path, "r") as fh:
            data = json.load(fh)

        raw_zones = data.get("zones", {})
        for zone_id, zone_data in raw_zones.items():
            self.zones[zone_id] = {
                "camera_id": zone_data.get("camera_id", ""),
                "allowed_direction": tuple(zone_data["allowed_direction"]),
                "angle_threshold_degrees": zone_data.get(
                    "angle_threshold_degrees", 100,
                ),
                "lane_polygon": [tuple(pt) for pt in zone_data["lane_polygon"]],
                "description": zone_data.get("description", ""),
            }

        logger.info("Loaded %d direction zone(s) from %s", len(self.zones), path)

    def _cleanup_stale(self, current_frame: int) -> None:
        """Remove position-history entries not updated recently.

        Args:
            current_frame: Current frame number.
        """
        stale_ids = []
        for tid, hist in self.position_history.items():
            if hist and (current_frame - hist[-1][2]) > _STALE_FRAME_THRESHOLD:
                stale_ids.append(tid)

        for tid in stale_ids:
            del self.position_history[tid]
            self._alerted.discard(tid)

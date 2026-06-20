"""Illegal parking detection: vehicles dwelling in no-parking zones beyond threshold.

Tracks how long each vehicle remains inside a configured no-parking polygon.
When dwell time exceeds the zone-specific threshold the detector emits a
violation record.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["IllegalParkingDetector"]

logger = logging.getLogger(__name__)

# COCO class IDs that qualify as "vehicles" for parking violations
_VEHICLE_CLASS_IDS = {
    2,   # car
    5,   # bus
    7,   # truck
}

# Track IDs not seen for this many frames are evicted from the dwell tracker
_STALE_FRAME_THRESHOLD = 30


class IllegalParkingDetector:
    """Detect vehicles parked in no-parking zones beyond a dwell threshold.

    No-parking zones and their thresholds are loaded from a JSON config file.
    The detector maintains an internal dwell tracker keyed by ``track_id``.
    """

    def __init__(
        self,
        config_path: str = "config/parking_zones.json",
        fps: int = 30,
    ) -> None:
        """Initialise the illegal-parking detector.

        Args:
            config_path: Path (relative to project root or absolute) to the
                parking zones JSON config.
            fps: Video frame rate used to convert frame counts to seconds.
        """
        self.fps = max(1, fps)
        self.zones: list[dict[str, Any]] = []
        self._load_config(config_path)

        # {track_id: {"zone_id": str, "first_seen_frame": int,
        #             "last_seen_frame": int, "alerted": bool}}
        self.dwell_tracker: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        detections: dict[str, Any],
        frame_number: int,
    ) -> list[dict[str, Any]]:
        """Run illegal-parking check for the current frame.

        Args:
            detections: Dict with at least:
                - ``xyxy``      : np.ndarray (N, 4)
                - ``class_id``  : np.ndarray (N,)
                - ``confidence``: np.ndarray (N,)
                - ``tracker_id``: np.ndarray (N,)  — required for tracking
            frame_number: Monotonically increasing frame counter.

        Returns:
            List of ViolationRecord dicts for every vehicle that exceeds the
            dwell threshold in a no-parking zone.
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
        seen_track_ids: set[int] = set()

        for idx in range(len(xyxy)):
            cid = int(class_ids[idx])
            if cid not in _VEHICLE_CLASS_IDS:
                continue

            tid = tracker_ids[idx]
            if tid is None:
                continue
            tid = int(tid)
            seen_track_ids.add(tid)

            bbox = xyxy[idx]
            conf = float(confidences[idx])
            centroid = self._compute_centroid(bbox)

            # Check each no-parking zone
            for zone in self.zones:
                polygon = zone["polygon"]
                zone_id = zone["zone_id"]
                threshold_sec = zone.get("dwell_threshold_seconds", 120)

                if not self._point_in_polygon(centroid, polygon):
                    # If the vehicle was previously in this zone, remove it
                    if tid in self.dwell_tracker and self.dwell_tracker[tid]["zone_id"] == zone_id:
                        del self.dwell_tracker[tid]
                    continue

                # Vehicle centroid is inside the zone
                if tid not in self.dwell_tracker:
                    self.dwell_tracker[tid] = {
                        "zone_id": zone_id,
                        "first_seen_frame": frame_number,
                        "last_seen_frame": frame_number,
                        "alerted": False,
                    }
                else:
                    self.dwell_tracker[tid]["last_seen_frame"] = frame_number

                entry = self.dwell_tracker[tid]
                dwell_frames = frame_number - entry["first_seen_frame"]
                dwell_seconds = dwell_frames / self.fps

                if dwell_seconds >= threshold_sec and not entry["alerted"]:
                    entry["alerted"] = True
                    violations.append(
                        {
                            "violation_type": "illegal_parking",
                            "confidence": round(conf, 4),
                            "vehicle_id": tid,
                            "vehicle_bbox": bbox.tolist(),
                            "frame_number": frame_number,
                            "detail": {
                                "zone_id": zone_id,
                                "zone_description": zone.get("description", ""),
                                "dwell_seconds": round(dwell_seconds, 2),
                                "threshold_seconds": threshold_sec,
                                "centroid": list(centroid),
                            },
                            "timestamp": time.time(),
                        },
                    )

        self._cleanup_stale(frame_number)
        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> None:
        """Load no-parking zone definitions from JSON.

        Args:
            config_path: Path to the parking zones config file.
        """
        path = Path(config_path)
        if not path.is_absolute():
            # Try resolving relative to this file's parent (project root)
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / config_path

        if not path.exists():
            logger.warning(
                "Parking zones config not found at %s – detector will be a no-op.",
                path,
            )
            return

        with open(path, "r") as fh:
            data = json.load(fh)

        self.zones = data.get("no_parking_zones", [])
        # Convert polygon lists to numpy for convenience but keep lists for
        # the ray-casting algorithm
        for zone in self.zones:
            zone["polygon"] = [tuple(pt) for pt in zone["polygon"]]

        logger.info("Loaded %d no-parking zone(s) from %s", len(self.zones), path)

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> bool:
        """Ray-casting algorithm to test if *point* is inside *polygon*.

        Casts a horizontal ray from the point to the right and counts the
        number of polygon edges it crosses.  An odd count means the point
        is inside.

        Args:
            point:   (x, y) coordinates.
            polygon: Ordered list of (x, y) vertices.

        Returns:
            True if the point lies inside the polygon.
        """
        x, y = point
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            # Check if the ray crosses the edge from vertex j to vertex i
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i

        return inside

    @staticmethod
    def _compute_centroid(bbox: np.ndarray) -> tuple[float, float]:
        """Compute the centroid of an [x1, y1, x2, y2] bounding box.

        Args:
            bbox: Bounding box array of shape (4,).

        Returns:
            (cx, cy) centroid coordinates.
        """
        cx = float((bbox[0] + bbox[2]) / 2.0)
        cy = float((bbox[1] + bbox[3]) / 2.0)
        return (cx, cy)

    def _cleanup_stale(self, current_frame: int) -> None:
        """Remove dwell tracker entries not seen for > _STALE_FRAME_THRESHOLD frames.

        Args:
            current_frame: Current frame number.
        """
        stale_ids = [
            tid
            for tid, entry in self.dwell_tracker.items()
            if (current_frame - entry["last_seen_frame"]) > _STALE_FRAME_THRESHOLD
        ]
        for tid in stale_ids:
            del self.dwell_tracker[tid]

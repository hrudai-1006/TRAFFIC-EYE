"""Red light violation: vehicles crossing stop-line when signal is red.

Uses a configurable signal ROI to classify the traffic-light colour via
HSV pixel counting, then checks whether any tracked vehicle crosses the
stop-line while the signal is red.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

__all__ = ["RedLightDetector"]

logger = logging.getLogger(__name__)

# COCO class IDs considered vehicles for red-light violations
_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# ---- HSV thresholds for traffic-light colour classification ----
# OpenCV uses H: 0-179, S: 0-255, V: 0-255
_RED_LOWER_1 = np.array([0, 100, 100], dtype=np.uint8)
_RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
_RED_LOWER_2 = np.array([170, 100, 100], dtype=np.uint8)
_RED_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

_YELLOW_LOWER = np.array([20, 100, 100], dtype=np.uint8)
_YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

_GREEN_LOWER = np.array([40, 100, 100], dtype=np.uint8)
_GREEN_UPPER = np.array([80, 255, 255], dtype=np.uint8)


class RedLightDetector:
    """Detect vehicles that cross the stop-line while the signal is red."""

    def __init__(
        self,
        config_path: str = "config/camera_locations.json",
        camera_id: str = "CAM_001",
    ) -> None:
        """Initialise the red-light detector.

        Args:
            config_path: Path to camera locations JSON config.
            camera_id: Camera identifier whose signal_roi and stop_line to use.
        """
        self.camera_id = camera_id
        self.signal_roi: dict[str, int] | None = None
        self.stop_line: dict[str, int] | None = None
        self._load_config(config_path)

        # Track IDs that have already been flagged (avoid duplicate alerts)
        self.crossed: set[int] = set()

        # Previous frame bottom-edge y-positions: {track_id: float}
        self.prev_positions: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        frame: np.ndarray,
        detections: dict[str, Any],
        frame_number: int,
    ) -> list[dict[str, Any]]:
        """Run red-light violation check.

        Args:
            frame: BGR image (H, W, 3).
            detections: Dict with xyxy, class_id, confidence, tracker_id.
            frame_number: Current frame counter.

        Returns:
            List of ViolationRecord dicts.
        """
        if self.signal_roi is None or self.stop_line is None:
            return []

        signal_color = self._detect_signal_color(frame)

        xyxy = np.asarray(detections.get("xyxy", []), dtype=np.float64)
        class_ids = np.asarray(detections.get("class_id", []), dtype=np.int32)
        confidences = np.asarray(
            detections.get("confidence", []), dtype=np.float64,
        )
        tracker_ids = detections.get("tracker_id", None)

        if xyxy.size == 0 or tracker_ids is None:
            return []

        tracker_ids = np.asarray(tracker_ids)

        stop_y = float(self.stop_line["y"])
        stop_x1 = float(self.stop_line["x1"])
        stop_x2 = float(self.stop_line["x2"])

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

            # Use the bottom-centre of the bbox as the reference point
            bottom_cx = float((bbox[0] + bbox[2]) / 2.0)
            bottom_y = float(bbox[3])

            # Only consider vehicles whose x-centroid is within the stop-line extent
            if bottom_cx < stop_x1 or bottom_cx > stop_x2:
                self.prev_positions[tid] = bottom_y
                continue

            prev_y = self.prev_positions.get(tid)
            self.prev_positions[tid] = bottom_y

            if prev_y is None:
                continue

            # Check crossing only when signal is red
            if signal_color == "red":
                if (
                    self._has_crossed_line(prev_y, bottom_y, stop_y)
                    and tid not in self.crossed
                ):
                    self.crossed.add(tid)
                    violations.append(
                        {
                            "violation_type": "red_light_violation",
                            "confidence": round(conf, 4),
                            "vehicle_id": tid,
                            "vehicle_bbox": bbox.tolist(),
                            "frame_number": frame_number,
                            "detail": {
                                "signal_color": signal_color,
                                "stop_line_y": stop_y,
                                "prev_bottom_y": round(prev_y, 2),
                                "curr_bottom_y": round(bottom_y, 2),
                                "frame_number": frame_number,
                            },
                            "timestamp": time.time(),
                        },
                    )

        return violations

    # ------------------------------------------------------------------
    # Signal colour detection
    # ------------------------------------------------------------------

    def _detect_signal_color(self, frame: np.ndarray) -> str:
        """Classify the traffic signal colour from the configured ROI.

        Crops the signal region, converts to HSV, then counts pixels
        matching red, yellow, and green thresholds.

        Args:
            frame: Full BGR frame (H, W, 3).

        Returns:
            One of ``'red'``, ``'yellow'``, ``'green'``, or ``'unknown'``.
        """
        if self.signal_roi is None:
            return "unknown"

        roi = self.signal_roi
        h, w = frame.shape[:2]

        x1 = max(0, roi["x1"])
        y1 = max(0, roi["y1"])
        x2 = min(w, roi["x2"])
        y2 = min(h, roi["y2"])

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Red wraps around H=0/180, so we use two ranges
        red_mask_1 = cv2.inRange(hsv, _RED_LOWER_1, _RED_UPPER_1)
        red_mask_2 = cv2.inRange(hsv, _RED_LOWER_2, _RED_UPPER_2)
        red_count = int(cv2.countNonZero(red_mask_1) + cv2.countNonZero(red_mask_2))

        yellow_mask = cv2.inRange(hsv, _YELLOW_LOWER, _YELLOW_UPPER)
        yellow_count = int(cv2.countNonZero(yellow_mask))

        green_mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
        green_count = int(cv2.countNonZero(green_mask))

        counts = {
            "red": red_count,
            "yellow": yellow_count,
            "green": green_count,
        }

        max_color = max(counts, key=counts.get)  # type: ignore[arg-type]
        max_count = counts[max_color]

        # Require a minimum number of pixels to make a call
        min_pixel_threshold = 5
        if max_count < min_pixel_threshold:
            return "unknown"

        return max_color

    # ------------------------------------------------------------------
    # Stop-line crossing logic
    # ------------------------------------------------------------------

    @staticmethod
    def _has_crossed_line(
        prev_y: float,
        curr_y: float,
        stop_line_y: float,
    ) -> bool:
        """Check if a vehicle has crossed the stop-line between two frames.

        Crossing is detected if the bottom-edge y-position transitions from
        above (< stop_line_y) to below (>= stop_line_y), **or** vice-versa.
        This handles both directions of travel.

        Args:
            prev_y: Previous bottom-edge y coordinate.
            curr_y: Current bottom-edge y coordinate.
            stop_line_y: Y coordinate of the stop-line.

        Returns:
            True if a crossing occurred.
        """
        above_then_below = prev_y < stop_line_y and curr_y >= stop_line_y
        below_then_above = prev_y >= stop_line_y and curr_y < stop_line_y
        return above_then_below or below_then_above

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> None:
        """Load signal ROI and stop-line for the configured camera.

        Args:
            config_path: Path to camera_locations.json.
        """
        path = Path(config_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / config_path

        if not path.exists():
            logger.warning(
                "Camera config not found at %s – red-light detector disabled.",
                path,
            )
            return

        with open(path, "r") as fh:
            data = json.load(fh)

        cameras = data.get("cameras", {})
        cam_cfg = cameras.get(self.camera_id)

        if cam_cfg is None:
            logger.warning(
                "Camera '%s' not found in config – red-light detector disabled.",
                self.camera_id,
            )
            return

        self.signal_roi = cam_cfg.get("signal_roi")
        self.stop_line = cam_cfg.get("stop_line")

        if self.signal_roi is None or self.stop_line is None:
            logger.warning(
                "Camera '%s' has no signal_roi or stop_line – "
                "red-light detector disabled.",
                self.camera_id,
            )

"""ANPR violation module: license plate detection and reading for all violation-flagged vehicles.

Provides two modes of use:
1. **Service mode** – other violation detectors call ``read_plate(frame, bbox)``
   to get plate text for a specific vehicle.
2. **Standalone mode** – ``check(frame, detections)`` scans all vehicles and
   creates ``anpr_flag`` violations for those with unreadable plates
   (confidence < 0.3).

When the ``models.plate_detector`` module is not yet available a lightweight
stub based on contour analysis is used.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import cv2
import numpy as np

__all__ = ["ANPRModule"]

logger = logging.getLogger(__name__)

# COCO class IDs considered vehicles
_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Minimum confidence to consider a plate reading valid
_MIN_PLATE_CONFIDENCE = 0.3

# Plate region heuristic: bottom portion of vehicle bbox
_PLATE_REGION_Y_START = 0.55
_PLATE_REGION_Y_END = 1.0
_PLATE_REGION_X_START = 0.10
_PLATE_REGION_X_END = 0.90


class _PlateDetectorStub:
    """Fallback plate detector using classical CV when the trained model is unavailable.

    Performs basic contour-based plate localisation and character segmentation.
    Not production-accurate, but provides a functional pipeline.
    """

    def detect_and_read(
        self,
        crop: np.ndarray,
    ) -> dict[str, Any] | None:
        """Detect and read a license plate from a vehicle crop.

        Args:
            crop: BGR image of the vehicle region.

        Returns:
            Dict with plate_text, plate_confidence, plate_bbox; or None.
        """
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edges = cv2.Canny(filtered, 30, 200)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        best_plate: dict[str, Any] | None = None
        best_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            # License plates are roughly rectangular (4 corners)
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect_ratio = cw / max(ch, 1)

                # Typical plate aspect ratio is between 2:1 and 5:1
                if 1.5 <= aspect_ratio <= 6.0 and area > best_area:
                    best_area = area
                    plate_crop = gray[y : y + ch, x : x + cw]

                    # Simple OCR placeholder: threshold and count white regions
                    _, thresh = cv2.threshold(
                        plate_crop,
                        0,
                        255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    )
                    white_ratio = np.sum(thresh > 0) / max(thresh.size, 1)

                    # Generate pseudo plate text from contour characteristics
                    plate_text = self._extract_plate_text(plate_crop)

                    best_plate = {
                        "plate_text": plate_text,
                        "plate_confidence": round(
                            min(0.5, white_ratio * 0.6), 4,
                        ),
                        "plate_bbox": [x, y, x + cw, y + ch],
                    }

        return best_plate

    @staticmethod
    def _extract_plate_text(plate_crop: np.ndarray) -> str:
        """Attempt basic character segmentation on a plate crop.

        This is a stub that returns a placeholder based on the image
        characteristics.  Replace with a real OCR engine (Tesseract,
        EasyOCR, PaddleOCR) for production use.

        Args:
            plate_crop: Grayscale image of the plate region.

        Returns:
            Detected plate text (or placeholder).
        """
        if plate_crop.size == 0:
            return ""

        # Threshold and find character-like contours
        _, binary = cv2.threshold(
            plate_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Filter contours that look like characters
        char_contours = []
        ph, pw = plate_crop.shape[:2]
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Characters should be a reasonable fraction of plate height
            if 0.2 * ph < ch < 0.9 * ph and 0.02 * pw < cw < 0.25 * pw:
                char_contours.append((x, y, cw, ch))

        # Sort by x position (left to right)
        char_contours.sort(key=lambda c: c[0])

        num_chars = len(char_contours)
        if num_chars >= 4:
            # Return a placeholder indicating character count detected
            return f"PLATE_{num_chars}CH"

        return ""


class ANPRModule:
    """Automatic Number Plate Recognition module.

    Can be used as a service (``read_plate``) or as a standalone violation
    checker (``check``).
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise the ANPR module.

        Args:
            model_path: Optional path to a trained plate detection model.
        """
        self.model = self._load_model(model_path)

    # ------------------------------------------------------------------
    # Service API (used by other violation detectors)
    # ------------------------------------------------------------------

    def read_plate(
        self,
        frame: np.ndarray,
        vehicle_bbox: list[float] | np.ndarray,
    ) -> dict[str, Any] | None:
        """Detect and read a license plate from a vehicle region.

        Args:
            frame: Full BGR frame (H, W, 3).
            vehicle_bbox: Vehicle bounding box [x1, y1, x2, y2].

        Returns:
            Dict with ``plate_text``, ``plate_confidence``, ``plate_bbox``
            (coordinates relative to the vehicle crop), or *None* if no plate
            is detected.
        """
        h, w = frame.shape[:2]
        bbox = np.asarray(vehicle_bbox, dtype=np.float64)

        # Extract vehicle crop
        vx1 = max(0, int(bbox[0]))
        vy1 = max(0, int(bbox[1]))
        vx2 = min(w, int(bbox[2]))
        vy2 = min(h, int(bbox[3]))

        if vx2 <= vx1 or vy2 <= vy1:
            return None

        vehicle_crop = frame[vy1:vy2, vx1:vx2]

        # Focus on the bottom portion where plates are typically located
        vh, vw = vehicle_crop.shape[:2]
        py1 = int(vh * _PLATE_REGION_Y_START)
        py2 = int(vh * _PLATE_REGION_Y_END)
        px1 = int(vw * _PLATE_REGION_X_START)
        px2 = int(vw * _PLATE_REGION_X_END)

        plate_region = vehicle_crop[py1:py2, px1:px2]

        if plate_region.size == 0:
            return None

        result = self.model.detect_and_read(plate_region)

        if result is None:
            return None

        # Adjust plate bbox coordinates to be relative to the full vehicle crop
        if result.get("plate_bbox"):
            pbbox = result["plate_bbox"]
            result["plate_bbox"] = [
                pbbox[0] + px1,
                pbbox[1] + py1,
                pbbox[2] + px1,
                pbbox[3] + py1,
            ]

        return result

    # ------------------------------------------------------------------
    # Standalone violation check
    # ------------------------------------------------------------------

    def check(
        self,
        frame: np.ndarray,
        detections: dict[str, Any],
        frame_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Scan all vehicles and flag those with unreadable plates.

        This is primarily used for auditing / suspicious-vehicle flagging.
        Normally other violation detectors call ``read_plate`` directly.

        Args:
            frame: BGR image (H, W, 3).
            detections: Dict with xyxy, class_id, confidence, tracker_id.
            frame_number: Optional current frame counter.

        Returns:
            List of ViolationRecord dicts with type ``anpr_flag`` for
            vehicles whose plates could not be read (confidence < 0.3).
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

        violations: list[dict[str, Any]] = []

        for idx in range(len(xyxy)):
            cid = int(class_ids[idx])
            if cid not in _VEHICLE_CLASS_IDS:
                continue

            bbox = xyxy[idx]
            det_conf = float(confidences[idx])
            tid = (
                int(tracker_ids[idx])
                if tracker_ids is not None and tracker_ids[idx] is not None
                else -1
            )

            plate_result = self.read_plate(frame, bbox)

            # Flag as violation only if plate is unreadable
            plate_conf = (
                float(plate_result["plate_confidence"])
                if plate_result is not None
                else 0.0
            )
            plate_text = (
                plate_result["plate_text"]
                if plate_result is not None
                else ""
            )

            if plate_conf < _MIN_PLATE_CONFIDENCE:
                violations.append(
                    {
                        "violation_type": "anpr_flag",
                        "confidence": round(det_conf * max(0.1, 1.0 - plate_conf), 4),
                        "vehicle_id": tid,
                        "vehicle_bbox": bbox.tolist(),
                        "frame_number": frame_number,
                        "detail": {
                            "plate_text": plate_text,
                            "plate_confidence": round(plate_conf, 4),
                            "plate_bbox": (
                                plate_result["plate_bbox"]
                                if plate_result is not None
                                else None
                            ),
                            "reason": "plate_unreadable",
                        },
                        "timestamp": time.time(),
                    },
                )

        return violations

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_path: str | None) -> Any:
        """Try to load the real plate detector; fall back to contour stub.

        Args:
            model_path: Optional path to model weights.

        Returns:
            Model with ``detect_and_read(crop) -> dict | None``.
        """
        try:
            from models.plate_detector import PlateDetector  # type: ignore[import-not-found]

            model = PlateDetector(model_path=model_path)
            logger.info("Loaded PlateDetector from models.plate_detector")
            return model
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "models.plate_detector not available – "
                "using contour-based stub plate detector.",
            )
            return _PlateDetectorStub()

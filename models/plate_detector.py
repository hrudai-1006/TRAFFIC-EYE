"""License plate detection (fine-tuned YOLO11n) + OCR (EasyOCR).

Detects license plates in full frames or within vehicle bounding-box crops,
then reads the plate text via EasyOCR with post-processing.
"""

import os
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Default weight path relative to project root
_DEFAULT_WEIGHT_PATH: str = os.path.join("models", "weights", "plate_best.pt")


class PlateDetector:
    """License plate detector + OCR reader.

    Uses a fine-tuned YOLO11n for plate localisation and EasyOCR for
    text recognition.  EasyOCR is run on CPU because it does not support
    Apple MPS.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        """Load the plate detection model and initialise the OCR reader.

        Args:
            model_path: Path to fine-tuned YOLO weights.  Falls back to
                ``models/weights/plate_best.pt`` when *None*.
            confidence: Minimum detection confidence for plates.
            device: Compute device override (``'mps'`` / ``'cpu'``).
        """
        # Device — MPS preferred, CPU fallback, never CUDA
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )

        self.confidence = confidence
        self.model_path = model_path or _DEFAULT_WEIGHT_PATH
        self.available = False
        self.model: Optional[YOLO] = None

        # --- Load YOLO plate-detection model ---
        if not os.path.isfile(self.model_path):
            print(
                f"[PlateDetector] WARNING: Weight file not found at "
                f"'{self.model_path}'. Plate detection unavailable until "
                f"you train or supply weights."
            )
        else:
            try:
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                self.available = True
                print(
                    f"[PlateDetector] Loaded weights from '{self.model_path}' "
                    f"on {self.device}"
                )
            except Exception as exc:
                print(f"[PlateDetector] Failed to load model: {exc}")

        # --- Initialise EasyOCR (CPU-only; MPS unsupported) ---
        self.ocr_reader = None
        try:
            import easyocr

            self.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[PlateDetector] EasyOCR reader initialised (CPU)")
        except ImportError:
            print(
                "[PlateDetector] WARNING: easyocr not installed. "
                "OCR will be unavailable. Install with: pip install easyocr"
            )
        except Exception as exc:
            print(f"[PlateDetector] WARNING: EasyOCR init failed: {exc}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect_plates(
        self,
        frame: np.ndarray,
        vehicle_boxes: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Detect license plates and read their text.

        If *vehicle_boxes* is provided, the plate detector is run on each
        vehicle crop (faster, higher precision).  Otherwise it runs on the
        full frame.

        Args:
            frame: Full BGR image (H, W, 3).
            vehicle_boxes: Optional array of shape ``(N, 4)`` with
                ``[x1, y1, x2, y2]`` vehicle bounding boxes.

        Returns:
            List of dicts::

                {
                    "plate_bbox": [x1, y1, x2, y2],   # in full-frame coords
                    "plate_text": str,
                    "plate_confidence": float,          # OCR confidence
                    "detection_confidence": float,      # YOLO confidence
                }
        """
        if not self.available:
            return []

        results: list[dict] = []

        if vehicle_boxes is not None and len(vehicle_boxes) > 0:
            results = self._detect_in_crops(frame, vehicle_boxes)
        else:
            results = self._detect_in_frame(frame)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_in_frame(self, frame: np.ndarray) -> list[dict]:
        """Run plate detection on the full frame."""
        preds = self.model(frame, conf=self.confidence, verbose=False)[0]
        results: list[dict] = []

        for det_box in preds.boxes:
            x1, y1, x2, y2 = map(int, det_box.xyxy[0].tolist())
            det_conf = float(det_box.conf[0])

            plate_crop = self._safe_crop(frame, x1, y1, x2, y2)
            text, ocr_conf = self._run_ocr(plate_crop)

            results.append(
                {
                    "plate_bbox": [x1, y1, x2, y2],
                    "plate_text": text,
                    "plate_confidence": ocr_conf,
                    "detection_confidence": det_conf,
                }
            )

        return results

    def _detect_in_crops(
        self, frame: np.ndarray, vehicle_boxes: np.ndarray
    ) -> list[dict]:
        """Run plate detection within each vehicle crop."""
        h_frame, w_frame = frame.shape[:2]
        results: list[dict] = []

        for vbox in vehicle_boxes:
            vx1, vy1, vx2, vy2 = map(int, vbox[:4])
            vx1 = max(0, vx1)
            vy1 = max(0, vy1)
            vx2 = min(w_frame, vx2)
            vy2 = min(h_frame, vy2)

            crop = frame[vy1:vy2, vx1:vx2]
            if crop.size == 0:
                continue

            preds = self.model(crop, conf=self.confidence, verbose=False)[0]

            for det_box in preds.boxes:
                # Plate coords are relative to the vehicle crop
                cx1, cy1, cx2, cy2 = map(int, det_box.xyxy[0].tolist())
                det_conf = float(det_box.conf[0])

                # Map back to full-frame coordinates
                fx1 = vx1 + cx1
                fy1 = vy1 + cy1
                fx2 = vx1 + cx2
                fy2 = vy1 + cy2

                plate_crop = self._safe_crop(frame, fx1, fy1, fx2, fy2)
                text, ocr_conf = self._run_ocr(plate_crop)

                results.append(
                    {
                        "plate_bbox": [fx1, fy1, fx2, fy2],
                        "plate_text": text,
                        "plate_confidence": ocr_conf,
                        "detection_confidence": det_conf,
                    }
                )

        return results

    @staticmethod
    def _safe_crop(
        frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Crop with boundary clamping to avoid empty arrays."""
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return frame[y1:y2, x1:x2]

    def _run_ocr(self, plate_crop: np.ndarray) -> tuple[str, float]:
        """Run EasyOCR on a plate crop and post-process the text.

        Args:
            plate_crop: BGR image of the license plate region.

        Returns:
            Tuple of ``(cleaned_text, confidence)``.  Returns
            ``('', 0.0)`` when OCR is unavailable or the crop is empty.
        """
        if self.ocr_reader is None or plate_crop.size == 0:
            return ("", 0.0)

        try:
            # Convert to grayscale for better OCR accuracy
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            # Light pre-processing: resize small crops for better OCR
            h, w = gray.shape[:2]
            if w < 100:
                scale = 100 / w
                gray = cv2.resize(
                    gray, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            ocr_results = self.ocr_reader.readtext(gray)

            if not ocr_results:
                return ("", 0.0)

            # Concatenate all detected text fragments
            texts: list[str] = []
            confidences: list[float] = []
            for _bbox, text, conf in ocr_results:
                texts.append(text)
                confidences.append(conf)

            raw_text = " ".join(texts)

            # Post-process: uppercase, strip, keep only alphanumeric + spaces
            cleaned = raw_text.strip().upper()
            cleaned = re.sub(r"[^A-Z0-9 ]", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return (cleaned, round(avg_conf, 4))

        except Exception as exc:
            print(f"[PlateDetector] OCR error: {exc}")
            return ("", 0.0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_yaml: str,
        epochs: int = 30,
        imgsz: int = 640,
        batch: int = 16,
    ) -> dict:
        """Fine-tune YOLO11n for license plate localisation.

        Args:
            data_yaml: Path to the dataset YAML (Ultralytics format).
            epochs: Number of training epochs.
            imgsz: Training image size.
            batch: Batch size.

        Returns:
            Dict with ``mAP50``, ``mAP50_95``, ``training_time`` keys.
        """
        print(f"[PlateDetector] Starting training for {epochs} epochs …")
        base_model = YOLO("yolo11n.pt")
        base_model.to(self.device)

        start = time.perf_counter()
        train_results = base_model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=str(self.device),
            project="runs/plate",
            name="train",
            exist_ok=True,
        )
        training_time = time.perf_counter() - start

        # Copy best weights to the canonical location
        best_src = Path("runs/plate/train/weights/best.pt")
        dest = Path(_DEFAULT_WEIGHT_PATH)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if best_src.exists():
            import shutil

            shutil.copy2(str(best_src), str(dest))
            print(f"[PlateDetector] Best weights saved to {dest}")

            # Reload the newly trained model
            self.model = YOLO(str(dest))
            self.model.to(self.device)
            self.available = True
        else:
            print("[PlateDetector] WARNING: best.pt not found after training.")

        # Extract metrics safely
        metrics: dict = {}
        try:
            metrics["mAP50"] = float(
                train_results.results_dict.get("metrics/mAP50(B)", 0.0)
            )
            metrics["mAP50_95"] = float(
                train_results.results_dict.get("metrics/mAP50-95(B)", 0.0)
            )
        except Exception:
            metrics["mAP50"] = 0.0
            metrics["mAP50_95"] = 0.0

        metrics["training_time"] = round(training_time, 2)
        print(f"[PlateDetector] Training complete: {metrics}")
        return metrics


if __name__ == "__main__":
    det = PlateDetector()
    print(f"Plate detector available: {det.available}")
    print(f"OCR available: {det.ocr_reader is not None}")

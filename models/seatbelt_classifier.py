"""Seatbelt detection via fine-tuned YOLO11n.

Detects whether vehicle occupants are wearing seatbelts by cropping the
cabin/windshield region (vertical 40–60 % of the vehicle bounding box) and
running a specialised YOLO11n model on the crop.
"""

import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Default weight path relative to project root
_DEFAULT_WEIGHT_PATH: str = os.path.join("models", "weights", "seatbelt_best.pt")


class SeatbeltDetector:
    """Seatbelt presence classifier built on a fine-tuned YOLO11n model.

    The detector expects vehicle bounding boxes (car, bus) from
    ``BaseDetector`` and inspects the windshield / cabin region for
    seatbelt presence.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        """Load the seatbelt detection model.

        Args:
            model_path: Path to fine-tuned YOLO weights.  Falls back to
                ``models/weights/seatbelt_best.pt`` when *None*.
            confidence: Minimum detection confidence.
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

        if not os.path.isfile(self.model_path):
            print(
                f"[SeatbeltDetector] WARNING: Weight file not found at "
                f"'{self.model_path}'. Seatbelt detection unavailable until "
                f"you train or supply weights."
            )
            return

        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.available = True
            print(
                f"[SeatbeltDetector] Loaded weights from '{self.model_path}' "
                f"on {self.device}"
            )
        except Exception as exc:
            print(f"[SeatbeltDetector] Failed to load model: {exc}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect_seatbelts(
        self,
        frame: np.ndarray,
        vehicle_boxes: np.ndarray,
    ) -> list[dict]:
        """Classify seatbelt presence for occupants of each vehicle box.

        The cabin / windshield area is approximated as the vertical slice
        from 40 % to 60 % of each vehicle bounding box height.

        Args:
            frame: Full BGR image (H, W, 3).
            vehicle_boxes: Array of shape ``(N, 4)`` with
                ``[x1, y1, x2, y2]`` bounding boxes for detected vehicles
                (typically cars and buses).

        Returns:
            List of dicts, one per vehicle box::

                {
                    "bbox": [x1, y1, x2, y2],       # original vehicle box
                    "wearing_seatbelt": bool,
                    "confidence": float,
                }

            If the model is unavailable every entry has
            ``wearing_seatbelt=False, confidence=0.0``.
        """
        results: list[dict] = []
        h_frame, w_frame = frame.shape[:2]

        for box in vehicle_boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)

            box_h = y2 - y1
            box_w = x2 - x1
            if box_h <= 0 or box_w <= 0:
                results.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "wearing_seatbelt": False,
                        "confidence": 0.0,
                    }
                )
                continue

            # Cabin / windshield region: vertical 40 %–60 % of the bbox
            cabin_y1 = y1 + int(box_h * 0.40)
            cabin_y2 = y1 + int(box_h * 0.60)
            cabin_y2 = max(cabin_y2, cabin_y1 + 1)  # ensure ≥1 px height

            cabin_crop = frame[cabin_y1:cabin_y2, x1:x2]

            if cabin_crop.size == 0 or not self.available:
                results.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "wearing_seatbelt": False,
                        "confidence": 0.0,
                    }
                )
                continue

            # Run seatbelt model on the cabin crop
            preds = self.model(
                cabin_crop, conf=self.confidence, verbose=False
            )[0]

            wearing = False
            best_conf = 0.0

            if len(preds.boxes) > 0:
                for det_box in preds.boxes:
                    cls_id = int(det_box.cls[0])
                    conf = float(det_box.conf[0])
                    # Convention: class 0 = seatbelt (wearing),
                    #             class 1 = no_seatbelt
                    # (adjust mapping if your dataset differs)
                    if cls_id == 0 and conf > best_conf:
                        wearing = True
                        best_conf = conf
                    elif cls_id == 1 and conf > best_conf:
                        wearing = False
                        best_conf = conf

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "wearing_seatbelt": wearing,
                    "confidence": best_conf,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_yaml: str,
        epochs: int = 80,
        imgsz: int = 640,
        batch: int = 16,
    ) -> dict:
        """Fine-tune YOLO11n on a seatbelt detection dataset.

        Args:
            data_yaml: Path to the dataset YAML (Ultralytics format).
            epochs: Number of training epochs.
            imgsz: Training image size.
            batch: Batch size.

        Returns:
            Dict with ``mAP50``, ``mAP50_95``, ``training_time`` keys.
        """
        print(f"[SeatbeltDetector] Starting training for {epochs} epochs …")
        base_model = YOLO("yolo11n.pt")
        base_model.to(self.device)

        start = time.perf_counter()
        train_results = base_model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=str(self.device),
            project="runs/seatbelt",
            name="train",
            exist_ok=True,
        )
        training_time = time.perf_counter() - start

        # Copy best weights to the canonical location
        best_src = Path("runs/seatbelt/train/weights/best.pt")
        dest = Path(_DEFAULT_WEIGHT_PATH)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if best_src.exists():
            import shutil

            shutil.copy2(str(best_src), str(dest))
            print(f"[SeatbeltDetector] Best weights saved to {dest}")

            # Reload the newly trained model
            self.model = YOLO(str(dest))
            self.model.to(self.device)
            self.available = True
        else:
            print("[SeatbeltDetector] WARNING: best.pt not found after training.")

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
        print(f"[SeatbeltDetector] Training complete: {metrics}")
        return metrics


if __name__ == "__main__":
    det = SeatbeltDetector()
    print(f"Seatbelt detector available: {det.available}")

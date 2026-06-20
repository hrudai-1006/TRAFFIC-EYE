"""Base vehicle/person detection using pretrained YOLO11n + ByteTrack tracking."""

import time
from typing import Optional

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


class BaseDetector:
    """Core detection and tracking engine using YOLO11n and ByteTrack.

    Provides real-time object detection filtered to traffic-relevant COCO classes,
    persistent multi-object tracking via ByteTrack, and frame annotation utilities.
    """

    # Traffic-relevant COCO class IDs -> human-readable names
    RELEVANT_CLASSES: dict[int, str] = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
    }

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the detector with YOLO11n and ByteTrack tracker.

        Args:
            model_name: YOLO model weight file name.
            confidence: Minimum detection confidence threshold.
            iou_threshold: IoU threshold for NMS.
            device: Compute device override. Auto-selects MPS > CPU if None.
        """
        # Device selection — MPS on Apple Silicon, CPU fallback. Never CUDA.
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        print(f"[BaseDetector] Using device: {self.device}")

        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Load YOLO11 model
        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
            print(f"[BaseDetector] Loaded model: {model_name}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_name}': {exc}"
            ) from exc

        # ByteTrack tracker for persistent object IDs across frames
        self.tracker = sv.ByteTrack()

        # Annotation helpers (instantiated once, reused per frame)
        self._box_annotator = sv.BoxAnnotator(thickness=2)
        self._label_annotator = sv.LabelAnnotator(
            text_scale=0.4, text_thickness=1, text_padding=4
        )

        # Set of relevant class IDs for fast membership checks
        self._relevant_ids: set[int] = set(self.RELEVANT_CLASSES.keys())

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLO inference on a single frame and filter to relevant classes.

        Args:
            frame: BGR image as a NumPy array (H, W, 3).

        Returns:
            ``sv.Detections`` containing only traffic-relevant objects.
            Tracker IDs are **not** assigned at this stage.
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            return detections

        # Keep only relevant COCO classes
        mask = np.array(
            [cid in self._relevant_ids for cid in detections.class_id],
            dtype=bool,
        )
        return detections[mask]

    # ------------------------------------------------------------------
    # Detection + Tracking
    # ------------------------------------------------------------------

    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        """Detect objects and update the ByteTrack tracker.

        Args:
            frame: BGR image as a NumPy array (H, W, 3).

        Returns:
            ``sv.Detections`` with persistent ``tracker_id`` assigned.
        """
        detections = self.detect(frame)

        if len(detections) == 0:
            return detections

        tracked = self.tracker.update_with_detections(detections)
        return tracked

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_class_name(self, class_id: int) -> str:
        """Return the human-readable name for a COCO class ID.

        Args:
            class_id: Integer COCO class ID.

        Returns:
            Class name string, or ``'unknown'`` if the ID is not in
            ``RELEVANT_CLASSES``.
        """
        return self.RELEVANT_CLASSES.get(class_id, "unknown")

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of *frame*.

        Labels follow the format: ``class_name ID:tracker_id conf:0.XX``

        Args:
            frame: Original BGR image.
            detections: Detections (optionally with tracker IDs).

        Returns:
            Annotated copy of the frame.
        """
        annotated = frame.copy()

        if len(detections) == 0:
            return annotated

        # Build label strings
        labels: list[str] = []
        for i in range(len(detections)):
            class_name = self.get_class_name(int(detections.class_id[i]))
            conf = float(detections.confidence[i])

            if detections.tracker_id is not None:
                tid = int(detections.tracker_id[i])
                label = f"{class_name} ID:{tid} conf:{conf:.2f}"
            else:
                label = f"{class_name} conf:{conf:.2f}"

            labels.append(label)

        annotated = self._box_annotator.annotate(
            scene=annotated, detections=detections
        )
        annotated = self._label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )
        return annotated


# ======================================================================
# Standalone video processing helper
# ======================================================================


def process_video(
    input_path: str,
    output_path: str,
    detector: Optional[BaseDetector] = None,
    max_frames: Optional[int] = None,
) -> None:
    """Process a video file with detection + tracking and save annotated output.

    Args:
        input_path: Path to the source video file.
        output_path: Destination path for the annotated video.
        detector: A ``BaseDetector`` instance. Created with defaults if *None*.
        max_frames: Stop after this many frames (process all if *None*).
    """
    if detector is None:
        detector = BaseDetector()

    video_info = sv.VideoInfo.from_video_path(input_path)
    print(
        f"[process_video] Input : {input_path}"
        f"  ({video_info.width}x{video_info.height} @ {video_info.fps:.1f} FPS, "
        f"~{video_info.total_frames} frames)"
    )

    frame_generator = sv.get_video_frames_generator(input_path)
    frame_count = 0
    start_time = time.perf_counter()

    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for frame in frame_generator:
            if max_frames is not None and frame_count >= max_frames:
                break

            detections = detector.detect_and_track(frame)
            annotated = detector.annotate_frame(frame, detections)
            sink.write_frame(annotated)

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.perf_counter() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"[process_video] Processed {frame_count} frames "
                    f"({current_fps:.1f} FPS)"
                )

    elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(
        f"[process_video] Done — {frame_count} frames in {elapsed:.1f}s "
        f"({avg_fps:.1f} FPS)"
    )
    print(f"[process_video] Output: {output_path}")


# ======================================================================
# Quick smoke-test when run directly
# ======================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m models.base_detection <input_video> <output_video>")
        sys.exit(1)

    process_video(sys.argv[1], sys.argv[2])

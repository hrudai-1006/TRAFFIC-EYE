"""Traffic-Eye AI — Model wrappers for detection, classification, and OCR."""

from models.base_detection import BaseDetector, process_video
from models.helmet_classifier import HelmetDetector
from models.seatbelt_classifier import SeatbeltDetector
from models.plate_detector import PlateDetector

__all__ = [
    "BaseDetector",
    "process_video",
    "HelmetDetector",
    "SeatbeltDetector",
    "PlateDetector",
]

"""Traffic-Eye AI — Violation Detection Modules."""

from violations.triple_riding import TripleRidingDetector
from violations.illegal_parking import IllegalParkingDetector
from violations.wrong_side import WrongSideDetector
from violations.red_light import RedLightDetector
from violations.helmet import HelmetViolationDetector
from violations.seatbelt import SeatbeltViolationDetector
from violations.anpr import ANPRModule

__all__ = [
    "TripleRidingDetector",
    "IllegalParkingDetector",
    "WrongSideDetector",
    "RedLightDetector",
    "HelmetViolationDetector",
    "SeatbeltViolationDetector",
    "ANPRModule",
]

"""
Traffic-Eye AI — Core Video Processor

Shared generator function that processes video frame by frame, yielding
progress after each frame.  Used by both pipeline.py (CLI) and
dashboard/app.py (live Streamlit UI).

This is the single source of truth for the per-frame detection pipeline.
Do NOT duplicate this logic elsewhere.
"""

import os
import sys
import inspect
import time
import json

import cv2
import numpy as np
import supervision as sv

# Ensure project root is on sys.path for relative imports
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CORE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.base_detection import BaseDetector
from violations.triple_riding import TripleRidingDetector
from violations.illegal_parking import IllegalParkingDetector
from violations.wrong_side import WrongSideDetector
from violations.red_light import RedLightDetector
from violations.helmet import HelmetViolationDetector
from violations.seatbelt import SeatbeltViolationDetector
from violations.anpr import ANPRModule
from intelligence.evidence_generator import EvidenceGenerator
from intelligence.violation_deduplicator import ViolationDeduplicator
from intelligence.risk_score import RiskScoreCalculator
from intelligence.repeat_offender import RepeatOffenderTracker
from agent.alert_router import AlertRouter


# ──────────────────────────────────────────────────────────────────────
# Utility: convert sv.Detections → dict understood by violation modules
# ──────────────────────────────────────────────────────────────────────

def detections_to_dict(detections: sv.Detections) -> dict:
    """Convert a supervision Detections object to a plain dict.

    Violation modules were designed to accept a dict with keys:
      xyxy, class_id, confidence, tracker_id
    This bridges the gap with the sv.Detections attribute-based API.
    """
    d = {
        "xyxy": detections.xyxy if detections.xyxy is not None else np.empty((0, 4)),
        "class_id": detections.class_id if detections.class_id is not None else np.array([]),
        "confidence": detections.confidence if detections.confidence is not None else np.array([]),
        "tracker_id": detections.tracker_id,  # may be None
    }
    return d


# ──────────────────────────────────────────────────────────────────────
# Core generator
# ──────────────────────────────────────────────────────────────────────

def process_video_stream(
    source_path: str,
    output_path: str | None = None,
    camera_id: str = "CAM_001",
    config_dir: str = "config",
    max_frames: int | None = None,
):
    """
    Generator that processes a video frame by frame and yields progress
    after each frame, so callers (CLI or Streamlit) can react incrementally.

    Yields a dict per frame:
        {
            "frame_idx": int,
            "total_frames": int,
            "annotated_frame": np.ndarray (BGR, ready for display),
            "new_violations": list[dict],  # evidence records generated this frame
            "fps_so_far": float,
            "elapsed_seconds": float,
            "violation_count": int,        # running total
            "seen_exceptions": set,
            "suppressed_counts": dict,
        }

    All initialization (detector, violation modules, intelligence layer,
    alert router) happens once before the frame loop begins.

    This function does NOT depend on argparse or any CLI-specific code.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Resolve config paths relative to project root
    if not os.path.isabs(config_dir):
        config_dir = os.path.join(project_root, config_dir)

    # ── 1. Base Detector ──────────────────────────────────────────
    print("\n[1/5] Loading Base Detector (YOLO11n + ByteTrack)...")
    base_detector = BaseDetector(model_name="yolo11n.pt")

    # ── 2. Violation Modules ──────────────────────────────────────
    print("[2/5] Loading Violation Modules...")

    camera_config_path = os.path.join(config_dir, "camera_locations.json")
    parking_config_path = os.path.join(config_dir, "parking_zones.json")
    direction_config_path = os.path.join(config_dir, "direction_vectors.json")

    triple_riding = TripleRidingDetector()
    illegal_parking = IllegalParkingDetector(config_path=parking_config_path)
    wrong_side = WrongSideDetector(config_path=direction_config_path)
    red_light = RedLightDetector(config_path=camera_config_path, camera_id=camera_id)
    helmet_det = HelmetViolationDetector()
    seatbelt_det = SeatbeltViolationDetector()
    anpr = ANPRModule()

    # Pre-compute the call signatures so we don't introspect on every frame
    violation_detectors = [
        triple_riding, illegal_parking, wrong_side,
        red_light, helmet_det, seatbelt_det,
    ]
    detector_signatures = {
        id(det): inspect.signature(det.check) for det in violation_detectors
    }

    # ── 3. Intelligence Layer ─────────────────────────────────────
    print("[3/5] Loading Intelligence Layer...")
    evidence_store_dir = os.path.join(project_root, "evidence_store")
    evidence_gen = EvidenceGenerator(
        output_dir=evidence_store_dir,
        config_path=camera_config_path,
        weights_config_path=os.path.join(config_dir, "violation_weights.json"),
    )
    risk_calc = RiskScoreCalculator(
        config_path=os.path.join(config_dir, "violation_weights.json"),
    )
    repeat_tracker = RepeatOffenderTracker(
        db_path=os.path.join(evidence_store_dir, "offenders.db"),
    )

    # ── 4. Alert Router ───────────────────────────────────────────
    print("[4/5] Loading Alert Router...")
    alert_router = AlertRouter(
        officer_config_path=os.path.join(config_dir, "officer_zones.json"),
        alert_log_path=os.path.join(evidence_store_dir, "alerts.json"),
    )

    # ── 5. Process Video ──────────────────────────────────────────
    print(f"[5/5] Processing video: {source_path}\n")

    video_info = sv.VideoInfo.from_video_path(source_path)
    frame_generator = sv.get_video_frames_generator(source_path)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=4)

    total_frames = video_info.total_frames or 0
    cooldown_frames = _load_deduplication_cooldown_frames(
        weights_config_path=os.path.join(config_dir, "violation_weights.json"),
        fps=getattr(video_info, "fps", None),
    )
    deduplicator = ViolationDeduplicator(cooldown_frames=cooldown_frames)
    start_time = time.perf_counter()
    violation_count = 0

    # Track exception types per module so each distinct failure prints once,
    # then repeat occurrences are counted silently and summarized at the end.
    seen_exceptions: set[tuple] = set()       # (module_name, exc_type_name)
    suppressed_counts: dict[tuple, int] = {}  # (module_name, exc_type_name) -> count

    # Open output video sink if an output path was provided
    sink = None
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sink = sv.VideoSink(output_path, video_info=video_info)
        sink.__enter__()

    try:
        for frame_idx, frame in enumerate(frame_generator):
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % 100 == 0:
                deduplicator.reset_stale_entries(frame_idx)

            # ── a. Detect & Track ─────────────────────────────────
            detections = base_detector.detect_and_track(frame)
            det_dict = detections_to_dict(detections)

            # ── b. Run Violation Modules ──────────────────────────
            all_violations: list[dict] = []
            for det_module in violation_detectors:
                try:
                    sig = detector_signatures[id(det_module)]
                    kwargs = {}
                    for pname in sig.parameters:
                        if pname == "self":
                            continue
                        if pname == "frame":
                            kwargs["frame"] = frame
                        elif pname == "detections":
                            kwargs["detections"] = det_dict
                        elif pname == "frame_number":
                            kwargs["frame_number"] = frame_idx
                        elif pname == "frame_shape":
                            kwargs["frame_shape"] = frame.shape

                    violations = det_module.check(**kwargs)
                    if violations:
                        all_violations.extend(violations)
                except Exception as exc:
                    exc_key = (det_module.__class__.__name__, type(exc).__name__)
                    if exc_key not in seen_exceptions:
                        seen_exceptions.add(exc_key)
                        suppressed_counts[exc_key] = 0
                        print(f"   ⚠️  {exc_key[0]}: {exc_key[1]}: {exc} (frame {frame_idx})")
                    else:
                        suppressed_counts[exc_key] += 1

            # ── c. Process Each Violation ─────────────────────────
            frame_evidence_records: list[dict] = []
            for v in all_violations:
                v_type = v.get("violation_type", "unknown")
                t_id = v.get("vehicle_id")

                # De-duplicate per (track_id, violation_type)
                if not deduplicator.should_log(t_id, v_type, frame_idx):
                    continue

                # ANPR on the violated vehicle
                plate_info = None
                vbbox = v.get("vehicle_bbox")
                if vbbox is not None:
                    try:
                        plate_info = anpr.read_plate(frame, vbbox)
                    except Exception:
                        pass

                plate_text = plate_info.get("plate_text") if plate_info else None
                prior_count = repeat_tracker.get_violation_count(plate_text) if plate_text else 0

                # Risk scoring
                risk_info = risk_calc.calculate(v_type, v["confidence"], prior_count)

                # Evidence generation
                record = evidence_gen.generate_evidence(
                    frame=frame,
                    violation_info=v,
                    camera_id=camera_id,
                    plate_info=plate_info,
                    risk_score=risk_info["risk_score"],
                    repeat_count=prior_count,
                    risk_category=risk_info["risk_category"],
                )

                violation_count += 1
                status_icon = "🔴" if record["status"] == "formal_record" else "🟡"
                print(
                    f"   {status_icon} [{frame_idx}] {v_type.upper()} | "
                    f"conf={record['confidence']:.2f} | "
                    f"risk={risk_info['risk_score']:.2f} ({risk_info['risk_category']}) | "
                    f"plate={plate_text or 'N/A'} | "
                    f"status={record['status']}"
                )

                # Record repeat offender
                if record["status"] == "formal_record" and plate_text:
                    repeat_tracker.record_violation(
                        plate_number=plate_text,
                        violation_type=v_type,
                        evidence_id=record["evidence_id"],
                        camera_id=record["camera_id"],
                        camera_location=record["camera_location"],
                        confidence=record["confidence"],
                        risk_score=record["risk_score"],
                    )
                    alert_router.route_alert(record)

                frame_evidence_records.append(record)

            # ── d. Annotate Frame ─────────────────────────────────
            labels = []
            if len(detections) > 0:
                for i in range(len(detections)):
                    cid = int(detections.class_id[i])
                    tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                    conf = float(detections.confidence[i])
                    cname = base_detector.get_class_name(cid)

                    # Tag with any active violations
                    v_tags = deduplicator.get_active_violation_types(tid, frame_idx)
                    tag_str = f" ⚠{','.join(v_tags)}" if v_tags else ""
                    labels.append(f"{cname} #{tid} {conf:.2f}{tag_str}")

            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            if sink is not None:
                sink.write_frame(annotated_frame)

            # Progress (CLI print)
            if frame_idx % 50 == 0 and frame_idx > 0:
                elapsed = time.perf_counter() - start_time
                fps = frame_idx / elapsed
                print(f"   📊 Frame {frame_idx}/{total_frames} | {fps:.1f} FPS | {violation_count} violations")

            # ── e. Yield progress to caller ───────────────────────
            elapsed = time.perf_counter() - start_time
            fps_so_far = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0

            yield {
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "annotated_frame": annotated_frame,
                "new_violations": frame_evidence_records,
                "fps_so_far": fps_so_far,
                "elapsed_seconds": elapsed,
                "violation_count": violation_count,
                "seen_exceptions": seen_exceptions,
                "suppressed_counts": suppressed_counts,
            }

    finally:
        if sink is not None:
            sink.__exit__(None, None, None)


def _load_deduplication_cooldown_frames(weights_config_path: str, fps) -> int:
    """Load the dedup cooldown, preferring a two-second FPS-derived window."""
    config_default = 48
    if os.path.exists(weights_config_path):
        try:
            with open(weights_config_path, "r") as f:
                data = json.load(f)
            config_default = int(
                data.get("deduplication_cooldown_frames", config_default)
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        fps_value = 0.0

    if fps_value > 0:
        return max(1, int(round(fps_value * 2)))
    return max(1, config_default)

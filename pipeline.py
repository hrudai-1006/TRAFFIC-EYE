"""
Traffic-Eye AI — Main Pipeline Orchestration (CLI)

Runs the end-to-end pipeline: video input -> base detection & tracking -> 
violation logic -> evidence generation -> intelligence -> alert routing.

This CLI wrapper delegates to core.processor.process_video_stream()
which is the single source of truth for the per-frame detection logic.
"""

import os
import sys
import argparse

import supervision as sv

# Ensure project root is on sys.path for relative imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.base_detection import process_video as _process_video_detect
from core.processor import process_video_stream


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Traffic-Eye AI Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output/final", help="Output directory")
    parser.add_argument("--config", type=str, default="config", help="Config directory")
    parser.add_argument("--camera-id", type=str, default="CAM_001", help="Camera ID from config")
    parser.add_argument("--mode", type=str, choices=["detect_only", "full"], default="full",
                        help="Pipeline mode")
    parser.add_argument("--benchmark", action="store_true", help="Print FPS benchmarks")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve config paths relative to project root
    config_dir = os.path.join(PROJECT_ROOT, args.config)

    os.makedirs(args.output, exist_ok=True)
    out_video_path = os.path.join(args.output, f"output_{os.path.basename(args.source)}")

    print("╔══════════════════════════════════════════╗")
    print("║       🚦  Traffic-Eye AI Pipeline        ║")
    print("╚══════════════════════════════════════════╝")

    # ── Detect-only mode ──────────────────────────────────────────
    if args.mode == "detect_only":
        print("\n[1/5] Loading Base Detector (YOLO11n + ByteTrack)...")
        from models.base_detection import BaseDetector
        base_detector = BaseDetector(model_name="yolo11n.pt")
        print("   Running in detect_only mode...")
        _process_video_detect(args.source, out_video_path, base_detector,
                              max_frames=args.max_frames)
        return

    # ── Full pipeline: consume the shared generator ───────────────
    # The generator handles ALL initialization (steps 1-4) and the
    # per-frame loop (step 5) internally, printing progress as it goes.
    # We just need to consume it and collect the final state.

    last_update = None
    for update in process_video_stream(
        source_path=args.source,
        output_path=out_video_path,
        camera_id=args.camera_id,
        config_dir=config_dir,
        max_frames=args.max_frames,
    ):
        last_update = update

    # ── Summary ───────────────────────────────────────────────────
    if last_update is None:
        print("\n  ⚠️  No frames processed.")
        return

    elapsed = last_update["elapsed_seconds"]
    processed = last_update["frame_idx"] + 1
    total_frames = last_update["total_frames"]
    violation_count = last_update["violation_count"]
    avg_fps = processed / elapsed if elapsed > 0 else 0.0
    evidence_store_dir = os.path.join(PROJECT_ROOT, "evidence_store")

    print("\n" + "═" * 55)
    print("  ✅  Pipeline Execution Complete")
    print("═" * 55)
    print(f"  📹 Input:           {args.source}")
    print(f"  📹 Output:          {out_video_path}")
    print(f"  🖼️  Frames:          {processed}")
    print(f"  ⏱️  Time:            {elapsed:.1f}s")
    print(f"  🚀 Avg FPS:         {avg_fps:.2f}")
    print(f"  🚨 Violations:      {violation_count}")
    print(f"  📁 Evidence Store:  {evidence_store_dir}")
    if args.benchmark:
        print(f"  📊 [BENCHMARK] Hardware=M4 MPS | FPS={avg_fps:.2f} (measured, not estimated)")

    # Report suppressed repeat exceptions
    suppressed_counts = last_update["suppressed_counts"]
    for exc_key, count in suppressed_counts.items():
        if count > 0:
            print(f"  ⚠️  {exc_key[0]} raised {exc_key[1]} {count} more time(s) after first occurrence (suppressed)")

    print("═" * 55)


# ──────────────────────────────────────────────────────────────────────
# Detect-only mode
# ──────────────────────────────────────────────────────────────────────

def process_video_detect_only(input_path, output_path, detector, benchmark=False):
    """Run only base detection and tracking for debugging."""
    _process_video_detect(input_path, output_path, detector)


if __name__ == "__main__":
    main()

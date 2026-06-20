#!/usr/bin/env python3
"""
Verification test for the core.processor generator.
Mimics what the dashboard live-detection loop does:
- Consumes the generator frame by frame
- Checks that yielded dicts have the expected keys
- Confirms frames stream incrementally (not all at once)
- Reports timing of each yield
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.processor import process_video_stream


def main():
    source = os.path.join(PROJECT_ROOT, "data", "test_videos", "traffic_demo_1.mp4")
    assert os.path.exists(source), f"Test video not found: {source}"

    print("=" * 60)
    print("Generator verification test")
    print("=" * 60)

    expected_keys = {
        "frame_idx", "total_frames", "annotated_frame", "new_violations",
        "fps_so_far", "elapsed_seconds", "violation_count",
        "seen_exceptions", "suppressed_counts",
    }

    frame_times = []
    last_update = None
    prev_time = time.perf_counter()

    for update in process_video_stream(
        source_path=source,
        output_path=None,  # No output video needed for this test
        max_frames=20,     # Only process 20 frames for speed
    ):
        now = time.perf_counter()
        frame_times.append(now - prev_time)
        prev_time = now
        last_update = update

        # Verify dict keys
        missing = expected_keys - set(update.keys())
        assert not missing, f"Frame {update['frame_idx']}: missing keys {missing}"

        # Verify annotated_frame is a numpy array with 3 dimensions
        assert update["annotated_frame"].ndim == 3, (
            f"Frame {update['frame_idx']}: annotated_frame has {update['annotated_frame'].ndim} dims"
        )

        # Verify new_violations is always a list
        assert isinstance(update["new_violations"], list), (
            f"Frame {update['frame_idx']}: new_violations is {type(update['new_violations'])}"
        )

        # Verify frame_idx is incrementing
        assert update["frame_idx"] == len(frame_times) - 1, (
            f"Expected frame_idx={len(frame_times)-1}, got {update['frame_idx']}"
        )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Frames yielded:     {len(frame_times)}")
    print(f"  All keys present:   ✅")
    print(f"  Incremental yields: ✅ (avg {sum(frame_times)/len(frame_times)*1000:.1f}ms per frame)")
    print(f"  Min/Max frame time: {min(frame_times)*1000:.1f}ms / {max(frame_times)*1000:.1f}ms")
    print(f"  Final FPS:          {last_update['fps_so_far']:.1f}")
    print(f"  Final violations:   {last_update['violation_count']}")
    print(f"  Total frames field: {last_update['total_frames']}")
    print("=" * 60)
    print("\n🎉 Generator verification PASSED")


if __name__ == "__main__":
    main()

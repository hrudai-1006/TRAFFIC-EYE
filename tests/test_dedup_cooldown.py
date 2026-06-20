#!/usr/bin/env python3
"""Verification test for sliding cooldown violation deduplication."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from intelligence.violation_deduplicator import ViolationDeduplicator


def test_sliding_cooldown():
    cooldown_frames = 48
    deduplicator = ViolationDeduplicator(cooldown_frames=cooldown_frames)

    first = deduplicator.should_log(5, "triple_riding", 10)
    repeat = deduplicator.should_log(5, "triple_riding", 15)
    expired_frame = 16 + cooldown_frames
    expired = deduplicator.should_log(5, "triple_riding", expired_frame)
    untracked = deduplicator.should_log(None, "triple_riding", 20)

    print(f"should_log(5, 'triple_riding', 10) = {first}")
    print(f"should_log(5, 'triple_riding', 15) = {repeat}")
    print(
        f"should_log(5, 'triple_riding', {expired_frame}) = {expired}"
    )
    print(f"should_log(None, 'triple_riding', 20) = {untracked}")
    print(
        "Note: after the frame-15 sighting, a sliding cooldown expires at "
        "frame 64 for cooldown_frames=48."
    )

    assert first is True
    assert repeat is False
    assert expired is True
    assert untracked is True


if __name__ == "__main__":
    test_sliding_cooldown()

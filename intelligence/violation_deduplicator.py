"""Time-windowed deduplication for repeated violation sightings."""

from __future__ import annotations

from typing import Any

__all__ = ["ViolationDeduplicator"]


class ViolationDeduplicator:
    """
    Suppresses repeated logging of the same ongoing violation across
    consecutive frames, keyed by (tracker_id, violation_type), using a
    sliding cooldown window rather than permanent suppression.

    A violation is considered a continuation of an already-logged event
    if the same tracker_id triggers the same violation_type again within
    `cooldown_frames` of the last time it was seen. Each time it's seen
    again within the window, the window slides forward, so a continuously
    true violation stays suppressed for its whole duration without
    re-logging. Once the gap since last seen exceeds `cooldown_frames`,
    the next occurrence is treated as a new event and is logged.
    """

    def __init__(self, cooldown_frames: int = 48):
        self.cooldown_frames = max(0, int(cooldown_frames))
        self._last_logged_frame: dict[tuple[int, str], int] = {}

    def should_log(
        self,
        tracker_id: Any,
        violation_type: str,
        current_frame_idx: int,
    ) -> bool:
        """
        Return True if this should be logged as a fresh event now.

        Always updates internal state to reflect this frame as the most
        recent sighting of this (tracker_id, violation_type), regardless
        of return value, so the cooldown window correctly slides forward.
        """
        if tracker_id is None or int(tracker_id) < 0:
            return True

        key = (int(tracker_id), violation_type)
        last_frame = self._last_logged_frame.get(key)
        current_frame_idx = int(current_frame_idx)

        if (
            last_frame is None
            or (current_frame_idx - last_frame) > self.cooldown_frames
        ):
            self._last_logged_frame[key] = current_frame_idx
            return True

        self._last_logged_frame[key] = current_frame_idx
        return False

    def get_active_violation_types(
        self,
        tracker_id: Any,
        current_frame_idx: int,
    ) -> list[str]:
        """Return violation types currently active for the tracked vehicle."""
        if tracker_id is None or int(tracker_id) < 0:
            return []

        tracker_id = int(tracker_id)
        current_frame_idx = int(current_frame_idx)
        active_types = [
            violation_type
            for (seen_tracker_id, violation_type), last_frame
            in self._last_logged_frame.items()
            if (
                seen_tracker_id == tracker_id
                and (current_frame_idx - last_frame) <= self.cooldown_frames
            )
        ]
        return sorted(active_types)

    def reset_stale_entries(
        self,
        current_frame_idx: int,
        max_age_frames: int = 600,
    ) -> None:
        """
        Prune entries not seen in a long time so the dict doesn't grow
        unbounded over a long video. Call periodically, not every frame.
        """
        current_frame_idx = int(current_frame_idx)
        stale_keys = [
            key
            for key, last_frame in self._last_logged_frame.items()
            if (current_frame_idx - last_frame) > max_age_frames
        ]
        for key in stale_keys:
            del self._last_logged_frame[key]

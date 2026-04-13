# backend/services/dwell_tracker.py
"""
Dwell-time tracker for parking violation detection (NFR2.1).

Uses greedy nearest-neighbour matching on bbox centre points to track
vehicles across consecutive frames.  A violation is only reported when
a vehicle's centre point stays inside a no-parking zone for N consecutive
frames (configurable via DWELL_THRESHOLD).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Maximum centre-point distance (px) for matching across frames.
_MATCH_DISTANCE_THRESHOLD = 80.0

# Frames without a match before a track is expired.
_MISSING_EXPIRE_FRAMES = 3


@dataclass
class VehicleTrack:
    """Tracks one vehicle's dwell state inside a no-parking zone."""

    track_id: int
    cx: float
    cy: float
    dwell_frames: int = 1
    missing_frames: int = 0
    reported: bool = False
    last_detection: Dict[str, Any] = field(default_factory=dict)


def _centre(bbox) -> Tuple[float, float]:
    """Return (cx, cy) from a bbox [x1, y1, x2, y2]."""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


class DwellTracker:
    """Per-source tracker: counts how many consecutive frames each vehicle
    stays in a no-parking zone before emitting a violation event.

    Parameters
    ----------
    dwell_threshold : int
        Number of consecutive in-zone frames before a violation is reported.
    """

    def __init__(self, dwell_threshold: int = 5):
        self.dwell_threshold = max(1, dwell_threshold)
        self._tracks: List[VehicleTrack] = []
        self._next_id: int = 0

    def update(
        self, in_zone_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Feed current-frame detections that are *already filtered* to be
        inside a no-parking zone.

        Returns the subset of detections whose corresponding tracks have
        reached ``dwell_threshold`` consecutive frames and have not yet been
        reported.  Each returned detection is enriched with a ``track_id`` key.
        """
        # Build centres for incoming detections
        incoming: List[Tuple[float, float, Dict[str, Any]]] = []
        for det in in_zone_detections:
            bbox = det.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            cx, cy = _centre(bbox)
            incoming.append((cx, cy, det))

        # Greedy nearest-neighbour matching (sorted by distance)
        matched_track_ids: set = set()
        matched_det_indices: set = set()
        pairs: List[Tuple[float, int, int]] = []

        for ti, track in enumerate(self._tracks):
            for di, (cx, cy, _det) in enumerate(incoming):
                d = _distance(track.cx, track.cy, cx, cy)
                if d <= _MATCH_DISTANCE_THRESHOLD:
                    pairs.append((d, ti, di))

        pairs.sort(key=lambda p: p[0])

        for _d, ti, di in pairs:
            if ti in matched_track_ids or di in matched_det_indices:
                continue
            matched_track_ids.add(ti)
            matched_det_indices.add(di)

            track = self._tracks[ti]
            cx, cy, det = incoming[di]
            track.cx = cx
            track.cy = cy
            track.dwell_frames += 1
            track.missing_frames = 0
            track.last_detection = det

        # Increment missing counter for unmatched tracks
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_ids:
                track.missing_frames += 1

        # Create new tracks for unmatched detections
        for di, (cx, cy, det) in enumerate(incoming):
            if di not in matched_det_indices:
                self._tracks.append(
                    VehicleTrack(
                        track_id=self._next_id,
                        cx=cx,
                        cy=cy,
                        last_detection=det,
                    )
                )
                self._next_id += 1

        # Expire old tracks
        self._tracks = [
            t for t in self._tracks if t.missing_frames < _MISSING_EXPIRE_FRAMES
        ]

        # Reset tracks that left the zone and came back
        for track in self._tracks:
            if track.missing_frames > 0 and track.reported:
                track.reported = False
                track.dwell_frames = 0

        # Collect newly-qualifying violations
        triggered: List[Dict[str, Any]] = []
        for track in self._tracks:
            if (
                track.dwell_frames >= self.dwell_threshold
                and not track.reported
                and track.missing_frames == 0
            ):
                track.reported = True
                det = dict(track.last_detection)
                det["track_id"] = track.track_id
                triggered.append(det)
                logger.info(
                    "Dwell violation triggered: track_id=%d, dwell=%d frames",
                    track.track_id,
                    track.dwell_frames,
                )

        return triggered

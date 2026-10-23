from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np

from config import Config
from splitting import Instance


@dataclass
class Track:
    id: int
    centroid: np.ndarray
    size: float
    missing: int = 0
    last_frame: int = 0


class Tracker:
    def __init__(self):
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, instances: List[Instance], config: Config, frame_index: int) -> None:
        pairs = []
        for track_index, track in enumerate(self.tracks):
            for instance_index, instance in enumerate(instances):
                distance = float(
                    np.hypot(
                        track.centroid[0] - instance.centroid[0],
                        track.centroid[1] - instance.centroid[1],
                    )
                )
                max_distance = max(
                    12.0,
                    config.track_max_distance_ratio * max(track.size, instance.size),
                )
                if distance <= max_distance:
                    pairs.append((distance, track_index, instance_index))
        pairs.sort(key=lambda item: item[0])

        matched_tracks: Set[int] = set()
        matched_instances: Set[int] = set()
        for _, track_index, instance_index in pairs:
            if track_index in matched_tracks or instance_index in matched_instances:
                continue
            track = self.tracks[track_index]
            instance = instances[instance_index]
            track.centroid = np.array(instance.centroid, dtype=np.float64)
            track.size = instance.size
            track.missing = 0
            track.last_frame = frame_index
            instance.track_id = track.id
            matched_tracks.add(track_index)
            matched_instances.add(instance_index)

        alive: List[Track] = []
        for track_index, track in enumerate(self.tracks):
            if track_index in matched_tracks:
                alive.append(track)
                continue
            track.missing += 1
            if track.missing <= config.track_max_missing:
                alive.append(track)
        self.tracks = alive

        for instance_index, instance in enumerate(instances):
            if instance_index in matched_instances:
                continue
            track = Track(
                id=self._next_id,
                centroid=np.array(instance.centroid, dtype=np.float64),
                size=instance.size,
                missing=0,
                last_frame=frame_index,
            )
            self._next_id += 1
            instance.track_id = track.id
            self.tracks.append(track)


class ContactStabilizer:
    def __init__(self, config: Config):
        self.config = config
        self._counts: dict[Tuple[int, int], int] = {}
        self._missing: dict[Tuple[int, int], int] = {}

    def update(self, pairs: List[Tuple[int, int]], frame_index: int) -> Set[Tuple[int, int]]:
        normalized = {tuple(sorted(pair)) for pair in pairs if pair[0] is not None and pair[1] is not None}
        for pair in list(self._counts.keys()):
            if pair in normalized:
                self._counts[pair] += 1
                self._missing[pair] = 0
            else:
                self._missing[pair] = self._missing.get(pair, 0) + 1
                if self._missing[pair] > self.config.contact_hysteresis:
                    del self._counts[pair]
                    del self._missing[pair]
        for pair in normalized:
            if pair not in self._counts:
                self._counts[pair] = 1
                self._missing[pair] = 0
        return {
            pair
            for pair, count in self._counts.items()
            if count >= self.config.contact_hysteresis
        }

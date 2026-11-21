from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import Config
from splitting import Instance


@dataclass
class Contact:
    a: int
    b: int
    distance: float
    point_a: Tuple[float, float]
    point_b: Tuple[float, float]


def _points(contour: np.ndarray, limit: int = 220) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit)
    return points[np.round(indices).astype(np.int32)]


def _closest_to_contour(
    points: np.ndarray, contour: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    segments = np.roll(contour, -1, axis=0) - contour
    lengths_sq = np.einsum("ij,ij->i", segments, segments)
    offsets = points[:, None, :] - contour
    projections = np.einsum("ijk,jk->ij", offsets, segments)
    fractions = np.clip(projections / np.where(lengths_sq > 1e-9, lengths_sq, 1.0), 0.0, 1.0)
    closest = contour + fractions[..., None] * segments
    delta = points[:, None, :] - closest
    distances_sq = np.einsum("ijk,ijk->ij", delta, delta)
    point_index, segment_index = np.unravel_index(distances_sq.argmin(), distances_sq.shape)
    return float(distances_sq[point_index, segment_index]), points[point_index], closest[point_index, segment_index]


def _min_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    forward, point_a, closest_b = _closest_to_contour(points_a, points_b)
    reverse, point_b, closest_a = _closest_to_contour(points_b, points_a)
    if reverse < forward:
        return float(np.sqrt(reverse)), closest_a, point_b
    return float(np.sqrt(forward)), point_a, closest_b


def _bbox_gap(a: Instance, b: Instance) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    gap_x = max(bx - (ax + aw), ax - (bx + bw), 0)
    gap_y = max(by - (ay + ah), ay - (by + bh), 0)
    return float(np.hypot(gap_x, gap_y))


def contact_gap(instances: List[Instance], config: Config) -> float:
    if not instances:
        return config.contact_gap_px
    median_size = float(np.median([instance.size for instance in instances]))
    return max(config.contact_gap_px, config.contact_gap_ratio * median_size)


def find_contacts(instances: List[Instance], config: Config) -> List[Contact]:
    if len(instances) < 2:
        return []
    gap = contact_gap(instances, config)
    median_size = float(np.median([instance.size for instance in instances]))
    same_group_gap = max(gap, 0.08 * median_size)
    points = {instance.id: _points(instance.contour) for instance in instances}
    contacts: List[Contact] = []
    for left_index in range(len(instances)):
        for right_index in range(left_index + 1, len(instances)):
            left = instances[left_index]
            right = instances[right_index]
            same_group = (
                left.split_group is not None
                and left.split_group == right.split_group
            )
            limit = same_group_gap if same_group else gap
            if _bbox_gap(left, right) > limit:
                continue
            distance, point_a, point_b = _min_distance(points[left.id], points[right.id])
            if distance <= limit:
                contacts.append(
                    Contact(
                        left.id,
                        right.id,
                        distance,
                        (float(point_a[0]), float(point_a[1])),
                        (float(point_b[0]), float(point_b[1])),
                    )
                )
    contacts.sort(key=lambda contact: (contact.a, contact.b))
    return contacts

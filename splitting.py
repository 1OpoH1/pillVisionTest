from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config


@dataclass
class Instance:
    id: int
    contour: np.ndarray
    area: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    size: float
    confidence: float = 1.0
    track_id: Optional[int] = None
    split_group: Optional[int] = None


@dataclass
class _Edge:
    a: int
    b: int
    prominence: float
    gradient_support: bool


def _make_instance(
    contour: np.ndarray,
    confidence: float,
    instance_id: int = 0,
    split_group: Optional[int] = None,
) -> Optional[Instance]:
    if contour is None or len(contour) < 3:
        return None
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        x, y, w, h = cv2.boundingRect(contour)
        centroid = (x + w / 2.0, y + h / 2.0)
    else:
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    bbox = cv2.boundingRect(contour)
    size = 2.0 * np.sqrt(area / np.pi)
    return Instance(instance_id, contour, area, centroid, bbox, size, confidence, None, split_group)


def compute_gradient(frame: np.ndarray, config: Config) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    d = config.gradient_blur
    if d > 1:
        if d % 2 == 0:
            d += 1
        gray = cv2.bilateralFilter(gray, d, 50, 50)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 99))
    if scale < 1.0:
        scale = max(float(magnitude.max()), 1.0)
    return np.clip(magnitude / scale * 255.0, 0, 255).astype(np.uint8)


def _h_maxima(dist: np.ndarray, height: float, config: Config) -> np.ndarray:
    marker = np.maximum(dist - height, 0.0)
    reconstructed = marker.copy()
    kernel = np.ones(
        (config.peak_reconstruction_kernel, config.peak_reconstruction_kernel),
        np.uint8,
    )
    for _ in range(config.peak_reconstruction_iterations):
        expanded = cv2.dilate(reconstructed, kernel)
        updated = np.minimum(expanded, dist)
        if float(np.max(np.abs(updated - reconstructed))) < 1e-3:
            reconstructed = updated
            break
        reconstructed = updated
    return dist - reconstructed


def _find_peaks(dist: np.ndarray, config: Config) -> List[Tuple[int, int, float]]:
    max_distance = float(dist.max())
    if max_distance < config.peak_min_absolute:
        return []
    smooth = dist
    if config.peak_smoothing_sigma > 0:
        smooth = cv2.GaussianBlur(dist, (0, 0), config.peak_smoothing_sigma)
    height = max(config.peak_min_absolute, config.peak_h_ratio * max_distance)
    maxima = _h_maxima(smooth, height, config)
    peak_mask = (maxima > 1e-3).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(peak_mask, connectivity=8)
    minimum_value = max(config.peak_min_absolute, config.peak_min_ratio * max_distance)
    peaks: List[Tuple[int, int, float]] = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        values = dist[ys, xs]
        index = int(np.argmax(values))
        value = float(values[index])
        if value < minimum_value:
            continue
        peaks.append((int(xs[index]), int(ys[index]), value))
    peaks.sort(key=lambda item: item[2], reverse=True)
    return peaks[: config.peak_max_count]


def _gradient_support(
    gradient: np.ndarray,
    region_a: np.ndarray,
    region_b: np.ndarray,
    boundary: np.ndarray,
    config: Config,
) -> bool:
    if not np.any(boundary):
        return False
    boundary_score = float(np.percentile(gradient[boundary], 75))
    inside = gradient[region_a | region_b]
    inside_score = float(np.median(inside)) if inside.size else 0.0
    return boundary_score > max(config.gradient_support_abs, inside_score * config.gradient_support_ratio)


class _UnionFind:
    def __init__(self, items: List[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _fill_small_holes(mask: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0:
        return mask
    total = int(np.count_nonzero(mask))
    if total == 0:
        return mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    result = mask.copy()
    max_area = max(1, int(ratio * total))
    for index, contour in enumerate(contours):
        if hierarchy[0, index, 3] == -1:
            continue
        if cv2.contourArea(contour) <= max_area:
            cv2.drawContours(result, [contour], -1, 255, -1)
    return result


def _absorb_small_regions(
    regions: Dict[int, np.ndarray],
    areas: Dict[int, int],
    minimum_area: int,
    kernel: np.ndarray,
) -> None:
    small = sorted(
        (label for label, area in areas.items() if area < minimum_area),
        key=lambda label: areas[label],
    )
    for label in small:
        if label not in regions:
            continue
        source = regions[label]
        source_u8 = source.astype(np.uint8) * 255
        source_dilated = cv2.dilate(source_u8, kernel)
        neighbors: List[Tuple[int, int]] = []
        for other in regions:
            if other == label:
                continue
            other_dilated = cv2.dilate(regions[other].astype(np.uint8) * 255, kernel)
            if np.any((source_dilated > 0) & (other_dilated > 0)):
                neighbors.append((areas[other], other))
        if not neighbors:
            del regions[label]
            del areas[label]
            continue
        _, target = max(neighbors)
        regions[target] = regions[target] | source
        areas[target] += areas[label]
        del regions[label]
        del areas[label]


def _split_blob(local_mask: np.ndarray, local_frame: np.ndarray, config: Config) -> List[Tuple[np.ndarray, float]]:
    local_mask = _fill_small_holes(local_mask, config.hole_fill_ratio)
    dist = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
    peaks = _find_peaks(dist, config)
    if len(peaks) <= 1:
        contour = _largest_contour(local_mask)
        return [(contour, 1.0)] if contour is not None else []

    markers = np.zeros(local_mask.shape, np.int32)
    markers[local_mask == 0] = 1
    for marker_id, (x, y, _) in enumerate(peaks, start=2):
        markers[y, x] = marker_id

    gradient = compute_gradient(local_frame, config)
    watershed_input = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    watershed = cv2.watershed(watershed_input, markers)

    peak_values: Dict[int, float] = {
        marker_id: value for marker_id, (_, _, value) in enumerate(peaks, start=2)
    }
    labels = sorted(
        label
        for label in np.unique(watershed[local_mask > 0]).tolist()
        if label not in (0, 1, -1)
    )
    regions: Dict[int, np.ndarray] = {}
    for label in labels:
        region = (watershed == label) & (local_mask > 0)
        if np.any(region):
            regions[label] = region
    if len(regions) <= 1:
        contour = _largest_contour(local_mask)
        return [(contour, 1.0)] if contour is not None else []

    region_areas = {label: int(np.count_nonzero(region)) for label, region in regions.items()}
    parent_area = max(int(np.count_nonzero(local_mask)), 1)
    kernel = np.ones((3, 3), np.uint8)
    minimum_part_area = max(
        config.min_part_area,
        int(config.min_part_area_ratio * parent_area),
    )
    _absorb_small_regions(regions, region_areas, minimum_part_area, kernel)
    if len(regions) <= 1:
        contour = _largest_contour(local_mask)
        return [(contour, 1.0)] if contour is not None else []
    edges: List[_Edge] = []
    labels_list = list(regions.keys())
    for left_index in range(len(labels_list)):
        for right_index in range(left_index + 1, len(labels_list)):
            left = labels_list[left_index]
            right = labels_list[right_index]
            left_u8 = regions[left].astype(np.uint8) * 255
            right_u8 = regions[right].astype(np.uint8) * 255
            boundary = (cv2.dilate(left_u8, kernel) > 0) & (cv2.dilate(right_u8, kernel) > 0)
            if not np.any(boundary):
                continue
            saddle = float(dist[boundary].max())
            left_peak = peak_values.get(left, float(dist[regions[left]].max()))
            right_peak = peak_values.get(right, float(dist[regions[right]].max()))
            denominator = max(min(left_peak, right_peak), 1e-6)
            prominence = float(np.clip(1.0 - saddle / denominator, 0.0, 1.0))
            support = _gradient_support(gradient, regions[left], regions[right], boundary, config)
            edges.append(_Edge(left, right, prominence, support))

    union_find = _UnionFind(labels_list)
    for edge in edges:
        smaller_area = min(region_areas[edge.a], region_areas[edge.b])
        area_ok = smaller_area >= max(
            config.min_part_area,
            int(config.min_part_area_ratio * parent_area),
        )
        split_by_neck = edge.prominence >= config.neck_prominence
        split_by_gradient = (
            edge.prominence >= config.prominence_gradient_min and edge.gradient_support
        )
        if not area_ok or not (split_by_neck or split_by_gradient):
            union_find.union(edge.a, edge.b)

    groups: Dict[int, List[int]] = {}
    for label in labels_list:
        root = union_find.find(label)
        groups.setdefault(root, []).append(label)

    if len(groups) <= 1:
        contour = _largest_contour(local_mask)
        return [(contour, 1.0)] if contour is not None else []

    results: List[Tuple[np.ndarray, float]] = []
    for group in groups.values():
        group_set = set(group)
        group_mask = np.zeros(local_mask.shape, np.uint8)
        for label in group:
            group_mask[regions[label]] = 255
        contour = _largest_contour(group_mask)
        if contour is None:
            continue
        external = [
            edge.prominence
            for edge in edges
            if (edge.a in group_set) != (edge.b in group_set)
        ]
        confidence = float(max(external)) if external else 0.5
        results.append((contour, confidence))
    return results


def split_mask(mask: np.ndarray, frame: np.ndarray, config: Config) -> List[Instance]:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    instances: List[Instance] = []
    for blob_index, contour in enumerate(contours):
        if cv2.contourArea(contour) < config.min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        local_mask = binary[y:y + h, x:x + w]
        local_frame = frame[y:y + h, x:x + w]
        parts = _split_blob(local_mask, local_frame, config)
        if len(parts) == 1:
            instance = _make_instance(contour, 1.0)
            if instance is not None:
                instances.append(instance)
            continue
        for part, confidence in parts:
            shifted = part.copy()
            shifted[:, :, 0] += x
            shifted[:, :, 1] += y
            instance = _make_instance(shifted, confidence, split_group=blob_index)
            if instance is not None and instance.area >= config.min_part_area:
                instances.append(instance)

    instances.sort(key=lambda instance: (instance.centroid[1], instance.centroid[0]))
    for index, instance in enumerate(instances, start=1):
        instance.id = index
    return instances

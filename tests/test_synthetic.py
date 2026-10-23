from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from contact import find_contacts
from segmentation import Segmenter
from splitting import split_mask


BACKGROUND = (245, 245, 245)
NUT = (55, 85, 130)
SHADOW = (205, 205, 205)


def make_config(**overrides) -> Config:
    config = Config()
    config.min_area = 180
    config.min_part_area = 120
    config.min_part_area_ratio = 0.03
    config.color_threshold = 25.0
    config.auto_color_threshold = False
    config.shadow_suppress = True
    config.contact_gap_px = 3.0
    config.contact_gap_ratio = 0.03
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def canvas() -> np.ndarray:
    return np.full((600, 800, 3), BACKGROUND, np.uint8)


def circle(image: np.ndarray, center, radius: int, color=NUT) -> None:
    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def ellipse(image: np.ndarray, center, axes, angle: float = 0.0, color=NUT) -> None:
    cv2.ellipse(image, center, axes, angle, 0, 360, color, -1, cv2.LINE_AA)


def crescent(image: np.ndarray, center, outer: int, inner: int, offset: int) -> None:
    circle(image, center, outer)
    inner_center = (center[0] + offset, center[1])
    circle(image, inner_center, inner, BACKGROUND)


def process(image: np.ndarray, config: Config):
    segmenter = Segmenter(config)
    mask = segmenter.foreground(image)
    instances = split_mask(mask, image, config)
    contacts = find_contacts(instances, config)
    return mask, instances, contacts


def check(name: str, condition: bool, detail: str) -> bool:
    if condition:
        print(f"OK   {name}: {detail}")
        return True
    print(f"FAIL {name}: {detail}")
    return False


def run() -> int:
    results = []

    image = canvas()
    circle(image, (350, 300), 50)
    circle(image, (450, 300), 50)
    _, instances, contacts = process(image, make_config())
    results.append(check("identical touching", len(instances) == 2 and len(contacts) == 1,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    ellipse(image, (300, 300), (65, 42), 15)
    crescent(image, (420, 300), 55, 42, 18)
    _, instances, contacts = process(image, make_config())
    results.append(check("different shapes touching", len(instances) == 2 and len(contacts) == 1,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    ellipse(image, (300, 300), (60, 40))
    ellipse(image, (445, 300), (85, 57))
    _, instances, contacts = process(image, make_config())
    results.append(check("same shape different size", len(instances) == 2 and len(contacts) == 1,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    circle(image, (180, 180), 35)
    circle(image, (250, 180), 35)
    ellipse(image, (360, 180), (45, 30))
    crescent(image, (530, 180), 45, 34, 12)
    ellipse(image, (230, 400), (70, 45))
    _, instances, contacts = process(image, make_config())
    results.append(check("mixed cluster", len(instances) == 5,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    crescent(image, (400, 300), 60, 45, 20)
    _, instances, _ = process(image, make_config())
    results.append(check("single concave object", len(instances) == 1,
                         f"objects={len(instances)}"))

    image = canvas()
    circle(image, (350, 300), 50)
    circle(image, (452, 300), 50)
    _, instances, contacts = process(image, make_config())
    results.append(check("small gap and contact", len(instances) == 2 and len(contacts) == 1,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    cv2.ellipse(image, (412, 312), (58, 38), 0, 0, 360, SHADOW, -1, cv2.LINE_AA)
    circle(image, (400, 300), 45)
    _, instances, _ = process(image, make_config())
    results.append(check("shadow suppression", len(instances) == 1,
                         f"objects={len(instances)}"))

    image = canvas()
    circle(image, (400, 300), 50)
    _, instances, _ = process(image, make_config())
    results.append(check("single round object", len(instances) == 1,
                         f"objects={len(instances)}"))

    image = canvas()
    ellipse(image, (400, 300), (75, 42), 25)
    _, instances, _ = process(image, make_config())
    results.append(check("single elongated object", len(instances) == 1,
                         f"objects={len(instances)}"))

    image = canvas()
    circle(image, (300, 300), 50)
    circle(image, (400, 300), 50)
    circle(image, (500, 300), 50)
    _, instances, contacts = process(image, make_config())
    results.append(check("three identical touching", len(instances) == 3 and len(contacts) == 2,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    circle(image, (300, 300), 50)
    circle(image, (400, 300), 50)
    circle(image, (350, 387), 50)
    _, instances, contacts = process(image, make_config())
    results.append(check("triangular cluster", len(instances) == 3 and len(contacts) == 3,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    circle(image, (300, 300), 50)
    circle(image, (400, 300), 50)
    circle(image, (300, 400), 50)
    circle(image, (400, 400), 50)
    _, instances, contacts = process(image, make_config())
    results.append(check("square cluster", len(instances) == 4 and len(contacts) == 4,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    circle(image, (300, 300), 55)
    circle(image, (373, 300), 18)
    _, instances, contacts = process(image, make_config())
    results.append(check("very different sizes touching", len(instances) == 2 and len(contacts) == 1,
                         f"objects={len(instances)} contacts={len(contacts)}"))

    image = canvas()
    circle(image, (400, 300), 50)
    circle(image, (400, 300), 28, BACKGROUND)
    _, instances, _ = process(image, make_config())
    results.append(check("highlight hole", len(instances) == 1,
                         f"objects={len(instances)}"))

    image = canvas()
    circle(image, (200, 200), 55)
    circle(image, (273, 200), 18)
    ellipse(image, (400, 200), (60, 35), 10)
    crescent(image, (610, 200), 55, 42, 18)
    ellipse(image, (400, 420), (70, 45), 30)
    _, instances, _ = process(image, make_config())
    results.append(check("mixed sizes and shapes", len(instances) == 5,
                         f"objects={len(instances)}"))

    image = canvas()
    _, instances, _ = process(image, make_config())
    results.append(check("empty scene", len(instances) == 0,
                         f"objects={len(instances)}"))

    image = canvas()
    rng = np.random.default_rng(7)
    for _ in range(80):
        center = (int(rng.integers(20, 780)), int(rng.integers(20, 580)))
        circle(image, center, 3)
    _, instances, _ = process(image, make_config())
    results.append(check("small noise removed", len(instances) == 0,
                         f"objects={len(instances)}"))

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(run())

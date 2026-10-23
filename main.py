from __future__ import annotations

import argparse
import math
import os
import random
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from calibration import estimate_scale
from config import Config
from contact import Contact, find_contacts
from segmentation import Segmenter
from sources import create_source, parse_region
from splitting import Instance, split_mask
from tracking import ContactStabilizer, Tracker
from visualize import draw_result, make_debug_views


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Поиск и подсчёт объектов на фото, видео или экране")
    parser.add_argument("--source", choices=["image", "video", "camera", "screen"], default="camera")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--background", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-result", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--min-part-area", type=int, default=None)
    parser.add_argument("--min-part-area-ratio", type=float, default=None)
    parser.add_argument("--color-threshold", type=float, default=None)
    parser.add_argument("--fixed-threshold", action="store_true")
    parser.add_argument("--neck-prominence", type=float, default=None)
    parser.add_argument("--peak-min-ratio", type=float, default=None)
    parser.add_argument("--contact-gap", type=float, default=None)
    parser.add_argument("--contact-gap-ratio", type=float, default=None)
    parser.add_argument("--no-shadow-suppress", action="store_true")
    parser.add_argument("--generate-test", action="store_true", help="Сгенерировать случайную сцену с орехами")
    parser.add_argument("--generate-count", type=int, default=None, help="Количество орехов при генерации")
    parser.add_argument("--generate-seed", type=int, default=None, help="Сид для генерации")
    parser.add_argument("--generate-out", type=str, default="test_images/generated.png", help="Путь для сохранения сгенерированной картинки")
    parser.add_argument("--auto-check", action="store_true", help="Проверить сгенерированную картинку и вывести счёт")
    return parser


def build_config(args: argparse.Namespace) -> Config:
    config = Config()
    if args.min_area is not None:
        config.min_area = max(1, args.min_area)
    if args.min_part_area is not None:
        config.min_part_area = max(1, args.min_part_area)
    if args.min_part_area_ratio is not None:
        config.min_part_area_ratio = float(np.clip(args.min_part_area_ratio, 0.01, 0.9))
    if args.color_threshold is not None:
        config.color_threshold = float(args.color_threshold)
        config.auto_color_threshold = False
    if args.fixed_threshold:
        config.auto_color_threshold = False
    if args.neck_prominence is not None:
        config.neck_prominence = float(np.clip(args.neck_prominence, 0.0, 0.95))
    if args.peak_min_ratio is not None:
        config.peak_min_ratio = float(np.clip(args.peak_min_ratio, 0.01, 0.9))
    if args.contact_gap is not None:
        config.contact_gap_px = max(0.0, args.contact_gap)
    if args.contact_gap_ratio is not None:
        config.contact_gap_ratio = max(0.0, args.contact_gap_ratio)
    if args.no_shadow_suppress:
        config.shadow_suppress = False
    return config


def contacts_text(contacts: List[Contact]) -> str:
    if not contacts:
        return "нет"
    return ", ".join(f"{contact.a}-{contact.b}" for contact in contacts)


def _distance_points(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _ellipse_extent(center, axes, angle):
    a, b = axes
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = math.hypot(a * cos_a, b * sin_a)
    dy = math.hypot(a * sin_a, b * cos_a)
    return max(dx, dy)


def generate_random_scene(
    count: Optional[int] = None,
    seed: Optional[int] = None,
    out_path: str = "test_images/generated.png",
) -> Tuple[str, int, np.ndarray]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if count is None:
        count = random.randint(3, 12)
    width = max(1024, 900)
    height = max(768, 700)
    bg = (245, 245, 245)
    colors = [(55, 85, 130), (70, 95, 145), (60, 80, 140), (75, 100, 155)]
    img = np.full((height, width, 3), bg, np.uint8)
    placed = []
    attempts = 0
    while len(placed) < count and attempts < 8000:
        attempts += 1
        shape_type = random.choice(["circle", "ellipse", "crescent"])
        base_r = random.randint(28, 65)
        candidate = None
        candidate_radius = 0
        if shape_type == "circle":
            r = base_r
            x = random.randint(r + 30, width - r - 30)
            y = random.randint(r + 30, height - r - 30)
            center = (x, y)
            candidate = ("circle", center, r, random.choice(colors), None)
            candidate_radius = r
        elif shape_type == "ellipse":
            axes = (random.randint(int(base_r * 0.7), int(base_r * 1.4)), random.randint(int(base_r * 0.5), int(base_r * 0.9)))
            angle = random.randint(0, 179)
            ext = _ellipse_extent((0, 0), axes, angle)
            x = random.randint(int(ext) + 35, width - int(ext) - 35)
            y = random.randint(int(ext) + 35, height - int(ext) - 35)
            candidate = ("ellipse", (x, y), axes, random.choice(colors), angle)
            candidate_radius = ext
        else:
            outer = random.randint(38, 62)
            inner = random.randint(int(outer * 0.55), int(outer * 0.8))
            offset = random.randint(int(outer * 0.15), int(outer * 0.35))
            x = random.randint(outer + 35, width - outer - 35)
            y = random.randint(outer + 35, height - outer - 35)
            candidate = ("crescent", (x, y), (outer, inner, offset), random.choice(colors), None)
            candidate_radius = outer

        if candidate is None:
            continue
        valid = True
        for obj in placed:
            if obj[0] == "circle":
                _, c2, r2, _, _ = obj
                r1 = candidate_radius
                if _distance_points(candidate[1], c2) < r1 + r2 - 1.0:
                    valid = False
                    break
            elif obj[0] == "ellipse":
                _, c2, axes2, _, angle2 = obj
                r1 = candidate_radius
                r2 = _ellipse_extent(c2, axes2, angle2)
                if _distance_points(candidate[1], c2) < r1 + r2 - 1.0:
                    valid = False
                    break
            else:
                _, c2, dims2, _, _ = obj
                r1 = candidate_radius
                r2 = dims2[0]
                if _distance_points(candidate[1], c2) < r1 + r2 - 1.0:
                    valid = False
                    break
        if valid:
            placed.append(candidate)

    for obj in placed:
        if obj[0] == "circle":
            _, center, r, color, _ = obj
            cv2.circle(img, center, r, color, -1, cv2.LINE_AA)
        elif obj[0] == "ellipse":
            _, center, axes, color, angle = obj
            cv2.ellipse(img, center, axes, float(angle), 0, 360, color, -1, cv2.LINE_AA)
        elif obj[0] == "crescent":
            _, center, dims, color, _ = obj
            outer, inner, offset = dims
            cv2.circle(img, center, outer, color, -1, cv2.LINE_AA)
            cv2.circle(img, (center[0] + offset, center[1]), inner, bg, -1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path, len(placed), img


def print_state(
    frame_index: int,
    instances: List[Instance],
    contacts: List[Contact],
    source_name: str,
    tracks: Optional[int] = None,
) -> None:
    scale = estimate_scale(instances)
    size_text = f"{scale.median_size:.1f}px" if scale is not None else "n/a"
    line = (
        f"[{source_name}] кадр {frame_index}: объектов={len(instances)}, "
        f"касаний={len(contacts)}, медианный размер={size_text}"
    )
    if tracks is not None:
        line += f", треков={tracks}"
    print(line)
    if contacts:
        print(f"  пары касаний: {contacts_text(contacts)}")


def map_track_pairs(
    contacts: List[Contact],
    instances: List[Instance],
    active_pairs: Set[Tuple[int, int]],
) -> List[Contact]:
    by_id = {instance.id: instance for instance in instances}
    result: List[Contact] = []
    for contact in contacts:
        left = by_id.get(contact.a)
        right = by_id.get(contact.b)
        if left is None or right is None or left.track_id is None or right.track_id is None:
            continue
        pair = tuple(sorted((left.track_id, right.track_id)))
        if pair in active_pairs:
            result.append(contact)
    return result


def run(args: argparse.Namespace) -> int:
    config = build_config(args)
    source = create_source(args.source, args.input, args.camera, parse_region(args.region))
    segmenter = Segmenter(config)
    if args.background:
        background = cv2.imread(args.background, cv2.IMREAD_COLOR)
        if background is None:
            source.release()
            raise FileNotFoundError(f"cannot read background: {args.background}")
        segmenter.set_background(background)

    tracker: Optional[Tracker] = Tracker() if source.is_live else None
    stabilizer = ContactStabilizer(config) if source.is_live else None
    frame_index = 0
    last_instances: List[Instance] = []
    last_contacts: List[Contact] = []
    last_display: Optional[np.ndarray] = None
    saved = False
    previous_count = -1
    previous_contact_count = -1

    print(f"Источник: {source.name}")
    if source.is_live and not args.no_gui:
        print("Клавиши: q - выход, b - запомнить фон, c - сбросить фон, s - сохранить кадр")
    if args.background:
        print(f"Фон загружен: {args.background}")

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                if frame_index == 0:
                    print("Не удалось получить кадр")
                    return 1
                break

            mask = segmenter.foreground(frame)
            instances = split_mask(mask, frame, config)
            raw_contacts = find_contacts(instances, config)
            active_contacts = raw_contacts
            active_pairs: Set[Tuple[int, int]] = set()
            track_count: Optional[int] = None

            if source.is_live and tracker is not None and stabilizer is not None:
                tracker.update(instances, config, frame_index)
                pairs = []
                by_id = {instance.id: instance for instance in instances}
                for contact in raw_contacts:
                    left = by_id.get(contact.a)
                    right = by_id.get(contact.b)
                    if left is None or right is None:
                        continue
                    if left.track_id is None or right.track_id is None:
                        continue
                    pairs.append((left.track_id, right.track_id))
                active_pairs = stabilizer.update(pairs, frame_index)
                active_contacts = map_track_pairs(raw_contacts, instances, active_pairs)
                track_count = len(tracker.tracks)

            last_instances = instances
            last_contacts = active_contacts
            scale = estimate_scale(instances)
            extra = None
            if scale is not None:
                extra = f"median {scale.median_size:.0f}px"
            display = draw_result(frame, instances, active_contacts, extra)
            last_display = display

            count_changed = len(instances) != previous_count
            contacts_changed = len(active_contacts) != previous_contact_count
            should_print = (
                source.single_frame
                or frame_index == 0
                or count_changed
                or contacts_changed
                or (not args.quiet and frame_index % max(1, args.print_every) == 0)
            )
            if should_print:
                print_state(frame_index, instances, active_contacts, source.name, track_count)
                previous_count = len(instances)
                previous_contact_count = len(active_contacts)

            if source.single_frame and args.save_result:
                saved = bool(cv2.imwrite(args.save_result, display))
                if not saved:
                    print(f"Не удалось сохранить результат: {args.save_result}")

            if not args.no_gui:
                cv2.imshow("Nuts counter", display)
                if args.debug:
                    for name, view in make_debug_views(frame, mask, config).items():
                        cv2.imshow(name, view)

            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break

            if source.single_frame:
                if not args.no_gui:
                    cv2.waitKey(0)
                break

            key = -1
            if not args.no_gui:
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b") and source.supports_background:
                segmenter.set_background(frame)
                print("Фон запомнен")
            if key == ord("c"):
                segmenter.clear_background()
                print("Фон сброшен")
            if key == ord("s") and args.save_result:
                if cv2.imwrite(args.save_result, display):
                    saved = True
                    print(f"Сохранено: {args.save_result}")
                else:
                    print(f"Не удалось сохранить: {args.save_result}")

        if args.save_result and not saved and last_display is not None and not source.single_frame:
            if cv2.imwrite(args.save_result, last_display):
                saved = True
                print(f"Сохранено: {args.save_result}")

        if source.is_live:
            print_state(frame_index, last_instances, last_contacts, source.name)
        return 0
    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
        return 0
    finally:
        source.release()
        if not args.no_gui:
            cv2.destroyAllWindows()


def check_image(path: str, config: Config) -> Tuple[int, int]:
    segmenter = Segmenter(config)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    mask = segmenter.foreground(image)
    instances = split_mask(mask, image, config)
    contacts = find_contacts(instances, config)
    return len(instances), len(contacts)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_test:
        count = args.generate_count
        out_path, actual_count, _ = generate_random_scene(count, args.generate_seed, args.generate_out)
        print(f"Сгенерировано: {out_path}")
        print(f"Фактических объектов нарисовано: {actual_count}")
        if args.auto_check:
            config = build_config(args)
            detected, contacts = check_image(out_path, config)
            print(f"Обнаружено алгоритмом: {detected}")
            print(f"Касаний: {contacts}")
            if detected == actual_count:
                print("СОВПАДАЕТ")
            else:
                print("НЕ СОВПАДАЕТ")
        return 0

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

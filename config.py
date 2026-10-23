from dataclasses import dataclass


@dataclass
class Config:
    min_area: int = 350
    color_threshold: float = 26.0
    auto_color_threshold: bool = True
    color_threshold_min: float = 22.0
    color_threshold_max: float = 90.0
    chroma_threshold: float = 4.0
    shadow_suppress: bool = True
    open_kernel: int = 3
    close_kernel: int = 5
    fill_holes: bool = False
    hole_fill_ratio: float = 0.0
    gradient_blur: int = 9
    peak_min_ratio: float = 0.05
    peak_min_absolute: float = 3.0
    peak_h_ratio: float = 0.08
    peak_smoothing_sigma: float = 1.0
    peak_reconstruction_kernel: int = 7
    peak_reconstruction_iterations: int = 512
    peak_max_count: int = 64
    neck_prominence: float = 0.35
    prominence_gradient_min: float = 0.1
    gradient_support_ratio: float = 1.25
    gradient_support_abs: float = 4.0
    min_part_area_ratio: float = 0.03
    min_part_area: int = 150
    contact_gap_ratio: float = 0.03
    contact_gap_px: float = 2.0
    track_max_distance_ratio: float = 0.75
    track_max_missing: int = 8
    contact_hysteresis: int = 2

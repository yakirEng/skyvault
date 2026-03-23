import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ResolutionConfig:
    gsd_cm: float
    canvas_px: int

@dataclass
class ObjectConfig:
    tennis_court_w_px: int
    tennis_court_h_px: int
    tennis_rotations: list[int]
    crosswalk_stripe_px: int
    crosswalk_gap_px: int
    crosswalk_len_px: int
    pool_min_px: int
    pool_max_px: int
    eucalyptus_min_px: int
    eucalyptus_max_px: int

@dataclass
class RoadConfig:
    width_px: int
    n_horizontal: tuple[int, int]
    n_vertical: tuple[int, int]
    padding_frac: float

@dataclass
class PipelineConfig:
    resolution: ResolutionConfig
    objects: ObjectConfig
    roads: RoadConfig
    class_ids: dict[str, int]
    ROAD_INTERNAL: int = 255

def _m_to_px(meters: float, gsd_cm: float) -> int:
    """Helper to convert meters to pixels based on Ground Sample Distance."""
    return round(meters / (gsd_cm / 100.0))

def load_config(path: str | Path = "pipeline_config.yaml") -> PipelineConfig:
    """
    Load and parse pipeline_config.yaml.
    Derives all _px fields using the provided gsd_cm.
    """
    config_path = Path(path)
    if not config_path.exists():
        # Fallback to check if it's in the parent directory (useful if running from inside pipeline/)
        fallback_path = Path(__file__).parent.parent / "pipeline_config.yaml"
        if fallback_path.exists():
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Configuration file not found at {path} or {fallback_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    # 1. Parse Resolution
    gsd_cm = float(raw_cfg["resolution"]["gsd_cm"])
    res_cfg = ResolutionConfig(
        gsd_cm=gsd_cm,
        canvas_px=int(raw_cfg["resolution"]["canvas_size"])
    )

    # 2. Parse and Convert Objects
    obj_raw = raw_cfg["objects"]
    obj_cfg = ObjectConfig(
        tennis_court_w_px=_m_to_px(obj_raw["tennis_court"]["width_m"], gsd_cm),
        tennis_court_h_px=_m_to_px(obj_raw["tennis_court"]["height_m"], gsd_cm),
        tennis_rotations=list(obj_raw["tennis_court"]["rotations"]),
        crosswalk_stripe_px=_m_to_px(obj_raw["crosswalk"]["stripe_width_m"], gsd_cm),
        crosswalk_gap_px=_m_to_px(obj_raw["crosswalk"]["stripe_gap_m"], gsd_cm),
        crosswalk_len_px=_m_to_px(obj_raw["crosswalk"]["total_length_m"], gsd_cm),
        pool_min_px=_m_to_px(obj_raw["pool"]["min_m"], gsd_cm),
        pool_max_px=_m_to_px(obj_raw["pool"]["max_m"], gsd_cm),
        eucalyptus_min_px=_m_to_px(obj_raw["eucalyptus"]["min_m"], gsd_cm),
        eucalyptus_max_px=_m_to_px(obj_raw["eucalyptus"]["max_m"], gsd_cm)
    )

    # 3. Parse and Convert Roads
    road_raw = raw_cfg["roads"]
    road_cfg = RoadConfig(
        width_px=_m_to_px(road_raw["width_m"], gsd_cm),
        n_horizontal=tuple(road_raw["n_horizontal"]),
        n_vertical=tuple(road_raw["n_vertical"]),
        padding_frac=float(road_raw["padding_frac"])
    )

    # 4. Parse Classes
    class_ids = {k: int(v) for k, v in raw_cfg["classes"].items()}

    # Assemble final config
    config = PipelineConfig(
        resolution=res_cfg,
        objects=obj_cfg,
        roads=road_cfg,
        class_ids=class_ids
    )

    # Validation: Ensure no derived pixel dimension is <= 0
    for dataclass_instance in [config.objects, config.roads]:
        for field_name, value in dataclass_instance.__dict__.items():
            if field_name.endswith("_px") and isinstance(value, int):
                if value <= 0:
                    raise ValueError(f"Derived pixel value for {field_name} is {value}. Must be > 0. Check gsd_cm or raw meter values.")

    return config
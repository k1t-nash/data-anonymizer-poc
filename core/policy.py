from dataclasses import dataclass
from typing import Optional, Tuple

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class PlatePolicy:
    name: str
    roi_expand_w: float
    roi_expand_h: float
    conf_min: float
    retries: int


# Предустановленные политики
STANDARD = PlatePolicy(
    name="standard",
    roi_expand_w=1.4,
    roi_expand_h=1.8,
    conf_min=0.25,
    retries=1,
)

AGGRESSIVE = PlatePolicy(
    name="aggressive",
    roi_expand_w=1.8,
    roi_expand_h=2.4,
    conf_min=0.15,
    retries=3,
)


def policy_for_bbox(
    *,
    bbox: BBox,
    zone: Optional[str],
) -> PlatePolicy:
    """
    Выбирает политику анонимизации по контексту сцены.
    - parking_zone_* -> AGGRESSIVE (больше expand, ниже порог)
    - всё остальное  -> STANDARD
    """
    if zone and zone.startswith("parking_zone"):
        return AGGRESSIVE
    return STANDARD
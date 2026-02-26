from typing import Optional, Tuple
from dataclasses import dataclass
from core.policy import PlatePolicy, BBox


@dataclass(frozen=True)
class SearchROI:
    x1: int
    y1: int
    x2: int
    y2: int
    source: str  # "bbox" | "bbox∩parking_roi"

    def as_tuple(self) -> BBox:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def _expand_bbox(
    bbox: BBox,
    expand_w: float,
    expand_h: float,
    image_w: int,
    image_h: int,
) -> BBox:
    """
    Расширяет bbox от центра на коэффициенты expand_w / expand_h.
    Результат всегда внутри кадра.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    nw = w * expand_w
    nh = h * expand_h

    nx1 = int(max(0, cx - nw / 2))
    ny1 = int(max(0, cy - nh / 2))
    nx2 = int(min(image_w, cx + nw / 2))
    ny2 = int(min(image_h, cy + nh / 2))

    return nx1, ny1, nx2, ny2


def plate_search_roi(
    *,
    bbox: BBox,
    policy: PlatePolicy,
    zone: Optional[str],
    image_w: int,
    image_h: int,
    parking_roi_bbox: Optional[BBox] = None,
) -> SearchROI:
    """
    Строит ROI для поиска номера.

    Логика:
    1. Расширяем bbox согласно политике.
    2. Если объект в parking_zone И есть parking_roi_bbox:
       берём пересечение expanded ∩ parking_roi.
    3. Если пересечение валидно (area > 0) → используем его.
    4. Иначе → используем expanded bbox.

    Возвращает SearchROI с полем source для дебага.
    """
    # Валидация входа
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        # Вырожденный bbox — возвращаем как есть
        return SearchROI(
            max(0, x1), max(0, y1),
            min(image_w, x2), min(image_h, y2),
            source="bbox"
        )

    expanded = _expand_bbox(
        bbox=bbox,
        expand_w=policy.roi_expand_w,
        expand_h=policy.roi_expand_h,
        image_w=image_w,
        image_h=image_h,
    )

    # Пытаемся пересечь с parking ROI
    if zone and zone.startswith("parking_zone") and parking_roi_bbox is not None:
        ex1, ey1, ex2, ey2 = expanded
        px1, py1, px2, py2 = parking_roi_bbox

        ix1 = max(ex1, px1)
        iy1 = max(ey1, py1)
        ix2 = min(ex2, px2)
        iy2 = min(ey2, py2)

        if ix1 < ix2 and iy1 < iy2:
            return SearchROI(ix1, iy1, ix2, iy2, source="bbox∩parking_roi")

    return SearchROI(expanded[0], expanded[1], expanded[2], expanded[3], source="bbox")
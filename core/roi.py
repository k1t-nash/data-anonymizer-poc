from pathlib import Path
from typing import List, Tuple, Optional
import json
import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class ROIZone:
    def __init__(self, name: str, points: List[Tuple[int, int]]):
        self.name = name
        # Минимум 3 точки для валидного полигона
        if len(points) < 3:
            raise ValueError(f"ROIZone '{name}' требует минимум 3 точки, получено {len(points)}")
        self.points = np.array(points, dtype=np.int32)

    def contains_bbox(self, bbox: BBox) -> bool:
        """
        Проверяет попадание центра bbox внутрь полигона.
        Возвращает True если центр на границе или внутри.
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        # >= 0 включает границу полигона
        return cv2.pointPolygonTest(self.points, (float(cx), float(cy)), False) >= 0

    def bounding_box(self) -> BBox:
        """Минимальный прямоугольник вокруг полигона."""
        x1 = int(self.points[:, 0].min())
        y1 = int(self.points[:, 1].min())
        x2 = int(self.points[:, 0].max())
        y2 = int(self.points[:, 1].max())
        return x1, y1, x2, y2


class ROIMap:
    def __init__(
        self,
        zones: List[ROIZone],
        image_width: int,
        image_height: int,
    ):
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Некорректные размеры изображения: {image_width}x{image_height}")
        self.zones = zones
        self.image_width = image_width
        self.image_height = image_height

    @classmethod
    def from_json(cls, path: Path) -> "ROIMap":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ROI конфиг не найден: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required = {"image_width", "image_height", "zones"}
        missing = required - data.keys()
        if missing:
            raise KeyError(f"ROI JSON не содержит обязательных полей: {missing}")

        zones = []
        for z in data["zones"]:
            name   = z["name"]
            points = [tuple(p) for p in z["points"]]
            try:
                zones.append(ROIZone(name=name, points=points))
            except ValueError as e:
                # Пропускаем невалидные зоны с предупреждением, не падаем
                import warnings
                warnings.warn(f"Пропуск зоны '{name}': {e}")

        return cls(
            zones=zones,
            image_width=data["image_width"],
            image_height=data["image_height"],
        )

    @classmethod
    def empty(cls, image_width: int, image_height: int) -> "ROIMap":
        """Пустая карта без зон — для случаев когда конфига нет."""
        return cls(zones=[], image_width=image_width, image_height=image_height)

    def classify_bbox(self, bbox: BBox) -> Optional[str]:
        """
        Возвращает имя первой зоны в которую попадает центр bbox.
        None если ни одна зона не совпала.
        """
        for zone in self.zones:
            if zone.contains_bbox(bbox):
                return zone.name
        return None

    def parking_union_bbox(self) -> Optional[BBox]:
        """
        Возвращает объединённый bbox всех parking_zone_* зон.
        None если парковочных зон нет.

        Исправлен баг оригинала: xs1/xs2/ys1/ys2 дублировали одни и те же точки —
        результат был правильным случайно. Теперь явно берём min/max по всем точкам.
        """
        parking_zones = [z for z in self.zones if z.name.startswith("parking_zone")]
        if not parking_zones:
            return None

        all_x: List[int] = []
        all_y: List[int] = []

        for z in parking_zones:
            all_x.extend(z.points[:, 0].tolist())
            all_y.extend(z.points[:, 1].tolist())

        return (
            int(min(all_x)),
            int(min(all_y)),
            int(max(all_x)),
            int(max(all_y)),
        )

    def has_parking_zones(self) -> bool:
        return any(z.name.startswith("parking_zone") for z in self.zones)

    def zone_names(self) -> List[str]:
        return [z.name for z in self.zones]
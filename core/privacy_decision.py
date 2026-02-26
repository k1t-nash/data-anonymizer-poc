from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from core.roi import ROIMap
from core.policy import PlatePolicy, policy_for_bbox
from core.plate_search_roi import plate_search_roi, SearchROI

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class PrivacyDecision:
    """
    Финальное решение по анонимизации одного объекта.
    Единственный выход decision layer — никакого ML, никакой обработки изображений.
    """
    # Идентификация
    frame_index:   int
    timestamp_sec: float
    object_type:   str        # "vehicle", "face"
    bbox:          BBox

    # Контекст
    zone:   Optional[str]     # "parking_zone_1" | None
    policy: PlatePolicy

    # Геометрия
    plate_search_roi: SearchROI

    # Действия
    actions: List[str]        # ["detect_plate", "blur_plate"]

    # Объяснение для аудита
    reason: str

    def should_blur(self) -> bool:
        return "blur_plate" in self.actions

    def should_detect(self) -> bool:
        return "detect_plate" in self.actions


class PrivacyDecisionEngine:
    """
    Единственная точка принятия решений:
    ROI → policy → plate search ROI → PrivacyDecision.

    Без ML. Без обработки изображений. Только контекст + геометрия + правила.
    """

    def __init__(self, roi_map: ROIMap) -> None:
        self.roi_map = roi_map

    def decide(
        self,
        *,
        frame_index:   int,
        timestamp_sec: float,
        bbox:          BBox,
        object_type:   str = "vehicle",
    ) -> PrivacyDecision:

        # Валидация bbox
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Некорректный bbox: {bbox} — x2 должен быть > x1, y2 > y1"
            )

        # 1. Классификация зоны
        zone = self.roi_map.classify_bbox(bbox)

        # 2. Выбор политики
        policy = policy_for_bbox(bbox=bbox, zone=zone)

        # 3. Parking ROI bbox (объединение всех парковочных зон)
        parking_bbox = None
        if zone and zone.startswith("parking_zone"):
            parking_bbox = self.roi_map.parking_union_bbox()

        # 4. Plate search ROI
        search_roi = plate_search_roi(
            bbox=bbox,
            policy=policy,
            zone=zone,
            image_w=self.roi_map.image_width,
            image_h=self.roi_map.image_height,
            parking_roi_bbox=parking_bbox,
        )

        # 5. Действия — всегда детектируем и блюрим
        #    Расширяемо: здесь можно добавить логику "не блюрить в зоне X"
        actions = ["detect_plate", "blur_plate"]

        # 6. Причина для аудит-лога
        reason = (
            f"object={object_type}, "
            f"zone={zone or 'flow'}, "
            f"policy={policy.name}, "
            f"search_roi={search_roi.source}, "
            f"roi_area={search_roi.area}px²"
        )

        return PrivacyDecision(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            object_type=object_type,
            bbox=bbox,
            zone=zone,
            policy=policy,
            plate_search_roi=search_roi,
            actions=actions,
            reason=reason,
        )

    def decide_batch(
        self,
        *,
        frame_index:   int,
        timestamp_sec: float,
        bboxes:        List[BBox],
        object_type:   str = "vehicle",
    ) -> List[PrivacyDecision]:
        """
        Батч версия для обработки всех объектов кадра за один вызов.
        Невалидные bbox пропускаются без падения.
        """
        decisions = []
        for bbox in bboxes:
            try:
                decisions.append(
                    self.decide(
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        bbox=bbox,
                        object_type=object_type,
                    )
                )
            except ValueError:
                continue
        return decisions
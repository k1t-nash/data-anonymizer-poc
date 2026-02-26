import re
import torch
import easyocr
import numpy as np
import supervision as sv
from typing import List, Tuple, Set, Dict
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction


class PlateDetector:
    def __init__(self, model_path: str, conf: float = 0.3):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.detection_model = UltralyticsDetectionModel(
            model_path=model_path,
            confidence_threshold=conf,
            device=self.device
        )

        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.tracker = sv.ByteTrack()

        # ID треков которые подтверждены как номера
        self.confirmed_ids: Set[int] = set()
        # Активные ID в текущем кадре — для очистки мёртвых треков
        self.active_ids: Set[int] = set()
        # Кэш OCR результатов по tid — не гоняем OCR дважды на одном треке
        self.ocr_cache: Dict[int, bool] = {}

    def _is_valid_geometry(self, bbox: np.ndarray) -> bool:
        """
        Номер всегда горизонтальный прямоугольник.
        Отсекаем капоты, колёса и прочий мусор по пропорциям.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if h <= 0 or w <= 0:
            return False
        ratio = w / h
        # Большинство номеров: соотношение сторон от 2.0 до 7.0
        # Минимальная ширина 30px — меньше это шум
        return 2.0 <= ratio <= 7.0 and w >= 30

    def _is_valid_ocr(self, crop: np.ndarray) -> bool:
        """
        OCR подтверждение: ищем хотя бы 2 символа латиница+цифры.
        Одиночный символ — слишком много ложных срабатываний.
        """
        try:
            results = self.reader.readtext(crop)
            for (_, text, prob) in results:
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) >= 2 and prob > 0.3:
                    return True
        except Exception:
            pass
        return False

    def _cleanup_dead_tracks(self, current_ids: Set[int]) -> None:
        """
        ByteTrack переиспользует ID после потери трека.
        Без очистки confirmed_ids растёт бесконечно — утечка памяти.
        """
        dead_ids = self.active_ids - current_ids
        self.confirmed_ids -= dead_ids
        for tid in dead_ids:
            self.ocr_cache.pop(tid, None)
        self.active_ids = current_ids

    def detect_in_frame(
        self,
        image: np.ndarray,
        vehicle_bboxes: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Возвращает список bbox (x1, y1, x2, y2) для анонимизации.
        """
        h_img, w_img = image.shape[:2]
        all_candidates = []

        for v_box in vehicle_bboxes:
            vx1, vy1, vx2, vy2 = map(int, v_box[:4])

            # Защита от выхода за границы кадра
            vx1 = max(0, vx1)
            vy1 = max(0, vy1)
            vx2 = min(w_img, vx2)
            vy2 = min(h_img, vy2)

            v_crop = image[vy1:vy2, vx1:vx2]
            if v_crop.size == 0:
                continue

            v_h, v_w = v_crop.shape[:2]

            # SAHI: тайлы не больше размера кропа — иначе деградирует в обычный inference
            slice_h = min(480, v_h)
            slice_w = min(480, v_w)

            # Если кроп совсем маленький — не слайсируем
            if v_h < 32 or v_w < 64:
                continue

            try:
                result = get_sliced_prediction(
                    v_crop,
                    self.detection_model,
                    slice_height=slice_h,
                    slice_width=slice_w,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    verbose=0
                )
            except Exception:
                continue

            for obj in result.object_prediction_list:
                px1, py1, px2, py2 = map(int, obj.bbox.to_xyxy())
                # Переводим координаты обратно в систему полного кадра
                abs_bbox = [
                    vx1 + px1,
                    vy1 + py1,
                    vx1 + px2,
                    vy1 + py2,
                    obj.score.value
                ]
                all_candidates.append(abs_bbox)

        if not all_candidates:
            return []

        # Трекинг
        detections = sv.Detections(
            xyxy=np.array([c[:4] for c in all_candidates], dtype=np.float32),
            confidence=np.array([c[4] for c in all_candidates], dtype=np.float32),
            class_id=np.zeros(len(all_candidates), dtype=int)
        )
        tracked = self.tracker.update_with_detections(detections)

        if len(tracked.xyxy) == 0:
            return []

        current_ids = set(tracked.tracker_id.tolist())
        self._cleanup_dead_tracks(current_ids)

        final_plates = []

        for i in range(len(tracked.xyxy)):
            tid = int(tracked.tracker_id[i])
            bbox = tracked.xyxy[i].astype(int)
            conf = float(tracked.confidence[i])

            self.active_ids.add(tid)

            # Геометрическая валидация — быстрый фильтр мусора
            if not self._is_valid_geometry(bbox):
                continue

            # Логика подтверждения трека
            if tid not in self.confirmed_ids and tid not in self.ocr_cache:
                # Высокий confidence — доверяем без OCR
                if conf >= 0.75:
                    self.confirmed_ids.add(tid)
                    self.ocr_cache[tid] = True
                else:
                    # OCR валидация
                    crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if crop.size > 0:
                        ocr_result = self._is_valid_ocr(crop)
                        self.ocr_cache[tid] = ocr_result
                        if ocr_result:
                            self.confirmed_ids.add(tid)

            # Блюрим если: подтверждён трекером ИЛИ высокий confidence от YOLO
            # Второй вариант — страховка для новых треков которые OCR ещё не видел
            if tid in self.confirmed_ids or conf >= 0.75:
                x1, y1, x2, y2 = bbox
                # Финальная защита границ
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)
                if x2 > x1 and y2 > y1:
                    final_plates.append((x1, y1, x2, y2))

        return final_plates
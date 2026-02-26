#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiandy v3.2 (father script): Plate anonymizer with ROI allowlist + size/aspect/area filters + optional debug boxes.

Идея:
- Есть модель номерных знаков (plate_yolo.pt).
- На каждом кадре детектим номера, фильтруем по геометрии и (опционально) по ROI.
- Дальше включаем "hold" (держим бокс несколько кадров, даже если детекция пропала).
- Рисуем либо блюр, либо мозаик, либо только боксы (режим boxes).

Этот файл — наш базовый "отец-скрипт".
Параметры:
  --in-video         входное видео
  --out-video        выходное видео
  --model            YOLO-модель номеров
  --imgsz            размер инференса
  --conf             conf threshold
  --iou              iou threshold
  --device           GPU/CPU (например "0")
  --mode             anonymize | boxes
  --track / --no-track   флаг, но в этой версии трекинг простой (через hold), без ID
  --vid-stride       шаг кадров для инференса (1 = каждый кадр)
  --hold             сколько кадров держать бокс после последней уверенной детекции
  --smooth           сглаживание координат (чем больше, тем мягче)
  --permit-roi-rel   ROI: x1,y1,x2,y2 в [0..1] для ЦЕНТРА бокса
  --max-area-rel     макс. доля площади кадра для бокса
  --min-w/h, max-w/h, aspect-min/max — фильтры по размерам / аспекту
  --expand-w/h       во сколько раз расширять бокс перед блюром
  --method           blur | mosaic
  --blur-k           ядро блюра (Gaussian)
  --mosaic           пиксель размер для мозаики
  --crf, --preset    качество кодека (libx264 fallback)
"""

from __future__ import annotations

import argparse
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO



def tiled_batch_detect(model, frame: np.ndarray,
                 imgsz: int, conf: float, iou: float, device: str,
                 tile_size: int = 640, overlap: float = 0.25,
                 merge_iou: float = 0.5,
                 batch_size: int = 8) -> np.ndarray:
    """
    SAHI-like sliced inference с БАТЧИНГОМ тайлов.
    Полнокадровый + тайловый инференс за минимум вызовов model.predict().
    """
    H, W = frame.shape[:2]
    stride = int(tile_size * (1.0 - overlap))
    all_boxes = []   # list of (x1, y1, x2, y2, conf)

    # --- полнокадровый инференс ---
    res_full = model.predict(
        source=frame, imgsz=imgsz, conf=conf, iou=iou,
        device=device, verbose=False
    )[0]
    if res_full.boxes is not None and len(res_full.boxes) > 0:
        xyxy = res_full.boxes.xyxy.cpu().numpy()
        confs = res_full.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            all_boxes.append((*xyxy[i], confs[i]))

    # --- нарезаем тайлы ---
    tiles = []
    offsets = []
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)
            if (x1 - x0) < tile_size // 2 or (y1 - y0) < tile_size // 2:
                continue
            tile = frame[y0:y1, x0:x1]
            tiles.append(tile)
            offsets.append((x0, y0))

    # --- батчевый инференс тайлов (передаём список, не np.stack) ---
    for batch_start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[batch_start:batch_start + batch_size]
        batch_offsets = offsets[batch_start:batch_start + batch_size]

        results = model.predict(
            source=batch_tiles, imgsz=imgsz, conf=conf, iou=iou,
            device=device, verbose=False,
        )

        for res, (x0, y0) in zip(results, batch_offsets):
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    bx1 = xyxy[i][0] + x0
                    by1 = xyxy[i][1] + y0
                    bx2 = min(xyxy[i][2] + x0, W)
                    by2 = min(xyxy[i][3] + y0, H)
                    all_boxes.append((bx1, by1, bx2, by2, confs[i]))

    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32)

    arr = np.array(all_boxes, dtype=np.float32)
    boxes_for_nms = arr[:, :4]
    scores = arr[:, 4]
    indices = _nms(boxes_for_nms, scores, merge_iou)
    return boxes_for_nms[indices]

def tiled_detect(model, frame: np.ndarray,
                 imgsz: int, conf: float, iou: float, device: str,
                 tile_size: int = 640, overlap: float = 0.25,
                 merge_iou: float = 0.5) -> np.ndarray:
    """
    SAHI-like sliced inference: разрезаем кадр на тайлы с перекрытием,
    детектим на каждом, собираем все боксы и делаем NMS.
    Возвращает np.ndarray shape (N, 4) — xyxy координаты в исходном кадре.
    """
    H, W = frame.shape[:2]
    stride = int(tile_size * (1.0 - overlap))
    all_boxes = []   # list of (x1, y1, x2, y2, conf)

    # --- полнокадровый инференс ---
    res_full = model.predict(
        source=frame, imgsz=imgsz, conf=conf, iou=iou,
        device=device, verbose=False
    )[0]
    if res_full.boxes is not None and len(res_full.boxes) > 0:
        xyxy = res_full.boxes.xyxy.cpu().numpy()
        confs = res_full.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            all_boxes.append((*xyxy[i], confs[i]))

    # --- тайловый инференс ---
    tiles_count = 0
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)
            # пропускаем слишком маленькие краевые тайлы
            if (x1 - x0) < tile_size // 2 or (y1 - y0) < tile_size // 2:
                continue

            tile = frame[y0:y1, x0:x1]
            res = model.predict(
                source=tile, imgsz=imgsz, conf=conf, iou=iou,
                device=device, verbose=False
            )[0]
            tiles_count += 1
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    # смещаем координаты обратно в систему полного кадра
                    bx1 = xyxy[i][0] + x0
                    by1 = xyxy[i][1] + y0
                    bx2 = xyxy[i][2] + x0
                    by2 = xyxy[i][3] + y0
                    all_boxes.append((bx1, by1, bx2, by2, confs[i]))

    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32)

    # print(f"Main frame: 1. Tiles count: {tiles_count}")

    # --- NMS для слияния дубликатов ---
    arr = np.array(all_boxes, dtype=np.float32)
    boxes_for_nms = arr[:, :4]
    scores = arr[:, 4]
    indices = _nms(boxes_for_nms, scores, merge_iou)
    return boxes_for_nms[indices]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
    """Простой greedy NMS на numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou_vals = inter / (union + 1e-6)
        inds = np.where(iou_vals <= iou_thr)[0]
        order = order[inds + 1]
    return keep


# ----------------- utils -----------------

def odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def clamp_xyxy(x1, y1, x2, y2, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(int(round(x1)), w - 1)))
    y1 = int(max(0, min(int(round(y1)), h - 1)))
    x2 = int(max(x1 + 1, min(int(round(x2)), w)))
    y2 = int(max(y1 + 1, min(int(round(y2)), h)))
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0


def expand_xyxy(x1: int, y1: int, x2: int, y2: int,
                w: int, h: int,
                expand_w: float, expand_h: float) -> Tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    nbw = bw * float(expand_w)
    nbh = bh * float(expand_h)
    nx1 = cx - nbw / 2.0
    ny1 = cy - nbh / 2.0
    nx2 = cx + nbw / 2.0
    ny2 = cy + nbh / 2.0
    return clamp_xyxy(nx1, ny1, nx2, ny2, w, h)


def parse_roi_rel(s: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("permit-roi-rel must be 'x1,y1,x2,y2' in [0..1]")
    vals = tuple(float(p) for p in parts)
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise ValueError("permit-roi-rel values must be in [0..1]")
    return vals  # type: ignore


def compute_rotation_angle(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Вычисляет угол наклона бокса относительно горизонтали.
    Возвращает угол в градусах.
    """
    width = x2 - x1
    height = y2 - y1
    if width == 0:
        return 90.0
    angle = np.degrees(np.arctan(height / width))
    return abs(angle)


def compute_aspect_ratio_rotated(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Вычисляет эффективное соотношение сторон с учетом возможного наклона.
    Для наклоненных объектов использует диагональ.
    """
    width = x2 - x1
    height = y2 - y1

    # Стандартное соотношение сторон
    aspect = float(width) / float(height + 1e-6)

    # Если объект сильно наклонен, используем обратное соотношение
    angle = compute_rotation_angle(x1, y1, x2, y2)
    if angle > 60:  # Сильный наклон
        aspect = 1.0 / (aspect + 1e-6)

    return aspect

def to_abs_roi(rel: Tuple[float, float, float, float],
               W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = rel[0] * W
    y1 = rel[1] * H
    x2 = rel[2] * W
    y2 = rel[3] * H
    return clamp_xyxy(x1, y1, x2, y2, W, H)


def center_in_roi(x1: int, y1: int, x2: int, y2: int,
                  roi: Tuple[int, int, int, int]) -> bool:
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    rx1, ry1, rx2, ry2 = roi
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def apply_blur(img: np.ndarray,
               x1: int, y1: int, x2: int, y2: int,
               k: int) -> None:
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = odd(max(3, int(k)))
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def apply_mosaic(img: np.ndarray,
                 x1: int, y1: int, x2: int, y2: int,
                 mosaic: int) -> None:
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    rh, rw = y2 - y1, x2 - x1
    mosaic = max(2, int(mosaic))
    mh = max(1, rh // mosaic)
    mw = max(1, rw // mosaic)
    if mw < 1 or mh < 1 or rw < 1 or rh < 1:
        return
    small = cv2.resize(roi, (mw, mh), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pix


# ------------- video writer (ffmpeg, NVENC + fallback) -------------

class FFmpegWriter:
    def __init__(self, out_path: Path,
                 w: int, h: int,
                 fps: float,
                 crf: int = 18,
                 preset: str = "veryfast"):
        self.out_path = Path(out_path)
        self.w, self.h = int(w), int(h)
        self.fps = float(fps) if fps and fps > 0 else 20.0
        self.proc: Optional[subprocess.Popen] = None

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        # Используем только libx264 для совместимости
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.w}x{self.h}",
            "-r", f"{self.fps:.2f}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.out_path),
        ]

        print(f"[INFO] Starting FFmpeg: {self.out_path}")
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            print(f"[ERROR] Failed to start FFmpeg: {e}")
            sys.exit(1)

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            return

        # Проверяем, жив ли процесс
        if self.proc.poll() is not None:
            stderr = self.proc.stderr.read().decode('utf-8', errors='ignore') if self.proc.stderr else ""
            print(f"[ERROR] FFmpeg died! Return code: {self.proc.returncode}")
            if stderr:
                print(f"[FFmpeg stderr]:\n{stderr}")
            sys.exit(1)

        try:
            self.proc.stdin.write(frame.data if frame.flags['C_CONTIGUOUS'] else frame.tobytes())
        except BrokenPipeError:
            stderr = self.proc.stderr.read().decode('utf-8', errors='ignore') if self.proc.stderr else ""
            print(f"[ERROR] FFmpeg pipe broken!")
            if stderr:
                print(f"[FFmpeg stderr]:\n{stderr}")
            sys.exit(1)

    def close(self) -> None:
        if self.proc is None:
            return

        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()

            # Ждем завершения
            self.proc.wait(timeout=30)

            if self.proc.returncode != 0:
                stderr = self.proc.stderr.read().decode('utf-8', errors='ignore') if self.proc.stderr else ""
                print(f"[WARNING] FFmpeg finished with code {self.proc.returncode}")
                if stderr:
                    print(f"[FFmpeg stderr]:\n{stderr}")
        except subprocess.TimeoutExpired:
            print("[WARNING] FFmpeg timeout, killing...")
            self.proc.kill()
            self.proc.wait()
        except Exception as e:
            print(f"[ERROR] Error closing FFmpeg: {e}")


# ----------------- simple temporal "track" state -----------------

@dataclass
class TrackedBox:
    x1: int
    y1: int
    x2: int
    y2: int
    ttl: int  # frames left to keep


def update_memory(memory: List[TrackedBox],
                  new_boxes: List[Tuple[int, int, int, int]],
                  hold: int,
                  smooth: int) -> List[TrackedBox]:
    """
    Простая схема:
    - декремент ttl у существующих боксов
    - новые боксы матчим по IoU, обновляем координаты (сглаживание)
    """
    iou_thr = 0.3
    alpha = 1.0 / max(1, float(smooth))  # для EMA

    # step 1: decay
    memory = [b for b in memory if b.ttl > 1]
    for b in memory:
        b.ttl -= 1

    # step 2: match & update
    for (x1, y1, x2, y2) in new_boxes:
        if not memory:
            memory.append(TrackedBox(x1, y1, x2, y2, ttl=hold))
            continue

        best_idx = -1
        best_iou = 0.0
        for idx, tb in enumerate(memory):
            i = iou_xyxy(
                (float(tb.x1), float(tb.y1), float(tb.x2), float(tb.y2)),
                (float(x1), float(y1), float(x2), float(y2)),
            )
            if i > best_iou:
                best_iou = i
                best_idx = idx

        if best_iou >= iou_thr and best_idx >= 0:
            tb = memory[best_idx]
            # EMA сглаживание
            tb.x1 = int(round((1 - alpha) * tb.x1 + alpha * x1))
            tb.y1 = int(round((1 - alpha) * tb.y1 + alpha * y1))
            tb.x2 = int(round((1 - alpha) * tb.x2 + alpha * x2))
            tb.y2 = int(round((1 - alpha) * tb.y2 + alpha * y2))
            tb.ttl = hold
        else:
            memory.append(TrackedBox(x1, y1, x2, y2, ttl=hold))

    return memory


# ----------------- main -----------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in-video", default=str(Path("~/traffic_data/raw/tiandy_000.mkv").expanduser()))
    ap.add_argument("--out-video", default=str(Path("~/traffic_data/anonymized/tiandy_000_plates_v3_2_anon.mp4").expanduser()))
    ap.add_argument("--model", default=str(Path("~/traffic_data/models/plates/plate_yolo.pt").expanduser()))

    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--mode", choices=["anonymize", "boxes"], default="anonymize")

    tg = ap.add_mutually_exclusive_group()
    tg.add_argument("--track", dest="track", action="store_true")
    tg.add_argument("--no-track", dest="track", action="store_false")
    ap.set_defaults(track=True)

    ap.add_argument("--tracker", default="bytetrack.yaml", help="(зарезервировано, в этой версии не используется)")
    ap.add_argument("--vid-stride", type=int, default=1)
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--smooth", type=int, default=3)

    ap.add_argument("--permit-roi-rel", default=None,
                    help="OPTIONAL: ONLY keep boxes whose CENTER is inside ROI x1,y1,x2,y2 in [0..1]")

    ap.add_argument("--max-area-rel", type=float, default=0.03,
                    help="max relative area of plate bbox vs frame")

    ap.add_argument("--min-w", type=int, default=9)
    ap.add_argument("--min-h", type=int, default=5)
    ap.add_argument("--max-w", type=int, default=256)
    ap.add_argument("--max-h", type=int, default=128)
    ap.add_argument("--aspect-min", type=float, default=1.5)
    ap.add_argument("--aspect-max", type=float, default=8.0)

    # --- параметры для наклоненных объектов ---
    ap.add_argument("--allow-tilted", type=lambda x: str(x).lower() == 'true', default=True,
                    help="разрешить детекцию сильно наклоненных объектов")
    ap.add_argument("--max-tilt-angle", type=float, default=75.0,
                    help="максимальный угол наклона объекта в градусах")
    ap.add_argument("--relaxed-aspect-for-tilted", type=lambda x: str(x).lower() == 'true', default=True,
                    help="использовать более мягкие ограничения аспекта для наклоненных объектов")

    # --- tiled inference (SAHI-like) ---
    ap.add_argument("--tiled", type=lambda x: str(x).lower() == 'true', default=True,
                    help="enable tiled/sliced inference for small objects")
    ap.add_argument("--tile-size", type=int,
                    default=960, # Из оптимальных можно еще выставить 640 (медленнее, но детальнее) или 1280/1536 (быстрее)
                    help="tile size in pixels for sliced inference")
    ap.add_argument("--tile-overlap", type=float,
                    default=0.15, # При увеличении размера кадра, лучше снизить до 0.10
                    help="overlap ratio between tiles [0..1)")
    ap.add_argument("--tile-merge-iou", type=float, default=0.5,
                    help="NMS IoU threshold to merge tile detections")
    ap.add_argument("--batched", type=lambda x: str(x).lower() == 'true', default=True,
                    help="enable batched inference for tiled inference")
    ap.add_argument("--tile-batch", type=int, default=8,
                    help="batch size for tiled inference")

    ap.add_argument("--expand-w", type=float, default=1.4)
    ap.add_argument("--expand-h", type=float, default=1.8)

    ap.add_argument("--method", choices=["mosaic", "blur"], default="blur")
    ap.add_argument("--mosaic", type=int, default=18)
    ap.add_argument("--blur-k", type=int, default=31)

    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="veryfast")

    ap.add_argument("--print-every", type=int, default=100)

    args = ap.parse_args()

    in_path = Path(args.in_video).expanduser()
    out_path = Path(args.out_video).expanduser()

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {in_path}", file=sys.stderr)
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    roi_abs: Optional[Tuple[int, int, int, int]] = None
    if args.permit_roi_rel:
        rel = parse_roi_rel(args.permit_roi_rel)
        roi_abs = to_abs_roi(rel, W, H)
        print(f"[INFO] permit-roi-rel -> abs ROI: {roi_abs}", flush=True)

    writer = FFmpegWriter(out_path, W, H, fps=fps, crf=args.crf, preset=args.preset)

    model = YOLO(str(Path(args.model).expanduser()))

    memory: List[TrackedBox] = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        run_det = (frame_idx % max(1, args.vid_stride) == 0)

        new_boxes: List[Tuple[int, int, int, int]] = []

        if run_det:
            if args.tiled:
                tiled_method = tiled_batch_detect if args.batched else tiled_detect
                # SAHI-like tiled inference — лучше для мелких объектов
                boxes_xyxy = tiled_method(
                    model, frame,
                    imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                    device=args.device,
                    tile_size=args.tile_size,
                    overlap=args.tile_overlap,
                    merge_iou=args.tile_merge_iou,
                )
            else:
                # стандартный полнокадровый инференс
                res = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False
                )[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                else:
                    boxes_xyxy = np.empty((0, 4), dtype=np.float32)

            for (x1f, y1f, x2f, y2f) in boxes_xyxy:
                x1, y1, x2, y2 = clamp_xyxy(x1f, y1f, x2f, y2f, W, H)
                w = x2 - x1
                h = y2 - y1
                if w < args.min_w or h < args.min_h:
                    continue
                if w > args.max_w or h > args.max_h:
                    continue
                area = float(w * h) / float(W * H + 1e-6)
                if area > args.max_area_rel:
                    continue
                # Проверка угла наклона и аспекта с учетом наклона
                if args.allow_tilted:
                    angle = compute_rotation_angle(x1, y1, x2, y2)

                    # Если объект сильно наклонен
                    if angle > args.max_tilt_angle:
                        continue

                    # Для наклоненных объектов используем адаптивные критерии
                    if args.relaxed_aspect_for_tilted and angle > 45:
                        # Смягчаем критерии аспекта для наклоненных объектов
                        asp = compute_aspect_ratio_rotated(x1, y1, x2, y2)
                        # Расширяем диапазон допустимых аспектов
                        relaxed_min = args.aspect_min * 0.5
                        relaxed_max = args.aspect_max * 1.5
                        if asp < relaxed_min or asp > relaxed_max:
                            continue
                    else:
                        asp = float(w) / float(h + 1e-6)
                        if asp < args.aspect_min or asp > args.aspect_max:
                            continue
                else:
                    asp = float(w) / float(h + 1e-6)
                    if asp < args.aspect_min or asp > args.aspect_max:
                        continue
                if roi_abs is not None and not center_in_roi(x1, y1, x2, y2, roi_abs):
                    continue
                new_boxes.append((x1, y1, x2, y2))

        # обновляем память (hold-схема)
        memory = update_memory(memory, new_boxes, hold=args.hold, smooth=args.smooth)

        # активные боксы на этот кадр
        active_boxes = [(b.x1, b.y1, b.x2, b.y2) for b in memory]

        if not active_boxes:
            writer.write(frame)
        else:
            out = frame.copy()
            if args.mode == "boxes":
                for (x1, y1, x2, y2) in active_boxes:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            else:
                for (x1, y1, x2, y2) in active_boxes:
                    ex1, ey1, ex2, ey2 = expand_xyxy(
                        x1, y1, x2, y2, W, H,
                        expand_w=args.expand_w,
                        expand_h=args.expand_h
                    )
                    if args.method == "blur":
                        apply_blur(out, ex1, ey1, ex2, ey2, k=args.blur_k)
                    else:
                        apply_mosaic(out, ex1, ey1, ex2, ey2, mosaic=args.mosaic)
            writer.write(out)

        if args.print_every > 0 and (frame_idx % args.print_every == 0):
            dt = time.time() - t0
            print(f"[v3.2 father] frame={frame_idx} active_boxes={len(active_boxes)} dt={dt:.1f}s", flush=True)

    cap.release()
    writer.close()
    dt = time.time() - t0
    print(f"[DONE] Saved: {out_path} (frames={frame_idx}, time={dt:.1f}s)")


if __name__ == "__main__":
    main()

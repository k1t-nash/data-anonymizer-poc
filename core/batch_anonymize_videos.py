import subprocess
import shutil
import logging
from pathlib import Path
import numpy as np
from core.plate_detector import PlateDetector
from core.vehicle_detector import VehicleDetector
from core.face_detector import FaceDetector
from core.video_probe import get_video_meta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("anonymize.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

INPUT_DIR   = Path("recording/incoming")
OUTPUT_BASE = Path("dataset_v1/production_packages")
BITRATE     = "50M"
CQ_LEVEL    = "19"


def anonymize_roi(roi: np.ndarray) -> np.ndarray:
    """
    Pixelate вместо GaussianBlur.
    Необратимо даже после 4K upscale — нельзя восстановить через деконволюцию.
    """
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi
    block = max(8, min(h, w) // 4)
    small = np.ascontiguousarray(
        roi[::block, ::block]  # быстрее чем resize для pixelate
    )
    # Растягиваем обратно через numpy repeat — нет зависимости от cv2 интерполяции
    result = np.repeat(np.repeat(small, block, axis=0), block, axis=1)
    return result[:h, :w]


def process_video(
    video_path: Path,
    v_det: VehicleDetector,
    p_det: PlateDetector,
    f_det: FaceDetector,
) -> bool:
    package_path = OUTPUT_BASE / f"Package_{video_path.stem}"
    if package_path.exists():
        shutil.rmtree(package_path)
    package_path.mkdir(parents=True)
    video_out = package_path / f"{video_path.stem}_anonymized.mp4"
    ffmpeg_log_path = package_path / "ffmpeg.log"

    # Получаем метаданные через ffprobe (точнее чем cv2)
    try:
        meta = get_video_meta(video_path)
        w, h, fps = meta.width, meta.height, meta.fps
        total_f = int(meta.duration_sec * fps)
    except Exception as e:
        log.error(f"Не удалось получить метаданные {video_path.name}: {e}")
        return False

    if w == 0 or h == 0 or fps == 0:
        log.error(f"Некорректные метаданные: {video_path.name}")
        return False

    log.info(f"Старт: {video_path.name} | {w}x{h} @ {fps:.2f}fps | ~{total_f} кадров")

    cmd_in = [
        "ffmpeg", "-hwaccel", "cuda",
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]

    cmd_out = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", CQ_LEVEL,
        "-b:v", BITRATE,
        "-maxrate", "80M",
        "-bufsize", "80M",
        "-pix_fmt", "yuv420p",
        str(video_out)
    ]

    frame_bytes = w * h * 3
    ffmpeg_log = open(ffmpeg_log_path, "w")
    proc_in  = subprocess.Popen(cmd_in,  stdout=subprocess.PIPE, stderr=ffmpeg_log)
    proc_out = subprocess.Popen(cmd_out, stdin=subprocess.PIPE,  stderr=ffmpeg_log)

    frame_idx = 0

    try:
        while True:
            raw = proc_in.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()

            # Пайплайн: машины → номера → лица
            v_boxes = v_det.detect(frame)
            plates  = p_det.detect_in_frame(frame, v_boxes)
            faces   = f_det.detect_in_frame(frame)

            for b in list(plates) + list(faces):
                x1, y1, x2, y2 = map(int, b[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    frame[y1:y2, x1:x2] = anonymize_roi(frame[y1:y2, x1:x2])

            proc_out.stdin.write(frame.tobytes())
            frame_idx += 1

            if frame_idx % 20 == 0:
                pct = frame_idx / total_f * 100 if total_f > 0 else 0
                print(f"\r[{video_path.name}] {frame_idx}/{total_f} ({pct:.1f}%)", end="", flush=True)

    except Exception as e:
        log.error(f"Ошибка при обработке {video_path.name}: {e}")
        return False

    finally:
        proc_in.stdout.close()
        proc_out.stdin.close()
        proc_in.wait()
        ret = proc_out.wait()
        ffmpeg_log.close()
        print()

        if ret != 0:
            log.error(f"ffmpeg encoder упал с кодом {ret}. См. {ffmpeg_log_path}")
            return False

    log.info(f"Готово: {video_path.name} → {video_out}")
    return True


def main():
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        log.warning(f"Нет .mp4 файлов в {INPUT_DIR}")
        return

    log.info(f"Найдено {len(videos)} видео. Загружаем модели...")

    # Модели загружаются ОДИН РАЗ для всех видео
    v_det = VehicleDetector()
    p_det = PlateDetector(model_path="models/plates/license_plate_detector.pt")
    f_det = FaceDetector(model_path="models/faces/yolov8n-face.pt")

    log.info("Модели загружены. Старт обработки.")

    ok, fail = 0, 0
    for vid in videos:
        if process_video(vid, v_det, p_det, f_det):
            ok += 1
        else:
            fail += 1

    log.info(f"Завершено. Успешно: {ok} | Ошибки: {fail}")


if __name__ == "__main__":
    main()

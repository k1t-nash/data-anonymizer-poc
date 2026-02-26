import subprocess
import sys
import logging
from pathlib import Path
from typing import Generator, Tuple
import numpy as np

log = logging.getLogger(__name__)

# RGB24 — 3 байта на пиксель
BYTES_PER_PIXEL = 3


def frame_stream(
    video: Path,
    width: int,
    height: int,
    use_cuda: bool = True,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Генератор кадров из видеофайла через ffmpeg.

    Yields: (frame_index, frame_np)
        frame_np — np.ndarray shape (height, width, 3), dtype uint8, RGB.

    Параметры:
        video     — путь к видеофайлу
        width     — ширина кадра в пикселях
        height    — высота кадра в пикселях
        use_cuda  — использовать CUDA декодирование (быстрее на GPU)
    """
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"Видеофайл не найден: {video}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Некорректные размеры: {width}x{height}")

    frame_size = width * height * BYTES_PER_PIXEL

    cmd = ["ffmpeg", "-loglevel", "error"]

    if use_cuda:
        cmd += ["-hwaccel", "cuda"]

    cmd += [
        "-i", str(video),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]

    # stderr в PIPE чтобы видеть ошибки ffmpeg, не глотать их
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    frame_idx = 0

    try:
        while True:
            raw = proc.stdout.read(frame_size)

            # Конец потока или неполный кадр
            if len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            yield frame_idx, frame
            frame_idx += 1

    except Exception as e:
        log.error(f"Ошибка чтения кадра {frame_idx} из {video.name}: {e}")
        raise

    finally:
        proc.stdout.close()

        # Читаем stderr перед wait() — иначе deadlock если буфер заполнится
        stderr_output = proc.stderr.read().decode("utf-8", errors="replace").strip()
        proc.stderr.close()
        ret = proc.wait()

        if ret != 0 and stderr_output:
            log.warning(f"ffmpeg завершился с кодом {ret}: {stderr_output}")

        log.debug(f"{video.name}: прочитано {frame_idx} кадров")


def main():
    if len(sys.argv) < 4:
        print("Usage: python core/frame_reader.py <video> <width> <height> [max_frames]")
        sys.exit(1)

    video   = Path(sys.argv[1])
    width   = int(sys.argv[2])
    height  = int(sys.argv[3])
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    if not video.exists():
        print(f"ERROR: файл не найден: {video}")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    count = 0

    for idx, frame in frame_stream(video, width, height):
        count += 1
        if idx % 100 == 0:
            print(f"Frame {idx} | shape={frame.shape} | dtype={frame.dtype}")
        if idx >= max_frames - 1:
            break

    print(f"Всего прочитано кадров: {count}")
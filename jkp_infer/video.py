from __future__ import annotations

from pathlib import Path
import tempfile


def probe_video_metadata(video_path: str | Path) -> dict[str, float | int | None]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is required for video metadata probing. Activate .venv-vlm first."
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration_sec = (frame_count / source_fps) if source_fps > 0 and frame_count > 0 else None
    return {
        "source_fps": source_fps if source_fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
        "duration_sec": duration_sec,
        "file_size_bytes": video_path.stat().st_size,
    }


def sample_video_frames(
    video_path: str | Path,
    *,
    num_frames: int | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is required for frame sampling. Activate .venv-vlm first."
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if output_dir is None:
        frame_dir = Path(tempfile.mkdtemp(prefix="jkp_infer_frames_"))
    else:
        frame_dir = Path(output_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Unable to read frames from {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0

    if fps is not None:
        if fps <= 0:
            capture.release()
            raise ValueError("fps must be > 0 when provided.")
        step_frames = max(1, int(round(source_fps / fps)))
        indices = list(range(0, frame_count, step_frames))
        if max_frames is not None:
            if max_frames <= 0:
                capture.release()
                raise ValueError("max_frames must be > 0 when provided.")
            indices = indices[:max_frames]
        if not indices:
            indices = [0]
    else:
        if num_frames is None:
            capture.release()
            raise ValueError("Either num_frames or fps must be provided.")
        if num_frames <= 1:
            indices = [0]
        else:
            indices = sorted({round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)})

    frame_paths: list[Path] = []
    for position, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        out_path = frame_dir / f"{video_path.stem}_frame_{position:02d}.jpg"
        cv2.imwrite(str(out_path), frame)
        frame_paths.append(out_path)
    capture.release()

    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frame_paths


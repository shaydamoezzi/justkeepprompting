from __future__ import annotations

from pathlib import Path
import tempfile


def sample_video_frames(
    video_path: str | Path,
    *,
    num_frames: int,
    output_dir: str | Path | None = None,
) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is required for sampled-frame mode. "
            "Run scripts/setup_vlm_env.sh first."
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if output_dir is None:
        frame_dir = Path(tempfile.mkdtemp(prefix="jkp_frames_"))
    else:
        frame_dir = Path(output_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Unable to read frames from {video_path}")

    if num_frames <= 1:
        indices = [0]
    else:
        indices = sorted({round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)})

    output_paths: list[Path] = []
    for pos, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        out_path = frame_dir / f"{video_path.stem}_frame_{pos:02d}.jpg"
        cv2.imwrite(str(out_path), frame)
        output_paths.append(out_path)
    capture.release()

    if not output_paths:
        raise RuntimeError(f"No frames were extracted from {video_path}")
    return output_paths

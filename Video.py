"""Video helperji za RV 9HPT izziv."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, Union

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


class VideoReader:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Ne morem odpreti videa: {self.path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps and fps > 0 else 30.0

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def info(self) -> VideoInfo:
        duration = self.frame_count / self.fps if self.frame_count > 0 else 0.0

        return VideoInfo(
            path=self.path,
            fps=self.fps,
            frame_count=self.frame_count,
            width=self.width,
            height=self.height,
            duration_s=duration,
        )

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        frame_idx = 0

        while True:
            ok, frame = self.cap.read()

            if not ok:
                break

            t = frame_idx / self.fps
            yield frame_idx, t, frame

            frame_idx += 1

    def release(self) -> None:
        self.cap.release()


class VideoWriter:
    def __init__(
        self,
        path: Union[str, Path],
        fps: float,
        width: int,
        height: int,
    ) -> None:
        self.path = str(path)

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            self.path,
            fourcc,
            float(fps),
            (int(width), int(height)),
        )

        if not self.writer.isOpened():
            raise OSError(f"Ne morem ustvariti izhodnega videa: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
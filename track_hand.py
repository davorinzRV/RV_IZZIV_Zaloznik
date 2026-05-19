"""Hand tracking for the RV nine-hole peg test challenge.

Tracker: MediaPipe Hands, 21 landmarks.

Tracked point:
- midpoint between thumb tip and index finger tip.

Coordinates are in image pixels in the original input frame.
If MediaPipe temporarily loses the hand, short gaps are filled
with the last known pinch position.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackResult:
    frame: int
    t: float
    visible: int
    x: float
    y: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    area: float
    score: float
    thumb_x: float
    thumb_y: float
    index_x: float
    index_y: float
    method: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class HandTracker:
    """Tracks midpoint between thumb tip and index finger tip."""

    def __init__(
        self,
        use_mediapipe: bool = True,
        min_detection_confidence: float = 0.20,
        min_tracking_confidence: float = 0.20,
        fallback_min_area: int = 1200,
        max_jump_px: float = 250.0,
        max_missing_fill: int = 8,
    ) -> None:
        # Parametra fallback_min_area in max_jump_px ostaneta zaradi kompatibilnosti,
        # ampak OpenCV fallbacka in zavračanja skokov ne uporabljamo več.
        self.fallback_min_area = fallback_min_area
        self.max_jump_px = max_jump_px

        # Koliko zaporednih manjkajočih frame-ov lahko zapolnimo z zadnjo znano točko.
        self.max_missing_fill = max_missing_fill
        self.missing_count = 0

        self.prev_center: Optional[Tuple[float, float]] = None
        self.last_result: Optional[TrackResult] = None

        self._mp_hands = None
        self._hands = None

        if use_mediapipe:
            try:
                import mediapipe as mp

                self._mp_hands = mp.solutions.hands
                self._hands = self._mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            except Exception:
                self._mp_hands = None
                self._hands = None

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()

    def process(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> TrackResult:
        result = self._track_mediapipe(frame_bgr, frame_idx, t)

        if result is None:
            self.missing_count += 1

            # Če MediaPipe za kratek čas izgubi roko, uporabimo zadnjo znano
            # pozicijo prijema. To izboljša zveznost trajektorije.
            if self.last_result is not None and self.missing_count <= self.max_missing_fill:
                predicted = TrackResult(
                    frame=frame_idx,
                    t=t,
                    visible=1,
                    x=self.last_result.x,
                    y=self.last_result.y,
                    bbox_x1=self.last_result.bbox_x1,
                    bbox_y1=self.last_result.bbox_y1,
                    bbox_x2=self.last_result.bbox_x2,
                    bbox_y2=self.last_result.bbox_y2,
                    area=self.last_result.area,
                    score=0.10,
                    thumb_x=self.last_result.thumb_x,
                    thumb_y=self.last_result.thumb_y,
                    index_x=self.last_result.index_x,
                    index_y=self.last_result.index_y,
                    method="predicted_pinch",
                )

                self.prev_center = (predicted.x, predicted.y)
                return predicted

            return self._empty_result(frame_idx, t, method="none")

        self.missing_count = 0

        if result.visible:
            self.prev_center = (result.x, result.y)
            self.last_result = result

        return result

    def _empty_result(self, frame_idx: int, t: float, method: str) -> TrackResult:
        nan = float("nan")

        return TrackResult(
            frame=frame_idx,
            t=t,
            visible=0,
            x=nan,
            y=nan,
            bbox_x1=nan,
            bbox_y1=nan,
            bbox_x2=nan,
            bbox_y2=nan,
            area=0.0,
            score=0.0,
            thumb_x=nan,
            thumb_y=nan,
            index_x=nan,
            index_y=nan,
            method=method,
        )

    def _track_mediapipe(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        t: float,
    ) -> Optional[TrackResult]:
        if self._hands is None:
            return None

        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        # MediaPipe landmark 4 = konica palca
        # MediaPipe landmark 8 = konica kazalca
        thumb_tip = pts[4]
        index_tip = pts[8]

        # Rdeča točka in glavna kinematika se računata iz sredine med palcem in kazalcem.
        centre = (thumb_tip + index_tip) / 2.0

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        area = float(max(0.0, (x2 - x1) * (y2 - y1)))

        score = 1.0
        if results.multi_handedness:
            score = float(results.multi_handedness[0].classification[0].score)

        return TrackResult(
            frame=frame_idx,
            t=t,
            visible=1,
            x=float(centre[0]),
            y=float(centre[1]),
            bbox_x1=float(x1),
            bbox_y1=float(y1),
            bbox_x2=float(x2),
            bbox_y2=float(y2),
            area=area,
            score=score,
            thumb_x=float(pts[4, 0]),
            thumb_y=float(pts[4, 1]),
            index_x=float(pts[8, 0]),
            index_y=float(pts[8, 1]),
            method="mediapipe_pinch",
        )


def draw_track(
    frame_bgr: np.ndarray,
    row: Dict[str, object],
    trail: Optional[Iterable[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Draw tracking overlay on a frame."""
    out = frame_bgr.copy()

    visible = int(row.get("visible", 0)) == 1
    method = str(row.get("method", ""))

    if visible:
        x = float(row["x"])
        y = float(row["y"])

        x1 = float(row["bbox_x1"])
        y1 = float(row["bbox_y1"])
        x2 = float(row["bbox_x2"])
        y2 = float(row["bbox_y2"])

        # Zelen okvir okoli roke.
        cv2.rectangle(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

        # Rdeča pika = sredina med palcem in kazalcem.
        cv2.circle(
            out,
            (int(x), int(y)),
            7,
            (0, 0, 255),
            -1,
        )

        tx = row.get("thumb_x", np.nan)
        ty = row.get("thumb_y", np.nan)
        ix = row.get("index_x", np.nan)
        iy = row.get("index_y", np.nan)

        # Modra pika = palec.
        if np.isfinite(tx) and np.isfinite(ty):
            cv2.circle(
                out,
                (int(float(tx)), int(float(ty))),
                5,
                (255, 0, 0),
                -1,
            )
            cv2.putText(
                out,
                "T",
                (int(float(tx)) + 5, int(float(ty)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )

        # Oranžna pika = kazalec.
        if np.isfinite(ix) and np.isfinite(iy):
            cv2.circle(
                out,
                (int(float(ix)), int(float(iy))),
                5,
                (0, 165, 255),
                -1,
            )
            cv2.putText(
                out,
                "I",
                (int(float(ix)) + 5, int(float(iy)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )

        # Če je točka napovedana in ne direktno zaznana, dodamo majhen napis.
        if method == "predicted_pinch":
            cv2.putText(
                out,
                "predicted",
                (int(x) + 10, int(y) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    if trail is not None:
        pts = [
            (int(px), int(py))
            for px, py in trail
            if np.isfinite(px) and np.isfinite(py)
        ]

        for p1, p2 in zip(pts[:-1], pts[1:]):
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

            # Ne rišemo zelo dolgih skokov v prikazu.
            if dist < 150:
                cv2.line(
                    out,
                    p1,
                    p2,
                    (0, 255, 255),
                    2,
                )

    cv2.putText(
        out,
        f"frame {row.get('frame')} | {method}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        3,
    )

    cv2.putText(
        out,
        f"frame {row.get('frame')} | {method}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        1,
    )

    return out
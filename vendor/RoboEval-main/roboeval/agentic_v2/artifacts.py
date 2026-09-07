"""Small artifact helpers shared by single trials and matrix evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from roboeval.agentic_v2.types import to_jsonable


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(path.resolve())


class FrameRecorder:
    """Record sparse simulator frames without influencing control decisions."""

    def __init__(
        self,
        env: Any,
        directory: Path,
        *,
        enabled: bool,
        every: int = 8,
    ) -> None:
        self.env = env
        self.directory = directory
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self.seen = 0
        self.paths: list[Path] = []

    def callback(self, step: int, observation: Any) -> None:
        self.seen += 1
        if self.enabled and self.seen % self.every == 0:
            self._save(observation, f"motion_{self.seen:06d}")

    def capture(self, label: str) -> str | None:
        if not self.enabled:
            return None
        return self._save(self.env.get_observation(), label)

    def _save(self, observation: Any, label: str) -> str | None:
        frame = _rgb_frame(observation)
        if frame is None:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{len(self.paths):05d}_{_safe_label(label)}.png"
        imageio.imwrite(path, frame)
        self.paths.append(path)
        return str(path.resolve())

    def write_gif(self, path: Path, *, duration: float = 0.10) -> str | None:
        if not self.paths:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(
            path,
            [imageio.imread(frame) for frame in self.paths],
            duration=max(0.02, float(duration)),
        )
        return str(path.resolve())


def _rgb_frame(observation: Any) -> np.ndarray | None:
    if not isinstance(observation, dict):
        return None
    value = observation.get("rgb_external")
    if value is None:
        return None
    frame = np.asarray(value)
    if frame.ndim != 3:
        return None
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.dtype != np.uint8:
        maximum = float(np.nanmax(frame)) if frame.size else 0.0
        scale = 255.0 if maximum <= 1.0 else 1.0
        frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
    return frame


def _safe_label(value: str) -> str:
    result = "".join(char if char.isalnum() else "_" for char in value.lower())
    return result.strip("_") or "frame"


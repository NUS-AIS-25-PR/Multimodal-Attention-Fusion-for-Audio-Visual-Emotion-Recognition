"""Synthetic decodable paired media; deliberately incomplete and never paper data."""
from pathlib import Path
import wave
import cv2
import numpy as np


def create_smoke_media(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for actor in range(1, 25):
        emotion = (actor - 1) % 8 + 1
        suffix = f"01-{emotion:02d}-01-01-01-{actor:02d}"
        with wave.open(str(root / f"03-{suffix}.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            signal = (1000 * np.sin(np.arange(4800) * (actor + 200) / 16000 * 2 * np.pi)).astype('<i2')
            stream.writeframes(signal.tobytes())
        writer = cv2.VideoWriter(str(root / f"02-{suffix}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 8, (32, 32))
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v encoder unavailable")
        for i in range(4):
            writer.write(np.full((32, 32, 3), actor * 8 + i, dtype=np.uint8))
        writer.release()

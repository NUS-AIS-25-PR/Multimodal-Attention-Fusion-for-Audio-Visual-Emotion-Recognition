"""WebSocket streaming session management for chunk-level inference."""

from __future__ import annotations

import base64
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import SAMPLE_RATE, STREAM_MAX_BUFFER_SECONDS, STREAM_STEP_SECONDS, STREAM_WINDOW_SECONDS
from .infer import EmotionPredictor


def decode_frame_b64(image_b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG frame into a BGR image."""
    encoded = image_b64.split(",", 1)[-1]
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode frame payload.")
    return frame


def decode_pcm16_b64(pcm_b64: str) -> np.ndarray:
    """Decode base64-encoded int16 PCM to float32 waveform in [-1, 1]."""
    raw = base64.b64decode(pcm_b64)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float32)
    return pcm / 32768.0


@dataclass
class StreamingEmotionSession:
    """Mutable streaming buffers and inference cadence for one WebSocket client."""

    predictor: EmotionPredictor
    window_seconds: float = STREAM_WINDOW_SECONDS
    step_seconds: float = STREAM_STEP_SECONDS
    max_buffer_seconds: float = STREAM_MAX_BUFFER_SECONDS
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    use_face_crop: bool = True
    waveform_sample_rate: int = SAMPLE_RATE
    frames: Deque[Tuple[float, np.ndarray]] = field(default_factory=deque)
    audio_chunks: Deque[np.ndarray] = field(default_factory=deque)
    audio_sample_count: int = 0
    last_prediction_ts: float = 0.0

    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        now = float(timestamp if timestamp is not None else time.time())
        self.frames.append((now, frame))
        self._prune_frames(now)

    def add_audio_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int,
        timestamp: Optional[float] = None,
    ) -> None:
        _ = timestamp
        self.waveform_sample_rate = int(sample_rate)
        self.audio_chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        self.audio_sample_count += int(chunk.size)
        self._prune_audio()

    def _prune_frames(self, now: float) -> None:
        cutoff = now - float(self.max_buffer_seconds)
        while self.frames and self.frames[0][0] < cutoff:
            self.frames.popleft()

    def _prune_audio(self) -> None:
        max_samples = max(1, int(self.waveform_sample_rate * self.max_buffer_seconds))
        while self.audio_sample_count > max_samples and self.audio_chunks:
            dropped = self.audio_chunks.popleft()
            self.audio_sample_count -= int(dropped.size)

    def ready_for_inference(self, now: Optional[float] = None) -> bool:
        now = float(now if now is not None else time.monotonic())
        enough_audio = self.audio_sample_count >= int(self.waveform_sample_rate * self.window_seconds)
        enough_frames = len(self.frames) >= 2
        cadence_ok = (now - self.last_prediction_ts) >= self.step_seconds
        return enough_audio and enough_frames and cadence_ok

    def build_window(self, now: Optional[float] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        now = float(now if now is not None else time.time())
        frame_cutoff = now - float(self.window_seconds)
        window_frames = [frame for ts, frame in self.frames if ts >= frame_cutoff]
        if not window_frames:
            window_frames = [frame for _, frame in self.frames]

        waveform = np.concatenate(list(self.audio_chunks), axis=0) if self.audio_chunks else np.zeros(0, dtype=np.float32)
        target_samples = max(1, int(self.waveform_sample_rate * self.window_seconds))
        if waveform.size > target_samples:
            waveform = waveform[-target_samples:]
        return window_frames, waveform

    def infer(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = float(now if now is not None else time.monotonic())
        frames, waveform = self.build_window()  # uses time.time() internally, not monotonic
        result = self.predictor.predict_stream(
            frames,
            waveform,
            waveform_sample_rate=self.waveform_sample_rate,
            use_face_crop=self.use_face_crop,
        )
        self.last_prediction_ts = now
        result["session_id"] = self.session_id
        result["window_seconds"] = self.window_seconds
        result["num_buffered_frames"] = len(frames)
        result["num_audio_samples"] = int(waveform.size)
        return result


class StreamingSessionManager:
    """Factory for per-client streaming sessions."""

    def __init__(self, predictor: EmotionPredictor) -> None:
        self.predictor = predictor
        self.sessions: Dict[str, StreamingEmotionSession] = {}

    def create_session(self, use_face_crop: bool = True) -> StreamingEmotionSession:
        session = StreamingEmotionSession(
            predictor=self.predictor,
            use_face_crop=use_face_crop,
        )
        self.sessions[session.session_id] = session
        return session

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

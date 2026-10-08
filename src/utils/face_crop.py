"""Face detection and cropping utilities using MediaPipe."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional

# Try modern API first, fallback to legacy
MEDIAPIPE_AVAILABLE = False
_detector_impl = None

try:
    import mediapipe as mp
    
    # Try modern tasks API
    try:
        from mediapipe.tasks import vision
        from mediapipe import core as media_core
        _detector_impl = "modern"
        MEDIAPIPE_AVAILABLE = True
    except (ImportError, AttributeError):
        # Fallback to legacy solutions API
        try:
            from mediapipe.python.solutions import face_detection
            _detector_impl = "legacy"
            MEDIAPIPE_AVAILABLE = True
        except ImportError:
            try:
                # Alternative path for legacy
                import mediapipe.solutions.face_detection as face_detection
                _detector_impl = "legacy"
                MEDIAPIPE_AVAILABLE = True
            except ImportError:
                MEDIAPIPE_AVAILABLE = False
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class MediaPipeFaceDetector:
    """Lightweight face detector using MediaPipe (modern API)."""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize MediaPipe face detector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold for detections
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not properly installed")
        
        # Use the appropriate API based on availability
        if _detector_impl == "modern":
            self._init_modern(min_detection_confidence)
        elif _detector_impl == "legacy":
            self._init_legacy(min_detection_confidence)
        else:
            raise RuntimeError("No compatible MediaPipe API found")
    
    def _init_modern(self, min_detection_confidence: float):
        """Initialize modern MediaPipe tasks API."""
        try:
            from mediapipe.tasks import vision
            from mediapipe import core as media_core
            
            base_options = media_core.BaseOptions(model_asset_path=None)
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_detection_confidence,
            )
            self.detector = vision.FaceDetector.create_from_options(options)
            self.api_type = "modern"
        except Exception as e:
            raise RuntimeError(f"Modern MediaPipe API initialization failed: {e}")
    
    def _init_legacy(self, min_detection_confidence: float):
        """Initialize legacy MediaPipe solutions API."""
        try:
            # Try different import paths
            try:
                from mediapipe.python.solutions import face_detection
            except ImportError:
                import mediapipe.solutions.face_detection as face_detection
            
            self.detector = face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=min_detection_confidence
            )
            self.api_type = "legacy"
        except Exception as e:
            raise RuntimeError(f"Legacy MediaPipe API initialization failed: {e}")
    
    def detect_face_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in a frame and return bounding box.
        
        Args:
            frame: BGR image frame (H, W, 3)
        
        Returns:
            Tuple (x1, y1, x2, y2) in pixel coordinates, or None if no face detected
        """
        if not hasattr(self, 'detector'):
            return None
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            if self.api_type == "modern":
                # Modern API
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = self.detector.detect(mp_image)
                
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.bounding_box
                    x1 = max(0, int(bbox.origin_x))
                    y1 = max(0, int(bbox.origin_y))
                    x2 = min(w, int(bbox.origin_x + bbox.width))
                    y2 = min(h, int(bbox.origin_y + bbox.height))
                    return (x1, y1, x2, y2)
            
            elif self.api_type == "legacy":
                # Legacy API
                results = self.detector.process(frame_rgb)
                
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    x1 = max(0, int(bbox.xmin * w))
                    y1 = max(0, int(bbox.ymin * h))
                    x2 = min(w, int((bbox.xmin + bbox.width) * w))
                    y2 = min(h, int((bbox.ymin + bbox.height) * h))
                    return (x1, y1, x2, y2)
        
        except Exception:
            pass
        
        return None
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'detector') and self.detector:
            try:
                self.detector.close()
            except:
                pass


def crop_with_padding(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad_ratio: float = 0.3,
) -> np.ndarray:
    """Crop face region from frame with padding.
    
    Args:
        frame: Input image frame (H, W, 3)
        bbox: Bounding box (x1, y1, x2, y2)
        pad_ratio: Padding ratio relative to bbox size (0.3 = 30% padding on each side)
    
    Returns:
        Cropped frame centered on face with padding
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    # Calculate bbox dimensions
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    # Apply padding (symmetric on all sides)
    pad_x = int(bbox_w * pad_ratio)
    pad_y = int(bbox_h * pad_ratio)
    
    # Clip to image boundaries
    x1_padded = max(0, x1 - pad_x)
    y1_padded = max(0, y1 - pad_y)
    x2_padded = min(w, x2 + pad_x)
    y2_padded = min(h, y2 + pad_y)
    
    cropped = frame[y1_padded:y2_padded, x1_padded:x2_padded]
    return cropped


def get_face_detector() -> Optional[MediaPipeFaceDetector]:
    """Get or initialize a singleton face detector instance.
    
    Returns:
        MediaPipeFaceDetector instance, or None if MediaPipe unavailable
    """
    if not hasattr(get_face_detector, '_instance'):
        try:
            get_face_detector._instance = MediaPipeFaceDetector()
        except Exception as e:
            # Silently fallback - face detection is optional
            get_face_detector._instance = None
    
    return get_face_detector._instance

"""Regressions for the existing legacy MediaPipe face-crop path."""
from types import SimpleNamespace
import unittest

import numpy as np
from mediapipe.framework.formats import detection_pb2, location_data_pb2

from utils.face_crop import MediaPipeFaceDetector, crop_with_padding


class FaceCropTests(unittest.TestCase):
    def detector(self, detections):
        detector = MediaPipeFaceDetector.__new__(MediaPipeFaceDetector)
        detector.api_type = 'legacy'
        detector.detector = SimpleNamespace(process=lambda frame: SimpleNamespace(detections=detections))
        return detector

    def test_relative_legacy_box_yields_nonempty_padded_crop(self):
        detection = detection_pb2.Detection()
        detection.location_data.format = location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX
        detection.location_data.relative_bounding_box.xmin = 0.25
        detection.location_data.relative_bounding_box.ymin = 0.25
        detection.location_data.relative_bounding_box.width = 0.5
        detection.location_data.relative_bounding_box.height = 0.5
        frame = np.zeros((100,200,3), dtype=np.uint8)
        bbox = self.detector([detection]).detect_face_bbox(frame)
        self.assertEqual(bbox, (50,25,150,75))
        self.assertEqual(crop_with_padding(frame,bbox,pad_ratio=0.3).shape, (80,160,3))

    def test_relative_legacy_box_clips_to_frame(self):
        detection = detection_pb2.Detection()
        box = detection.location_data.relative_bounding_box
        box.xmin = -0.25; box.ymin = -0.25; box.width = 1.5; box.height = 1.5
        self.assertEqual(self.detector([detection]).detect_face_bbox(np.zeros((100,200,3),dtype=np.uint8)),
                         (0,0,200,100))

    def test_no_detection_remains_none(self):
        self.assertIsNone(self.detector([]).detect_face_bbox(np.zeros((100,200,3),dtype=np.uint8)))


if __name__ == '__main__':
    unittest.main()

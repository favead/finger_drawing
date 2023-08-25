"""
Wrapper for mediapipe hand detector
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from finger_drawing.utils import get_absolute_path


class MediaPipeHandDetector:
    """
    Hand detection using mediapipe library
    """

    def __init__(self, task_path: str, **kwargs) -> None:
        self.task_path = get_absolute_path(task_path)
        self.detector = self._create_hand_detector()

    def _create_hand_detector(self) -> vision.HandLandmarker:
        """
        Create hand landmark object
        """
        base_options = python.BaseOptions(model_asset_path=self.task_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        return self.detector

    def get_hand_box(self, image: np.ndarray) -> np.ndarray:
        """
        Running the image through the detector
        """
        H, W, _ = image.shape
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mediapipe_image)
        hand_landmarks_list = detection_result.hand_landmarks
        if len(hand_landmarks_list) == 0:
            return np.empty((0, 2))
        x_coordinates = np.array(
            [int(landmark.x * W) for landmark in hand_landmarks_list[0]]
        ).reshape(-1, 1)
        y_coordinates = np.array(
            [
                int(landmark.y * H)
                for landmark in detection_result.hand_landmarks[0]
            ]
        ).reshape(-1, 1)
        coordinates = np.concatenate((x_coordinates, y_coordinates), axis=1)
        return coordinates

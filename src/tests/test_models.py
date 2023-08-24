"""
Unit tests for models
"""
from pathlib import Path

from hydra import initialize, compose
import numpy as np
from finger_drawing.hand_detection_models import build_hand_detection_model
from finger_drawing.hand_detection_models.media_pipe_detector import (
    MediaPipeHandDetector,
)
import cv2 as cv


def test_model_build() -> None:
    """
    Testing the build_model method
    """
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config", overrides=["model=test"])
        builded_model = build_hand_detection_model(cfg)
        media_pipe_detector = MediaPipeHandDetector(cfg["model"]["task_path"])
        assert isinstance(builded_model, MediaPipeHandDetector)
        assert isinstance(media_pipe_detector, MediaPipeHandDetector)
    return None


def test_model_inference() -> None:
    """
    Testing the model detection quality
    """
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=["model=test", "input=test", "output=test"],
        )
        hand_detection_model = build_hand_detection_model(cfg)
        input_image_path = str(Path(cfg["input"]["path"]))
        bgr_input_image = cv.imread(input_image_path)
        rgb_input_image = bgr_input_image[:, :, ::-1]
        landmarks = hand_detection_model.get_hand_box(
            rgb_input_image.astype(np.uint8)
        )
        for landmark in landmarks:
            bgr_input_image = cv.circle(
                bgr_input_image,
                (landmark[0], landmark[1]),
                5,
                color=(255, 255, 255),
            )
        cv.imshow("Test example", bgr_input_image)
        cv.imwrite(cfg["output"]["path"], bgr_input_image)
        assert landmarks.shape[0] == 21
        assert landmarks.shape[1] == 2
        return None

"""
Application entry point
"""
import cv2 as cv
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from finger_drawing.hand_detection_models import build_hand_detection_model

from finger_drawing.hand_detection_models.media_pipe_detector import (
    MediaPipeHandDetector,
)
from finger_drawing.video_utils import VideoHandler


class App:
    """
    Finger drawing application
    """

    def __init__(
        self, video_handler: VideoHandler, hand_detector: MediaPipeHandDetector
    ) -> None:
        self.video_handler = video_handler
        self.hand_detector = hand_detector
        self.W, self.H = self.video_handler.get_frame_shape()

    def run(self) -> None:
        """
        Run webcam finger detection
        """
        draw_frame = np.zeros((self.H, self.W)).astype(np.uint8)
        for frame in self.video_handler.get_frame():
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            draw_frame = self._handle_frame(rgb_frame, draw_frame)
            self.video_handler.write_frame(draw_frame, color=None)
        return None

    def _handle_frame(
        self, frame: np.ndarray, draw_frame: np.ndarray
    ) -> np.ndarray:
        """
        Drawing a finger trace on a frame
        """
        landmarks = self.hand_detector.get_hand_box(frame)
        for i, landmark in enumerate(landmarks):
            if i == 8:
                if landmark[0] < 30:
                    draw_frame = np.zeros((self.H, self.W)).astype(np.uint8)
                draw_frame = cv.circle(
                    draw_frame,
                    tuple(landmark),
                    4,
                    color=(255, 255, 255),
                    thickness=3,
                )
        return draw_frame


@hydra.main(
    version_base=None, config_path="../../configs", config_name="config"
)
def run(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    video_handler = VideoHandler(**cfg["video_handler"])
    hand_detection_model = build_hand_detection_model(cfg)
    app = App(video_handler, hand_detection_model)
    app.run()
    return None

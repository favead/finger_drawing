"""
Build hand detection models
"""
from finger_drawing.hand_detection_models.media_pipe_detector import (
    MediaPipeHandDetector,
)
from omegaconf import DictConfig, OmegaConf


def build_hand_detection_model(cfg: DictConfig) -> MediaPipeHandDetector | None:
    OmegaConf.to_yaml(cfg)
    if cfg["model"]["model_type"] == "MediaPipeHandDetector":
        return MediaPipeHandDetector(**cfg["model"])
    else:
        return None

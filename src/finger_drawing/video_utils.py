"""
Video utils
"""
from typing import Generator, Tuple

import cv2 as cv
import numpy as np


class VideoHandler:
    """
    OpenCV video flow handler
    """

    def __init__(
        self, video_input_path: str | None, video_output_path: str | None
    ) -> None:
        self.video_input_path = video_input_path
        self.video_output_path = video_output_path
        self._open_input_video()
        self._open_output_writer()

    def _open_input_video(self) -> cv.VideoCapture:
        """
        Open input buffer for loading frames
        """
        if self.video_input_path is None:
            self.video_input_path = 0
        self.video_capture = cv.VideoCapture(self.video_input_path)
        return self.video_capture

    def _open_output_writer(self, fps: int = 30) -> cv.VideoWriter | None:
        """
        Open output buffer for writing frames
        """
        self.out_writer = None
        if self.video_output_path:
            height = int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
            width = int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            self.out_writer = cv.VideoWriter(
                self.video_output_path, fourcc, fps, (width, height)
            )
        return self.out_writer

    def get_frame_shape(self) -> Tuple[int, int]:
        """
        Get height and width of frame
        """
        height = int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        return width, height

    def get_frame(self) -> Generator[np.ndarray, None, None]:
        """
        Load frame from video flow
        """
        cv.namedWindow("Frame", cv.WINDOW_NORMAL)
        while True:
            ret, frame = self.video_capture.read()
            if (not ret) | (cv.waitKey(25) & 0xFF == ord("q")):
                break
            yield frame
        self.video_capture.release()
        cv.destroyAllWindows()
        return None

    def write_frame(
        self, frame: np.ndarray, color: int = cv.COLOR_RGB2BGR
    ) -> None:
        """
        Write frame
        """
        bgr_frame = cv.cvtColor(frame, color)
        if self.video_output_path is None:
            cv.imshow("Frame", bgr_frame)
            cv.resizeWindow("Frame", 1000, 800)
        else:
            self.out_writer.write(bgr_frame)
        return None

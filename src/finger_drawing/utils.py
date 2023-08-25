"""
Non-video utils
"""
from os.path import dirname, abspath
from pathlib import Path


def get_absolute_path(related_path: str) -> str:
    """
    Convert relative path to absolute
    """
    root_dir = dirname(dirname(dirname(abspath(__file__))))
    absolute_path = str(Path(root_dir, related_path))
    return absolute_path

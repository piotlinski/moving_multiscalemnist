"""Tool to create a sequence of moving multiscale MNIST."""
from typing import Generator, List, Tuple

from PIL import Image

from moving_multiscalemnist.digit import Digit

Bbox = Tuple[float, float, float, float]


def get_bbox_coords(
    bbox: Tuple[int, int, int, int], x1: int, y1: int, image_size: Tuple[int, int]
) -> Bbox:
    """Calculate XYWH bbox based on bbox coordinates, location and image size."""
    x2 = x1 + bbox[2]
    y2 = y1 + bbox[3]
    x1 = x1 + bbox[0]
    y1 = y1 + bbox[1]

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return x / image_size[0], y / image_size[1], w / image_size[0], h / image_size[1]


def prepare_sequence(
    digits: List[Digit], n_frames: int, image_size: Tuple[int, int]
) -> Generator[Tuple[Image.Image, List[Bbox], List[int]], None, None]:
    """Prepare sequence of images of moving digits.

    :param digits: digits to be put in the image
    :param n_frames: number of frames to generate
    :param image_size: target image size
    :return: sequence of frames with bounding boxes
    """
    background = Image.new("RGB", image_size)
    for _ in range(n_frames):
        frame = background.copy()
        bboxes = []
        labels = []
        for digit in digits:
            frame.paste(digit.image, box=(digit.x1, digit.y1))
            bbox = get_bbox_coords(digit.bbox, digit.x1, digit.y1, image_size)
            bboxes.append(bbox)
            labels.append(digit.label)
            digit.update()
        yield frame, bboxes, labels

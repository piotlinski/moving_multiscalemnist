"""Tool to create a sequence of moving multiscale MNIST."""
import json
from pathlib import Path
from typing import Generator, Iterable, List, Tuple

import numpy as np
from PIL import Image

from moving_multiscalemnist.digit import Digit


def get_bbox_coords(
    bbox: Tuple[int, int, int, int], x1: int, y1: int, image_size: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """Calculate XYWH bbox based on bbox coordinates, location and image size."""
    x2 = x1 + bbox[2]
    y2 = y1 + bbox[3]
    x1 = x1 + bbox[0]
    y1 = y1 + bbox[1]

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return (
        min(1.0, max(0.0, x / image_size[0])),
        min(1.0, max(0.0, y / image_size[1])),
        min(1.0, max(0.0, w / image_size[0])),
        min(1.0, max(0.0, h / image_size[1])),
    )


def prepare_sequence(
    digits: List[Digit], n_frames: int, image_size: Tuple[int, int]
) -> Generator[
    Tuple[Image.Image, List[Tuple[float, float, float, float]], List[int], List[int]],
    None,
    None,
]:
    """Prepare sequence of images of moving digits.

    :param digits: digits to be put in the image
    :param n_frames: number of frames to generate
    :param image_size: target image size
    :return: sequence of frames with bounding boxes, labels and track ids
    """
    background = Image.new("RGB", image_size)
    for _ in range(n_frames):
        frame = background.copy()
        bboxes = []
        labels = []
        ids = []
        for idx, digit in enumerate(digits):
            mask = Image.fromarray(255 * (np.array(digit.image) > 100).astype(np.uint8))
            frame.paste(digit.image, box=(digit.x1, digit.y1), mask=mask)
            bbox = get_bbox_coords(digit.bbox, digit.x1, digit.y1, image_size)
            bboxes.append(bbox)
            labels.append(digit.label)
            ids.append(idx)
            digit.update()
        yield frame, bboxes, labels, ids


def save_sequence(
    sequence: Iterable[
        Tuple[
            Image.Image, List[Tuple[float, float, float, float]], List[int], List[int]
        ]
    ],
    directory: str,
    idx: int,
):
    """"""
    idx_str = f"{idx:06d}"
    path = Path(directory).joinpath(idx_str)
    path.mkdir(parents=True, exist_ok=True)
    annotation = []
    for idx, (frame, bboxes, labels, ids) in enumerate(sequence):
        with path.joinpath(f"{idx:06d}.jpg").open("wb") as fp:
            frame.save(fp)
        annotation.append({"bboxes": bboxes, "labels": labels, "ids": ids})
    with path.joinpath("annotations.json").open("w") as fp:
        json.dump(annotation, fp, indent=2)

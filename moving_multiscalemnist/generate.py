"""Generate dataset."""
import logging
from pathlib import Path
from typing import Generator, Iterable, List, Tuple

import numpy as np
from PIL.Image import Image
from tqdm import trange

from moving_multiscalemnist.digit import Digit
from moving_multiscalemnist.mnist import fetch_mnist
from moving_multiscalemnist.prepare import prepare_dataset
from moving_multiscalemnist.sequence import prepare_sequence, save_sequence

logger = logging.getLogger(__name__)


def shuffle_subset(
    subset: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle subset in unison."""
    images, labels = subset
    indices = np.random.permutation(len(labels))
    return images[indices], labels[indices]


def generate_subset(
    subset: Tuple[np.ndarray, np.ndarray],
    n_sequences: int,
    n_frames: int,
    min_digits: int,
    max_digits: int,
    image_size: Tuple[int, int],
    sizes: Tuple[int, ...],
    oscillations: Tuple[float, ...],
    oscillations_variances: Tuple[float, ...],
    fps: int,
    velocity: float,
) -> Generator[
    Iterable[
        Tuple[Image, List[Tuple[float, float, float, float]], List[int], List[int]]
    ],
    None,
    None,
]:
    """Generate subset of moving multiscale MNIST.

    :param subset: tuple of images and labels
    :param n_sequences: number of sequences to generate
    :param n_frames: number of frames in each sequence
    :param min_digits: minimum number of digits in sequence
    :param max_digits: maximum number of digits in sequence
    :param image_size: frame size
    :param sizes: available digit sizes
    :param oscillations: digit size change periods coefficients
    :param oscillations_variances: proportion in which size oscillates
    :param fps: number of frames in one second (period length)
    :param velocity: velocity of digit movement
    :return: generator of sequences of frames, boxes, labels and track ids
    """
    images, labels = shuffle_subset(subset)
    start_idx = 0
    for _ in trange(n_sequences, desc="Generating"):
        n_digits = np.random.randint(min_digits, max_digits + 1)
        if start_idx + n_digits > len(labels):
            start_idx = 0

        end_idx = start_idx + n_digits
        digits = [
            Digit(
                image=images[idx],
                label=labels[idx].item(),
                image_size=image_size,
                sizes=sizes,
                oscillations=oscillations,
                oscillations_variances=oscillations_variances,
                fps=fps,
                velocity=velocity,
            )
            for idx in range(start_idx, end_idx)
        ]
        yield prepare_sequence(digits, n_frames=n_frames, image_size=image_size)

        start_idx = end_idx


def generate_dataset(
    data_dir: str,
    train_size: int,
    test_size: int,
    n_frames: int,
    min_digits: int,
    max_digits: int,
    image_size: Tuple[int, int],
    sizes: Tuple[int, ...],
    oscillations: Tuple[float, ...],
    oscillations_variances: Tuple[float, ...],
    fps: int,
    velocity: float,
):
    """Generate sequences and save to file."""
    mnist = fetch_mnist(data_dir)

    logger.info("Generating train dataset.")
    for idx, sequence in enumerate(
        generate_subset(
            subset=mnist["train"],
            n_sequences=train_size,
            n_frames=n_frames,
            min_digits=min_digits,
            max_digits=max_digits,
            image_size=image_size,
            sizes=sizes,
            oscillations=oscillations,
            oscillations_variances=oscillations_variances,
            fps=fps,
            velocity=velocity,
        )
    ):
        save_sequence(sequence, directory="dataset/train", idx=idx)

    logger.info("Generating test dataset.")
    for idx, sequence in enumerate(
        generate_subset(
            subset=mnist["test"],
            n_sequences=test_size,
            n_frames=n_frames,
            min_digits=min_digits,
            max_digits=max_digits,
            image_size=image_size,
            sizes=sizes,
            oscillations=oscillations,
            oscillations_variances=oscillations_variances,
            fps=fps,
            velocity=velocity,
        )
    ):
        save_sequence(sequence, directory="dataset/test", idx=idx)

    logger.info("Generating annotations.")
    prepare_dataset(Path("dataset"), train_folder="train", test_folder="test")

    logger.info("Done.")

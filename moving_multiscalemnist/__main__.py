import argparse
import logging

import numpy as np

from moving_multiscalemnist.defaults import (
    DATA_DIR,
    FPS,
    IMAGE_SIZE,
    MAX_DIGITS,
    MIN_DIGITS,
    N_FRAMES,
    OSCILLATIONS,
    OSCILLATIONS_VARIANCES,
    SEED,
    SIZES,
    TEST_SIZE,
    TRAIN_SIZE,
    VELOCITY,
)
from moving_multiscalemnist.generate import generate_dataset

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", help="Random seed", type=int, default=SEED)
    parser.add_argument("--data-dir", "-d", help="MNIST location", default=DATA_DIR)
    parser.add_argument(
        "--train-size", "-trs", help="Train subset size", type=int, default=TRAIN_SIZE
    )
    parser.add_argument(
        "--test-size", "-tss", help="Test subset size", type=int, default=TEST_SIZE
    )
    parser.add_argument(
        "--n-frames",
        "-nf",
        help="Number of frames in a sequence",
        type=int,
        default=N_FRAMES,
    )
    parser.add_argument(
        "--min-digits",
        help="Minimum number of digits in a sequence",
        type=int,
        default=MIN_DIGITS,
    )
    parser.add_argument(
        "--max-digits",
        help="Maximum number of digits in a sequence",
        type=int,
        default=MAX_DIGITS,
    )
    parser.add_argument(
        "--image-size",
        "-is",
        help="Output image size",
        nargs=2,
        type=int,
        default=IMAGE_SIZE,
    )
    parser.add_argument(
        "--sizes", "-ss", help="Digit sizes", nargs="+", type=int, default=SIZES
    )
    parser.add_argument(
        "--oscillations",
        "-o",
        help="Size oscillation period",
        nargs="+",
        type=float,
        default=OSCILLATIONS,
    )
    parser.add_argument(
        "--oscillations-variances",
        "-ov",
        help="Size oscillation variances",
        nargs="+",
        type=float,
        default=OSCILLATIONS_VARIANCES,
    )
    parser.add_argument(
        "--fps", help="Frames per second (period length)", type=float, default=FPS
    )
    parser.add_argument(
        "--velocity", "-v", help="Digit velocity", type=float, default=VELOCITY
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    generate_dataset(
        data_dir=args.data_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        n_frames=args.n_frames,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        image_size=args.image_size,
        sizes=args.sizes,
        oscillations=args.oscillations,
        oscillations_variances=args.oscillations_variances,
        fps=args.fps,
        velocity=args.velocity,
    )

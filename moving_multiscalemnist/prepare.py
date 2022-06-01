"""Prepare annotations for YOLOv4."""
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Set, Tuple

from tqdm import tqdm

TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
NAMES_FILE = "obj.names"


def handle_sequence(path: Path) -> Tuple[List[Path], Set[int]]:
    """Handle single sequence directory."""
    labels = set()
    with path.joinpath("annotations.json").open("r") as fp:
        annotations = json.load(fp)
    images = list(sorted(path.glob("*.jpg")))
    if len(annotations) != len(images):
        raise ValueError("Unequal number of images and annotations")
    for image, annotation in zip(images, annotations):
        with image.with_suffix(".txt").open("w") as fp:
            for box, label in zip(annotation["bboxes"], annotation["labels"]):
                labels.add(label)
                x, y, w, h = box
                fp.write(f"{label} {x} {y} {w} {h}\n")
    return images, labels


def handle_subset(path: Path) -> Tuple[List[Path], Set[int]]:
    """Handle subset of sequences."""
    images = []
    labels = set()
    for sequence in tqdm(list(sorted(path.glob("*")))):
        seq_images, seq_labels = handle_sequence(sequence)
        images.extend(seq_images)
        labels.update(seq_labels)
    return images, labels


def prepare_dataset(path: Path, train_folder: str, test_folder: str):
    train_images, train_labels = handle_subset(path.joinpath(train_folder))
    test_images, test_labels = handle_subset(path.joinpath(test_folder))

    with path.joinpath(TRAIN_FILE).open("w") as fp:
        for file in train_images:
            fp.write(f"data/train/{str(file).split('train/')[-1]}\n")

    with path.joinpath(TEST_FILE).open("w") as fp:
        for file in test_images:
            fp.write(f"data/test/{str(file).split('test/')[-1]}\n")

    with path.joinpath(NAMES_FILE).open("w") as fp:
        for label in sorted(train_labels.union(test_labels)):
            fp.write(f"{label}\n")

    with path.joinpath("obj.data").open("w") as fp:
        fp.write(f"classes = {max(len(train_labels), len(test_labels))}\n")
        fp.write(f"train = data/{TRAIN_FILE}\n")
        fp.write(f"valid = data/{TEST_FILE}\n")
        fp.write(f"names = data/{NAMES_FILE}\n")
        fp.write("backup = data/\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", help="dataset directory")
    parser.add_argument("--train", default="train", help="train subdirectory")
    parser.add_argument("--test", default="test", help="test subdirectory")
    args = parser.parse_args()

    prepare_dataset(Path(args.dataset), train_folder=args.train, test_folder=args.test)

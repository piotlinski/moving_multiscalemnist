"""Test digit class."""
from unittest.mock import patch

import numpy as np
import pytest

from moving_multiscalemnist.digit import Digit


@pytest.fixture
@patch("numpy.random.uniform", return_value=0.5)
@patch("numpy.random.choice", side_effect=[32, 1.0, 1])
def sample_digit(_choice_mock, _uniform_mock):
    """Return sample digit object."""
    image = np.random.random((27, 27))
    return Digit(image, label=3, image_size=(512, 512), sizes=(32, 64))


def test_digit_properties(sample_digit):
    """Verify if digit properties are calculated correctly."""
    assert sample_digit.x1 == 240  # 0.5 * 512 - 32 / 2
    assert sample_digit.y1 == 240
    assert sample_digit.size == 32


def test_digit_update(sample_digit):
    """Verify if digit position is updated correctly."""
    start_x1 = sample_digit.x1
    start_y1 = sample_digit.y1
    start_size = sample_digit.size

    ret = sample_digit.update()
    assert ret is sample_digit

    assert sample_digit.x1 > start_x1
    assert sample_digit.y1 > start_y1
    assert sample_digit.size > start_size
    assert sample_digit._t == 1


def test_digit_bounce(sample_digit):
    """Verify if bounce is performed correctly."""
    vel_x = sample_digit._vel_x
    vel_y = sample_digit._vel_y

    for _ in range(10):  # should exceed image
        sample_digit.update()

    assert sample_digit.shall_bounce_horizontally()
    assert sample_digit.shall_bounce_vertically()

    sample_digit.update()
    assert sample_digit._vel_x == -vel_x
    assert sample_digit._vel_y == -vel_y


def test_digit_oscillate(sample_digit):
    """Verify if digit size is oscillating."""
    start_size = sample_digit.size

    for _ in range(3):
        sample_digit.update()
    assert sample_digit.size > start_size

    for _ in range(2):
        sample_digit.update()
    assert sample_digit.size == start_size

    for _ in range(3):
        sample_digit.update()
    assert sample_digit.size < start_size

    for _ in range(2):
        sample_digit.update()
    assert sample_digit.size == start_size

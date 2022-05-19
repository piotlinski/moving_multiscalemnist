from unittest.mock import patch

import numpy as np
import pytest

from moving_multiscalemnist.digit import Digit


@pytest.fixture
@patch("numpy.random.uniform", return_value=0.5)
@patch("numpy.random.choice", side_effect=[32, 1.0, 0.5, 1])
def sample_digit(_choice_mock, _uniform_mock):
    """Return sample digit object."""
    image = np.zeros((32, 32))
    image[10:20, 8:23] = 0.2
    return Digit(image, label=3, image_size=(128, 128), sizes=(32, 64))

"""Handler for moving digit."""
from typing import Tuple

import numpy as np
from PIL import Image


class Digit:
    """Single digit from MNIST."""

    def __init__(
        self,
        image: np.ndarray,
        label: int,
        image_size: Tuple[int, int],
        sizes: Tuple[int, ...],
        oscillations: Tuple[float, ...] = (0.8, 1.0, 1.2, 1.4),
        fps: int = 10,
    ):
        """
        :param image: digit image as np array
        :param label: digit label
        :param image_size: target image size (width, height)
        :param sizes: available digit sizes
        :param oscillations: oscillation period factor
        :param fps: number of frames per second (period)
        """
        self.image = Image.fromarray(image)
        self.label = label

        self._size = np.random.choice(sizes)
        self._x = np.random.uniform(0, 1)
        self._y = np.random.uniform(0, 1)

        self._image_width, self._image_height = image_size

        self._vel_x = np.random.uniform(-1, 1)
        self._vel_y = np.random.uniform(-1, 1)

        self._osc_t = np.random.choice(oscillations)
        self._osc_dir = np.random.choice([1, -1])

        self._t: int = 0
        self._T: int = fps

        self._resize()

    def _sin(self) -> float:
        return round(1 + np.sin(self._t / (self._T * self._osc_t) * 2 * np.pi) / 2, 2)

    @property
    def x1(self) -> int:
        return int(self._x * self._image_width - self.image.width / 2)

    @property
    def y1(self) -> int:
        return int(self._y * self._image_height - self.image.height / 2)

    @property
    def size(self) -> int:
        return int(self._size * self._sin())

    def shall_bounce_horizontally(self) -> bool:
        return self._x <= 0 or 1 <= self._x

    def shall_bounce_vertically(self) -> bool:
        return self._y <= 0 or 1 <= self._y

    def _update_position(self):
        self._x += self._vel_x / self._T
        self._y += self._vel_y / self._T

    def _resize(self):
        self.image = self.image.resize((self.size, self.size))

    def update(self) -> "Digit":
        self._update_position()
        self._resize()

        if self.shall_bounce_horizontally():
            self._vel_x = -self._vel_x
        if self.shall_bounce_vertically():
            self._vel_y = -self._vel_y

        self._t += 1

        return self

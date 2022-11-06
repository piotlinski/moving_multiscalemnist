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
        oscillations_variances: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
        fps: int = 10,
    ):
        """
        :param image: digit image as np array
        :param label: digit label
        :param image_size: target image size (width, height)
        :param sizes: available digit sizes
        :param oscillations: oscillation period factor
        :param oscillations_variances: proportion in which size oscillates
        :param fps: number of frames per second (period)
        """
        self._image = Image.fromarray(image)
        self.label = label

        self._t: int = 0
        self._T: int = fps

        self._image_width, self._image_height = image_size

        self._size = np.random.choice(sizes)
        self._osc_t = np.random.choice(oscillations)
        self._osc_var = np.random.choice(oscillations_variances)
        self._osc_dir = np.random.choice([1, -1])

        self._x = np.random.uniform(self.x_margin, 1 - self.x_margin)
        self._y = np.random.uniform(self.y_margin, 1 - self.y_margin)

        self._vel_x = np.random.uniform(-1, 1)
        self._vel_y = np.random.uniform(-1, 1)

        self._x_bounce = -1
        self._y_bounce = -1
        self._bounce_thresh = 10

    @property
    def image(self) -> Image.Image:
        return self._image.resize((self.size, self.size))

    def _sin(self) -> float:
        return round(
            1 + np.sin(self._t / (self._T * self._osc_t) * 2 * np.pi) * self._osc_var, 2
        )

    @property
    def x1(self) -> int:
        return int(self._x * self._image_width - self.image.width / 2)

    @property
    def y1(self) -> int:
        return int(self._y * self._image_height - self.image.height / 2)

    @property
    def size(self) -> int:
        return int(self._size * self._sin())

    @property
    def x_margin(self) -> float:
        return self.size / (4 * self._image_width)

    @property
    def y_margin(self) -> float:
        return self.size / (4 * self._image_height)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Calculate X1Y1X2Y2 bbox coordinates.

        .. note: MNIST digits are usually smaller than the image, hence for tight bbox
            coordinates need to be recalculated
        """
        coords = np.where(np.array(self.image) > 0)
        white_ys = coords[0]
        white_xs = coords[1]
        return white_xs.min(), white_ys.min(), white_xs.max(), white_ys.max()

    def shall_bounce_horizontally(self) -> bool:
        shall_bounce = self._x < self.x_margin or 1 - self.x_margin < self._x
        if shall_bounce:
            self._x_bounce += 1
            if self._x_bounce == self._bounce_thresh:
                self._x_bounce = 0

        return shall_bounce and self._x_bounce == 0

    def shall_bounce_vertically(self) -> bool:
        shall_bounce = self._y < self.y_margin or 1 - self.y_margin < self._y
        if shall_bounce:
            self._y_bounce += 1
            if self._y_bounce == self._bounce_thresh:
                self._y_bounce = 0

        return shall_bounce and self._y_bounce == 0

    def _update_position(self):
        self._x += self._vel_x / self._T
        self._y += self._vel_y / self._T

    def update(self) -> "Digit":
        if self.shall_bounce_horizontally():
            self._vel_x = -self._vel_x
        if self.shall_bounce_vertically():
            self._vel_y = -self._vel_y

        self._update_position()

        self._t += 1

        return self

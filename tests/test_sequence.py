"""Test creating sequence of digits."""
import pytest

from moving_multiscalemnist.sequence import get_bbox_coords, prepare_sequence


@pytest.mark.parametrize(
    "bbox, x1, y1, image_size, expected",
    [
        ((10, 12, 18, 20), 13, 15, (100, 100), (0.27, 0.31, 0.08, 0.08)),
        ((16, 32, 80, 160), 16, 32, (512, 512), (0.125, 0.25, 0.125, 0.25)),
        ((23, 17, 29, 31), 18, 5, (416, 416), (0.10577, 0.06971, 0.01442, 0.03365)),
    ],
)
def test_get_bbox_coords(bbox, x1, y1, image_size, expected):
    """Verify if bounding box coordinates are calculated correctly."""
    x, y, w, h = get_bbox_coords(bbox, x1, y1, image_size)
    x_exp, y_exp, w_exp, h_exp = expected
    assert x == pytest.approx(x_exp, abs=1e-5)
    assert y == pytest.approx(y_exp, abs=1e-5)
    assert w == pytest.approx(w_exp, abs=1e-5)
    assert h == pytest.approx(h_exp, abs=1e-5)


def test_prepare_sequence(sample_digit):
    """Verify if sequence contains prepared digits images."""
    (frame_0, bboxes_0, labels_0), (frame_1, bboxes_1, labels_1) = list(
        prepare_sequence([sample_digit], n_frames=2, image_size=(128, 128))
    )

    assert frame_0.getdata() != frame_1.getdata()

    assert bboxes_0[0][0] < bboxes_1[0][0]
    assert bboxes_0[0][1] < bboxes_1[0][1]
    assert bboxes_0[0][2] == bboxes_1[0][2]
    assert bboxes_0[0][3] == bboxes_1[0][3]

    assert labels_0[0] == labels_1[0] == sample_digit.label

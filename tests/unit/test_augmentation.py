import pytest

torch = pytest.importorskip("torch")

from trajectory_mlops.models.dataset import augment  # noqa: E402

pytestmark = pytest.mark.ml


def test_time_shift_only_moves_columns():
    image = torch.arange(10.0).repeat(3, 4, 1) / 10  # 3 channels, 4 rows, 10 time steps
    shifted = augment(image, max_shift=0.3, noise_std=0.0)
    assert shifted.shape == image.shape
    assert sorted(shifted[0, 0].tolist()) == sorted(image[0, 0].tolist())  # same values, moved
    assert torch.equal(shifted[0, 0], shifted[2, 3])  # every row moves by the same amount


def test_noise_keeps_pixels_between_0_and_1():
    image = torch.ones(3, 8, 8)
    noisy = augment(image, max_shift=0.0, noise_std=0.5)
    assert noisy.min() >= 0 and noisy.max() <= 1
    assert not torch.equal(noisy, image)


def test_zero_settings_change_nothing():
    image = torch.rand(3, 8, 8)
    assert torch.equal(augment(image, max_shift=0.0, noise_std=0.0), image)

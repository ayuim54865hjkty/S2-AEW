import torch

from s2aew.models import HamidaEtAl
from s2aew.runner import S2AEWPipeline
from s2aew.targets import make_spatial_spectral_targets
from s2aew.verification import verify_s2aew


def test_keyed_targets_are_deterministic():
    a = make_spatial_spectral_targets("k", 9, 5)
    b = make_spatial_spectral_targets("k", 9, 5)
    assert torch.equal(a.spatial, b.spatial)
    assert torch.equal(a.spectral, b.spectral)
    assert set(a.spatial.tolist()).issubset({-1.0, 1.0})


def test_pipeline_smoke_cpu():
    torch.manual_seed(0)
    image_size = 11
    channels = 16
    model = HamidaEtAl(input_channels=channels, n_classes=4, patch_size=image_size)
    image = torch.rand(1, 1, channels, image_size, image_size)

    pipeline = S2AEWPipeline(model, image_size=image_size, num_channels=channels, key="unit-test", device="cpu")
    watermarked = pipeline.generate(image, steps=1, eps=0.05)
    report = pipeline.verify(model, watermarked)

    assert watermarked.shape == image.shape
    assert 0.0 <= report.joint.wsr <= 100.0
    assert 0.0 <= report.joint.p_value <= 1.0


def test_verification_report_shapes():
    torch.manual_seed(1)
    image_size = 11
    channels = 16
    model = HamidaEtAl(input_channels=channels, n_classes=4, patch_size=image_size)
    image = torch.rand(1, 1, channels, image_size, image_size)
    targets = make_spatial_spectral_targets("verify", image_size * image_size, channels)

    report = verify_s2aew(model, image, targets.spatial, targets.spectral, image_size, channels, device="cpu")
    assert report.spatial.total == image_size * image_size
    assert report.spectral.total == channels
    assert report.joint.total == image_size * image_size + channels

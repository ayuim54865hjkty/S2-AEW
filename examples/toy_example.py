import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s2aew.generation import generate_s2aew_sample
from s2aew.models import HamidaEtAl
from s2aew.targets import make_spatial_spectral_targets
from s2aew.verification import verify_s2aew


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 11
    num_channels = 16
    num_classes = 4

    torch.manual_seed(7)
    model = HamidaEtAl(input_channels=num_channels, n_classes=num_classes, patch_size=image_size).to(device)
    model.eval()

    image = torch.rand(1, 1, num_channels, image_size, image_size, device=device)
    targets = make_spatial_spectral_targets(
        key="demo-private-key",
        spatial_bits=image_size * image_size,
        spectral_bits=num_channels,
        device=device,
    )

    watermarked = generate_s2aew_sample(
        model=model,
        image=image,
        target_spatial=targets.spatial,
        target_spectral=targets.spectral,
        image_size=image_size,
        num_channels=num_channels,
        steps=5,
        eps=0.1,
    )
    report = verify_s2aew(
        model=model,
        image=watermarked,
        target_spatial=targets.spatial,
        target_spectral=targets.spectral,
        image_size=image_size,
        num_channels=num_channels,
    )

    print("Spatial WSR:  %.2f%%, p=%.3g" % (report.spatial.wsr, report.spatial.p_value))
    print("Spectral WSR: %.2f%%, p=%.3g" % (report.spectral.wsr, report.spectral.p_value))
    print("Joint WSR:    %.2f%%, p=%.3g" % (report.joint.wsr, report.joint.p_value))


if __name__ == "__main__":
    main()

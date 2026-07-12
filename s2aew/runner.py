import torch

from .generation import generate_s2aew_sample
from .targets import S2AEWTargets, make_spatial_spectral_targets
from .verification import S2AEWReport, verify_s2aew


class S2AEWPipeline:
    """High-level pipeline that mirrors the three method stages in the paper."""

    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int,
        num_channels: int,
        key: str,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.image_size = image_size
        self.num_channels = num_channels
        self.targets = make_spatial_spectral_targets(
            key=key,
            spatial_bits=image_size * image_size,
            spectral_bits=num_channels,
            device=self.device,
        )

    def generate(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        return generate_s2aew_sample(
            model=self.model,
            image=image.to(self.device),
            target_spatial=self.targets.spatial,
            target_spectral=self.targets.spectral,
            image_size=self.image_size,
            num_channels=self.num_channels,
            **kwargs,
        )

    def verify(self, suspicious_model: torch.nn.Module, image: torch.Tensor, targets: S2AEWTargets | None = None) -> S2AEWReport:
        active_targets = targets or self.targets
        return verify_s2aew(
            model=suspicious_model,
            image=image.to(self.device),
            target_spatial=active_targets.spatial,
            target_spectral=active_targets.spectral,
            image_size=self.image_size,
            num_channels=self.num_channels,
            device=self.device,
        )

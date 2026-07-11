from dataclasses import dataclass

import torch
from scipy.stats import binomtest

from .explanation import SpatialLimeCarrier, SpectralLimeCarrier


@dataclass(frozen=True)
class BranchReport:
    wsr: float
    matched: int
    total: int
    p_value: float
    evidence_score: float


@dataclass(frozen=True)
class S2AEWReport:
    spatial: BranchReport
    spectral: BranchReport
    joint: BranchReport


def _branch_report(weights: torch.Tensor, target: torch.Tensor) -> BranchReport:
    recovered = torch.where(weights.flatten() > 0, 1.0, -1.0)
    target = target.flatten().to(recovered.device).float()
    matched = int((recovered == target).sum().item())
    total = int(target.numel())
    p_value = float(binomtest(matched, total, p=0.5, alternative="greater").pvalue)
    if p_value <= 0:
        evidence = float("inf")
    else:
        evidence = float(-torch.log10(torch.tensor(p_value)).item())
    return BranchReport(
        wsr=100.0 * matched / total,
        matched=matched,
        total=total,
        p_value=p_value,
        evidence_score=evidence,
    )


def verify_s2aew(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_spatial: torch.Tensor,
    target_spectral: torch.Tensor,
    image_size: int,
    num_channels: int,
    lam: float = 1e-3,
    device: torch.device | str | None = None,
) -> S2AEWReport:
    """Verify S2-AEW ownership evidence from a suspicious model."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    image = image.to(device).float()

    with torch.no_grad():
        label = int(model(image).argmax(dim=1).item())

    spatial_carrier = SpatialLimeCarrier(image_size, num_channels, device=device, lam=lam)
    spectral_carrier = SpectralLimeCarrier(num_channels, device=device, lam=lam)

    spatial_weights = spatial_carrier.explain(model, image, label)
    spectral_weights = spectral_carrier.explain(model, image, label)

    spatial = _branch_report(spatial_weights, target_spatial)
    spectral = _branch_report(spectral_weights, target_spectral)

    joint_weights = torch.cat([spatial_weights.flatten(), spectral_weights.flatten()])
    joint_target = torch.cat([target_spatial.flatten(), target_spectral.flatten()])
    joint = _branch_report(joint_weights, joint_target)
    return S2AEWReport(spatial=spatial, spectral=spectral, joint=joint)

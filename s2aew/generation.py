import torch
import torch.nn.functional as F

from .explanation import SpatialLimeCarrier, SpectralLimeCarrier


def _hinge_sign_loss(weights: torch.Tensor, target: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    weights = weights.flatten()
    target = target.flatten().to(weights.device).float()
    return torch.relu(margin - weights * target).sum()


def generate_s2aew_sample(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_spatial: torch.Tensor,
    target_spectral: torch.Tensor,
    image_size: int,
    num_channels: int,
    eps: float = 0.3,
    lr: float = 3e-3,
    steps: int = 200,
    lam: float = 1e-3,
    spectral_weight: float = 2.2,
    keep_weight: float = 0.1,
    margin: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a prediction-preserving S2-AEW watermark sample.

    Args:
        model: Trained HSI classifier that accepts ``[B,1,C,H,W]`` inputs.
        image: Clean HSI patch with shape ``[1,1,C,H,W]`` or compatible.
        target_spatial: Secret spatial target signs in ``{-1,+1}``.
        target_spectral: Secret spectral target signs in ``{-1,+1}``.
        image_size: Spatial patch size.
        num_channels: Number of HSI spectral bands.
        eps: L-infinity perturbation bound.
        lr: Adam learning rate.
        steps: Optimization steps.
        lam: Ridge term used by the local surrogate decoder.
        spectral_weight: Relative weight of the spectral explanation branch.
        keep_weight: KL penalty for preserving the original prediction.
        margin: Sign-alignment hinge margin.
        device: Torch device.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    clean = image.detach().clone().to(device).float()
    adv = clean.detach().clone().requires_grad_(True)

    spatial = SpatialLimeCarrier(image_size, num_channels, device=device, lam=lam)
    spectral = SpectralLimeCarrier(num_channels, device=device, lam=lam)

    with torch.no_grad():
        reference_logits = model(clean)
        label = int(reference_logits.argmax(dim=1).item())

    optimizer = torch.optim.Adam([adv], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        spatial_weights = spatial.explain(model, adv, label)
        spectral_weights = spectral.explain(model, adv, label)
        spatial_loss = _hinge_sign_loss(spatial_weights, target_spatial, margin) / target_spatial.numel()
        spectral_loss = _hinge_sign_loss(spectral_weights, target_spectral, margin) / target_spectral.numel()

        logits = model(adv)
        keep_loss = F.kl_div(
            F.log_softmax(logits, dim=1),
            F.softmax(reference_logits, dim=1),
            reduction="batchmean",
        )

        loss = spatial_loss + spectral_weight * spectral_loss + keep_weight * keep_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta = torch.clamp(adv - clean, min=-eps, max=eps)
            adv.copy_(clean + delta)

    return adv.detach()

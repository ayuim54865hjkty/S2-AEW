import argparse
import json

import torch

from .config import load_config
from .models import ChenEtAl, HamidaEtAl, HeEtAl
from .runner import S2AEWPipeline


MODEL_REGISTRY = {
    "ChenEtAl": ChenEtAl,
    "HamidaEtAl": HamidaEtAl,
    "HeEtAl": HeEtAl,
}


def build_model(name: str, num_channels: int, num_classes: int, image_size: int) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model architecture: {name}")
    return MODEL_REGISTRY[name](input_channels=num_channels, n_classes=num_classes, patch_size=image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight S2-AEW pipeline check.")
    parser.add_argument("--config", default="configs/paviau_hamida.yaml", help="Path to YAML config.")
    parser.add_argument("--toy-steps", type=int, default=5, help="Steps for the random-input smoke test.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        cfg.model.architecture,
        cfg.dataset.num_channels,
        cfg.dataset.num_classes,
        cfg.dataset.image_size,
    ).to(device)
    if cfg.model.checkpoint:
        model.load_state_dict(torch.load(cfg.model.checkpoint, map_location=device))
    model.eval()

    image = torch.rand(1, 1, cfg.dataset.num_channels, cfg.dataset.image_size, cfg.dataset.image_size, device=device)
    pipeline = S2AEWPipeline(
        model=model,
        image_size=cfg.dataset.image_size,
        num_channels=cfg.dataset.num_channels,
        key=cfg.watermark.key,
        device=device,
    )
    watermarked = pipeline.generate(
        image,
        eps=cfg.watermark.eps,
        lr=cfg.watermark.lr,
        steps=args.toy_steps,
        lam=cfg.watermark.lam,
        spectral_weight=cfg.watermark.spectral_weight,
        keep_weight=cfg.watermark.keep_weight,
    )
    report = pipeline.verify(model, watermarked)
    print(json.dumps({
        "spatial_wsr": report.spatial.wsr,
        "spectral_wsr": report.spectral.wsr,
        "joint_wsr": report.joint.wsr,
        "joint_p_value": report.joint.p_value,
        "joint_evidence_score": report.joint.evidence_score,
    }, indent=2))


if __name__ == "__main__":
    main()

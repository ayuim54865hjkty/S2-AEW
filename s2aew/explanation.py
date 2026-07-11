import math

import numpy as np
import torch


class SpatialLimeCarrier:
    """Spatial explanation carrier for HSI patches.

    The carrier partitions an HSI patch into spatial blocks and measures how
    the target-class logit changes when each block is masked. With
    ``num_masks=None``, the implementation uses deterministic leave-one-block
    perturbations, which gives one verifiable sign per spatial block.
    """

    def __init__(
        self,
        image_size: int,
        num_channels: int,
        spatial_bits: int | None = None,
        num_masks: int | None = None,
        lam: float = 1e-3,
        device: torch.device | str | None = None,
    ) -> None:
        self.image_size = image_size
        self.num_channels = num_channels
        self.spatial_bits = spatial_bits or image_size * image_size
        self.num_masks = num_masks or self.spatial_bits
        self.lam = lam
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.block_size = max(1, int(image_size / math.sqrt(self.spatial_bits)))
        self.masks, self.design = self._build_masks(num_masks)
        self.masks = self.masks.to(self.device)
        self.design = self.design.to(self.device)
        self.decoder = self._build_decoder(self.design)

    def _build_masks(self, num_masks: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        masks = []
        design = []
        row_num = max(1, int(self.image_size / self.block_size))

        if num_masks is None:
            for bit_idx in range(self.spatial_bits):
                flat = np.ones(self.spatial_bits, dtype=np.float32)
                flat[bit_idx] = 0.0
                design.append(flat)

                mask = np.ones((self.num_channels, self.image_size, self.image_size), dtype=np.float32)
                row = bit_idx // row_num
                col = bit_idx % row_num
                mask[
                    :,
                    row * self.block_size : (row + 1) * self.block_size,
                    col * self.block_size : (col + 1) * self.block_size,
                ] = 0.0
                masks.append(mask)
        else:
            rng = np.random.default_rng()
            for _ in range(num_masks):
                flat = rng.integers(0, 2, size=self.spatial_bits).astype(np.float32)
                design.append(flat)
                mask = np.ones((self.num_channels, self.image_size, self.image_size), dtype=np.float32)
                for bit_idx, keep in enumerate(flat):
                    if keep:
                        continue
                    row = bit_idx // row_num
                    col = bit_idx % row_num
                    mask[
                        :,
                        row * self.block_size : (row + 1) * self.block_size,
                        col * self.block_size : (col + 1) * self.block_size,
                    ] = 0.0
                masks.append(mask)

        return torch.from_numpy(np.stack(masks)), torch.from_numpy(np.stack(design))

    def _build_decoder(self, design: torch.Tensor) -> torch.Tensor:
        design = design.float()
        gram = design.T @ design
        gram = gram + torch.eye(self.spatial_bits, device=design.device) * self.lam
        return torch.linalg.solve(gram, design.T)

    def explain(self, model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor | int | None = None) -> torch.Tensor:
        model.eval()
        image = image.to(self.device).float()
        if image.ndim == 5:
            image_4d = image.squeeze(0).squeeze(0)
        elif image.ndim == 4:
            image_4d = image.squeeze(0)
        elif image.ndim == 3:
            image_4d = image
        else:
            raise ValueError("Expected image shape [C,H,W], [1,C,H,W], or [1,1,C,H,W].")

        if label is None:
            with torch.no_grad():
                label = int(model(image_4d.unsqueeze(0).unsqueeze(0)).argmax(dim=1).item())
        elif isinstance(label, torch.Tensor):
            label = int(label.item())

        masked = self.masks * image_4d.unsqueeze(0)
        logits = model(masked.unsqueeze(1))
        predictions = logits[:, label].unsqueeze(-1)
        return self.decoder @ predictions


class SpectralLimeCarrier:
    """Spectral explanation carrier with one leave-one-band bit per channel."""

    def __init__(
        self,
        num_channels: int,
        lam: float = 1e-3,
        device: torch.device | str | None = None,
    ) -> None:
        self.num_channels = num_channels
        self.lam = lam
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        design = torch.ones(num_channels, num_channels)
        design.fill_diagonal_(0.0)
        self.design = design.to(self.device)
        gram = self.design.T @ self.design
        gram = gram + torch.eye(num_channels, device=self.device) * lam
        self.decoder = torch.linalg.solve(gram, self.design.T)

    def explain(self, model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor | int | None = None) -> torch.Tensor:
        model.eval()
        image = image.to(self.device).float()
        if image.ndim == 5:
            image_4d = image.squeeze(0).squeeze(0)
        elif image.ndim == 4:
            image_4d = image.squeeze(0)
        elif image.ndim == 3:
            image_4d = image
        else:
            raise ValueError("Expected image shape [C,H,W], [1,C,H,W], or [1,1,C,H,W].")

        if label is None:
            with torch.no_grad():
                label = int(model(image_4d.unsqueeze(0).unsqueeze(0)).argmax(dim=1).item())
        elif isinstance(label, torch.Tensor):
            label = int(label.item())

        masks = self.design[:, :, None, None]
        masked = masks * image_4d.unsqueeze(0)
        logits = model(masked.unsqueeze(1))
        predictions = logits[:, label].unsqueeze(-1)
        return self.decoder @ predictions

import numpy as np
import torch


def standardize_cube(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Band-wise z-score normalization for an HSI cube."""
    cube = cube.astype(np.float32, copy=False)
    mean = cube.reshape(-1, cube.shape[-1]).mean(axis=0)
    std = cube.reshape(-1, cube.shape[-1]).std(axis=0)
    return (cube - mean) / (std + eps)


def extract_patch(cube: np.ndarray, row: int, col: int, patch_size: int) -> np.ndarray:
    """Extract a zero-padded `C x H x W` patch from an `H x W x C` cube."""
    radius = patch_size // 2
    padded = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    patch = padded[row : row + patch_size, col : col + patch_size, :]
    return np.moveaxis(patch, -1, 0).astype(np.float32)


def to_model_input(patch: np.ndarray | torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert a `C x H x W` patch to `[1, 1, C, H, W]` model input."""
    if isinstance(patch, np.ndarray):
        tensor = torch.from_numpy(patch)
    else:
        tensor = patch
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError("Expected patch shape [C,H,W], [1,C,H,W], or [1,1,C,H,W].")
    return tensor.float().to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

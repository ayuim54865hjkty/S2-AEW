from dataclasses import dataclass
import hashlib
import hmac

import torch


@dataclass(frozen=True)
class S2AEWTargets:
    spatial: torch.Tensor
    spectral: torch.Tensor


def _keyed_signs(key: str | bytes, namespace: str, length: int, device: torch.device | str | None = None) -> torch.Tensor:
    if isinstance(key, str):
        key_bytes = key.encode("utf-8")
    else:
        key_bytes = key

    bits = []
    counter = 0
    while len(bits) < length:
        msg = f"{namespace}:{counter}".encode("utf-8")
        digest = hmac.new(key_bytes, msg, hashlib.sha256).digest()
        for byte in digest:
            for shift in range(8):
                bits.append(1 if ((byte >> shift) & 1) else -1)
                if len(bits) == length:
                    break
            if len(bits) == length:
                break
        counter += 1

    return torch.tensor(bits, dtype=torch.float32, device=device)


def make_spatial_spectral_targets(
    key: str | bytes,
    spatial_bits: int,
    spectral_bits: int,
    device: torch.device | str | None = None,
) -> S2AEWTargets:
    """Generate deterministic private target signs from a secret key."""
    return S2AEWTargets(
        spatial=_keyed_signs(key, "s2aew-spatial", spatial_bits, device=device),
        spectral=_keyed_signs(key, "s2aew-spectral", spectral_bits, device=device),
    )

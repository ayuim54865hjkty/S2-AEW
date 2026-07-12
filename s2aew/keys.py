from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KeyCandidate:
    index: int
    predicted: int
    confidence: float
    margin: float


def prediction_margin(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = logits.softmax(dim=1)
    top2 = probs.topk(k=2, dim=1).values
    confidence = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    return confidence, margin


@torch.no_grad()
def select_key_candidates(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor | None = None,
    max_keys: int = 24,
    min_margin: float = 0.2,
) -> list[KeyCandidate]:
    """Select stable correctly classified samples for watermark generation."""
    model.eval()
    logits = model(inputs)
    preds = logits.argmax(dim=1)
    confidence, margin = prediction_margin(logits)

    candidates: list[KeyCandidate] = []
    for idx in range(inputs.shape[0]):
        if labels is not None and int(preds[idx]) != int(labels[idx]):
            continue
        if float(margin[idx]) < min_margin:
            continue
        candidates.append(
            KeyCandidate(
                index=idx,
                predicted=int(preds[idx]),
                confidence=float(confidence[idx]),
                margin=float(margin[idx]),
            )
        )
    candidates.sort(key=lambda item: (item.margin, item.confidence), reverse=True)
    return candidates[:max_keys]

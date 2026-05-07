"""Named MLP layer-size presets for the PPO actor & critic.

Reference these from configs:
    learner:
      arch: medium

Larger networks need much more data to train. For 45 days of single-GPU training,
'small' or 'medium' are the realistic choices.
"""
from __future__ import annotations

ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "tiny":   (128, 128),
    "small":  (256, 256, 256),
    "medium": (512, 512, 512),
    "large":  (1024, 1024, 1024),
    # Nexto-ish — needs lots of compute. Don't pick this for the class final.
    "xlarge": (2048, 2048, 1024, 1024),
}


def get_layer_sizes(name: str) -> tuple[int, ...]:
    if name not in ARCHITECTURES:
        raise KeyError(f"Unknown arch {name!r}. Available: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name]

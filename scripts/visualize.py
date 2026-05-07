#!/usr/bin/env python
"""Watch a checkpoint play in the rlviser visualizer.

Requires rlviser_py installed and the visualizer binary running. See:
https://github.com/VirxEC/rlviser
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rlbot.utils.logging import get_logger


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    log = get_logger("rlbot.visualize")
    log.warning("visualize: stub — connect rlviser_py to env.render() once integrated.")
    log.info(f"Would play {args.episodes} eps of {args.checkpoint}")


if __name__ == "__main__":
    main()

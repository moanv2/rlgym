#!/usr/bin/env python
"""Export a trained checkpoint into a deployable RLBot bot folder.

Final-week task. See docs/roadmap_45_days.md week 5.
"""
from __future__ import annotations

import argparse

from rlbot.utils.logging import get_logger


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="Output bot folder (RLBot package)")
    args = p.parse_args()

    log = get_logger("rlbot.export")
    log.warning("export_model: stub — implement once policy load + RLBot adapter is in.")
    log.info(f"Would export {args.checkpoint} -> {args.out}")


if __name__ == "__main__":
    main()

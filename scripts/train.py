#!/usr/bin/env python
"""Thin wrapper so you can run `python scripts/train.py ...` instead of `-m rlbot.training.train`."""

from rlbot.training.train import main

if __name__ == "__main__":
    main()

"""Seeding has to be deterministic across the libs we use, otherwise reproducibility
claims are a lie."""
from __future__ import annotations

import random

from rlbot.utils.seeding import seed_everything


def test_seed_python_random():
    seed_everything(123)
    a = [random.random() for _ in range(5)]
    seed_everything(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_numpy():
    np = __import__("numpy")
    seed_everything(7)
    a = np.random.rand(5)
    seed_everything(7)
    b = np.random.rand(5)
    assert (a == b).all()

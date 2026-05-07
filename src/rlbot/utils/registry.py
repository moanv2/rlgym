"""Generic name -> factory registry for components selectable from YAML.

Example:
    from rlbot.utils.registry import Registry
    REWARDS = Registry("rewards")

    @REWARDS.register("velocity_player_to_ball")
    class VelocityPlayerToBallReward(RewardFunction): ...

    cls = REWARDS.get("velocity_player_to_ball")
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, type[T] | Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
        def decorator(obj: type[T] | Callable[..., T]) -> type[T] | Callable[..., T]:
            if key in self._items:
                raise ValueError(f"{self.name}: '{key}' already registered")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> type[T] | Callable[..., T]:
        if key not in self._items:
            raise KeyError(
                f"{self.name}: '{key}' not registered. Available: {sorted(self._items)}"
            )
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

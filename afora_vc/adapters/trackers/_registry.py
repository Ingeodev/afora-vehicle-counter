from typing import Any

from ...domain.ports.tracker import Tracker


class TrackerRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, tracker_class: type) -> None:
        cls._registry[name] = tracker_class

    @classmethod
    def create(cls, backend: str, **kwargs: Any) -> Tracker:
        if backend not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Tracker backend '{backend}' no registrado. Disponibles: {available}"
            )
        instance = cls._registry[backend](**kwargs)
        if not isinstance(instance, Tracker):
            raise TypeError(f"{backend} adapter does not satisfy Tracker protocol")
        return instance

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())

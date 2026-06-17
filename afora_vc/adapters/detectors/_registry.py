from typing import Any

from ...domain.ports.detector import Detector


class DetectorRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, detector_class: type) -> None:
        cls._registry[name] = detector_class

    @classmethod
    def create(cls, backend: str, **kwargs: Any) -> Detector:
        if backend not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Backend '{backend}' no registrado. Disponibles: {available}"
            )
        instance = cls._registry[backend](**kwargs)
        if not isinstance(instance, Detector):
            raise TypeError(f"{backend} adapter does not satisfy Detector protocol")
        return instance

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())

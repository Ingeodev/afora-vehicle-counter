from __future__ import annotations

from typing import Mapping, Sequence

COCO_VEHICLE_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
)


def _iter_class_names(names) -> list[tuple[int, str]]:
    if isinstance(names, Mapping):
        pairs: list[tuple[int, str]] = []
        for key, value in names.items():
            try:
                pairs.append((int(key), str(value)))
            except (TypeError, ValueError):
                continue
        return pairs
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return [(idx, str(value)) for idx, value in enumerate(names)]
    return []


def resolve_vehicle_classes(names=None) -> dict[int, str]:
    """Resuelve class_id -> nombre usando los nombres reales del modelo.

    El modelo puede tener un orden de clases distinto al de COCO (por ejemplo,
    modelos fine-tuned). Se conservan solo las clases vehiculares conocidas.
    """
    wanted = {name.lower() for name in VEHICLE_CLASS_NAMES}
    resolved = {
        class_id: class_name.strip()
        for class_id, class_name in _iter_class_names(names)
        if class_name.strip().lower() in wanted
    }
    return resolved or dict(COCO_VEHICLE_CLASSES)

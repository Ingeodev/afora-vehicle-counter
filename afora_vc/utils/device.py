from ..domain.enums import DeviceType
from ..domain.exceptions import DeviceNotAvailableError


def detect_device() -> DeviceType:
    """Auto-detecta el mejor dispositivo disponible."""
    try:
        import torch

        if torch.cuda.is_available():
            return DeviceType.CUDA
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceType.MPS
    except ImportError:
        pass

    return DeviceType.CPU


def validate_device(requested: str) -> DeviceType:
    """Valida y normaliza un device string, lanzando error si no esta disponible."""
    try:
        device = DeviceType(requested.lower())
    except ValueError:
        raise DeviceNotAvailableError(requested)

    if device == DeviceType.CUDA:
        try:
            import torch
            if not torch.cuda.is_available():
                raise DeviceNotAvailableError("cuda (no CUDA-capable GPU detected)")
        except ImportError:
            raise DeviceNotAvailableError("cuda (torch not installed)")

    if device == DeviceType.MPS:
        try:
            import torch
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise DeviceNotAvailableError("mps (Apple Silicon required)")
        except ImportError:
            raise DeviceNotAvailableError("mps (torch not installed)")

    return device

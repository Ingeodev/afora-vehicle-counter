"""Excepciones tipadas del dominio."""


class AforosError(Exception):
    """Error base de la libreria."""


class VideoNotFoundError(AforosError):
    def __init__(self, path: str):
        super().__init__(f"Video no encontrado: {path}")
        self.path = path


class VideoOpenError(AforosError):
    def __init__(self, path: str):
        super().__init__(f"No se pudo abrir el video: {path}")
        self.path = path


class ConfigurationError(AforosError):
    def __init__(self, message: str, path: str | None = None):
        super().__init__(f"{message}: {path}" if path else message)
        self.path = path


class OutputDirectoryError(AforosError):
    def __init__(self, path: str, detail: str = ""):
        super().__init__(f"No se pudo crear directorio de salida: {path}. {detail}")
        self.path = path


class ProcessingError(AforosError):
    def __init__(self, message: str, video_path: str | None = None):
        msg = f"[{video_path}] {message}" if video_path else message
        super().__init__(msg)
        self.video_path = video_path


class BackendNotInstalledError(AforosError):
    def __init__(self, backend: str, install_hint: str):
        super().__init__(
            f"Backend '{backend}' no esta instalado. "
            f"Instalar con: pip install {install_hint}"
        )
        self.backend = backend
        self.install_hint = install_hint


class BackendNotRegisteredError(AforosError):
    def __init__(self, backend: str, available: list[str]):
        super().__init__(
            f"Backend '{backend}' no registrado. "
            f"Disponibles: {', '.join(available)}"
        )
        self.backend = backend
        self.available = available


class DeviceNotAvailableError(AforosError):
    def __init__(self, device: str):
        super().__init__(f"Dispositivo no disponible: {device}")
        self.device = device

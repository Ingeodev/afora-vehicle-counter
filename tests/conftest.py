from pathlib import Path

import pytest


@pytest.fixture
def assets_dir() -> Path:
    return Path(__file__).parent / "assets"


@pytest.fixture
def output_dir(tmp_path) -> Path:
    return tmp_path / "output"


@pytest.fixture
def ensure_assets(assets_dir: Path) -> None:
    required = [
        assets_dir / "video.mp4",
        assets_dir / "zones.json",
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        pytest.skip(
            f"Assets faltantes: {', '.join(missing)}. "
            f"Ubicar en tests/assets/"
        )

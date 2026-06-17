#!/usr/bin/env python3
"""Genera los assets de prueba ejecutando la libreria v1 sobre un video."""

import subprocess
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets"
EXPECTED_DIR = ASSETS_DIR / "expected"
V1_MODULE = "mglon_vehicle_counter.cli"


def main():
    if not (ASSETS_DIR / "video.mp4").exists():
        print("ERROR: tests/assets/video.mp4 no existe.")
        print("Coloca un video de prueba en tests/assets/video.mp4")
        sys.exit(1)

    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", V1_MODULE, "process",
        str(ASSETS_DIR / "video.mp4"),
        "--zones", str(ASSETS_DIR / "zones.json"),
        "--output", str(EXPECTED_DIR),
    ]

    if (ASSETS_DIR / "mask.png").exists():
        cmd.extend(["--mask", str(ASSETS_DIR / "mask.png")])

    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    generated = list(EXPECTED_DIR.iterdir())
    print(f"\nAssets generados ({len(generated)} archivos):")
    for f in sorted(generated):
        rel = f.relative_to(EXPECTED_DIR.parent.parent)
        size = f.stat().st_size
        print(f"  {rel} ({size:,} bytes)")


if __name__ == "__main__":
    main()

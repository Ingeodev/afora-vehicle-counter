"""Instalador inteligente: detecta CUDA y elige la variante optima de PyTorch.

Uso desde el proyecto (local):
    python scripts/install.py rtdetr

Uso desde PyPI (publicado):
    python scripts/install.py rtdetr --pypi
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple


CUDA_VARIANTS = {
    "11.8": ("torch-cu118", "https://download.pytorch.org/whl/cu118"),
    "12.1": ("torch-cu121", "https://download.pytorch.org/whl/cu121"),
    "12.4": ("torch-cu124", "https://download.pytorch.org/whl/cu124"),
    "12.6": ("torch-cu126", "https://download.pytorch.org/whl/cu126"),
}


def detect_cuda() -> Tuple[str, str] | None:
    """Detecta CUDA y retorna (variant_name, index_url) o None si solo CPU.

    Nota: torch-auto instala torch desde PyPI, el cual ya incluye CUDA support.
    Las variantes torch-cuXXX son solo si se quiere pin a un CUDA toolkit especifico
    usando --extra-index-url del repositorio de PyTorch.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.split("\n"):
            if "CUDA Version" not in line:
                continue
            version = line.split("CUDA Version:")[-1].strip().rstrip()
            parts = version.split(".")
            major_minor = ".".join(parts[:2]) if len(parts) >= 2 else version

            if major_minor in CUDA_VARIANTS:
                return CUDA_VARIANTS[major_minor]

            # CUDA > 12.6 (e.g. 13.2): torch-auto instala torch de PyPI,
            # que ya incluye CUDA support. Funciona por backward compatibility.
            print(f"  CUDA {version} detectado — usando torch-auto (PyPI con CUDA incluido)")
            print(f"  (CUDA {version} es compatible con CUDA 12.x via backward compatibility)")
            return ("torch-auto", None)

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def build_pip_command(
    backends: list[str],
    cuda_variant: Tuple[str, str] | None,
    local: bool = True,
) -> list[str]:
    extras = backends.copy()
    if cuda_variant:
        extras.append(cuda_variant[0])
    else:
        extras.append("torch-auto")

    if local:
        project_root = Path(__file__).resolve().parent.parent
        pkg_spec = str(project_root)
    else:
        pkg_spec = "afora-vehicle-counter"

    cmd = [
        sys.executable, "-m", "pip", "install",
        "-e", f"{pkg_spec}[{','.join(extras)}]",
    ]

    if cuda_variant and cuda_variant[1]:
        cmd.extend(["--extra-index-url", cuda_variant[1]])

    return cmd


def main():
    print("=" * 60)
    print("  afora-vehicle-counter - Instalador automatico")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser(description="Instalador de afora-vehicle-counter")
    parser.add_argument(
        "backends", nargs="*", default=["yolo", "rtdetr"],
        help="Backends a instalar: yolo, dfine, rtdetr, tracking",
    )
    parser.add_argument("--pypi", action="store_true", help="Instalar desde PyPI en vez de local")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar el comando")
    args = parser.parse_args()

    print(f"\n  Backends solicitados: {', '.join(args.backends)}")
    print(f"  Detectando CUDA...\n")

    cuda = detect_cuda()

    if cuda:
        print(f"  ==> Variante CUDA: {cuda[0]}")
        if cuda[1]:
            print(f"      Index URL:    {cuda[1]}")
    else:
        print(f"  ==> CUDA no detectado -- instalando torch-auto")
        print(f"      (PyPI con CUDA incluido si hay GPU)")

    pip_cmd = build_pip_command(args.backends, cuda, local=not args.pypi)

    print(f"\n  Comando:\n")
    print(f"    {' '.join(pip_cmd)}\n")

    if not args.dry_run:
        print(f"  Ejecutando...\n")
        sys.stdout.flush()
        result = subprocess.run(pip_cmd)
        if result.returncode == 0:
            print(f"\n  ==> Instalacion completada")
        else:
            print(f"\n  ==> Error durante la instalacion (codigo: {result.returncode})")
            sys.exit(result.returncode)
    else:
        print(f"  [dry-run] No se ejecuto el comando.\n")


if __name__ == "__main__":
    main()

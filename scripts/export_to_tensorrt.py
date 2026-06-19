"""
Exporta un modelo RT-DETR (.pt) a TensorRT (.engine) usando Ultralytics.
Uso: python scripts/export_to_tensorrt.py [--half] [--batch N] [--model MODELO.pt]
"""

import argparse
import sys
from pathlib import Path


def export_model(model_path: str, half: bool = False, batch: int = 1) -> str:
    from ultralytics import RTDETR

    model = RTDETR(model_path)
    out = model.export(format="engine", half=half, imgsz=640, batch=batch)
    dest = Path(out)
    stem = dest.stem
    suffix_b = f"-b{batch}"
    if suffix_b not in stem:
        dest = dest.with_stem(f"{stem}{suffix_b}")
        Path(out).replace(dest)
    print(f"Exportado exitosamente: {dest}")
    return str(dest)


def main():
    parser = argparse.ArgumentParser(
        description="Exporta modelo RT-DETR a TensorRT"
    )
    parser.add_argument(
        "--model",
        default="models/rtdetr-l.pt",
        help="Ruta al modelo .pt (default: models/rtdetr-l.pt)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Exportar en FP16 (media precision)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Tamano del batch (default: 1). Ej: --batch 16 para hasta 16 imagenes por inferencia",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: no se encuentra el modelo {args.model}")
        sys.exit(1)

    export_model(args.model, half=args.half, batch=args.batch)


if __name__ == "__main__":
    main()

r"""
Equivalente Python del Google Apps Script que separa videos por dia.

Uso:
    python separar_videos_por_dia.py "C:\ruta\a\carpeta"          # separar
    python separar_videos_por_dia.py --rollback "C:\ruta\a\carpeta"   # deshacer
"""

import os
import re
import shutil
import sys
from datetime import datetime


EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
LOG_NAME = "_separacion_log.txt"


def format_datetime(segment: str) -> tuple[str, str]:
    """'20260413040150' -> ('2026-04-13', '2026-04-13_15.40.13')"""
    if len(segment) < 14:
        raise ValueError(f"Segmento demasiado corto: {segment}")
    date_part = f"{segment[0:4]}-{segment[4:6]}-{segment[6:8]}"
    formatted = f"{date_part}_{segment[8:10]}.{segment[10:12]}.{segment[12:14]}"
    return date_part, formatted


def get_or_create_folder(parent: str, name: str) -> str:
    folder = os.path.join(parent, name)
    os.makedirs(folder, exist_ok=True)
    return folder


def rollback(root_path: str) -> None:
    log_path = os.path.join(root_path, LOG_NAME)
    if not os.path.isfile(log_path):
        print(f"[ERROR] No se encuentra el log: {log_path}")
        sys.exit(1)

    lines = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            orig, dest = parts[0].strip(), parts[1].strip()
            lines.append((orig, dest))

    # Revertir en orden inverso para respetar dependencias de carpetas
    errores = 0
    for orig, dest in reversed(lines):
        if os.path.isfile(dest):
            try:
                os.makedirs(os.path.dirname(orig), exist_ok=True)
                shutil.move(dest, orig)
                print(f"[ROLLBACK] {os.path.basename(dest)} -> {os.path.basename(orig)}")
            except Exception as e:
                print(f"[ERROR] No se pudo revertir {os.path.basename(dest)}: {e}")
                errores += 1
        else:
            print(f"[SKIP] No existe: {dest}")

    # Eliminar carpetas de fecha vacias
    for item in os.listdir(root_path):
        full = os.path.join(root_path, item)
        if os.path.isdir(full) and item != os.path.basename(root_path):
            try:
                os.rmdir(full)
            except OSError:
                pass  # No vacia, la dejamos

    # Eliminar el log
    if errores == 0:
        os.remove(log_path)
        print(f"[OK] Log eliminado: {LOG_NAME}")
    else:
        print(f"[WARN] {errores} error(es) al revertir. Revise manualmente.")

    print(f"[OK] Rollback completado.")


def split_folder(root_path: str) -> list[dict]:
    root_name = os.path.basename(root_path.rstrip("\\/"))
    results = []
    movimientos = []

    if not os.path.isdir(root_path):
        print(f"[ERROR] La ruta no existe o no es una carpeta: {root_path}")
        return results

    files = [f for f in os.listdir(root_path)
             if os.path.isfile(os.path.join(root_path, f))]

    video_files = [f for f in files
                   if os.path.splitext(f)[1].lower() in EXTENSIONES_VIDEO]

    if not video_files:
        print(f"[WARN] No se encontraron videos en: {root_path}")
        return results

    current_date = None
    current_folder = None

    for filename in video_files:
        parts = filename.split("_")
        if len(parts) < 3:
            print(f"[SKIP] Formato inesperado (menos de 3 segmentos): {filename}")
            continue

        segment = parts[2]
        segment = re.sub(r"\..+$", "", segment)

        try:
            date_part, formatted_ts = format_datetime(segment)
        except (ValueError, IndexError) as e:
            print(f"[SKIP] No se pudo parsear fecha en '{filename}': {e}")
            continue

        if date_part != current_date:
            current_folder = get_or_create_folder(root_path, date_part)
            current_date = date_part

        ext = os.path.splitext(filename)[1]
        new_name = f"{root_name}_{formatted_ts}{ext}"
        src = os.path.join(root_path, filename)
        dst = os.path.join(current_folder, new_name)

        # Evitar colision de nombres
        counter = 1
        orig_dst = dst
        while os.path.exists(dst):
            base = os.path.splitext(orig_dst)[0]
            dst = f"{base}_{counter}{ext}"
            counter += 1

        shutil.move(src, dst)

        movimientos.append((src, dst))
        results.append({
            "status": "success",
            "original": filename,
            "destino": dst,
        })
        print(f"[OK] {filename} -> {os.path.basename(dst)} en {date_part}")

    # Escribir log
    if movimientos:
        log_path = os.path.join(root_path, LOG_NAME)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Log de separacion de videos por dia\n")
            f.write(f"# Ejecutado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Formato: ruta_original | ruta_nueva\n")
            f.write("-" * 60 + "\n")
            for orig, dest in movimientos:
                f.write(f"{orig} | {dest}\n")
        print(f"[OK] Log guardado: {LOG_NAME}")

    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--rollback":
        if len(sys.argv) < 3:
            print("Uso: python separar_videos_por_dia.py --rollback <ruta_carpeta>")
            sys.exit(1)
        rollback(sys.argv[2])
    else:
        folder = sys.argv[1]
        if folder == "--help" or folder == "-h":
            print(__doc__)
            sys.exit(0)
        print(f"Separando videos en: {folder}")
        results = split_folder(folder)
        print(f"\nProcesados {len(results)} videos.")


if __name__ == "__main__":
    main()

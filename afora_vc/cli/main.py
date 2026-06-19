import sys

from ..application.facade import process_video, process_directory
from ..domain.exceptions import AforosError


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Contador de vehiculos multi-backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # process
    p = subparsers.add_parser("process", help="Procesar un video")
    p.add_argument("video", help="Ruta al video de entrada")
    p.add_argument("--zones", "-z", help="Archivo JSON de zonas")
    p.add_argument("--mask", "-m", help="Mascara de region")
    p.add_argument("--output", "-o", default="./output", help="Carpeta de salida")
    p.add_argument("--base-time", "-t", default="00:00", help="Hora base (HH:MM)")
    p.add_argument("--device", "-d", choices=["cpu", "cuda", "mps"], help="Dispositivo")
    p.add_argument("--model", default="models/yolov8s.pt", help="Modelo a usar")
    p.add_argument("--backend", "-b", default="yolo", help="Backend de deteccion (yolo, dfine, rtdetr)")
    p.add_argument("--tracker", default="botsort", help="Backend de tracking (botsort, bytetrack)")
    p.add_argument("--reduce", "-r", type=int, default=1, help="Factor de reduccion")
    p.add_argument("--max-minutes", type=float, help="Limite de minutos")
    p.add_argument("--strategy", choices=["box", "seg"], default="box", help="Estrategia")
    p.add_argument("--night-enhance", action="store_true", help="Mejora nocturna")
    p.add_argument("--date", default="2025-01-01", help="Fecha del video (YYYY-MM-DD)")
    p.add_argument("--model-config", type=str, help="YAML con parametros del modelo")
    p.add_argument("--deblurring", action="store_true", help="Deblurring agresivo")
    p.add_argument("--no-video", action="store_true", help="No guardar video")
    p.add_argument("--quiet", "-q", action="store_true", help="Modo silencioso")
    p.add_argument("--output-name", type=str, help="Nombre del video de salida")
    p.add_argument("--batch-size", type=int, default=1, help="Tamano de batch")
    p.add_argument("--tracker-config", type=str, help="YAML de config del tracker")

    # scan
    s = subparsers.add_parser("scan", help="Procesar directorio de videos")
    s.add_argument("directory", help="Directorio raiz")
    s.add_argument("--output", "-o", default="./output", help="Carpeta base de salida")
    s.add_argument("--recursive", "-R", action="store_true")
    s.add_argument("--device", "-d", choices=["cpu", "cuda", "mps"])
    s.add_argument("--model", default="models/yolov8s.pt", help="Modelo")
    s.add_argument("--backend", "-b", default="yolo", help="Backend de deteccion")
    s.add_argument("--tracker", default="botsort", help="Backend de tracking")
    s.add_argument("--reduce", "-r", type=int, default=1)
    s.add_argument("--max-minutes", type=float)
    s.add_argument("--zones", "-z", help="Archivo JSON de zonas (global)")
    s.add_argument("--mask", "-m", help="Mascara global")
    s.add_argument("--strategy", choices=["box", "seg"], default="box")
    s.add_argument("--night-enhance", action="store_true")
    s.add_argument("--date", default="2025-01-01")
    s.add_argument("--model-config", type=str)
    s.add_argument("--deblurring", action="store_true")
    s.add_argument("--yes", "-y", action="store_true", help="Confirmar automaticamente")
    s.add_argument("--quiet", "-q", action="store_true")
    s.add_argument("--batch-size", type=int, default=1)
    s.add_argument("--output-name", type=str, help="Nombre del video de salida")
    s.add_argument("--tracker-config", type=str)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "process":
            return _cmd_process(args)
        elif args.command == "scan":
            return _cmd_scan(args)
    except AforosError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error inesperado: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_process(args):
    result = process_video(
        video=args.video,
        zones=args.zones,
        mask=args.mask,
        output=args.output,
        base_time=args.base_time,
        date=args.date,
        device=args.device,
        model=args.model,
        reduce=args.reduce,
        max_minutes=args.max_minutes,
        strategy=args.strategy,
        night_enhance=args.night_enhance,
        model_config=args.model_config,
        deblurring=args.deblurring,
        no_video=args.no_video,
        quiet=args.quiet,
        backend=args.backend,
        tracker_backend=args.tracker,
        output_name=args.output_name,
        batch_size=args.batch_size,
        tracker_config=args.tracker_config,
    )

    if not args.quiet:
        print(f"\n✅ Completado: {result.total_detections} detecciones")
    return 0


def _cmd_scan(args):
    results = process_directory(
        root_path=args.directory,
        output_base=args.output,
        recursive=args.recursive,
        device=args.device,
        model=args.model,
        reduce=args.reduce,
        max_minutes=args.max_minutes,
        zones=args.zones,
        mask=args.mask,
        strategy=args.strategy,
        night_enhance=args.night_enhance,
        model_config=args.model_config,
        deblurring=args.deblurring,
        no_video=False,
        quiet=args.quiet,
        backend=args.backend,
        tracker_backend=args.tracker,
        output_name=args.output_name,
        batch_size=args.batch_size,
        tracker_config=args.tracker_config,
    )

    if not args.quiet:
        total = sum(r.total_detections for r in results)
        print(f"\n✅ Completado: {len(results)} videos, {total} detecciones")
    return 0


if __name__ == "__main__":
    sys.exit(main())

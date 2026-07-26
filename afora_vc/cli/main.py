import os
import sys

from ..application.facade import process_video, process_directory, process_from_config, check_config
from ..application.batch_config import BatchConfig
from ..domain.exceptions import AforosError
from ..domain.counting.services import VehicleClassesResolver
from ..application.count_pipeline import GroupPipeline, export_chart
from ..output.count_exporter import DefaultFlowExporter
from ..utilities.video_sorter import sort_videos_by_date, rollback_sort
from ..utils.display import DisplayManager


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
    p.add_argument("--pipeline-mode", default="sequential", choices=["sequential", "pipelined"], help="Modo de pipeline")
    p.add_argument("--profile", action="store_true", help="Habilitar profiling por paso")
    p.add_argument("--profile-path", type=str, help="Ruta del CSV de profiling")
    p.add_argument("--queue-maxsize", type=int, default=4, help="Tamano maximo de cola (modo pipelined)")
    p.add_argument("--audit", action="store_true", help="Auditar detecciones por etapa del pipeline")

    # run (batch config)
    r = subparsers.add_parser("run", help="Procesar desde archivo de configuracion YAML")
    r.add_argument("config", help="Ruta al archivo .yaml de configuracion")
    r.add_argument("--quiet", "-q", action="store_true", help="Modo silencioso")
    r.add_argument("--graph", choices=["none", "image", "web"], help="Tipo de grafico (sorescribe el YAML)")

    # utils
    utils_parser = subparsers.add_parser("utils", help="Utility tools")
    utils_sub = utils_parser.add_subparsers(dest="util_cmd", help="Utility command")

    gbd = utils_sub.add_parser("group-by-date", help="Group video files into date-based subdirectories")
    gbd.add_argument("path", help="Folder containing video files")
    gbd.add_argument("--rollback", "-r", action="store_true", help="Undo a previous group-by-date operation")

    # group
    g = subparsers.add_parser("group", help="Agrupar detecciones de una carpeta procesada")
    g.add_argument("--folder", "-f", help="Carpeta con detections.csv")
    g.add_argument("--config", "-c", required=True, help="YAML con configuracion (usa seccion count:)")
    g.add_argument("--model-config", help="YAML con config del modelo (vehicle_classes)")
    g.add_argument("--output", "-o", help="Carpeta de salida (por defecto: la misma --folder)")
    g.add_argument("--graph", choices=["none", "image", "web"], help="Tipo de grafico (sorescribe el YAML)")

    # check
    ck = subparsers.add_parser("check", help="Previsualizar configuracion por video sin procesar")
    ck.add_argument("config", help="Ruta al archivo .yaml de configuracion")

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
    s.add_argument("--base-time", "-t", default="00:00", help="Hora base (HH:MM)")
    s.add_argument("--queue-maxsize", type=int, default=4, help="Tamano maximo de cola (modo pipelined)")
    s.add_argument("--profile", action="store_true", help="Habilitar profiling por paso")
    s.add_argument("--profile-path", type=str, help="Ruta del CSV de profiling")
    s.add_argument("--audit", action="store_true", help="Auditar detecciones por etapa del pipeline")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    _display = DisplayManager(verbose=not getattr(args, 'quiet', False), quiet=getattr(args, 'quiet', False))

    try:
        if args.command == "process":
            return _cmd_process(args)
        elif args.command == "utils":
            return _cmd_utils(args)
        elif args.command == "scan":
            return _cmd_scan(args)
        elif args.command == "group":
            return _cmd_group(args)
        elif args.command == "check":
            return _cmd_check(args)
        elif args.command == "run":
            return _cmd_run(args)
    except AforosError as e:
        _display.show_error(str(e), context="AforosError")
        return 1
    except Exception as e:
        _display.show_error(str(e), context="inesperado")
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
        pipeline_mode=args.pipeline_mode,
        profile=args.profile,
        profile_path=args.profile_path,
        queue_maxsize=args.queue_maxsize,
        audit=args.audit,
    )

    if not args.quiet:
        print(f"\n✅ Completado: {result.total_detections} detecciones")
    return 0


def _cmd_utils(args):
    if args.util_cmd == "group-by-date":
        if args.rollback:
            result = rollback_sort(args.path)
        else:
            result = sort_videos_by_date(args.path)
        total = len(result.moved)
        skipped = len(result.skipped)
        errs = len(result.errors)
        if not getattr(args, "quiet", False) or errs:
            print(f"\nProcessed: {total} moved, {skipped} skipped, {errs} errors")
        return 0 if errs == 0 else 1
    return 0


def _cmd_group(args):
    batch_cfg = BatchConfig.from_yaml(args.config)
    if batch_cfg.count is None:
        print("Error: El YAML no tiene seccion 'count:'")
        return 1

    # CLI --graph sobreescribe el YAML
    if args.graph is not None:
        import dataclasses
        batch_cfg.count = dataclasses.replace(batch_cfg.count, graph=args.graph)

    folder = args.folder or batch_cfg.output
    out_dir = args.output or folder
    resolver = VehicleClassesResolver.from_model_config(args.model_config)

    pipeline = GroupPipeline(batch_cfg.count, resolver=resolver)
    data = pipeline.execute(folder, output_dir=out_dir)

    exporter = DefaultFlowExporter()
    out_path = os.path.join(out_dir, "aggregated_counts.csv")
    exporter.write(data, batch_cfg.count, resolver, out_path)

    # Fecha desde el nombre de la carpeta o el config
    date = os.path.basename(os.path.normpath(folder))
    chart_path = export_chart(data, batch_cfg.count, resolver, out_dir, date=date)

    if not getattr(args, "quiet", False):
        print(f"CSV: {out_path}")
        if chart_path:
            print(f"Grafico: {chart_path}")
    return 0


def _cmd_run(args):
    config_obj = BatchConfig.from_yaml(args.config)
    config_obj._source_path = os.path.abspath(args.config)
    if args.graph is not None and config_obj.count is not None:
        import dataclasses
        config_obj.count = dataclasses.replace(config_obj.count, graph=args.graph)
    results = process_from_config(config_obj)
    if not args.quiet:
        total = sum(r.total_detections for r in results)
        print(f"Completados {len(results)} videos, {total} detecciones totales")
        if config_obj.count and config_obj.count.graph != "none":
            print(f"Grafico: {config_obj.count.graph}")
    return 0


def _cmd_check(args):
    check_config(args.config)
    return 0


def _cmd_scan(args):
    results = process_directory(
        root_path=args.directory,
        output_base=args.output,
        recursive=args.recursive,
        global_mask=args.mask,
        global_zones=args.zones,
        device=args.device,
        model=args.model,
        reduce=args.reduce,
        max_minutes=args.max_minutes,
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
        base_time=args.base_time,
        queue_maxsize=args.queue_maxsize,
        profile=args.profile,
        profile_path=args.profile_path,
        audit=args.audit,
    )

    if not args.quiet:
        total = sum(r.total_detections for r in results)
        print(f"Completados {len(results)} videos, {total} detecciones totales")
    return 0


if __name__ == "__main__":
    sys.exit(main())

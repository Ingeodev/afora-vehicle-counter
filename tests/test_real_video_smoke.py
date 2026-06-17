import pytest


@pytest.mark.slow
class TestRealVideoSmoke:
    """Verifica que el pipeline procesa 20 segundos de video real sin errores."""

    def test_process_20_seconds(self, assets_dir, output_dir):
        from afora_vc.application.facade import process_video

        result = process_video(
            video=str(assets_dir / "video.mp4"),
            zones=str(assets_dir / "zones.json"),
            mask=str(assets_dir / "mask.png"),
            output=str(output_dir),
            model="yolov8s.pt",
            max_minutes=20 / 60,
            quiet=True,
            no_video=True,
        )

        assert result.total_detections > 0, "Debe detectar al menos un vehiculo"
        assert len(result.zone_counts) > 0, "Debe haber conteos por zona"
        assert result.frames_processed > 0, "Debe procesar al menos un frame"

    def test_csv_output_20_seconds(self, assets_dir, output_dir):
        from afora_vc.application.facade import process_video
        import pandas as pd

        process_video(
            video=str(assets_dir / "video.mp4"),
            zones=str(assets_dir / "zones.json"),
            mask=str(assets_dir / "mask.png"),
            output=str(output_dir),
            model="yolov8s.pt",
            max_minutes=20 / 60,
            quiet=True,
            no_video=True,
        )

        csv_dir = output_dir / "csv"
        detections_csv = csv_dir / "detections.csv"
        summary_csv = csv_dir / "summary.csv"

        assert detections_csv.exists(), "Falta detections.csv"
        assert summary_csv.exists(), "Falta summary.csv"

        det_df = pd.read_csv(detections_csv)
        assert not det_df.empty, "detections.csv no debe estar vacio"
        assert "zone" in det_df.columns
        assert "vehicle_id" in det_df.columns

from afora_vc.application.config import BatchConfig, PipelineConfig


class TestBatchConfig:
    def test_default_disabled(self):
        cfg = BatchConfig()
        assert cfg.enabled is False
        assert cfg.batch_size == 4
        assert cfg.queue_maxsize == 4

    def test_roundtrip_yaml(self, tmp_path):
        cfg = PipelineConfig()
        cfg.batch.enabled = True
        cfg.batch.batch_size = 8
        cfg.batch.queue_maxsize = 16

        path = tmp_path / "config.yaml"
        cfg.to_yaml(str(path))

        loaded = PipelineConfig.from_yaml(str(path))
        assert loaded.batch.enabled is True
        assert loaded.batch.batch_size == 8
        assert loaded.batch.queue_maxsize == 16

from afora_vc.utils.device import detect_device, validate_device
from afora_vc.domain.enums import DeviceType
from afora_vc.domain.exceptions import DeviceNotAvailableError


class TestDetectDevice:
    def test_returns_enum(self):
        device = detect_device()
        assert isinstance(device, DeviceType)
        assert device in (DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS)


class TestValidateDevice:
    def test_cpu_always_valid(self):
        device = validate_device("cpu")
        assert device == DeviceType.CPU

    def test_invalid_device_name_raises(self):
        try:
            validate_device("xenomorph")
            assert False, "Should have raised"
        except DeviceNotAvailableError:
            pass

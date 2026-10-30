from afora_vc.domain.classes import COCO_VEHICLE_CLASSES, resolve_vehicle_classes
from afora_vc.domain.counting.services import VehicleClassesResolver


class TestResolveVehicleClasses:
    def test_uses_model_class_ids_instead_of_coco(self):
        names = {
            0: "car",
            1: "motorcycle",
            2: "bus",
            3: "truck",
            4: "bicycle",
            5: "person",
        }

        assert resolve_vehicle_classes(names) == names

    def test_coco_names_include_person(self):
        names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            7: "truck",
        }

        assert resolve_vehicle_classes(names) == {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

    def test_falls_back_to_coco_when_model_has_no_known_classes(self):
        assert resolve_vehicle_classes(None) == COCO_VEHICLE_CLASSES
        assert resolve_vehicle_classes({0: "airplane"}) == COCO_VEHICLE_CLASSES


class TestVehicleClassesResolverDicts:
    def test_include_dict_trusts_the_configured_name(self):
        resolver = VehicleClassesResolver()

        # En COCO el id 5 es bus, pero el YAML declara explicitamente person.
        assert resolver.resolve_group_include({5: "person"}) == ["person"]

    def test_exclude_dict_trusts_the_configured_name(self):
        resolver = VehicleClassesResolver()

        assert resolver.resolve_group_exclude({0: "person"}) == {"person"}

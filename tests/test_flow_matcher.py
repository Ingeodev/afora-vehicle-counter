import csv

import pytest

from afora_vc.domain.counting.entities import (
    CountGroupConfig,
    VehiclePath,
    ZonePattern,
)
from afora_vc.domain.counting.services import (
    FlowMatcher,
    InterventionFileExporter,
    VehicleClassesResolver,
)

VEHICULOS = ["car", "motorcycle", "bus", "truck", "bicycle"]


def _grupo(nombre: str, patterns: list[list[str]], include="all", exclude="none"):
    return CountGroupConfig(
        name=nombre,
        access="sur",
        include=include,
        exclude=exclude,
        zone_paths=patterns,
    )


def _path(zones: list[str], vt: str = "car") -> VehiclePath:
    return VehiclePath(
        vehicle_id="v1",
        vehicle_type=vt,
        zones=zones,
        first_seen_seconds=0,
        last_seen_seconds=0,
        first_seen_str="08:00:00",
        last_seen_str="08:00:05",
    )


def _matchear(grupos: list[CountGroupConfig], path: VehiclePath) -> VehiclePath:
    return FlowMatcher(grupos, VehicleClassesResolver()).process([path])[0]


class TestZonePattern:
    @pytest.mark.parametrize(
        "zonas, patron, esperado",
        [
            # El patron literal exige match completo (nada de subsecuencia).
            (["A", "B", "C"], ["A", "B", "C"], True),
            (["A", "B", "C", "D"], ["A", "B", "C"], False),
            (["A", "B", "C"], ["A", "B", "C", "D"], False),
            # '_' es exactamente una zona.
            (["A", "B", "C"], ["_", "B", "C"], True),
            (["A", "B", "C", "D"], ["_", "B", "C"], False),
            (["A", "B", "C", "D"], ["_", "B", "C", "_"], True),
            (["L", "M", "B", "C", "I"], ["_", "B", "C", "_"], False),
            (["A", "B", "D"], ["A", "_", "D"], True),
            (["A", "L", "D"], ["A", "_", "D"], True),
            (["A", "D"], ["A", "_", "D"], False),
            # '+' es una o mas zonas, en cualquier posicion.
            (["L", "C", "D"], ["L", "+", "D"], True),
            (["L", "D", "F", "G", "D"], ["L", "+", "D"], True),
            (["L", "D"], ["L", "+", "D"], False),
            (["A", "D"], ["+", "D"], True),
            (["A", "F", "R", "D"], ["+", "D"], True),
            (["A", "B", "C", "D"], ["+", "D"], True),
            (["D"], ["+", "D"], False),
            (["A", "B", "C"], ["A", "+"], True),
            (["A", "B", "C", "D"], ["A", "+"], True),
            (["B", "C", "D"], ["A", "+"], False),
            (["A"], ["A", "+"], False),
            # '+' no cruza el inicio del patron.
            (["B", "C"], ["+", "B", "C"], False),
            (["A", "B", "C"], ["+", "B", "C"], True),
            (["L", "M", "B", "C"], ["+", "B", "C"], True),
        ],
    )
    def test_match_completo_con_comodines(self, zonas, patron, esperado):
        assert ZonePattern.compile(patron).matches(zonas) is esperado

    def test_no_permite_el_separador_en_una_zona(self):
        with pytest.raises(ValueError):
            ZonePattern.compile(["A,B", "C"])


class TestPrecedencia:
    def test_el_patron_literal_le_gana_al_comodin(self):
        grupos = [
            _grupo("amplio", [["+", "D"]]),
            _grupo("literal", [["A", "B", "C", "D"]]),
        ]
        assert _matchear(grupos, _path(["A", "B", "C", "D"])).flow_name == "literal"

    def test_gana_el_que_tiene_menos_comodines_de_secuencia(self):
        grupos = [
            _grupo("con_mas", [["+", "B", "C"]]),
            _grupo("con_una", [["_", "B", "C"]]),
        ]
        assert _matchear(grupos, _path(["A", "B", "C"])).flow_name == "con_una"

    def test_gana_el_que_fija_mas_posiciones(self):
        grupos = [
            _grupo("dos", [["_", "_", "D"]]),
            _grupo("tres", [["_", "_", "_", "D"]]),
        ]
        assert _matchear(grupos, _path(["A", "B", "C", "D"])).flow_name == "tres"

    def test_empate_lo_define_el_orden_del_yaml(self):
        # [A,+] y [+,D] son igual de especificos para [A,B,D].
        assert _matchear(
            [_grupo("primero", [["A", "+"]]), _grupo("segundo", [["+", "D"]])],
            _path(["A", "B", "D"]),
        ).flow_name == "primero"
        assert _matchear(
            [_grupo("primero", [["+", "D"]]), _grupo("segundo", [["A", "+"]])],
            _path(["A", "B", "D"]),
        ).flow_name == "primero"

    def test_los_ultimos_grupos_ya_no_quedan_sombreados(self):
        grupos = [
            _grupo("1", [["A"], ["A", "F"]]),
            _grupo("2", [["B", "E"], ["E", "B"]]),
            _grupo("8", [["C", "B", "A", "F"], ["C", "B", "A"], ["B", "A"]]),
            _grupo("92", [["B", "E", "D"], ["E", "B", "D"]]),
        ]
        assert _matchear(grupos, _path(["C", "B", "A", "F"])).flow_name == "8"
        assert _matchear(grupos, _path(["B", "E", "D"])).flow_name == "92"


class TestTiposDeVehiculo:
    def test_filtra_por_include_y_exclude(self):
        grupos = [
            _grupo("vehiculos", [["A", "+"]], include=VEHICULOS, exclude="person"),
        ]
        assert _matchear(grupos, _path(["A", "B"], "car")).flow_name == "vehiculos"
        assert _matchear(grupos, _path(["A", "B"], "person")).flow_name is None

    def test_persona_usa_su_propio_flujo(self):
        grupos = [
            _grupo("vehiculos", [["A", "+"]], include=VEHICULOS, exclude="person"),
            _grupo("personas", [["+", "E", "B"]], include="person",
                   exclude=VEHICULOS),
        ]
        assert _matchear(
            grupos, _path(["A", "F", "E", "B"], "person")
        ).flow_name == "personas"


class TestSinMatch:
    def test_devuelve_unknown_cuando_no_hay_patron(self):
        grupos = [_grupo("8", [["C", "B", "A", "F"]])]
        path = _matchear(grupos, _path(["B", "A", "F"]))
        assert path.flow_name is None
        assert path.flow_pattern is None

    def test_path_vacio_no_matchea(self):
        grupos = [_grupo("1", [["A"]])]
        assert _matchear(grupos, _path([])).flow_name is None


class TestPatronAsignado:
    def test_guarda_el_patron_que_matcheo(self):
        grupos = [_grupo("8", [["+", "B", "A", "F"]])]
        path = _matchear(grupos, _path(["D", "C", "B", "A", "F"]))
        assert path.flow_name == "8"
        assert path.flow_pattern == "[+,B,A,F]"


class TestPatronesMuertos:
    def test_reporta_los_patrones_que_no_matchearon(self):
        grupos = [
            _grupo("vivo", [["A"]]),
            _grupo("muerto", [["Z", "Z"]]),
        ]
        matcher = FlowMatcher(grupos, VehicleClassesResolver())
        matcher.process([_path(["A"])])
        assert matcher.dead_patterns() == [("muerto", "[Z,Z]")]


def test_reporte_de_trayectorias_sin_match(tmp_path):
    paths = [
        _path(["B"]),
        _path(["B"]),
        _path(["C", "D"], vt="truck"),
        _path(["A"]),
    ]
    paths[-1].flow_name = "1"
    paths[-1].flow_pattern = "[A]"

    InterventionFileExporter().process(paths, str(tmp_path))

    with open(tmp_path / "unmatched_paths.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0] == {
        "zones": "[B]",
        "vehicle_type": "car",
        "count": "2",
        "hora_min": "08:00:00",
        "hora_max": "08:00:05",
    }
    assert rows[1]["zones"] == "[C, D]"
    assert rows[1]["count"] == "1"

    with open(tmp_path / "detections.csv", newline="", encoding="utf-8") as f:
        detections = list(csv.DictReader(f))
    assert detections[-1]["patron"] == "[A]"
    assert detections[0]["patron"] == ""


def test_borra_el_reporte_viejo_si_todo_matcheo(tmp_path):
    (tmp_path / "unmatched_paths.csv").write_text("viejo")

    path = _path(["A"])
    path.flow_name = "1"
    InterventionFileExporter().process([path], str(tmp_path))

    assert not (tmp_path / "unmatched_paths.csv").exists()

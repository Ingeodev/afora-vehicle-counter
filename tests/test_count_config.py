import pytest

from afora_vc.application.batch_config import BatchConfig
from afora_vc.domain.exceptions import ConfigurationError


def _config(count: dict) -> BatchConfig:
    return BatchConfig.from_dict({"count": count})


def _grupo(zones: list[list[str]], name: str = "1") -> dict:
    return {"name": name, "access": "sur", "include": "all",
            "exclude": "none", "zones": zones}


class TestPatrones:
    def test_acepta_comodines(self):
        cfg = _config({"groups": [_grupo([["_", "B", "C"], ["A", "+"]])]})
        assert cfg.count.groups[0].zone_paths == [["_", "B", "C"], ["A", "+"]]

    def test_grupo_sin_zones(self):
        with pytest.raises(ConfigurationError, match="no tiene 'zones'"):
            _config({"groups": [{"name": "1", "access": "sur"}]})

    def test_patron_vacio(self):
        with pytest.raises(ConfigurationError, match="vacio o invalido"):
            _config({"groups": [_grupo([[]])]})

    def test_patron_sin_zonas_literales(self):
        with pytest.raises(ConfigurationError, match="no tiene zonas literales"):
            _config({"groups": [_grupo([["+", "_"]])]})

    def test_rechaza_el_comodin_de_asterisco(self):
        with pytest.raises(ConfigurationError, match="no son"):
            _config({"groups": [_grupo([["*", "B"]])]})

    def test_rechaza_el_comodin_de_porcentaje(self):
        with pytest.raises(ConfigurationError, match="no son"):
            _config({"groups": [_grupo([["%", "B"]])]})

    def test_rechaza_dos_mas_seguidos(self):
        with pytest.raises(ConfigurationError, match="repite"):
            _config({"groups": [_grupo([["A", "+", "+", "B"]])]})

    def test_rechaza_zona_con_coma(self):
        with pytest.raises(ConfigurationError, match="no puede contener"):
            _config({"groups": [_grupo([["A,B"]])]})

    def test_rechaza_patron_repetido_entre_grupos(self):
        with pytest.raises(ConfigurationError, match="repetido"):
            _config({"groups": [
                _grupo([["A", "B"]], name="1"),
                _grupo([["A", "B"]], name="2"),
            ]})

    def test_dos_guiones_bajos_siguen_siendo_validos(self):
        # [_,_] fija exactamente dos zonas; no es lo mismo que '+'.
        cfg = _config({"groups": [_grupo([["_", "_", "D"]])]})
        assert cfg.count.groups[0].zone_paths == [["_", "_", "D"]]


class TestClaves:
    def test_clave_desconocida_en_count(self):
        with pytest.raises(ConfigurationError, match="en 'count'"):
            _config({"grups": []})

    def test_clave_desconocida_en_un_grupo(self):
        with pytest.raises(ConfigurationError, match="en el grupo"):
            _config({"groups": [
                {"name": "1", "access": "sur", "zone": [["A"]]},
            ]})

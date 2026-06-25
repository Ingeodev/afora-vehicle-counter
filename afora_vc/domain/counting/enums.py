from enum import Enum


class Access(str, Enum):
    NORTE = "norte"
    SUR = "sur"
    ESTE = "este"
    OESTE = "oeste"

    @classmethod
    def display_order(cls) -> list[str]:
        return ["norte", "sur", "oeste", "este"]

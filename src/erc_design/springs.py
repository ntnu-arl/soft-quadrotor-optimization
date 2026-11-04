from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml


@dataclass(frozen=True)
class Spring:
    """Tension-spring data in SI units.

    Values are per physical spring. ``count`` is applied by the designer to
    build the equivalent parallel spring bundle used by one ERC.
    """

    spring_id: str
    max_length_m: float
    stiffness_n_per_m: float
    max_extension_m: float
    tension_at_rest_n: float = 0.0
    maximum_load_n: float | None = None
    count: int = 1
    description: str = ""
    material: str = ""
    wire_diameter_m: float | None = None
    external_diameter_m: float | None = None
    free_length_m: float | None = None

    @property
    def min_length_m(self) -> float:
        """Spring length at the lower end of the usable extension interval."""

        return self.max_length_m - self.max_extension_m

    @property
    def total_stiffness_n_per_m(self) -> float:
        return self.stiffness_n_per_m * self.count

    @property
    def total_tension_at_rest_n(self) -> float:
        return self.tension_at_rest_n * self.count

    @property
    def total_max_tension_n(self) -> float:
        if self.maximum_load_n is not None:
            return self.maximum_load_n * self.count
        return (
            self.tension_at_rest_n + self.stiffness_n_per_m * self.max_extension_m
        ) * self.count

    @property
    def spring_constant_n_per_m(self) -> float:
        """Catalog-facing alias for stiffness."""

        return self.stiffness_n_per_m

    def safe_extension_m(self, safety_factor: float) -> float:
        """Extension available after applying the MATLAB-style safety factor."""

        if safety_factor <= 0.0:
            raise ValueError("safety_factor must be positive")
        return self.max_length_m / safety_factor - self.min_length_m

    def energy_from_extension_j(self, extension_m: float) -> float:
        """Energy stored by the equivalent spring bundle from ``min_length_m``."""

        return (
            self.total_tension_at_rest_n * extension_m
            + 0.5 * self.total_stiffness_n_per_m * extension_m**2
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "Spring":
        spring_id = str(data.get("id", data.get("spring_id", "")))
        if not spring_id:
            raise ValueError("spring entry is missing 'id'")
        stiffness = data.get("spring_constant_n_per_m", data.get("stiffness_n_per_m"))
        if stiffness is None:
            raise ValueError(
                f"spring '{spring_id}' is missing 'spring_constant_n_per_m'"
            )
        return cls(
            spring_id=spring_id,
            max_length_m=float(data["max_length_m"]),
            stiffness_n_per_m=float(stiffness),
            max_extension_m=float(data["max_extension_m"]),
            tension_at_rest_n=float(data.get("tension_at_rest_n", 0.0)),
            maximum_load_n=_optional_float(data.get("maximum_load_n")),
            count=int(data.get("count", 1)),
            description=str(data.get("description", "")),
            material=str(data.get("material", "")),
            wire_diameter_m=_optional_float(data.get("wire_diameter_m")),
            external_diameter_m=_optional_float(data.get("external_diameter_m")),
            free_length_m=_optional_float(data.get("free_length_m")),
        )


class SpringCatalog:
    """Named spring collection loaded from YAML."""

    def __init__(self, springs: Iterable[Spring]):
        self._springs = {spring.spring_id: spring for spring in springs}
        if len(self._springs) == 0:
            raise ValueError("SpringCatalog requires at least one spring")

    def __iter__(self):
        return iter(self._springs.values())

    def __len__(self) -> int:
        return len(self._springs)

    def __getitem__(self, spring_id: str) -> Spring:
        return self._springs[spring_id]

    def get(self, spring_id: str) -> Spring | None:
        return self._springs.get(spring_id)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpringCatalog":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        entries = raw["springs"] if isinstance(raw, Mapping) else raw
        return cls(Spring.from_mapping(entry) for entry in entries)


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    return float(value)

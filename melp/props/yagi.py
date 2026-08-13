from __future__ import annotations

from pathlib import Path

import numpy as np
from manimlib import Group, VectorizedPoint, color_to_rgba

from .mesh import TriangleMeshMobject


YAGI_MODEL_PATH = Path(__file__).resolve().parent / "static" / "yagi_mesh.npz"
ELEMENT_NAMES = (
    "reflector",
    "driven_element",
    "director_1",
    "director_2",
    "director_3",
)

METAL_COLOR = "#AEB8BC"
DARK_METAL_COLOR = "#69747A"
PLASTIC_COLOR = "#252B2E"


class Yagi(Group):
    """Five-element Yagi with a right-facing boom and indexed reference points."""

    def __init__(self, boom_length=4, color=None, opacity=1, **kwargs):
        if boom_length <= 0:
            raise ValueError("boom_length must be positive")
        if not YAGI_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing Yagi mesh: {YAGI_MODEL_PATH}")

        with np.load(YAGI_MODEL_PATH) as mesh:
            points = mesh["points"]
            normals = mesh["normals"]
            names = [str(name) for name in mesh["part_names"]]
            starts = mesh["part_point_starts"]
            counts = mesh["part_point_counts"]
            self.parts = {
                name: self._make_part(
                    name,
                    points[start : start + count],
                    normals[start : start + count],
                    color,
                    opacity,
                )
                for name, start, count in zip(names, starts, counts)
            }
            reference_names = tuple(str(name) for name in mesh["reference_names"])
            reference_values = mesh["reference_points"]

        self.boom = self.parts["Boom"]
        self.reflector = self.parts["Element_4"]
        self.driven_element = Group(
            self.parts["Radiator"],
            self.parts["Radiator001"],
        )
        self.directors = Group(
            self.parts["Element_3"],
            self.parts["Element_2"],
            self.parts["Element_1"],
        )
        self.elements = Group(
            self.reflector,
            self.driven_element,
            *self.directors,
        )
        self.mount = self.parts["Plate"]
        self.cable = Group(self.parts["RG-58"], self.parts["RG-001"])

        self.reference_names = reference_names
        self._references_by_name = {
            name: VectorizedPoint(point)
            for name, point in zip(reference_names, reference_values)
        }
        self.reference_points = tuple(
            self._references_by_name[name] for name in reference_names
        )
        self.element_names = ELEMENT_NAMES
        self.element_points = tuple(
            self._references_by_name[name] for name in self.element_names
        )
        self.references = Group(*self.reference_points)
        self.body = Group(*self.parts.values())

        self.parts.update(
            {
                "body": self.body,
                "boom": self.boom,
                "elements": self.elements,
                "reflector": self.reflector,
                "driven_element": self.driven_element,
                "directors": self.directors,
                "mount": self.mount,
                "cable": self.cable,
            }
        )

        super().__init__(self.body, self.references, **kwargs)
        self.scale(boom_length, about_point=np.zeros(3))

    @staticmethod
    def _make_part(name, points, normals, color, opacity):
        part_color = color or Yagi._get_part_color(name)
        rgba = np.tile(color_to_rgba(part_color, opacity), (len(points), 1))
        return TriangleMeshMobject(points, normals, rgba)

    @staticmethod
    def _get_part_color(name):
        if name.startswith("End Cap") or name.startswith("Cap") or name == "RG-58":
            return PLASTIC_COLOR
        if (
            name.startswith("159734")
            or name.startswith("Radiator Hub")
            or name.startswith("Radiator Joint")
            or name.startswith("screw")
            or name == "M4 Washer DIN"
        ):
            return DARK_METAL_COLOR
        return METAL_COLOR

    def __getitem__(self, value):
        if isinstance(value, str):
            try:
                return self.parts[value]
            except KeyError:
                raise KeyError(
                    f"Unknown Yagi part {value!r}. Call get_parts() for the "
                    "available names."
                ) from None
        return super().__getitem__(value)

    def get_parts(self):
        return list(self.parts)

    def get_reference_point(self, reference):
        if isinstance(reference, str):
            try:
                anchor = self._references_by_name[reference]
            except KeyError:
                available = ", ".join(self.reference_names)
                raise KeyError(
                    f"Unknown Yagi reference {reference!r}. Available: {available}"
                ) from None
        else:
            anchor = self.reference_points[reference]
        return anchor.get_center()

    def get_element_point(self, index):
        return self.element_points[index].get_center()

    def get_boom_direction(self):
        direction = (
            self.get_reference_point("boom_front")
            - self.get_reference_point("boom_back")
        )
        return direction / np.linalg.norm(direction)

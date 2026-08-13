from pathlib import Path

import numpy as np
from manimlib import Group, RIGHT, VectorizedPoint, color_to_rgba

from ..animation_helpers import VT
from .mesh import TriangleMeshMobject

ROUTER_MODEL_PATH = Path(__file__).resolve().parent / "static" / "router_mesh.npz"
MODEL_SOURCE = "https://sketchfab.com/3d-models/698cb8a679c3403e97dc806e55ea3938"
MODEL_AUTHOR = "Jerovdl"


def _update_antenna_control(tracker):
    for ancestor in reversed(tracker.get_ancestors()):
        if isinstance(ancestor, Router):
            ancestor._update_antenna_angle()
            return


class Router(Group):
    """Texture-free router model with antennas that tilt out from its face."""

    def __init__(self, height=4, color="#87949A", opacity=1, **kwargs):
        if height <= 0:
            raise ValueError("height must be positive")
        if not ROUTER_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing router mesh: {ROUTER_MODEL_PATH}")

        rgba = color_to_rgba(color, opacity)
        with np.load(ROUTER_MODEL_PATH) as mesh:
            self.body = self._make_part(mesh, "body", rgba)
            self.left_antenna = self._make_part(mesh, "left_antenna", rgba)
            self.right_antenna = self._make_part(mesh, "right_antenna", rgba)

        self.antennas = Group(self.left_antenna, self.right_antenna)
        self._left_pivot = VectorizedPoint([-0.867, -0.6782, 0.0927])
        self._right_pivot = VectorizedPoint([0.867, -0.6773, 0.0927])
        self._axis_point = VectorizedPoint(self._left_pivot.get_center() + RIGHT)
        self.antenna_angle = VT(0)
        self._last_antenna_angle = 0
        self.references = Group(
            self._left_pivot,
            self._right_pivot,
            self._axis_point,
            self.antenna_angle,
        )

        super().__init__(self.body, self.antennas, self.references, **kwargs)
        self.center().set_height(height)
        self.antenna_angle.add_updater(_update_antenna_control, call=False)

    @staticmethod
    def _make_part(mesh, name, rgba):
        points = mesh[f"{name}_points"]
        normals = mesh[f"{name}_normals"]
        colors = np.tile(rgba, (len(points), 1))
        return TriangleMeshMobject(points, normals, colors)

    def get_antenna_axis(self):
        axis = self._axis_point.get_center() - self._left_pivot.get_center()
        return axis / np.linalg.norm(axis)

    def _update_antenna_angle(self):
        angle = ~self.antenna_angle
        delta = angle - self._last_antenna_angle
        if delta:
            axis = self.get_antenna_axis()
            self.left_antenna.rotate(
                delta,
                axis=axis,
                about_point=self._left_pivot.get_center(),
            )
            self.right_antenna.rotate(
                delta,
                axis=axis,
                about_point=self._right_pivot.get_center(),
            )
        self._last_antenna_angle = angle
        return self

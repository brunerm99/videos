from __future__ import annotations

import moderngl
import numpy as np
from manimlib import Mobject


class TriangleMeshMobject(Mobject):
    render_primitive = moderngl.TRIANGLES
    shader_folder = "surface"
    data_dtype = np.dtype(
        [
            ("point", np.float32, (3,)),
            ("du_point", np.float32, (3,)),
            ("dv_point", np.float32, (3,)),
            ("rgba", np.float32, (4,)),
        ]
    )
    pointlike_data_keys = ["point", "du_point", "dv_point"]

    def __init__(self, points, normals, rgba, **kwargs):
        self.mesh_points = np.asarray(points, dtype=np.float32)
        self.mesh_normals = np.asarray(normals, dtype=np.float32)
        self.mesh_rgba = np.asarray(rgba, dtype=np.float32)
        super().__init__(depth_test=True, shading=(0.25, 0.45, 0.2), **kwargs)

    def init_points(self):
        self.set_points(self.mesh_points)

        reference = np.zeros_like(self.mesh_normals)
        reference[:, 2] = 1
        nearly_vertical = np.abs(self.mesh_normals[:, 2]) > 0.9
        reference[nearly_vertical] = np.array([0, 1, 0])

        tangent_u = np.cross(reference, self.mesh_normals)
        tangent_u /= np.linalg.norm(tangent_u, axis=1, keepdims=True)
        tangent_v = np.cross(self.mesh_normals, tangent_u)

        self.data["du_point"][:] = self.mesh_points + tangent_u
        self.data["dv_point"][:] = self.mesh_points + tangent_v

    def init_colors(self):
        self.data["rgba"][:] = self.mesh_rgba

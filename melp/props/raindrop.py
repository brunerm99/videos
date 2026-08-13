import numpy as np
from manimlib import (
    BLUE,
    ORIGIN,
    OUT,
    RIGHT,
    TAU,
    UP,
    Group,
    ParametricSurface,
    VectorizedPoint,
    straight_path,
)

from ..animation_helpers import VT


MIN_D = 1.5
MAX_D = 6.0


def _validate_D(D):
    try:
        D = float(D)
    except (TypeError, ValueError) as exc:
        raise ValueError("D must be a finite number from 1.5 to 6 mm") from exc
    if not np.isfinite(D) or not MIN_D <= D <= MAX_D:
        raise ValueError("D must be a finite number from 1.5 to 6 mm")
    return D


def _c1(D):
    return (0.02914 * D**2 + 0.9263 * D + 0.07791) / np.pi


def _c2(D):
    return -0.01938 * D**2 + 0.4698 * D + 0.09538


def _c3(D):
    return -0.06123 * D**3 + 1.3880 * D**2 - 10.41 * D + 28.34


def _c4(D):
    if D > 4:
        return -0.01352 * D**3 + 0.2014 * D**2 - 0.8964 * D + 1.226
    return 0.0


def _half_width(D, y):
    c2 = _c2(D)
    y_over_c2 = np.clip(y / c2, -1.0, 1.0)
    term1 = np.sqrt(np.maximum(0.0, 1.0 - y_over_c2**2))
    term2 = np.arccos(np.clip(y / (_c3(D) * c2), -1.0, 1.0))
    term3 = _c4(D) * y_over_c2**2 + 1.0
    return _c1(D) * term1 * term2 * term3


def _update_D_control(tracker):
    for ancestor in reversed(tracker.get_ancestors()):
        if isinstance(ancestor, RainDrop):
            ancestor._update_D_geometry()
            return


class _DiameterTracker(VT):
    def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
        start_D = mobject1.get_value()
        target_D = mobject2.get_value()
        result = super().interpolate(mobject1, mobject2, alpha, path_func)
        if start_D == target_D:
            self.set_value(start_D)
        return result


class RainDrop(ParametricSurface):
    """Closed 3D surface from the Thurai equivalent-diameter drop profile."""

    def __init__(
        self,
        D=1.5,
        unit_size=1.0,
        color=BLUE,
        opacity=0.85,
        shading=(0.15, 0.35, 0.2),
        resolution=(96, 49),
        depth_test=True,
        **kwargs,
    ):
        D = _validate_D(D)
        unit_size = float(unit_size)
        if not np.isfinite(unit_size) or unit_size <= 0:
            raise ValueError("unit_size must be positive and finite")

        self.unit_size = unit_size
        self._shape_D = D
        self._last_D = D

        super().__init__(
            self._surface_point,
            u_range=(0, TAU),
            v_range=(-1, 1),
            resolution=resolution,
            color=color,
            opacity=opacity,
            shading=shading,
            depth_test=depth_test,
            **kwargs,
        )

        self._reference_scale = 1e-3
        self._origin_anchor = VectorizedPoint(ORIGIN)
        self._x_anchor = VectorizedPoint(self._reference_scale * RIGHT)
        self._y_anchor = VectorizedPoint(self._reference_scale * UP)
        self._z_anchor = VectorizedPoint(self._reference_scale * OUT)
        self.D = _DiameterTracker(D)
        self.references = Group(
            self._origin_anchor,
            self._x_anchor,
            self._y_anchor,
            self._z_anchor,
            self.D,
        )
        self.add(self.references)
        self.D.add_updater(_update_D_control, call=False)

    def _surface_point(self, azimuth, normalized_y):
        y = _c2(self._shape_D) * normalized_y
        radius = _half_width(self._shape_D, y)
        return self.unit_size * np.array(
            [
                radius * np.cos(azimuth),
                y,
                -radius * np.sin(azimuth),
            ]
        )

    def _get_reference_transform(self):
        origin = self._origin_anchor.get_center()
        basis = np.column_stack(
            [
                self._x_anchor.get_center() - origin,
                self._y_anchor.get_center() - origin,
                self._z_anchor.get_center() - origin,
            ]
        )
        return origin, basis / self._reference_scale

    def _update_D_geometry(self):
        D = _validate_D(~self.D)
        if D == self._last_D:
            return self

        origin, basis = self._get_reference_transform()
        self._shape_D = D
        self.init_points()
        for key in self.pointlike_data_keys:
            self.data[key][:] = self.data[key] @ basis.T + origin
        self._last_D = D
        return self

    def get_D(self):
        return float(~self.D)

    def set_D(self, D):
        self.D.set_value(_validate_D(D))
        return self._update_D_geometry()

from __future__ import annotations

import os
from collections.abc import Mapping
from itertools import count
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image
from manimlib import (
    Circle,
    DEG,
    Dot,
    Group,
    ImageMobject,
    Line,
    MED_SMALL_BUFF,
    ORIGIN,
    Rectangle,
    RIGHT,
    Text,
    UP,
    VGroup,
    VectorizedPoint,
)
from manimlib.mobject.mobject import Mobject

from ..animation_helpers import VT


__all__ = ["PPIGrid", "field_to_rgba", "polar_to_cartesian"]


_SHADER_DIR = Path(__file__).resolve().parent / "shaders" / "ppi_image"
_TEXTURE_TOKENS = count(1)
FONT = os.getenv("FONT", "")
_FIELD_KEYS = (
    "field_data",
    "data",
    "values",
    "reflectivity_dbz",
    "velocity_ms",
    "zdr_db",
    "rhohv",
    "phidp_deg",
    "kdp_deg_per_km",
)
_RESERVED_ARRAY_KEYS = {
    "valid_mask",
    "azimuth_grid_deg",
    "range_grid_km",
    "x_coords_km",
    "y_coords_km",
}


def _as_float_array(data):
    if np.ma.isMaskedArray(data):
        masked = np.ma.asarray(data, dtype=np.float32)
        values = np.ma.filled(masked, np.nan)
        valid_mask = ~np.ma.getmaskarray(masked)
    else:
        values = np.asarray(data, dtype=np.float32)
        valid_mask = np.ones(values.shape, dtype=bool)
    return values, valid_mask & np.isfinite(values)


def _as_valid_mask(valid_mask, shape):
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    if isinstance(valid_mask, tuple):
        result = np.zeros(shape, dtype=bool)
        result[valid_mask] = True
        return result
    result = np.asarray(valid_mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(
            f"valid_mask has shape {result.shape}; expected {shape}."
        )
    return result


def _cartesian_grids(shape, max_range_km):
    y_count, x_count = shape
    x_coords_km = np.linspace(
        -max_range_km,
        max_range_km,
        x_count,
        dtype=np.float32,
    )
    y_coords_km = np.linspace(
        max_range_km,
        -max_range_km,
        y_count,
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(x_coords_km, y_coords_km)
    range_grid_km = np.hypot(grid_x, grid_y).astype(np.float32)
    azimuth_grid_deg = (
        np.degrees(np.arctan2(grid_x, grid_y)) % 360.0
    ).astype(np.float32)
    return x_coords_km, y_coords_km, range_grid_km, azimuth_grid_deg


def _extract_field(data, field_key):
    if not isinstance(data, Mapping):
        return data, {}, field_key

    metadata = dict(data)
    resolved_key = field_key or metadata.get("field_key")
    if resolved_key is not None:
        if resolved_key not in metadata:
            raise KeyError(f"PPI field {resolved_key!r} is not in the metadata.")
        return metadata[resolved_key], metadata, resolved_key

    for candidate in _FIELD_KEYS:
        if candidate in metadata:
            return metadata[candidate], metadata, candidate

    candidates = []
    for key, value in metadata.items():
        if key in _RESERVED_ARRAY_KEYS:
            continue
        try:
            if np.asanyarray(value).ndim == 2:
                candidates.append(key)
        except (TypeError, ValueError):
            continue

    if len(candidates) == 1:
        resolved_key = candidates[0]
        return metadata[resolved_key], metadata, resolved_key
    if candidates:
        joined = ", ".join(repr(key) for key in candidates)
        raise KeyError(
            "More than one 2D field is present in the metadata "
            f"({joined}); pass field_key=... explicitly."
        )
    raise KeyError(
        "No 2D PPI field was found. Pass a 2D array or field_key=... "
        "for a metadata mapping."
    )


def _get_colormap(cmap_name, fallback_cmap_name="viridis"):
    if callable(cmap_name):
        return cmap_name

    from matplotlib import colormaps

    try:
        return colormaps[cmap_name]
    except KeyError:
        try:
            import cmweather  # noqa: F401
        except ImportError:
            pass
        else:
            try:
                return colormaps[cmap_name]
            except KeyError:
                pass

    try:
        return colormaps[fallback_cmap_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown colormap {cmap_name!r} and fallback "
            f"{fallback_cmap_name!r}."
        ) from exc


def field_to_rgba(
    field_data,
    *,
    valid_mask=None,
    vmin=None,
    vmax=None,
    cmap_name="turbo",
    fallback_cmap_name="viridis",
    min_alpha=1.0,
):
    """Map arbitrary scalar radar data to an RGBA uint8 image."""
    values, intrinsic_valid = _as_float_array(field_data)
    if values.ndim != 2:
        raise ValueError(f"Radar field must be 2D; got shape {values.shape}.")
    if not 0.0 <= min_alpha <= 1.0:
        raise ValueError("min_alpha must be between 0 and 1.")

    valid = intrinsic_valid & _as_valid_mask(valid_mask, values.shape)
    finite = values[valid]
    if vmin is None:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("vmin and vmax must be finite.")
    if vmax < vmin:
        raise ValueError("vmax must be greater than or equal to vmin.")
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    normalized = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    normalized[~valid] = 0.0
    rgba = np.asarray(
        _get_colormap(cmap_name, fallback_cmap_name)(normalized),
        dtype=float,
    )
    if rgba.shape != (*values.shape, 4):
        raise ValueError(
            "The colormap callable must return an RGBA array with shape "
            f"{(*values.shape, 4)}; got {rgba.shape}."
        )

    rgba = np.uint8(np.clip(rgba, 0.0, 1.0) * 255.0)
    rgba[..., 3] = 0
    rgba[..., 3][valid] = np.uint8(round(255 * min_alpha))
    return rgba


def _circular_delta_deg(first, second):
    return ((first - second + 180.0) % 360.0) - 180.0


def polar_to_cartesian(
    field_data,
    azimuths_deg,
    ranges_km,
    *,
    valid_mask=None,
    max_range_km=None,
    resolution=600,
    azimuth_tolerance_deg=None,
):
    """Rasterize native ``(ray, gate)`` data onto a north-up PPI grid.

    Azimuth follows the meteorological convention: 0 degrees is north/up and
    90 degrees is east/right. Sampling is nearest-ray and nearest-gate so the
    displayed pixels retain the source gates without interpolation.
    """
    values, intrinsic_valid = _as_float_array(field_data)
    azimuths_deg = np.asarray(azimuths_deg, dtype=np.float32).reshape(-1)
    ranges_km = np.asarray(ranges_km, dtype=np.float32).reshape(-1)

    expected_shape = (len(azimuths_deg), len(ranges_km))
    if values.shape != expected_shape:
        raise ValueError(
            "Polar radar data must have shape (len(azimuths_deg), "
            f"len(ranges_km)) = {expected_shape}; got {values.shape}."
        )
    if len(azimuths_deg) == 0 or len(ranges_km) == 0:
        raise ValueError("Polar radar coordinates cannot be empty.")
    if not np.all(np.isfinite(azimuths_deg)):
        raise ValueError("azimuths_deg must contain only finite values.")
    if not np.all(np.isfinite(ranges_km)):
        raise ValueError("ranges_km must contain only finite values.")
    if np.any(ranges_km < 0):
        raise ValueError("ranges_km cannot contain negative values.")
    if int(resolution) != resolution or resolution < 2:
        raise ValueError("resolution must be an integer of at least 2.")
    resolution = int(resolution)

    native_valid = intrinsic_valid & _as_valid_mask(valid_mask, values.shape)

    azimuth_order = np.argsort(azimuths_deg % 360.0)
    azimuths_deg = (azimuths_deg[azimuth_order] % 360.0).astype(np.float32)
    values = values[azimuth_order]
    native_valid = native_valid[azimuth_order]

    range_order = np.argsort(ranges_km)
    ranges_km = ranges_km[range_order]
    values = values[:, range_order]
    native_valid = native_valid[:, range_order]
    if len(ranges_km) > 1 and np.any(np.diff(ranges_km) <= 0):
        raise ValueError("ranges_km must contain unique gate positions.")

    if max_range_km is None:
        max_range_km = float(ranges_km[-1])
    max_range_km = float(max_range_km)
    if not np.isfinite(max_range_km) or max_range_km <= 0:
        raise ValueError("max_range_km must be positive and finite.")

    (
        x_coords_km,
        y_coords_km,
        range_grid_km,
        azimuth_grid_deg,
    ) = _cartesian_grids((resolution, resolution), max_range_km)

    right_index = np.searchsorted(azimuths_deg, azimuth_grid_deg, side="left")
    right_index %= len(azimuths_deg)
    left_index = (right_index - 1) % len(azimuths_deg)
    right_error = np.abs(
        _circular_delta_deg(azimuth_grid_deg, azimuths_deg[right_index])
    )
    left_error = np.abs(
        _circular_delta_deg(azimuth_grid_deg, azimuths_deg[left_index])
    )
    use_right = right_error < left_error
    ray_index = np.where(use_right, right_index, left_index)
    azimuth_error = np.where(use_right, right_error, left_error)

    if azimuth_tolerance_deg is None:
        if len(azimuths_deg) == 1:
            azimuth_tolerance_deg = 180.0
        else:
            spacings = np.diff(
                np.concatenate([azimuths_deg, [azimuths_deg[0] + 360.0]])
            )
            typical_spacings = np.sort(spacings)[:-1]
            if typical_spacings.size == 0:
                typical_spacings = spacings
            azimuth_tolerance_deg = max(
                0.5,
                1.5 * float(np.median(typical_spacings)),
            )
            azimuth_tolerance_deg = min(180.0, azimuth_tolerance_deg)
    azimuth_tolerance_deg = float(azimuth_tolerance_deg)
    if not 0 < azimuth_tolerance_deg <= 180:
        raise ValueError("azimuth_tolerance_deg must be in (0, 180].")

    if len(ranges_km) == 1:
        half_width = max(float(ranges_km[0]), max_range_km) / 2.0
        range_edges = np.array(
            [max(0.0, ranges_km[0] - half_width), ranges_km[0] + half_width]
        )
    else:
        midpoints = (ranges_km[:-1] + ranges_km[1:]) / 2.0
        first_half_width = (ranges_km[1] - ranges_km[0]) / 2.0
        last_half_width = (ranges_km[-1] - ranges_km[-2]) / 2.0
        range_edges = np.concatenate(
            [
                [max(0.0, ranges_km[0] - first_half_width)],
                midpoints,
                [ranges_km[-1] + last_half_width],
            ]
        )

    gate_index = np.searchsorted(range_edges, range_grid_km, side="right") - 1
    grid_valid = (
        (range_grid_km <= max_range_km)
        & (range_grid_km >= range_edges[0])
        & (gate_index >= 0)
        & (gate_index < len(ranges_km))
        & (azimuth_error <= azimuth_tolerance_deg)
    )

    cartesian = np.full((resolution, resolution), np.nan, dtype=np.float32)
    safe_gate_index = np.clip(gate_index, 0, len(ranges_km) - 1)
    sampled_valid = native_valid[ray_index, safe_gate_index]
    grid_valid &= sampled_valid
    cartesian[grid_valid] = values[ray_index[grid_valid], gate_index[grid_valid]]

    return {
        "field_data": cartesian,
        "field_key": "field_data",
        "valid_mask": grid_valid,
        "azimuth_grid_deg": azimuth_grid_deg,
        "range_grid_km": range_grid_km,
        "x_coords_km": x_coords_km,
        "y_coords_km": y_coords_km,
        "max_range_km": max_range_km,
    }


class _PPIImageMobject(ImageMobject):
    shader_folder = str(_SHADER_DIR)

    def __init__(
        self,
        rgba,
        *,
        height,
        azimuth_grid_deg=None,
        scan_start_deg=0.0,
        scan_progress_deg=360.0,
        data_opacity=1.0,
        **kwargs,
    ):
        rgba = np.asarray(rgba, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"RGBA image must have shape (H, W, 4); got {rgba.shape}.")
        self.rgba = np.ascontiguousarray(rgba)
        self.image = Image.fromarray(self.rgba, mode="RGBA")
        self.height = float(height)
        if azimuth_grid_deg is None:
            azimuth_grid_deg = np.zeros(self.rgba.shape[:2], dtype=np.float32)
        azimuth_grid_deg = np.asarray(azimuth_grid_deg, dtype=np.float32)
        if azimuth_grid_deg.shape != self.rgba.shape[:2]:
            raise ValueError(
                "azimuth_grid_deg must match the RGBA image shape "
                f"{self.rgba.shape[:2]}; got {azimuth_grid_deg.shape}."
            )
        if not np.all(np.isfinite(azimuth_grid_deg)):
            raise ValueError("azimuth_grid_deg must contain only finite values.")
        self.azimuth_fraction = np.ascontiguousarray(
            (azimuth_grid_deg % 360.0) / 360.0,
            dtype=np.float32,
        )
        self._texture_token = next(_TEXTURE_TOKENS)
        self._initial_scan_start_deg = float(scan_start_deg)
        self._initial_scan_progress_deg = float(scan_progress_deg)
        self._initial_data_opacity = float(data_opacity)
        Mobject.__init__(self, **kwargs)

    def init_uniforms(self):
        super().init_uniforms()
        self.uniforms.update(
            texture_token=self._texture_token,
            scan_start_deg=self._initial_scan_start_deg,
            scan_progress_deg=self._initial_scan_progress_deg,
            data_opacity=self._initial_data_opacity,
        )

    def init_shader_wrapper(self, ctx):
        super().init_shader_wrapper(ctx)
        texture = ctx.texture(
            size=self.image.size,
            components=4,
            data=self.image.tobytes(),
        )
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture.repeat_x = False
        texture.repeat_y = False
        self.shader_wrapper.add_texture("Texture", texture)

        azimuth_texture = ctx.texture(
            size=self.image.size,
            components=1,
            data=self.azimuth_fraction.tobytes(),
            dtype="f4",
        )
        azimuth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        azimuth_texture.repeat_x = False
        azimuth_texture.repeat_y = False
        self.shader_wrapper.add_texture("AzimuthTexture", azimuth_texture)

    def copy(self, deep=False):
        result = super().copy(deep=deep)
        result._texture_token = next(_TEXTURE_TOKENS)
        result.uniforms["texture_token"] = result._texture_token
        result.shader_wrapper = None
        return result

    def bind_trackers(self, scan_start, scan_progress, data_opacity):
        self._scan_start_tracker = scan_start
        self._scan_progress_tracker = scan_progress
        self._data_opacity_tracker = data_opacity
        if _update_ppi_image not in self.updaters:
            self.add_updater(_update_ppi_image, call=False)
        return _update_ppi_image(self)

    def set_display(self, scan_start_deg, scan_progress_deg, data_opacity):
        self.set_uniforms(
            {
                "scan_start_deg": float(scan_start_deg),
                "scan_progress_deg": float(scan_progress_deg),
                "data_opacity": float(data_opacity),
            }
        )
        return self


def _update_ppi_image(image):
    return image.set_display(
        ~image._scan_start_tracker,
        ~image._scan_progress_tracker,
        ~image._data_opacity_tracker,
    )


def _format_tick_label(value):
    if np.isclose(value, 0.0):
        value = 0.0
    if abs(value) >= 10 or np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _text(label, font, color):
    kwargs = {"color": color}
    if font:
        kwargs["font"] = font
    return Text(label, **kwargs)


class PPIGrid(Group):
    """A ManimGL PPI display for 2D rasters or native polar radar data.

    Cartesian rasters are displayed in their existing array order. Use
    :meth:`from_polar` to nearest-neighbor rasterize native ray/gate data.
    """

    def __init__(
        self,
        data,
        field_key=None,
        *,
        valid_mask=None,
        vmin=None,
        vmax=None,
        cmap_name=None,
        fallback_cmap_name="viridis",
        max_range_km=None,
        radius=2.0,
        range_rings=None,
        azimuth_step_deg=45,
        azimuth_grid_deg=None,
        show_cardinal_labels=True,
        show_colorbar=True,
        colorbar_ticks=None,
        colorbar_label=None,
        background_color="#09151D",
        axis_color="#D6EEF7",
        label_color="#B7D4E2",
        min_alpha=0.78,
        font=FONT,
        init_scan_progress=360.0,
        init_data_opacity=1.0,
        scan_start_deg=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        field_data, metadata, resolved_field_key = _extract_field(data, field_key)
        field_data, intrinsic_valid = _as_float_array(field_data)
        if field_data.ndim != 2:
            raise ValueError(f"PPIGrid expects a 2D field; got {field_data.shape}.")

        if max_range_km is None:
            max_range_km = metadata.get("max_range_km", 150.0)
        max_range_km = float(max_range_km)
        radius = float(radius)
        if max_range_km <= 0 or not np.isfinite(max_range_km):
            raise ValueError("max_range_km must be positive and finite.")
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError("radius must be positive and finite.")
        if int(azimuth_step_deg) != azimuth_step_deg or not 0 < azimuth_step_deg <= 360:
            raise ValueError("azimuth_step_deg must be an integer in [1, 360].")
        azimuth_step_deg = int(azimuth_step_deg)

        (
            default_x_coords,
            default_y_coords,
            default_range_grid,
            default_azimuth_grid,
        ) = _cartesian_grids(field_data.shape, max_range_km)
        if azimuth_grid_deg is None:
            azimuth_grid_deg = metadata.get("azimuth_grid_deg", default_azimuth_grid)
        azimuth_grid_deg = np.asarray(azimuth_grid_deg, dtype=np.float32)
        if azimuth_grid_deg.shape != field_data.shape:
            raise ValueError(
                "azimuth_grid_deg must match the field shape "
                f"{field_data.shape}; got {azimuth_grid_deg.shape}."
            )

        range_grid_km = np.asarray(
            metadata.get("range_grid_km", default_range_grid),
            dtype=np.float32,
        )
        if range_grid_km.shape != field_data.shape:
            raise ValueError(
                "range_grid_km must match the field shape "
                f"{field_data.shape}; got {range_grid_km.shape}."
            )

        if valid_mask is None:
            valid_mask = metadata.get("valid_mask")
        valid_mask = (
            intrinsic_valid
            & _as_valid_mask(valid_mask, field_data.shape)
            & (range_grid_km <= max_range_km)
        )

        finite = field_data[valid_mask]
        if vmin is None:
            vmin = metadata.get("vmin")
        if vmax is None:
            vmax = metadata.get("vmax")
        if vmin is None:
            vmin = float(np.nanmin(finite)) if finite.size else 0.0
        if vmax is None:
            vmax = float(np.nanmax(finite)) if finite.size else 1.0
        vmin = float(vmin)
        vmax = float(vmax)
        if vmax < vmin:
            raise ValueError("vmax must be greater than or equal to vmin.")
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        if cmap_name is None:
            cmap_name = metadata.get("cmap_name", "turbo")
        fallback_cmap_name = metadata.get(
            "fallback_cmap_name",
            fallback_cmap_name,
        )
        if not 0.0 <= min_alpha <= 1.0:
            raise ValueError("min_alpha must be between 0 and 1.")
        if not 0.0 <= init_data_opacity <= 1.0:
            raise ValueError("init_data_opacity must be between 0 and 1.")

        self.field_data = field_data
        self.field_key = resolved_field_key
        self.valid_mask = valid_mask
        self.azimuth_grid_deg = azimuth_grid_deg
        self.range_grid_km = range_grid_km
        self.x_coords_km = np.asarray(
            metadata.get("x_coords_km", default_x_coords),
            dtype=np.float32,
        )
        self.y_coords_km = np.asarray(
            metadata.get("y_coords_km", default_y_coords),
            dtype=np.float32,
        )
        if self.x_coords_km.shape != (field_data.shape[1],):
            raise ValueError(
                "x_coords_km must be a 1D coordinate for every image column; "
                f"got shape {self.x_coords_km.shape}."
            )
        if self.y_coords_km.shape != (field_data.shape[0],):
            raise ValueError(
                "y_coords_km must be a 1D coordinate for every image row; "
                f"got shape {self.y_coords_km.shape}."
            )
        self.max_range_km = max_range_km
        self.radius = radius
        self.vmin = vmin
        self.vmax = vmax
        self.cmap_name = cmap_name
        self.fallback_cmap_name = fallback_cmap_name
        self.min_alpha = float(min_alpha)
        self.font = font
        self.metadata = metadata

        rgba = field_to_rgba(
            field_data,
            valid_mask=valid_mask,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            fallback_cmap_name=fallback_cmap_name,
            min_alpha=min_alpha,
        )
        self.rgba = rgba

        self.scan_progress = VT(float(init_scan_progress))
        self.scan_start = VT(float(scan_start_deg))
        self.data_opacity_tracker = VT(float(init_data_opacity))

        self.background = Circle(
            radius=radius,
            fill_color=background_color,
            fill_opacity=1,
            stroke_width=0,
        ).set_z_index(0)
        self.border = Circle(radius=radius).set_stroke(
            axis_color,
            width=1.6,
            opacity=0.76,
        ).set_z_index(3)
        self.origin_marker = Dot(
            ORIGIN,
            radius=0.018 * radius,
            color=axis_color,
        ).set_z_index(3)
        self._coordinate_points = VGroup(
            VectorizedPoint(ORIGIN),
            VectorizedPoint(radius * RIGHT),
            VectorizedPoint(radius * UP),
        )

        self.data_image = _PPIImageMobject(
            rgba,
            height=radius * 2,
            azimuth_grid_deg=azimuth_grid_deg,
            scan_start_deg=~self.scan_start,
            scan_progress_deg=~self.scan_progress,
            data_opacity=~self.data_opacity_tracker,
        )
        self.data_image.stretch_to_fit_width(radius * 2)
        self.data_image.move_to(self.origin_marker).set_z_index(1)
        self.data_image.bind_trackers(
            self.scan_start,
            self.scan_progress,
            self.data_opacity_tracker,
        )

        self.grid_lines = VGroup()
        for azimuth in range(0, 360, azimuth_step_deg):
            theta = azimuth * DEG
            endpoint = radius * (np.sin(theta) * RIGHT + np.cos(theta) * UP)
            is_primary = azimuth % 90 == 0
            self.grid_lines.add(
                Line(
                    ORIGIN,
                    endpoint,
                    stroke_color=axis_color,
                    stroke_width=1.2 if is_primary else 0.8,
                    stroke_opacity=0.16 if is_primary else 0.08,
                )
            )
        self.grid_lines.set_z_index(2)

        if range_rings is None:
            range_rings = (
                max_range_km / 3.0,
                2.0 * max_range_km / 3.0,
                max_range_km,
            )
        self.range_rings = VGroup()
        for ring_km in range_rings:
            ring_km = float(ring_km)
            if not 0 < ring_km <= max_range_km:
                raise ValueError(
                    "Every range ring must be in (0, max_range_km]."
                )
            self.range_rings.add(
                Circle(radius=radius * ring_km / max_range_km).set_stroke(
                    axis_color,
                    width=1.1 if np.isclose(ring_km, max_range_km) else 0.8,
                    opacity=0.16 if np.isclose(ring_km, max_range_km) else 0.09,
                )
            )
        self.range_rings.set_z_index(2)

        self.cardinal_labels = VGroup()
        if show_cardinal_labels:
            for label, direction in (
                ("N", UP),
                ("E", RIGHT),
                ("S", -UP),
                ("W", -RIGHT),
            ):
                self.cardinal_labels.add(
                    _text(label, font, axis_color)
                    .scale(0.13 * radius)
                    .move_to(direction * radius * 1.09)
                )
        self.cardinal_labels.set_z_index(4)

        self.ppi = Group(
            self.background,
            self.data_image,
            self.grid_lines,
            self.range_rings,
            self.border,
            self.cardinal_labels,
            self.origin_marker,
            self._coordinate_points,
        )
        self.add(self.ppi)

        self.colorbar = Group()
        self.colorbar_ticks = VGroup()
        self.colorbar_label = None
        if show_colorbar:
            gradient = np.linspace(vmax, vmin, 420, dtype=np.float32)[:, None]
            colorbar_rgba = field_to_rgba(
                np.repeat(gradient, 28, axis=1),
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
                fallback_cmap_name=fallback_cmap_name,
                min_alpha=1.0,
            )
            colorbar_image = _PPIImageMobject(
                colorbar_rgba,
                height=radius * 1.4,
            )
            colorbar_frame = Rectangle(
                width=colorbar_image.get_width() + 0.08,
                height=colorbar_image.get_height() + 0.08,
                fill_opacity=0,
                stroke_color=axis_color,
                stroke_opacity=0.42,
                stroke_width=0.8,
            ).move_to(colorbar_image)
            self.colorbar = Group(colorbar_frame, colorbar_image).next_to(
                self.border,
                RIGHT,
                MED_SMALL_BUFF,
            )

            if colorbar_ticks is None:
                colorbar_ticks = np.linspace(vmin, vmax, 5)
            for tick in colorbar_ticks:
                tick = float(tick)
                tick_y = (
                    colorbar_image.get_bottom()[1]
                    + colorbar_image.get_height() * (tick - vmin) / (vmax - vmin)
                )
                tick_anchor = np.array(
                    [colorbar_image.get_right()[0], tick_y, 0.0]
                )
                tick_line = Line(
                    tick_anchor + RIGHT * 0.01,
                    tick_anchor + RIGHT * 0.07,
                    stroke_color=axis_color,
                    stroke_width=0.7,
                    stroke_opacity=0.6,
                )
                tick_text = _text(
                    _format_tick_label(tick),
                    font,
                    label_color,
                ).scale(0.105 * radius)
                tick_text.next_to(tick_line, RIGHT, buff=0.03)
                self.colorbar_ticks.add(VGroup(tick_line, tick_text))

            self.add(self.colorbar, self.colorbar_ticks)

            if colorbar_label is not None:
                self.colorbar_label = _text(
                    str(colorbar_label),
                    font,
                    label_color,
                ).scale(0.12 * radius)
                self.colorbar_label.next_to(self.colorbar, UP, buff=0.06)
                self.add(self.colorbar_label)

    def copy(self, deep=False):
        result = super().copy(deep=deep)
        result.scan_progress = self.scan_progress.copy(deep=deep)
        result.scan_start = self.scan_start.copy(deep=deep)
        result.data_opacity_tracker = self.data_opacity_tracker.copy(deep=deep)
        result.data_image.bind_trackers(
            result.scan_start,
            result.scan_progress,
            result.data_opacity_tracker,
        )
        return result

    @classmethod
    def from_polar(
        cls,
        field_data,
        azimuths_deg,
        ranges_km,
        *,
        valid_mask=None,
        max_range_km=None,
        resolution=600,
        azimuth_tolerance_deg=None,
        **kwargs,
    ):
        data = polar_to_cartesian(
            field_data,
            azimuths_deg,
            ranges_km,
            valid_mask=valid_mask,
            max_range_km=max_range_km,
            resolution=resolution,
            azimuth_tolerance_deg=azimuth_tolerance_deg,
        )
        return cls(data, **kwargs)

    def range_azimuth_to_point(self, range_km, azimuth_deg):
        center_point, east_point, north_point = self._coordinate_points
        center = center_point.get_center()
        east_vector = east_point.get_center() - center
        north_vector = north_point.get_center() - center
        theta = float(azimuth_deg) * DEG
        scaled_range = float(range_km) / self.max_range_km
        return center + scaled_range * (
            east_vector * np.sin(theta) + north_vector * np.cos(theta)
        )

    range_azimuth_to_graph_point = range_azimuth_to_point
    ra2gp = range_azimuth_to_point

    def value_to_rgba(self, value, *, alpha=None):
        alpha = self.min_alpha if alpha is None else float(alpha)
        values = np.asarray(value, dtype=np.float32)
        scalar = values.ndim == 0
        values = np.atleast_2d(values)
        rgba = field_to_rgba(
            values,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap_name=self.cmap_name,
            fallback_cmap_name=self.fallback_cmap_name,
            min_alpha=alpha,
        )
        return rgba[0, 0] if scalar else rgba

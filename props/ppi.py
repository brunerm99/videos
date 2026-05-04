import os

import numpy as np
from manim import *
from MF_Tools import VT

FONT = os.getenv("FONT", "")


def _blend_radar_rgba(base_rgba, next_rgba, update_mask):
    return np.where(update_mask[..., None], next_rgba, base_rgba)


def _nexrad_relative_sweep_progress_mask(
    azimuth_grid_deg, start_theta_deg, sweep_progress_deg
):
    relative_deg = (azimuth_grid_deg - start_theta_deg) % 360.0
    return relative_deg <= sweep_progress_deg


def _get_colormap(cmap_name, fallback_cmap_name):
    from matplotlib import colormaps

    try:
        return colormaps[cmap_name]
    except KeyError:
        return colormaps[fallback_cmap_name]


def _field_from_metadata(metadata, field_key):
    if field_key is not None:
        return metadata[field_key]

    for candidate in (
        "field_data",
        "reflectivity_dbz",
        "velocity_ms",
        "zdr_db",
        "rhohv",
        "phidp_deg",
        "kdp_deg_per_km",
    ):
        if candidate in metadata:
            return metadata[candidate]

    raise KeyError("No PPI field found. Pass field_key=... for this metadata dict.")


def _as_float_array(data):
    if np.ma.isMaskedArray(data):
        return np.ma.filled(data, np.nan).astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def _circular_mask(shape):
    y_count, x_count = shape
    y = np.linspace(1.0, -1.0, y_count, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, x_count, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x * grid_x + grid_y * grid_y <= 1.0


def _field_to_rgba(
    field_data,
    valid_mask,
    vmin,
    vmax,
    cmap_name,
    fallback_cmap_name,
    min_alpha,
):
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    normalized = np.clip((field_data - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _get_colormap(cmap_name, fallback_cmap_name)
    rgba = (cmap(normalized) * 255).astype(np.uint8)

    alpha = np.zeros(field_data.shape, dtype=np.uint8)
    alpha[valid_mask] = np.uint8(255 * min_alpha)
    rgba[..., 3] = alpha
    return rgba


def _colorbar_rgba(height_px, width_px, vmin, vmax, cmap_name, fallback_cmap_name):
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    gradient = np.linspace(vmax, vmin, height_px, dtype=np.float32)[:, None]
    cmap = _get_colormap(cmap_name, fallback_cmap_name)
    rgba = (cmap((gradient - vmin) / (vmax - vmin)) * 255).astype(np.uint8)
    rgba = np.repeat(rgba, width_px, axis=1)
    rgba[..., 3] = 255
    return rgba


def _polar_ppi_point(center, radius, max_range_km, azimuth_deg, range_km):
    scaled_radius = radius * (range_km / max_range_km)
    theta = azimuth_deg * DEGREES
    return (
        center
        + RIGHT * (scaled_radius * np.sin(theta))
        + UP * (scaled_radius * np.cos(theta))
    )


def _format_tick_label(value):
    if np.isclose(value, 0.0):
        value = 0.0
    if abs(value) >= 10 or np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:.1f}"


class PPIGrid(Group):
    def __init__(
        self,
        data,
        field_key=None,
        valid_mask=None,
        vmin=None,
        vmax=None,
        cmap_name="turbo",
        fallback_cmap_name="viridis",
        max_range_km=150.0,
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
        init_scan_progress=1,
        init_data_opacity=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(data, dict):
            metadata = data
            field_data = _field_from_metadata(metadata, field_key)
            if valid_mask is None:
                valid_mask = metadata.get("valid_mask")
            if vmin is None:
                vmin = metadata.get("vmin")
            if vmax is None:
                vmax = metadata.get("vmax")
            azimuth_grid_deg = metadata.get("azimuth_grid_deg")
            max_range_km = metadata.get("max_range_km", max_range_km)
        else:
            field_data = data

        if azimuth_grid_deg is None:
            raise ValueError("azimuth_grid_deg must be in dict or class arg")

        field_data = _as_float_array(field_data)
        if field_data.ndim != 2:
            raise ValueError("PPIGrid expects a 2D data array or metadata dict.")

        if valid_mask is None:
            valid_mask = np.isfinite(field_data) & _circular_mask(field_data.shape)
        elif isinstance(valid_mask, tuple):
            tuple_mask = np.zeros(field_data.shape, dtype=bool)
            tuple_mask[valid_mask] = True
            valid_mask = tuple_mask & np.isfinite(field_data)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(field_data)

        finite = field_data[valid_mask]
        if vmin is None:
            vmin = float(np.nanmin(finite)) if finite.size else 0.0
        if vmax is None:
            vmax = float(np.nanmax(finite)) if finite.size else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        if range_rings is None:
            range_rings = (
                max_range_km / 3.0,
                2.0 * max_range_km / 3.0,
                max_range_km,
            )
        if colorbar_ticks is None:
            colorbar_ticks = np.linspace(vmin, vmax, 2)

        plot_fill = ManimColor(background_color)
        axis_color = ManimColor(axis_color)
        label_color = ManimColor(label_color)

        rgba = _field_to_rgba(
            field_data,
            valid_mask,
            vmin,
            vmax,
            cmap_name,
            fallback_cmap_name,
            min_alpha,
        )

        self.background = Circle(
            radius=radius,
            fill_color=plot_fill,
            fill_opacity=1,
            stroke_width=0,
        )
        self.border = Circle(radius=radius).set_stroke(
            axis_color, width=1.6, opacity=0.76
        )
        self.origin_marker = Dot(ORIGIN, radius=0.018 * radius, color=axis_color)

        self.grid_lines = VGroup()
        for azimuth in range(0, 360, azimuth_step_deg):
            is_primary = azimuth % 90 == 0
            self.grid_lines.add(
                Line(
                    ORIGIN,
                    _polar_ppi_point(
                        ORIGIN,
                        radius,
                        max_range_km,
                        azimuth,
                        max_range_km,
                    ),
                    stroke_color=axis_color,
                    stroke_width=1.2 if is_primary else 0.8,
                    stroke_opacity=0.16 if is_primary else 0.08,
                )
            )

        self.range_rings = VGroup()
        for ring_km in range_rings:
            ring_radius = radius * ring_km / max_range_km
            self.range_rings.add(
                Circle(radius=ring_radius).set_stroke(
                    axis_color,
                    width=1.1 if np.isclose(ring_km, max_range_km) else 0.8,
                    opacity=0.16 if np.isclose(ring_km, max_range_km) else 0.09,
                )
            )

        self.cardinal_labels = VGroup()
        if show_cardinal_labels:
            for label, azimuth in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
                self.cardinal_labels.add(
                    Text(label, font=font, color=axis_color)
                    .scale(0.13 * radius)
                    .move_to(
                        _polar_ppi_point(
                            ORIGIN,
                            radius,
                            max_range_km,
                            azimuth,
                            max_range_km * 1.09,
                        )
                    )
                )

        self.scan_progress = VT(init_scan_progress)
        self.data_opacity_tracker = VT(init_data_opacity)

        blank_rgba = np.zeros_like(rgba)

        def current_scan_rgba():
            progress_mask = _nexrad_relative_sweep_progress_mask(
                azimuth_grid_deg,
                start_theta_deg=0.0,
                sweep_progress_deg=~self.scan_progress,
            )
            return _blend_radar_rgba(blank_rgba, rgba, progress_mask)

        def make_data_image():
            image = ImageMobject(current_scan_rgba(), image_mode="RGBA")
            image.scale_to_fit_height(radius * 2)
            image.move_to(self.origin_marker.get_center())
            image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            image.set_opacity(~self.data_opacity_tracker)
            return image

        # self.data_image = ImageMobject(rgba, image_mode="RGBA")
        # self.data_image.scale_to_fit_height(radius * 2)
        # self.data_image.move_to(ORIGIN)
        # self.data_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.data_image = always_redraw(make_data_image)

        self.ppi = Group(
            self.background,
            self.grid_lines,
            self.data_image,
            self.range_rings,
            self.border,
            self.cardinal_labels,
            self.origin_marker,
        )
        self.add(self.ppi)

        self.colorbar = Group()
        self.colorbar_ticks = VGroup()
        self.colorbar_label = None
        if show_colorbar:
            colorbar = ImageMobject(
                _colorbar_rgba(
                    height_px=420,
                    width_px=28,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_name=cmap_name,
                    fallback_cmap_name=fallback_cmap_name,
                ),
                image_mode="RGBA",
            ).scale_to_fit_height(radius * 1.4)
            colorbar.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

            colorbar_frame = Rectangle(
                width=colorbar.width + 0.08,
                height=colorbar.height + 0.08,
                fill_opacity=0,
                stroke_color=axis_color,
                stroke_opacity=0.42,
                stroke_width=0.8,
            ).move_to(colorbar)

            self.colorbar = Group(colorbar_frame, colorbar).next_to(
                self.cardinal_labels, RIGHT, MED_SMALL_BUFF
            )

            for tick in colorbar_ticks:
                tick_y = colorbar.get_bottom()[1] + colorbar.height * (
                    (tick - vmin) / (vmax - vmin)
                )
                tick_anchor = np.array([colorbar.get_right()[0], tick_y, 0])
                tick_line = Line(
                    tick_anchor + RIGHT * 0.01,
                    tick_anchor + RIGHT * 0.07,
                    stroke_color=axis_color,
                    stroke_width=0.7,
                    stroke_opacity=0.6,
                )
                tick_text = Text(
                    _format_tick_label(tick), font=font, color=label_color
                ).scale(0.105 * radius)
                tick_text.next_to(tick_line, RIGHT, buff=0.03)
                self.colorbar_ticks.add(VGroup(tick_line, tick_text))

            self.add(self.colorbar, self.colorbar_ticks)

            if colorbar_label is not None:
                self.colorbar_label = Text(
                    colorbar_label, font=font, color=label_color
                ).scale(0.12 * radius)
                self.colorbar_label.next_to(self.colorbar, UP, buff=0.06)
                self.add(self.colorbar_label)

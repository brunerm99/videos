from __future__ import annotations

from pathlib import Path

import numpy as np
from manimlib import Group, Mobject, ORIGIN, OUT, RIGHT, VectorizedPoint

from ..animation_helpers import VT
from .mesh import TriangleMeshMobject


MODEL_PATH = Path(__file__).resolve().parent / "static" / "wsr88_mesh.npz"


class WSR88Model(Group):
    parts: dict[str, Mobject]
    body: Group
    fixed_body: Group
    dome: Group
    upper_dome: Mobject
    lower_dome: Mobject
    radome_panels: tuple[Mobject, ...]
    radome_panels_indexed: bool
    antenna: Group
    feed_horn: Group
    feed_horn_mesh: Mobject
    base: Group
    base_sections: tuple[Mobject, ...]
    base_section_1: Mobject
    base_section_2: Mobject
    base_section_3: Mobject
    base_section_4: Mobject
    base_section_5: Mobject
    base_section_6: Mobject
    references: Group

    def get_scan_pivot(self):
        return self.get_elevation_pivot()

    def get_azimuth_pivot(self):
        return self._azimuth_pivot_anchor.get_center()

    def get_elevation_pivot(self):
        return self._elevation_pivot_anchor.get_center()

    def get_elevation_axis_direction(self):
        direction = (
            self._elevation_axis_anchor.get_center()
            - self._elevation_pivot_anchor.get_center()
        )
        return direction / np.linalg.norm(direction)

    def get_scan_up_direction(self):
        direction = (
            self._scan_up_anchor.get_center() - self._azimuth_pivot_anchor.get_center()
        )
        return direction / np.linalg.norm(direction)

    def get_scan_forward_direction(self):
        direction = (
            self._scan_forward_anchor.get_center()
            - self._azimuth_pivot_anchor.get_center()
        )
        return direction / np.linalg.norm(direction)

    def get_feed_horn_origin(self):
        return self._beam_origin_anchor.get_center()

    def get_feed_horn_direction(self):
        direction = (
            self._beam_axis_anchor.get_center() - self._beam_origin_anchor.get_center()
        )
        return direction / np.linalg.norm(direction)

    def get_beam_origin(self):
        return self.get_feed_horn_origin()

    def get_beam_direction(self):
        return self.get_feed_horn_direction()

    def get_dome_lift_vector(self):
        return (
            self._dome_raised_anchor.get_center()
            - self._dome_seated_anchor.get_center()
        )

    @property
    def beam_origin(self):
        return self.get_beam_origin()

    @property
    def beam_direction(self):
        return self.get_beam_direction()

    @property
    def feed_horn_origin(self):
        return self.get_feed_horn_origin()

    @property
    def feed_horn_direction(self):
        return self.get_feed_horn_direction()

    @property
    def dome_lift_vector(self):
        return self.get_dome_lift_vector()

    def set_base_section_count(self, count):
        if not 0 <= count <= len(self.base_sections):
            raise ValueError(
                f"count must be between 0 and {len(self.base_sections)}, got {count}"
            )
        for index, section in enumerate(self.base_sections):
            section.set_opacity(index < count)
        return self

    def get_scan_angles_to_point(self, point):
        direction = np.asarray(point, dtype=float) - self.get_scan_pivot()
        if np.linalg.norm(direction) < 1e-9:
            raise ValueError("Cannot track a point at the radar scan pivot")

        up = self.get_scan_up_direction()
        forward = self.get_scan_forward_direction()
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        forward = np.cross(left, up)

        forward_component = np.dot(direction, forward)
        left_component = np.dot(direction, left)
        up_component = np.dot(direction, up)
        az = np.arctan2(left_component, forward_component)
        el = np.arctan2(
            up_component,
            np.hypot(forward_component, left_component),
        )
        return az, el

    def point_at(self, point):
        az, el = self.get_scan_angles_to_point(point)
        self.az.set_value(az)
        self.el.set_value(el)
        self._update_scan_angles()
        return self

    def enable_scan_controls(self):
        azimuth_pivot = self.get_azimuth_pivot()
        up = self.get_scan_up_direction()
        beam_direction = self.get_beam_direction()

        initial_az = np.arctan2(beam_direction[1], beam_direction[0])
        self.antenna.rotate(-initial_az, axis=up, about_point=azimuth_pivot)

        beam_direction = self.get_beam_direction()
        horizontal = beam_direction - np.dot(beam_direction, up) * up
        horizontal /= np.linalg.norm(horizontal)
        initial_el = np.arctan2(
            np.dot(beam_direction, up),
            np.dot(beam_direction, horizontal),
        )
        self.antenna.rotate(
            -initial_el,
            axis=self.get_elevation_axis_direction(),
            about_point=self.get_elevation_pivot(),
        )

        self.az = VT(0)
        self.el = VT(0)
        self._last_scan_az = 0.0
        self._last_scan_el = 0.0
        self.add_updater(WSR88Model._update_scan_angles)
        return self

    def _update_scan_angles(self):
        az = ~self.az
        el = ~self.el
        az_delta = az - self._last_scan_az
        el_delta = el - self._last_scan_el
        azimuth_pivot = self.get_azimuth_pivot()
        up = self.get_scan_up_direction()

        if az_delta:
            self.antenna.rotate(
                az_delta,
                axis=up,
                about_point=azimuth_pivot,
            )

        if el_delta:
            self.antenna.rotate(
                el_delta,
                axis=self.get_elevation_axis_direction(),
                about_point=self.get_elevation_pivot(),
            )

        self._last_scan_az = az
        self._last_scan_el = el
        return self


def get_wsr88_model(
    height=5.6,
    dome_raised=False,
    base_section_count=None,
    indexed_radome_panels=False,
):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run melp/props/export_wsr88_mesh.py "
            "through Blender first."
        )

    with np.load(MODEL_PATH) as mesh:
        points = mesh["points"]
        normals = mesh["normals"]
        rgba = mesh["rgba"]
        object_names = [str(name) for name in mesh["included_objects"]]
        object_starts = mesh["object_point_starts"]
        object_counts = mesh["object_point_counts"]
        dome_part_names = {str(name) for name in mesh["dome_part_names"]}
        antenna_part_names = {str(name) for name in mesh["antenna_part_names"]}
        base_part_names = [str(name) for name in mesh["base_part_names"]]
        upper_dome_part_name = (
            str(mesh["upper_dome_part_name"].item())
            if "upper_dome_part_name" in mesh
            else "Icosphere.001"
        )
        lower_dome_part_name = (
            str(mesh["lower_dome_part_name"].item())
            if "lower_dome_part_name" in mesh
            else "Icosphere"
        )
        feed_horn_part_name = str(mesh["feed_horn_part_name"].item())
        beam_origin = mesh["beam_origin"].copy()
        beam_direction = mesh["beam_direction"].copy()
        elevation_pivot = mesh["elevation_pivot"].copy()
        elevation_axis = mesh["elevation_axis"].copy()
        dome_lift_vector = mesh["dome_lift_vector"].copy()
        reference_model_depth = float(mesh["reference_model_depth"])

        indexed_shell_names = {
            upper_dome_part_name,
            lower_dome_part_name,
        }
        parts = {}
        for name, start, count in zip(object_names, object_starts, object_counts):
            if indexed_radome_panels and name in indexed_shell_names:
                continue
            end = start + count
            parts[name] = TriangleMeshMobject(
                points[start:end],
                normals[start:end],
                rgba[start:end],
            )

        if indexed_radome_panels:
            panel_keys = (
                "radome_panel_object_names",
                "radome_panel_polygon_indices",
                "radome_panel_point_starts",
                "radome_panel_point_counts",
                "upper_dome_part_name",
                "lower_dome_part_name",
            )
            missing_panel_keys = [key for key in panel_keys if key not in mesh]
            if missing_panel_keys:
                missing = ", ".join(missing_panel_keys)
                raise ValueError(
                    "The WSR-88 mesh does not contain indexed radome panels "
                    f"({missing}). Re-export it with "
                    "melp/props/export_wsr88_mesh.py."
                )

            panel_object_names = [
                str(name) for name in mesh["radome_panel_object_names"]
            ]
            panel_polygon_indices = mesh["radome_panel_polygon_indices"]
            panel_point_starts = mesh["radome_panel_point_starts"]
            panel_point_counts = mesh["radome_panel_point_counts"]
            panel_metadata_lengths = {
                len(panel_object_names),
                len(panel_polygon_indices),
                len(panel_point_starts),
                len(panel_point_counts),
            }
            if len(panel_metadata_lengths) != 1:
                raise ValueError("Radome panel metadata arrays have different lengths")

            for shell_name in indexed_shell_names:
                shell_panels = []
                shell_polygon_ids = []
                for object_name, polygon_id, start, count in zip(
                    panel_object_names,
                    panel_polygon_indices,
                    panel_point_starts,
                    panel_point_counts,
                ):
                    if object_name != shell_name:
                        continue
                    start = int(start)
                    count = int(count)
                    end = start + count
                    if count <= 0 or count % 3 != 0:
                        raise ValueError(
                            f"Radome panel {polygon_id} on {shell_name!r} has "
                            f"an invalid point count: {count}"
                        )
                    if start < 0 or end > len(points):
                        raise ValueError(
                            f"Radome panel {polygon_id} on {shell_name!r} "
                            "points outside the exported mesh: "
                            f"[{start}, {end})"
                        )
                    shell_panels.append(
                        TriangleMeshMobject(
                            points[start:end],
                            normals[start:end],
                            rgba[start:end],
                        )
                    )
                    shell_polygon_ids.append(int(polygon_id))

                if not shell_panels:
                    raise ValueError(
                        f"No indexed radome panels were exported for {shell_name!r}"
                    )
                shell = Group(*shell_panels)
                shell.panels = tuple(shell_panels)
                shell.polygon_ids = tuple(shell_polygon_ids)
                parts[shell_name] = shell

        upper_dome = parts[upper_dome_part_name]
        lower_dome = parts[lower_dome_part_name]
        if not indexed_radome_panels:
            upper_dome.panels = ()
            upper_dome.polygon_ids = ()
            lower_dome.panels = ()
            lower_dome.polygon_ids = ()

        beam_origin_anchor = VectorizedPoint(beam_origin)
        beam_axis_anchor = VectorizedPoint(beam_origin + 0.05 * beam_direction)
        elevation_pivot_anchor = VectorizedPoint(elevation_pivot)
        elevation_axis_anchor = VectorizedPoint(elevation_pivot + 0.05 * elevation_axis)
        feed_horn_mesh = parts[feed_horn_part_name]
        feed_horn = Group(feed_horn_mesh, beam_origin_anchor, beam_axis_anchor)

        antenna_members = []
        for name in object_names:
            if name not in antenna_part_names:
                continue
            if name == feed_horn_part_name:
                antenna_members.append(feed_horn)
            else:
                antenna_members.append(parts[name])
        antenna = Group(
            *antenna_members,
            elevation_pivot_anchor,
            elevation_axis_anchor,
        )

        base_sections = tuple(parts[name] for name in base_part_names)
        base = Group(*base_sections)
        dome = Group(*(parts[name] for name in object_names if name in dome_part_names))
        fixed_part_names = [
            name
            for name in object_names
            if name not in dome_part_names
            and name not in antenna_part_names
            and name not in base_part_names
        ]
        fixed_body = Group(*(parts[name] for name in fixed_part_names))
        body = Group(fixed_body, antenna, base)

        azimuth_pivot = fixed_body.get_center()
        dome_seated_anchor = VectorizedPoint(ORIGIN)
        dome_raised_anchor = VectorizedPoint(dome_lift_vector)
        azimuth_pivot_anchor = VectorizedPoint(azimuth_pivot)
        scan_up_anchor = VectorizedPoint(azimuth_pivot + 0.05 * OUT)
        scan_forward_anchor = VectorizedPoint(azimuth_pivot + 0.05 * RIGHT)
        references = Group(
            dome_seated_anchor,
            dome_raised_anchor,
            azimuth_pivot_anchor,
            scan_up_anchor,
            scan_forward_anchor,
        )
        model = WSR88Model(body, dome, references)

        model.parts = parts
        model.body = body
        model.fixed_body = fixed_body
        model.dome = dome
        model.dome.parts = {
            name: parts[name] for name in object_names if name in dome_part_names
        }
        model.upper_dome = upper_dome
        model.lower_dome = lower_dome
        model.dome.panels = upper_dome.panels
        model.dome.polygon_ids = upper_dome.polygon_ids
        model.radome_panels = upper_dome.panels + lower_dome.panels
        model.radome_panels_indexed = indexed_radome_panels
        model.antenna = antenna
        model.antenna.parts = {
            name: parts[name] for name in object_names if name in antenna_part_names
        }
        model.feed_horn = feed_horn
        model.feed_horn_mesh = feed_horn_mesh
        model.base = base
        model.base.parts = {name: parts[name] for name in base_part_names}
        model.base_sections = base_sections
        for index, section in enumerate(base_sections, start=1):
            setattr(model, f"base_section_{index}", section)

        model.references = references
        model._beam_origin_anchor = beam_origin_anchor
        model._beam_axis_anchor = beam_axis_anchor
        model._elevation_pivot_anchor = elevation_pivot_anchor
        model._elevation_axis_anchor = elevation_axis_anchor
        model._dome_seated_anchor = dome_seated_anchor
        model._dome_raised_anchor = dome_raised_anchor
        model._azimuth_pivot_anchor = azimuth_pivot_anchor
        model._scan_pivot_anchor = azimuth_pivot_anchor
        model._scan_up_anchor = scan_up_anchor
        model._scan_forward_anchor = scan_forward_anchor

    model.dome.shift(-dome_lift_vector)

    scale_factor = height / reference_model_depth
    model.scale(scale_factor, about_point=ORIGIN)
    if dome_raised:
        model.dome.shift(model.dome_lift_vector)
    if base_section_count is not None:
        model.set_base_section_count(base_section_count)
    model.enable_scan_controls()
    return model


def place_wsr88_model(height, shift, dome_raised=False):
    radar = get_wsr88_model(height=height, dome_raised=dome_raised)
    azimuth = np.arctan2(radar.beam_direction[1], radar.beam_direction[0])
    rotation = -azimuth

    radar.rotate(rotation, axis=OUT, about_point=ORIGIN)
    radar.shift(shift)
    return radar, radar.beam_origin


def place_wsr88_on_terrain(site, terrain, height=2.25, rotation: float = 0.0):
    site_x, site_y = site
    radar = get_wsr88_model(height=height)
    radar.rotate(rotation, axis=OUT, about_point=ORIGIN)
    base_z = radar.get_bounding_box()[0, 2]
    radar.shift(
        np.array(
            [
                site_x,
                site_y,
                terrain.height_at(site_x, site_y) - base_z + 0.015,
            ]
        )
    )
    return radar

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import bpy
import numpy as np


MODEL_COLORS = {
    "tower_light": np.array([0.75, 0.82, 0.85, 1.0], dtype=np.float32),
    "tower_mid": np.array([0.49, 0.59, 0.64, 1.0], dtype=np.float32),
    "tower_dark": np.array([0.25, 0.34, 0.38, 1.0], dtype=np.float32),
    "dish": np.array([0.91, 0.95, 0.96, 1.0], dtype=np.float32),
    "radome": np.array([0.82, 0.88, 0.90, 1.0], dtype=np.float32),
}

EXCLUDED_OBJECTS = {"Plane", "Cube.045"}
UPPER_DOME_OBJECT = "Icosphere.001"
LOWER_DOME_OBJECT = "Icosphere"
ANTENNA_ROOT_OBJECT = "UP_DOWN"
FEED_HORN_OBJECT = "Cylinder.030"
ELEVATION_TRUNNION_OBJECT = "Cylinder.021"
BASE_OBJECT = "Cylinder.088"
BASE_SECTION_Z_EDGES = np.array(
    [
        -0.849642,
        -6.059300,
        -11.229366,
        -16.387077,
        -21.544791,
        -26.709178,
        -32.017429,
    ],
    dtype=np.float32,
)
REFERENCE_LOWER_Z = -10.23


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--clip-z", type=float, default=None)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def get_face_color(obj, material_name):
    collection_names = " ".join(collection.name for collection in obj.users_collection)
    descriptor = f"{obj.name} {material_name} {collection_names}".lower()

    if "dome" in descriptor or "icosphere" in descriptor:
        return MODEL_COLORS["radome"]
    if "dishbar" in descriptor or "screw" in descriptor or "used metal" in descriptor:
        return MODEL_COLORS["tower_mid"]
    if "dish" in descriptor:
        return MODEL_COLORS["dish"]
    if "base" in descriptor or "metal.007" in descriptor:
        return MODEL_COLORS["tower_dark"]
    if "pole" in descriptor or "metal" in descriptor:
        return MODEL_COLORS["tower_mid"]
    return MODEL_COLORS["tower_light"]


def clip_polygon_to_z(polygon, clip_z, keep_above):
    clipped = []

    for current, following in zip(polygon, polygon[1:] + polygon[:1]):
        current_point, current_normal = current
        next_point, next_normal = following
        if keep_above:
            current_inside = current_point[2] >= clip_z
            next_inside = next_point[2] >= clip_z
        else:
            current_inside = current_point[2] <= clip_z
            next_inside = next_point[2] <= clip_z

        if current_inside and next_inside:
            clipped.append(following)
            continue

        if current_inside == next_inside:
            continue

        alpha = (clip_z - current_point[2]) / (next_point[2] - current_point[2])
        point = current_point + alpha * (next_point - current_point)
        normal = current_normal + alpha * (next_normal - current_normal)
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal /= normal_length
        intersection = (point, normal)

        if current_inside:
            clipped.append(intersection)
        else:
            clipped.extend((intersection, following))

    return clipped


def clip_triangle_to_z_range(vertices, normals, min_z=None, max_z=None):
    polygon = list(zip(vertices, normals))
    if min_z is not None:
        polygon = clip_polygon_to_z(polygon, min_z, keep_above=True)
    if max_z is not None and polygon:
        polygon = clip_polygon_to_z(polygon, max_z, keep_above=False)
    return polygon


def get_radome_panel_group_indices(mesh):
    vertex_points = np.array([vertex.co for vertex in mesh.vertices], dtype=float)
    design = np.column_stack((2 * vertex_points, np.ones(len(vertex_points))))
    target = np.sum(vertex_points**2, axis=1)
    dome_center = np.linalg.lstsq(design, target, rcond=None)[0][:3]

    face_centers = np.array([polygon.center for polygon in mesh.polygons], dtype=float)
    face_normals = np.array([polygon.normal for polygon in mesh.polygons], dtype=float)
    radial_directions = face_centers - dome_center
    radial_directions /= np.linalg.norm(radial_directions, axis=1, keepdims=True)
    surface_faces = set(
        np.flatnonzero(np.abs(np.sum(face_normals * radial_directions, axis=1)) > 0.8)
    )

    edge_faces = {}
    for polygon in mesh.polygons:
        for edge_key in polygon.edge_keys:
            edge_faces.setdefault(tuple(sorted(edge_key)), []).append(polygon.index)

    adjacency = [set() for _ in mesh.polygons]
    for face_indices in edge_faces.values():
        for first in face_indices:
            adjacency[first].update(
                second for second in face_indices if second != first
            )

    surface_components = []
    remaining = set(surface_faces)
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current] & remaining & surface_faces:
                remaining.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        surface_components.append(component)

    if len(surface_components) < 2:
        raise RuntimeError("Unable to find logical panels in the radome mesh")

    shared_seam_and_backing_component = max(
        surface_components,
        key=lambda component: (len(component), -min(component)),
    )
    panel_components = [
        component
        for component in surface_components
        if component is not shared_seam_and_backing_component
    ]
    if any(len(component) != 1 for component in panel_components):
        component_sizes = sorted(len(component) for component in panel_components)
        raise RuntimeError(
            "Expected each logical radome panel face to be isolated; "
            f"got component sizes {component_sizes}"
        )

    panel_seeds = sorted(next(iter(component)) for component in panel_components)
    distances = np.full(len(mesh.polygons), np.inf)
    owners = np.full(len(mesh.polygons), -1, dtype=np.int64)
    queue = []
    for panel_index, face_index in enumerate(panel_seeds):
        distances[face_index] = 0.0
        owners[face_index] = panel_index
        heapq.heappush(queue, (0.0, panel_index, face_index))

    while queue:
        distance, panel_index, face_index = heapq.heappop(queue)
        if distance != distances[face_index] or panel_index != owners[face_index]:
            continue

        for neighbor in adjacency[face_index]:
            edge_length = np.linalg.norm(
                face_centers[face_index] - face_centers[neighbor]
            )
            candidate = (distance + edge_length, panel_index)
            existing = (distances[neighbor], owners[neighbor])
            if candidate >= existing:
                continue
            distances[neighbor], owners[neighbor] = candidate
            heapq.heappush(queue, (candidate[0], panel_index, neighbor))

    if np.any(owners < 0):
        missing = np.flatnonzero(owners < 0).tolist()
        raise RuntimeError(f"Radome faces are disconnected from all panels: {missing}")

    for panel_index, face_index in enumerate(panel_seeds):
        panel_faces = set(np.flatnonzero(owners == panel_index))
        reached = {face_index}
        stack = [face_index]
        while stack:
            current = stack.pop()
            for neighbor in (adjacency[current] & panel_faces) - reached:
                reached.add(neighbor)
                stack.append(neighbor)
        if reached != panel_faces:
            raise RuntimeError(f"Logical radome panel {panel_index} is disconnected")

    return owners


def evaluated_triangles(obj, depsgraph, min_z=None, max_z=None):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    if mesh is None:
        return [], [], [], []

    mesh.calc_loop_triangles()
    matrix_world = evaluated.matrix_world
    normal_matrix = matrix_world.to_3x3().inverted().transposed()
    orientation = -1.0 if matrix_world.to_3x3().determinant() < 0 else 1.0
    points = []
    normals = []
    colors = []
    polygon_point_ranges = []
    panel_group_indices = (
        get_radome_panel_group_indices(mesh)
        if obj.name in {UPPER_DOME_OBJECT, LOWER_DOME_OBJECT}
        else None
    )

    triangles_by_polygon = {polygon.index: [] for polygon in mesh.polygons}
    for triangle in mesh.loop_triangles:
        triangles_by_polygon[triangle.polygon_index].append(triangle)

    for polygon in mesh.polygons:
        polygon_start = len(points)
        material_name = ""
        if polygon.material_index < len(obj.material_slots):
            material = obj.material_slots[polygon.material_index].material
            material_name = material.name if material is not None else ""
        face_color = get_face_color(obj, material_name)

        for triangle in triangles_by_polygon[polygon.index]:
            vertices = np.array(
                [matrix_world @ mesh.vertices[index].co for index in triangle.vertices],
                dtype=np.float32,
            )
            vertex_normals = np.array(
                [
                    orientation * (normal_matrix @ mesh.loops[index].normal)
                    for index in triangle.loops
                ],
                dtype=np.float32,
            )
            vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True)

            clipped = clip_triangle_to_z_range(
                vertices,
                vertex_normals,
                min_z,
                max_z,
            )
            if len(clipped) < 3:
                continue

            for index in range(1, len(clipped) - 1):
                triangle_vertices = (clipped[0], clipped[index], clipped[index + 1])
                for point, normal in triangle_vertices:
                    points.append(point)
                    normals.append(normal)
                    colors.append(face_color)

        polygon_count = len(points) - polygon_start
        if polygon_count:
            polygon_point_ranges.append(
                (
                    polygon.index,
                    -1
                    if panel_group_indices is None
                    else panel_group_indices[polygon.index],
                    polygon_start,
                    polygon_count,
                )
            )

    evaluated.to_mesh_clear()
    return points, normals, colors, polygon_point_ranges


def get_dish_reference(scene, depsgraph):
    dish = scene.objects.get("Circle")
    if dish is None:
        raise RuntimeError("Unable to find the dish object named 'Circle'.")

    evaluated = dish.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    points = np.array(
        [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices],
        dtype=np.float32,
    )
    evaluated.to_mesh_clear()

    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    direction = axes[-1]
    if direction[0] > 0:
        direction *= -1
    return center, direction


def get_feed_horn_reference(scene, depsgraph):
    feed_horn = scene.objects.get(FEED_HORN_OBJECT)
    if feed_horn is None:
        raise RuntimeError(
            f"Unable to find the feed horn object named {FEED_HORN_OBJECT!r}."
        )

    _, dish_direction = get_dish_reference(scene, depsgraph)
    evaluated = feed_horn.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    points = np.array(
        [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices],
        dtype=np.float32,
    )

    adjacency = [set() for _ in points]
    for edge in mesh.edges:
        first, second = edge.vertices
        adjacency[first].add(second)
        adjacency[second].add(first)

    components = []
    remaining = set(range(len(points)))
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(np.fromiter(component, dtype=np.int64))

    head_indices = max(
        components,
        key=lambda indices: np.mean(points[indices] @ dish_direction),
    )
    head_points = points[head_indices]
    head_center = head_points.mean(axis=0)
    _, _, axes = np.linalg.svd(head_points - head_center, full_matrices=False)
    direction = axes[-1]
    if np.dot(direction, dish_direction) < 0:
        direction *= -1

    axial_positions = head_points @ direction
    axial_range = np.ptp(axial_positions)
    front_tolerance = max(1e-5, 1e-3 * axial_range)
    front_points = head_points[
        axial_positions >= axial_positions.max() - front_tolerance
    ]
    origin = front_points.mean(axis=0)

    evaluated.to_mesh_clear()
    return origin, direction


def get_elevation_reference(scene, depsgraph):
    name = ELEVATION_TRUNNION_OBJECT
    obj = scene.objects.get(name)
    if obj is None:
        raise RuntimeError(f"Unable to find the object named {name!r}.")

    evaluated = obj.evaluated_get(depsgraph)
    matrix = np.array(evaluated.matrix_world, dtype=np.float32)
    pivot = matrix[:3, 3].copy()
    axis = matrix[:3, :3] @ np.array([0, 0, 1], dtype=np.float32)
    axis /= np.linalg.norm(axis)
    return pivot, axis


def main():
    args = parse_args()
    scene = bpy.context.scene
    if args.frame is not None:
        scene.frame_set(args.frame)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    beam_origin, beam_direction = get_feed_horn_reference(scene, depsgraph)

    upper_dome = scene.objects.get(UPPER_DOME_OBJECT)
    lower_dome = scene.objects.get(LOWER_DOME_OBJECT)
    if upper_dome is None or lower_dome is None:
        raise RuntimeError("Unable to find the upper and lower radome objects.")
    dome_part_names = [
        upper_dome.name,
        *(child.name for child in upper_dome.children_recursive),
    ]
    antenna_root = scene.objects.get(ANTENNA_ROOT_OBJECT)
    if antenna_root is None:
        raise RuntimeError(
            f"Unable to find the antenna root named {ANTENNA_ROOT_OBJECT!r}."
        )
    elevation_pivot, elevation_axis = get_elevation_reference(scene, depsgraph)
    expected_elevation_axis = np.cross(beam_direction, np.array([0, 0, 1]))
    if np.dot(elevation_axis, expected_elevation_axis) < 0:
        elevation_axis *= -1
    antenna_source_names = {
        child.name for child in antenna_root.children_recursive
    }
    dome_lift_vector = np.array(
        upper_dome.matrix_world.translation - lower_dome.matrix_world.translation,
        dtype=np.float32,
        copy=True,
    )

    all_points = []
    all_normals = []
    all_colors = []
    included_objects = []
    object_point_starts = []
    object_point_counts = []
    radome_panel_object_names = []
    radome_panel_polygon_indices = []
    radome_panel_group_indices = []
    radome_panel_point_starts = []
    radome_panel_point_counts = []

    def append_part(name, points, normals, colors, polygon_point_ranges=()):
        if not points:
            return
        part_start = len(all_points)
        object_point_starts.append(part_start)
        object_point_counts.append(len(points))
        all_points.extend(points)
        all_normals.extend(normals)
        all_colors.extend(colors)
        included_objects.append(name)

        if name in {UPPER_DOME_OBJECT, LOWER_DOME_OBJECT}:
            for (
                polygon_index,
                panel_group_index,
                point_start,
                point_count,
            ) in polygon_point_ranges:
                radome_panel_object_names.append(name)
                radome_panel_polygon_indices.append(polygon_index)
                radome_panel_group_indices.append(panel_group_index)
                radome_panel_point_starts.append(part_start + point_start)
                radome_panel_point_counts.append(point_count)

    for obj in scene.objects:
        if obj.name in EXCLUDED_OBJECTS or obj.hide_render:
            continue
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            continue

        if obj.name == BASE_OBJECT:
            for index, (max_z, min_z) in enumerate(
                zip(BASE_SECTION_Z_EDGES[:-1], BASE_SECTION_Z_EDGES[1:]),
                start=1,
            ):
                if index == 1:
                    max_z = None
                if index == len(BASE_SECTION_Z_EDGES) - 1:
                    min_z = None
                if args.clip_z is not None:
                    min_z = args.clip_z if min_z is None else max(min_z, args.clip_z)
                if max_z is not None and min_z is not None and min_z >= max_z:
                    continue
                points, normals, colors, polygon_point_ranges = evaluated_triangles(
                    obj,
                    depsgraph,
                    min_z=min_z,
                    max_z=max_z,
                )
                append_part(
                    f"base_section_{index}",
                    points,
                    normals,
                    colors,
                    polygon_point_ranges,
                )
            continue

        points, normals, colors, polygon_point_ranges = evaluated_triangles(
            obj,
            depsgraph,
            min_z=args.clip_z,
        )
        append_part(obj.name, points, normals, colors, polygon_point_ranges)

    points = np.asarray(all_points, dtype=np.float32)
    normals = np.asarray(all_normals, dtype=np.float32)
    rgba = np.asarray(all_colors, dtype=np.float32)
    if len(points) == 0:
        raise RuntimeError("No renderable model triangles were found.")

    lower = points.min(axis=0)
    upper = points.max(axis=0)
    normalization_height = upper[2] - REFERENCE_LOWER_Z
    center_xy = (lower[:2] + upper[:2]) / 2

    seated_upper_z = -np.inf
    dome_part_name_set = set(dome_part_names)
    for name, start, count in zip(
        included_objects,
        object_point_starts,
        object_point_counts,
    ):
        part_upper_z = points[start : start + count, 2].max()
        if name in dome_part_name_set:
            part_upper_z -= dome_lift_vector[2]
        seated_upper_z = max(seated_upper_z, part_upper_z)
    reference_model_depth = (
        seated_upper_z - REFERENCE_LOWER_Z
    ) / normalization_height

    points[:, 0] -= center_xy[0]
    points[:, 1] -= center_xy[1]
    points[:, 2] -= REFERENCE_LOWER_Z
    points /= normalization_height
    beam_origin[:2] -= center_xy
    beam_origin[2] -= REFERENCE_LOWER_Z
    beam_origin /= normalization_height
    elevation_pivot[:2] -= center_xy
    elevation_pivot[2] -= REFERENCE_LOWER_Z
    elevation_pivot /= normalization_height
    dome_lift_vector /= normalization_height

    antenna_part_names = [
        name for name in included_objects if name in antenna_source_names
    ]
    base_part_names = [
        name for name in included_objects if name.startswith("base_section_")
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=points,
        normals=normals,
        rgba=rgba,
        beam_origin=beam_origin,
        beam_direction=beam_direction,
        elevation_pivot=elevation_pivot,
        elevation_axis=elevation_axis,
        source_lower=lower,
        source_upper=upper,
        source_file=Path(bpy.data.filepath).name,
        source_frame=scene.frame_current,
        clip_z=np.nan if args.clip_z is None else args.clip_z,
        normalization_lower_z=REFERENCE_LOWER_Z,
        normalization_height=normalization_height,
        reference_model_depth=reference_model_depth,
        included_objects=np.asarray(included_objects),
        object_point_starts=np.asarray(object_point_starts, dtype=np.int64),
        object_point_counts=np.asarray(object_point_counts, dtype=np.int64),
        radome_panel_object_names=np.asarray(radome_panel_object_names),
        radome_panel_polygon_indices=np.asarray(
            radome_panel_polygon_indices,
            dtype=np.int64,
        ),
        radome_panel_group_indices=np.asarray(
            radome_panel_group_indices,
            dtype=np.int64,
        ),
        radome_panel_point_starts=np.asarray(
            radome_panel_point_starts,
            dtype=np.int64,
        ),
        radome_panel_point_counts=np.asarray(
            radome_panel_point_counts,
            dtype=np.int64,
        ),
        upper_dome_part_name=UPPER_DOME_OBJECT,
        lower_dome_part_name=LOWER_DOME_OBJECT,
        dome_part_names=np.asarray(dome_part_names),
        dome_lift_vector=dome_lift_vector,
        antenna_part_names=np.asarray(antenna_part_names),
        feed_horn_part_name=FEED_HORN_OBJECT,
        base_part_names=np.asarray(base_part_names),
        base_section_z_edges=BASE_SECTION_Z_EDGES,
    )

    triangle_count = len(points) // 3
    print(
        f"Exported {triangle_count:,} triangles from {len(included_objects)} objects "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()

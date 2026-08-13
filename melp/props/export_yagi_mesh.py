from __future__ import annotations

import argparse
import sys
from pathlib import Path

import FreeCAD as App
import Import
import numpy as np


DEFAULT_SOURCE = Path.home() / "media" / "models" / "yagi.STEP"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "static" / "yagi_mesh.npz"
LINEAR_DEFLECTION = 0.18
SMOOTH_ANGLE = np.deg2rad(40)

ELEMENT_LABELS = {
    "reflector": ("Element_4",),
    "driven_element": ("Radiator", "Radiator001"),
    "director_1": ("Element_3",),
    "director_2": ("Element_2",),
    "director_3": ("Element_1",),
}
REFERENCE_NAMES = (
    "origin",
    "boom_back",
    "reflector",
    "driven_element",
    "director_1",
    "director_2",
    "director_3",
    "boom_front",
    "mount_center",
    "feed_point",
    "cable_end",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(arguments)


def _triangle_data(shape):
    vertices, triangles = shape.tessellate(LINEAR_DEFLECTION)
    vertices = np.asarray(
        [[point.x, point.y, point.z] for point in vertices],
        dtype=np.float64,
    )
    triangles = np.asarray(triangles, dtype=np.int64)

    triangle_points = vertices[triangles]
    weighted_normals = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    normal_lengths = np.linalg.norm(weighted_normals, axis=1)
    valid = normal_lengths > 1e-12
    triangles = triangles[valid]
    triangle_points = triangle_points[valid]
    weighted_normals = weighted_normals[valid]
    face_normals = weighted_normals / normal_lengths[valid, None]

    incident_faces = [[] for _ in vertices]
    for face_index, triangle in enumerate(triangles):
        for vertex_index in triangle:
            incident_faces[vertex_index].append(face_index)

    cosine_threshold = np.cos(SMOOTH_ANGLE)
    corner_normals = np.empty_like(triangle_points)
    for face_index, triangle in enumerate(triangles):
        for corner_index, vertex_index in enumerate(triangle):
            candidates = np.asarray(incident_faces[vertex_index], dtype=np.int64)
            aligned = candidates[
                face_normals[candidates] @ face_normals[face_index]
                >= cosine_threshold
            ]
            normal = weighted_normals[aligned].sum(axis=0)
            corner_normals[face_index, corner_index] = normal / np.linalg.norm(normal)

    return (
        triangle_points.reshape((-1, 3)).astype(np.float32),
        corner_normals.reshape((-1, 3)).astype(np.float32),
    )


def _bounds(objects, labels):
    boxes = [objects[label].Shape.BoundBox for label in labels]
    lower = np.array(
        [
            min(box.XMin for box in boxes),
            min(box.YMin for box in boxes),
            min(box.ZMin for box in boxes),
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            max(box.XMax for box in boxes),
            max(box.YMax for box in boxes),
            max(box.ZMax for box in boxes),
        ],
        dtype=np.float64,
    )
    return lower, upper


def _center(objects, labels):
    lower, upper = _bounds(objects, labels)
    return (lower + upper) / 2


def _to_manim_coordinates(values, boom_length):
    values = np.asarray(values, dtype=np.float32)
    return values[..., [2, 0, 1]] / boom_length


def main():
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing Yagi STEP source: {source}")

    document = App.newDocument("YagiExport")
    Import.insert(str(source), document.Name)
    document.recompute()

    objects = {
        obj.Label: obj
        for obj in document.Objects
        if obj.TypeId == "Part::Feature"
        and hasattr(obj, "Shape")
        and not obj.Shape.isNull()
    }
    if "Boom" not in objects:
        raise RuntimeError("The STEP assembly does not contain the expected Boom part")

    part_names = []
    point_starts = []
    point_counts = []
    all_points = []
    all_normals = []
    for name, obj in objects.items():
        points, normals = _triangle_data(obj.Shape)
        if not len(points):
            continue
        part_names.append(name)
        point_starts.append(sum(len(values) for values in all_points))
        point_counts.append(len(points))
        all_points.append(points)
        all_normals.append(normals)

    points = np.concatenate(all_points)
    normals = np.concatenate(all_normals)
    boom_lower, boom_upper = _bounds(objects, ("Boom",))
    reference_boom_length = boom_upper[2] - boom_lower[2]
    if reference_boom_length <= 0:
        raise RuntimeError("The exported boom has no length along its source Z axis")

    element_centers = {
        name: _center(objects, labels)
        for name, labels in ELEMENT_LABELS.items()
    }
    mount_center = _center(objects, ("Plate",))
    feed_point = _center(
        objects,
        ("Radiator Hub", "Radiator Hub001", "Radiator Hub002"),
    )
    cable_lower, _ = _bounds(objects, ("RG-58",))
    references = {
        "origin": np.zeros(3),
        "boom_back": np.array([0, 0, boom_lower[2]]),
        **element_centers,
        "boom_front": np.array([0, 0, boom_upper[2]]),
        "mount_center": mount_center,
        "feed_point": feed_point,
        "cable_end": np.array([0, 0, cable_lower[2]]),
    }
    reference_points = np.asarray(
        [references[name] for name in REFERENCE_NAMES],
        dtype=np.float32,
    )

    points = _to_manim_coordinates(points, reference_boom_length)
    normals = normals[..., [2, 0, 1]]
    reference_points = _to_manim_coordinates(
        reference_points,
        reference_boom_length,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points=points,
        normals=normals,
        part_names=np.asarray(part_names),
        part_point_starts=np.asarray(point_starts, dtype=np.int64),
        part_point_counts=np.asarray(point_counts, dtype=np.int64),
        reference_names=np.asarray(REFERENCE_NAMES),
        reference_points=reference_points,
        element_names=np.asarray(tuple(ELEMENT_LABELS)),
        source_file=source.name,
        source_boom_length=reference_boom_length,
        axis_mapping="source XYZ -> manim YZX",
    )

    print(
        f"Exported {len(points) // 3:,} triangles from {len(part_names)} parts "
        f"to {output}"
    )
    App.closeDocument(document.Name)


if __name__ in {"__main__", Path(__file__).stem}:
    main()

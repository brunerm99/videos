from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from zipfile import ZipFile

import numpy as np


DEFAULT_SOURCE = Path.home() / "media/models/cloud-low-poly-large.zip"
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BINARY_CHUNK = 0x004E4942
COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}
GLTF_TO_MANIM = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ],
    dtype=np.float32,
)
CLOUD_ORIENTATION = np.array(
    [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
) @ GLTF_TO_MANIM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_source(path):
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".zip":
        return path.read_bytes(), path.name

    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".glb")]
        if len(members) != 1:
            raise ValueError(f"Expected one GLB in {path}, found {len(members)}")
        return archive.read(members[0]), members[0]


def parse_glb(data):
    magic, version, total_length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or total_length != len(data):
        raise ValueError("Expected a complete glTF 2.0 binary file")

    chunks = {}
    offset = 12
    while offset < len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks[chunk_type] = data[offset : offset + length]
        offset += length

    document = json.loads(chunks[GLB_JSON_CHUNK].decode("utf-8"))
    return document, chunks[GLB_BINARY_CHUNK]


def load_accessor(document, binary, accessor_index):
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    components = TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    packed_stride = dtype.itemsize * components
    stride = view.get("byteStride", packed_stride)

    if stride == packed_stride:
        values = np.frombuffer(
            binary,
            dtype=dtype,
            count=count * components,
            offset=offset,
        ).reshape(count, components)
    else:
        values = np.ndarray(
            shape=(count, components),
            dtype=dtype,
            buffer=binary,
            offset=offset,
            strides=(stride, dtype.itemsize),
        )
    return values.copy()


def quaternion_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )


def node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float32).reshape(4, 4).T

    matrix = np.identity(4, dtype=np.float32)
    rotation = quaternion_matrix(node.get("rotation", (0, 0, 0, 1)))
    scale = np.asarray(node.get("scale", (1, 1, 1)), dtype=np.float32)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = node.get("translation", (0, 0, 0))
    return matrix


def extract_triangles(document, binary):
    points = []
    normals = []
    scene = document["scenes"][document.get("scene", 0)]

    def visit(node_index, parent_matrix):
        node = document["nodes"][node_index]
        matrix = parent_matrix @ node_matrix(node)

        if "mesh" in node:
            for primitive in document["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Only triangle primitives are supported")
                attributes = primitive["attributes"]
                positions = load_accessor(
                    document,
                    binary,
                    attributes["POSITION"],
                ).astype(np.float32)
                vertex_normals = load_accessor(
                    document,
                    binary,
                    attributes["NORMAL"],
                ).astype(np.float32)
                if "indices" in primitive:
                    indices = load_accessor(
                        document,
                        binary,
                        primitive["indices"],
                    ).reshape(-1)
                else:
                    indices = np.arange(len(positions))

                linear = matrix[:3, :3]
                positions = positions @ linear.T + matrix[:3, 3]
                normal_matrix = np.linalg.inv(linear).T
                vertex_normals = vertex_normals @ normal_matrix.T
                vertex_normals /= np.linalg.norm(
                    vertex_normals,
                    axis=1,
                    keepdims=True,
                )

                points.append(positions[indices] @ CLOUD_ORIENTATION.T)
                normals.append(vertex_normals[indices] @ CLOUD_ORIENTATION.T)

        for child_index in node.get("children", ()):
            visit(child_index, matrix)

    for node_index in scene["nodes"]:
        visit(node_index, np.identity(4, dtype=np.float32))

    return np.concatenate(points), np.concatenate(normals)


def main():
    args = parse_args()
    source_data, archive_member = read_source(args.input)
    document, binary = parse_glb(source_data)
    points, normals = extract_triangles(document, binary)

    lower = points.min(axis=0)
    upper = points.max(axis=0)
    source_dimensions = upper - lower
    normalization_width = source_dimensions[0]
    points -= (lower + upper) / 2
    points /= normalization_width

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=points.astype(np.float32),
        normals=normals.astype(np.float32),
        source_file=str(args.input.expanduser().resolve()),
        source_member=archive_member,
        source_dimensions=source_dimensions.astype(np.float32),
        normalization_width=normalization_width,
    )
    print(
        f"Exported {len(points) // 3:,} triangles from {archive_member} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()

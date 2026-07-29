import base64
import struct

from mojo_tinygltf import Model


def model_with_accessor(
    data,
    *,
    component_type=5126,
    accessor_type="SCALAR",
    count=1,
    byte_offset=0,
    byte_stride=None,
    normalized=False,
    sparse=None,
):
    view = {"buffer": 0, "byteLength": len(data)}
    if byte_stride is not None:
        view["byteStride"] = byte_stride
    accessor = {
        "bufferView": 0,
        "byteOffset": byte_offset,
        "componentType": component_type,
        "count": count,
        "type": accessor_type,
    }
    if normalized:
        accessor["normalized"] = True
    if sparse is not None:
        accessor["sparse"] = sparse
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(data)}],
        "bufferViews": [view],
        "accessors": [accessor],
    }
    return Model(document, [data])


def triangle_model(positions, indices):
    position_bytes = struct.pack("<" + "f" * len(positions), *positions)
    index_bytes = struct.pack("<" + "H" * len(indices), *indices)
    binary = position_bytes + index_bytes
    document = {
        "asset": {"version": "2.0", "generator": "mojo-tinygltf tests"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {
                "buffer": 0,
                "byteOffset": len(position_bytes),
                "byteLength": len(index_bytes),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions) // 3,
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    return Model(document, [binary])


def data_uri(data):
    return "data:application/octet-stream;base64," + base64.b64encode(data).decode()

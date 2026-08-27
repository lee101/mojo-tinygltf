import struct

import numpy as np
import pytest

from mojo_tinygltf import GltfError, Model


@pytest.mark.parametrize(
    "index_type,index_fmt",
    [(5121, "BB"), (5123, "HH"), (5125, "II")],
)
def test_sparse_accessor_all_valid_index_types(index_type, index_fmt):
    base = struct.pack("<" + "f" * 12, *range(12))
    indices = struct.pack("<" + index_fmt, 1, 3)
    values = struct.pack("<ffffff", 10, 11, 12, 30, 31, 32)
    buffers = [base, indices, values]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(x)} for x in buffers],
        "bufferViews": [
            {"buffer": 0, "byteLength": len(base)},
            {"buffer": 1, "byteLength": len(indices)},
            {"buffer": 2, "byteLength": len(values)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "sparse": {
                    "count": 2,
                    "indices": {"bufferView": 1, "componentType": index_type},
                    "values": {"bufferView": 2},
                },
            }
        ],
    }
    expected = np.arange(12, dtype=float).reshape(4, 3)
    expected[1] = [10, 11, 12]
    expected[3] = [30, 31, 32]
    np.testing.assert_array_equal(Model(document, buffers).accessor(0), expected)


def test_sparse_accessor_without_base_starts_at_zero():
    buffers = [bytes([0, 2]), struct.pack("<hhhh", -32768, 32767, 0, 16384)]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(x)} for x in buffers],
        "bufferViews": [
            {"buffer": 0, "byteLength": 2},
            {"buffer": 1, "byteLength": 8},
        ],
        "accessors": [
            {
                "componentType": 5122,
                "normalized": True,
                "count": 3,
                "type": "VEC2",
                "sparse": {
                    "count": 2,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }
    expected = [[-1, 1], [0, 0], [0, 16384 / 32767]]
    np.testing.assert_allclose(Model(document, buffers).accessor(0), expected)


def test_sparse_float_vec3_simd_scalar_tail_preserves_rows():
    indices = struct.pack("<II", 1, 3)
    values = struct.pack("<ffffff", 10, 11, 12, 30, 31, 32)
    buffers = [indices, values]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(x)} for x in buffers],
        "bufferViews": [
            {"buffer": 0, "byteLength": len(indices)},
            {"buffer": 1, "byteLength": len(values)},
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 5,
                "type": "VEC3",
                "sparse": {
                    "count": 2,
                    "indices": {"bufferView": 0, "componentType": 5125},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }
    expected = np.zeros((5, 3))
    expected[1] = [10, 11, 12]
    expected[3] = [30, 31, 32]
    np.testing.assert_array_equal(Model(document, buffers).accessor(0), expected)


def test_sparse_out_of_range_index_rejected():
    buffers = [bytes([4]), struct.pack("<f", 1)]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(x)} for x in buffers],
        "bufferViews": [
            {"buffer": 0, "byteLength": 1},
            {"buffer": 1, "byteLength": 4},
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 4,
                "type": "SCALAR",
                "sparse": {
                    "count": 1,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }
    with pytest.raises(GltfError, match="invalid sparse accessor"):
        Model(document, buffers).accessor(0)


def test_sparse_invalid_signed_index_type():
    buffers = [bytes([0]), struct.pack("<f", 1)]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(x)} for x in buffers],
        "bufferViews": [
            {"buffer": 0, "byteLength": 1},
            {"buffer": 1, "byteLength": 4},
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 1,
                "type": "SCALAR",
                "sparse": {
                    "count": 1,
                    "indices": {"bufferView": 0, "componentType": 5120},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }
    with pytest.raises(GltfError, match="invalid sparse index type"):
        Model(document, buffers).accessor(0)


def test_sparse_indices_must_be_strictly_increasing():
    buffers = [bytes([1, 1]), struct.pack("<ff", 2, 3)]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(x)} for x in buffers],
        "bufferViews": [
            {"buffer": 0, "byteLength": 2},
            {"buffer": 1, "byteLength": 8},
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
                "sparse": {
                    "count": 2,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }
    with pytest.raises(GltfError, match="invalid sparse accessor"):
        Model(document, buffers).accessor(0)

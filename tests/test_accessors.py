import struct

import numpy as np
import pytest

from mojo_tinygltf import GltfError
from mojo_tinygltf._lib import addr, lib

from helpers import model_with_accessor


@pytest.mark.parametrize(
    "component_type,fmt,values,expected",
    [
        (5120, "bbb", [-128, 0, 127], [-128, 0, 127]),
        (5121, "BBB", [0, 128, 255], [0, 128, 255]),
        (5122, "hhh", [-32768, 0, 32767], [-32768, 0, 32767]),
        (5123, "HHH", [0, 32768, 65535], [0, 32768, 65535]),
        (5124, "iii", [-2147483648, 0, 2147483647], [-2147483648, 0, 2147483647]),
        (5125, "III", [0, 2147483648, 4294967295], [0, 2147483648, 4294967295]),
        (5126, "fff", [-1.25, 0.0, 3.5], [-1.25, 0.0, 3.5]),
        (5130, "ddd", [-1.25, 0.0, 3.5], [-1.25, 0.0, 3.5]),
    ],
)
def test_all_component_types(component_type, fmt, values, expected):
    model = model_with_accessor(
        struct.pack("<" + fmt, *values),
        component_type=component_type,
        count=3,
    )
    np.testing.assert_allclose(model.accessor(0), expected)


@pytest.mark.parametrize(
    "component_type,fmt,values,expected",
    [
        (5120, "bbb", [-128, 0, 127], [-1.0, 0.0, 1.0]),
        (5121, "BBB", [0, 128, 255], [0.0, 128 / 255, 1.0]),
        (5122, "hhh", [-32768, 0, 32767], [-1.0, 0.0, 1.0]),
        (5123, "HHH", [0, 32768, 65535], [0.0, 32768 / 65535, 1.0]),
    ],
)
def test_normalized_integer_formula(component_type, fmt, values, expected):
    model = model_with_accessor(
        struct.pack("<" + fmt, *values),
        component_type=component_type,
        count=3,
        normalized=True,
    )
    np.testing.assert_allclose(model.accessor(0), expected, rtol=0, atol=1e-15)


def test_normalized_int16_simd_tail():
    values = [-32768, -12345, -1, 0, 1, 12345, 32766, 32767, -23456]
    model = model_with_accessor(
        struct.pack("<" + "h" * len(values), *values),
        component_type=5122,
        count=len(values),
        normalized=True,
    )
    expected = np.maximum(np.asarray(values, dtype=np.float64) / 32767.0, -1.0)
    np.testing.assert_array_equal(model.accessor(0), expected)


@pytest.mark.parametrize("count", [999_999, 1_000_003])
def test_float32_decode_parallel_threshold(monkeypatch, count):
    import mojo_tinygltf.model as model_module

    values = np.linspace(-2, 2, count, dtype=np.float32)
    monkeypatch.setattr(model_module, "_available_workers", lambda: 2)
    model = model_with_accessor(
        values.tobytes(),
        component_type=5126,
        count=count,
    )
    np.testing.assert_array_equal(
        model.accessor(0), values.astype(np.float64)
    )


def test_interleaved_vec3_stride_and_offset():
    data = struct.pack("<IfffIfff", 99, 1, 2, 3, 88, 4, 5, 6)
    model = model_with_accessor(
        data,
        accessor_type="VEC3",
        component_type=5126,
        count=2,
        byte_offset=4,
        byte_stride=16,
    )
    np.testing.assert_array_equal(model.accessor(0), [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize(
    "accessor_type,data,expected",
    [
        ("MAT2", bytes([1, 2, 0, 0, 3, 4, 0, 0]), [[1, 2, 3, 4]]),
        (
            "MAT3",
            bytes([1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]),
            [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        ),
    ],
)
def test_small_integer_matrix_column_padding(accessor_type, data, expected):
    model = model_with_accessor(
        data,
        accessor_type=accessor_type,
        component_type=5121,
        count=1,
    )
    np.testing.assert_array_equal(model.accessor(0), expected)


def test_byte_stride_matches_upstream_accessor_utility():
    assert lib().mtg_byte_stride(5126, 3, 0) == 12
    assert lib().mtg_byte_stride(5123, 3, 16) == 16
    assert lib().mtg_byte_stride(5126, 3, 14) == -1
    assert lib().mtg_byte_stride(-1, 3, 0) == -1


def test_scalar_shape_and_dtype_conversion():
    model = model_with_accessor(struct.pack("<HH", 2, 7), component_type=5123, count=2)
    value = model.accessor(0, dtype=np.uint16)
    assert value.shape == (2,)
    assert value.dtype == np.uint16
    np.testing.assert_array_equal(value, [2, 7])


def test_empty_accessor_without_buffer_view():
    document = {
        "asset": {"version": "2.0"},
        "accessors": [{"componentType": 5126, "count": 0, "type": "VEC3"}],
    }
    from mojo_tinygltf import Model

    value = Model(document).accessor(0)
    assert value.shape == (0, 3)


def test_buffer_view_zero_copy_and_copy():
    model = model_with_accessor(b"abcdef", count=1, component_type=5121)
    view = model.buffer_view(0)
    assert isinstance(view, memoryview)
    assert view.tobytes() == b"abcdef"
    assert model.buffer_view(0, copy_data=True) == b"abcdef"


def test_accessor_bounds_rejected_before_ffi():
    model = model_with_accessor(b"\0" * 8, accessor_type="VEC3", count=1)
    with pytest.raises(GltfError, match="exceeds bufferView"):
        model.accessor(0)


def test_invalid_accessor_metadata():
    model = model_with_accessor(b"\0" * 4)
    model["accessors"][0]["type"] = "VEC5"
    with pytest.raises(GltfError, match="unsupported type"):
        model.accessor(0)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("byteOffset", -1, "non-negative"),
        ("byteOffset", "0", "non-negative"),
    ],
)
def test_accessor_offsets_are_strict_integers(field, value, message):
    model = model_with_accessor(b"\0" * 4)
    model["accessors"][0][field] = value
    with pytest.raises(GltfError, match=message):
        model.accessor(0)


def test_negative_accessor_and_buffer_view_indices_are_rejected():
    model = model_with_accessor(b"\0" * 4)
    with pytest.raises(GltfError, match="accessor"):
        model.accessor(-1)
    with pytest.raises(GltfError, match="bufferView"):
        model.buffer_view(-1)


def test_float_accessor_cannot_request_normalization():
    model = model_with_accessor(b"\0" * 4)
    model["accessors"][0]["normalized"] = True
    with pytest.raises(GltfError, match="cannot normalize"):
        model.accessor(0)


def test_decode_ffi_rejects_short_source_and_destination_lengths():
    source = np.zeros(3, dtype=np.float32)
    destination = np.empty(3, dtype=np.float64)
    common = (1, 3, 5126, 3, 0, 12, 0, 1)
    assert (
        lib().mtg_decode_accessor(
            addr(source), source.nbytes - 1, addr(destination), destination.size, *common
        )
        == -4
    )
    assert (
        lib().mtg_decode_accessor(
            addr(source), source.nbytes, addr(destination), destination.size - 1, *common
        )
        == -3
    )

import json
import struct

import numpy as np
import pytest
import trimesh

from mojo_tinygltf import GltfError, Model, load, loads

from helpers import data_uri, triangle_model


def test_ascii_data_uri_load_and_unknown_fields_roundtrip():
    raw = struct.pack("<fff", 1, 2, 3)
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["VENDOR_test"],
        "extensions": {"VENDOR_test": {"answer": 42}},
        "extras": {"kept": [1, None, "x"]},
        "buffers": [{"byteLength": len(raw), "uri": data_uri(raw)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(raw)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}
        ],
    }
    parsed = loads(json.dumps(document))
    np.testing.assert_array_equal(parsed.accessor(0), [[1, 2, 3]])
    assert json.loads(parsed.to_json(pretty=False, embed_buffers=False)) == document


def test_percent_encoded_data_uri():
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 3, "uri": "data:application/octet-stream,A%20B"}],
    }
    assert loads(json.dumps(document)).buffers == [b"A B"]


def test_external_buffer_load(tmp_path):
    (tmp_path / "mesh data.bin").write_bytes(b"1234")
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 4, "uri": "mesh%20data.bin"}],
    }
    path = tmp_path / "model.gltf"
    path.write_text(json.dumps(document))
    assert load(path).buffers == [b"1234"]


def test_gltf_save_writes_external_buffer(tmp_path):
    source = triangle_model([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2])
    path = tmp_path / "triangle.gltf"
    source.save(path)
    document = json.loads(path.read_text())
    assert document["buffers"][0]["uri"] == "triangle.bin"
    assert load(path).buffers == source.buffers


def test_glb_roundtrip_preserves_model_and_buffer():
    source = triangle_model([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2])
    encoded = source.to_glb()
    parsed = loads(encoded)
    assert parsed.binary
    assert parsed.document == source.document
    assert parsed.buffers == source.buffers
    np.testing.assert_array_equal(parsed.accessor(1), [0, 1, 2])


def test_empty_model_glb_matches_upstream_issue_382_expectation():
    source = Model({"asset": {"version": "2.0"}})
    parsed = loads(source.to_glb())
    assert parsed.document == source.document
    assert parsed.buffers == []


def test_empty_node_serialization_matches_upstream_issue_457():
    source = Model(
        {
            "asset": {"version": "2.0"},
            "nodes": [{}],
            "scenes": [{"nodes": [0]}],
        }
    )
    encoded = json.loads(source.to_json(pretty=False))
    assert encoded["nodes"] == [{}]
    assert encoded["scenes"] == [{"nodes": [0]}]


@pytest.mark.parametrize(
    "payload,message",
    [
        (b"", "Too short"),
        (b"x" * 20, "Invalid magic"),
        (
            b"glTF" + struct.pack("<IIII", 2, 1000, 4, 0x4E4F534A),
            "Invalid glTF binary",
        ),
        (
            b"glTF" + struct.pack("<IIII", 2, 20, 1, 0x4E4F534A),
            "Invalid glTF binary",
        ),
    ],
)
def test_glb_header_errors_from_upstream(payload, message):
    with pytest.raises(GltfError, match=message):
        loads(payload, binary=True)


def test_missing_asset_version_rejected():
    with pytest.raises(GltfError, match="asset"):
        loads("{}")


def test_buffer_shorter_than_declared_rejected():
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 4, "uri": data_uri(b"12")}],
    }
    with pytest.raises(GltfError, match="exceeds"):
        loads(json.dumps(document))


def test_buffer_view_stride_rule_matches_upstream_parser():
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 14, "uri": data_uri(b"\0" * 14)}],
        "bufferViews": [
            {"buffer": 0, "byteLength": 14, "byteStride": 14}
        ],
    }
    with pytest.raises(GltfError, match="multiple of 4"):
        loads(json.dumps(document))


def test_trimesh_cross_loads_generated_glb(tmp_path):
    source = triangle_model([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2])
    path = tmp_path / "triangle.glb"
    source.save(path)
    scene = trimesh.load(path, force="scene", process=False)
    mesh = next(iter(scene.geometry.values()))
    np.testing.assert_allclose(
        np.asarray(mesh.vertices),
        source.accessor(0),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(np.asarray(mesh.faces).ravel(), source.accessor(1))

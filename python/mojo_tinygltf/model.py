"""glTF 2.0/GLB model IO and decoded data access."""

from __future__ import annotations

import base64
import copy
import json
import os
import struct
from collections import deque
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

import numpy as np

from ._lib import addr, lib

COMPONENT_BYTE = 5120
COMPONENT_UNSIGNED_BYTE = 5121
COMPONENT_SHORT = 5122
COMPONENT_UNSIGNED_SHORT = 5123
COMPONENT_INT = 5124
COMPONENT_UNSIGNED_INT = 5125
COMPONENT_FLOAT = 5126
COMPONENT_DOUBLE = 5130

_COMPONENT_SIZE = {
    COMPONENT_BYTE: 1,
    COMPONENT_UNSIGNED_BYTE: 1,
    COMPONENT_SHORT: 2,
    COMPONENT_UNSIGNED_SHORT: 2,
    COMPONENT_INT: 4,
    COMPONENT_UNSIGNED_INT: 4,
    COMPONENT_FLOAT: 4,
    COMPONENT_DOUBLE: 8,
}
_TYPE_CODE = {
    "SCALAR": 65,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 34,
    "MAT3": 35,
    "MAT4": 36,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _element_size(component_type: int, accessor_type: str) -> int:
    size = _COMPONENT_SIZE[component_type]
    if accessor_type.startswith("MAT"):
        rows = int(accessor_type[-1])
        return rows * ((rows * size + 3) // 4 * 4)
    return _TYPE_COMPONENTS[accessor_type] * size


def _available_workers() -> int:
    if hasattr(os, "sched_getaffinity"):
        return min(8, len(os.sched_getaffinity(0)))
    return min(8, os.cpu_count() or 1)


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GltfError(f"{label} must be a non-negative integer")
    return value


def _list_index(value: Any, length: int, label: str) -> int:
    value = _nonnegative_int(value, label)
    if value >= length:
        raise GltfError(f"invalid {label}")
    return value


class GltfError(ValueError):
    """A malformed glTF document or invalid reference."""


def _data_uri(uri: str) -> bytes:
    if not uri.startswith("data:") or "," not in uri:
        raise GltfError("invalid data URI")
    header, payload = uri.split(",", 1)
    try:
        return (
            base64.b64decode(payload, validate=True)
            if header.endswith(";base64")
            else unquote_to_bytes(payload)
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise GltfError("failed to decode data URI") from exc


# tinygltf: tiny_gltf.h TinyGLTF::LoadBinaryFromMemory
def _parse_glb(data: bytes, *, strict: bool = True) -> tuple[dict[str, Any], bytes]:
    if len(data) < 20:
        raise GltfError("Too short data size for glTF Binary.")
    if data[:4] != b"glTF":
        raise GltfError("Invalid magic.")
    _, version, length = struct.unpack_from("<4sII", data)
    json_length, json_type = struct.unpack_from("<II", data, 12)
    header_and_json = 20 + json_length
    if (
        header_and_json > len(data)
        or json_length < 1
        or length > len(data)
        or header_and_json > length
        or json_type != 0x4E4F534A
    ):
        raise GltfError("Invalid glTF binary.")
    if header_and_json % 4:
        raise GltfError("JSON Chunk end does not aligned to a 4-byte boundary.")
    binary = b""
    if header_and_json != length:
        if header_and_json + 8 > length:
            raise GltfError("Insufficient storage space for Chunk1(BIN data).")
        bin_length, bin_type = struct.unpack_from("<II", data, header_and_json)
        if bin_type != 0x004E4942:
            raise GltfError("Invalid chunkType for Chunk1.")
        if bin_length:
            if bin_length < 4:
                raise GltfError("Insufficient Chunk1(BIN) data size.")
            if bin_length % 4 and strict:
                raise GltfError("BIN Chunk end is not aligned to a 4-byte boundary.")
            end = header_and_json + 8 + bin_length
            if end > length:
                raise GltfError("BIN Chunk data length exceeds the GLB size.")
            binary = data[header_and_json + 8 : end]
    try:
        document = json.loads(data[20:header_and_json].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GltfError(f"invalid GLB JSON: {exc}") from exc
    if version != 2:
        raise GltfError(f"unsupported GLB version {version}")
    return document, binary


class Model(MutableMapping[str, Any]):
    """A lossless JSON document plus its resolved binary buffers."""

    def __init__(
        self,
        document: dict[str, Any],
        buffers: list[bytes] | None = None,
        *,
        base_dir: str | Path | None = None,
        binary: bool = False,
    ):
        if not isinstance(document, dict):
            raise GltfError("Root element is not a JSON object")
        asset = document.get("asset")
        if not isinstance(asset, dict) or not isinstance(asset.get("version"), str):
            raise GltfError('"asset" object not found in .gltf or missing version')
        self.document = document
        self.buffers = list(buffers or [])
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.binary = binary
        if len(self.buffers) != len(document.get("buffers", [])):
            if document.get("buffers"):
                raise GltfError("resolved buffer count does not match document")

    def __getitem__(self, key: str) -> Any:
        return self.document[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.document[key] = value

    def __delitem__(self, key: str) -> None:
        del self.document[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.document)

    def __len__(self) -> int:
        return len(self.document)

    @classmethod
    def from_dict(
        cls, document: dict[str, Any], buffers: list[bytes] | None = None
    ) -> "Model":
        return cls(copy.deepcopy(document), buffers)

    def buffer_view(self, index: int, *, copy_data: bool = False):
        try:
            views = self.document["bufferViews"]
            index = _list_index(index, len(views), "bufferView")
            view = views[index]
            buffer_index = _list_index(
                view["buffer"], len(self.buffers), f"bufferView {index} buffer"
            )
            buffer = self.buffers[buffer_index]
        except (KeyError, IndexError, TypeError) as exc:
            raise GltfError(f"invalid bufferView {index}") from exc
        offset = _nonnegative_int(
            view.get("byteOffset", 0), f"bufferView {index} byteOffset"
        )
        length = _nonnegative_int(
            view["byteLength"], f"bufferView {index} byteLength"
        )
        if offset < 0 or length < 1 or offset + length > len(buffer):
            raise GltfError(f"bufferView {index} exceeds buffer bounds")
        result = memoryview(buffer)[offset : offset + length]
        return bytes(result) if copy_data else result

    def accessor(self, index: int, *, dtype=None) -> np.ndarray:
        try:
            accessors = self.document["accessors"]
            index = _list_index(index, len(accessors), "accessor")
            accessor = accessors[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise GltfError(f"invalid accessor {index}") from exc
        component_type = accessor.get("componentType")
        accessor_type = accessor.get("type")
        if component_type not in _COMPONENT_SIZE:
            raise GltfError(f"accessor {index} has invalid componentType")
        if accessor_type not in _TYPE_COMPONENTS:
            raise GltfError(f"accessor {index} has unsupported type")
        count = accessor.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise GltfError(f"accessor {index} has invalid count")
        normalized = accessor.get("normalized", False)
        if not isinstance(normalized, bool):
            raise GltfError(f"accessor {index} has invalid normalized flag")
        if normalized and component_type in (COMPONENT_FLOAT, COMPONENT_DOUBLE):
            raise GltfError(f"accessor {index} cannot normalize floating-point data")
        components = _TYPE_COMPONENTS[accessor_type]
        buffer_view_index = accessor.get("bufferView")
        workers = _available_workers()
        result = (
            np.empty((count, components), dtype=np.float64)
            if buffer_view_index is not None
            else np.zeros((count, components), dtype=np.float64)
        )
        if count and buffer_view_index is not None:
            try:
                views = self.document["bufferViews"]
                buffer_view_index = _list_index(
                    buffer_view_index, len(views), f"accessor {index} bufferView"
                )
                view = views[buffer_view_index]
                buffer_index = _list_index(
                    view["buffer"], len(self.buffers), f"bufferView {buffer_view_index} buffer"
                )
                raw = np.frombuffer(self.buffers[buffer_index], dtype=np.uint8)
            except (KeyError, IndexError, TypeError) as exc:
                raise GltfError(
                    f"accessor[{index}] invalid bufferView"
                ) from exc
            view_offset = _nonnegative_int(
                view.get("byteOffset", 0), f"bufferView {buffer_view_index} byteOffset"
            )
            view_length = _nonnegative_int(
                view["byteLength"], f"bufferView {buffer_view_index} byteLength"
            )
            accessor_offset = _nonnegative_int(
                accessor.get("byteOffset", 0), f"accessor {index} byteOffset"
            )
            view_stride = _nonnegative_int(
                view.get("byteStride", 0), f"bufferView {buffer_view_index} byteStride"
            )
            stride = (
                view_stride
                if view_stride
                else _element_size(component_type, accessor_type)
            )
            upstream_stride = int(
                lib().mtg_byte_stride(
                    component_type, _TYPE_CODE[accessor_type], view_stride
                )
            )
            if upstream_stride < 0:
                raise GltfError(f"accessor {index} has invalid byteStride")
            element_size = _element_size(component_type, accessor_type)
            end = (
                view_offset
                + accessor_offset
                + (count - 1) * stride
                + element_size
            )
            if (
                view_offset < 0
                or view_length < 1
                or accessor_offset < 0
                or end > view_offset + view_length
                or end > raw.size
            ):
                raise GltfError(f"accessor {index} exceeds bufferView bounds")
            status = lib().mtg_decode_accessor(
                addr(raw),
                raw.size,
                addr(result),
                result.size,
                count,
                components,
                component_type,
                _TYPE_CODE[accessor_type],
                view_offset + accessor_offset,
                stride,
                int(normalized),
                workers,
            )
            if status:
                raise GltfError(f"Mojo accessor decode failed ({status})")
        sparse = accessor.get("sparse")
        if sparse is not None:
            self._apply_sparse(index, accessor, sparse, result)
        if accessor_type == "SCALAR":
            result = result[:, 0]
        return result.astype(dtype, copy=False) if dtype is not None else result

    def _apply_sparse(
        self,
        index: int,
        accessor: dict[str, Any],
        sparse: dict[str, Any],
        result: np.ndarray,
    ) -> None:
        try:
            sparse_count = sparse["count"]
            indices = sparse["indices"]
            values = sparse["values"]
            index_type = indices["componentType"]
            views = self.document["bufferViews"]
            indices_view_index = _list_index(
                indices["bufferView"], len(views), f"accessor {index} sparse indices bufferView"
            )
            values_view_index = _list_index(
                values["bufferView"], len(views), f"accessor {index} sparse values bufferView"
            )
            indices_view = views[indices_view_index]
            values_view = views[values_view_index]
            indices_buffer = _list_index(
                indices_view["buffer"], len(self.buffers), "sparse indices buffer"
            )
            values_buffer = _list_index(
                values_view["buffer"], len(self.buffers), "sparse values buffer"
            )
            indices_raw = np.frombuffer(
                self.buffers[indices_buffer], dtype=np.uint8
            )
            values_raw = np.frombuffer(
                self.buffers[values_buffer], dtype=np.uint8
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise GltfError(f"accessor {index} has invalid sparse storage") from exc
        if (
            not isinstance(sparse_count, int)
            or isinstance(sparse_count, bool)
            or sparse_count < 0
            or sparse_count > accessor["count"]
        ):
            raise GltfError(f"accessor {index} has invalid sparse count")
        if index_type not in (
            COMPONENT_UNSIGNED_BYTE,
            COMPONENT_UNSIGNED_SHORT,
            COMPONENT_UNSIGNED_INT,
        ):
            raise GltfError(f"accessor {index} has invalid sparse index type")
        components = _TYPE_COMPONENTS[accessor["type"]]
        indices_view_offset = _nonnegative_int(
            indices_view.get("byteOffset", 0), "sparse indices bufferView byteOffset"
        )
        values_view_offset = _nonnegative_int(
            values_view.get("byteOffset", 0), "sparse values bufferView byteOffset"
        )
        index_offset = indices_view_offset + _nonnegative_int(
            indices.get("byteOffset", 0), "sparse indices byteOffset"
        )
        value_offset = values_view_offset + _nonnegative_int(
            values.get("byteOffset", 0), "sparse values byteOffset"
        )
        index_end = index_offset + sparse_count * _COMPONENT_SIZE[index_type]
        value_end = (
            value_offset
            + sparse_count
            * _element_size(accessor["componentType"], accessor["type"])
        )
        indices_limit = indices_view_offset + _nonnegative_int(
            indices_view["byteLength"], "sparse indices bufferView byteLength"
        )
        values_limit = values_view_offset + _nonnegative_int(
            values_view["byteLength"], "sparse values bufferView byteLength"
        )
        if (
            index_offset < indices_view_offset
            or index_end > indices_limit
            or index_end > indices_raw.size
        ):
            raise GltfError(f"accessor {index} sparse indices exceed bounds")
        if (
            value_offset < values_view_offset
            or value_end > values_limit
            or value_end > values_raw.size
        ):
            raise GltfError(f"accessor {index} sparse values exceed bounds")
        if not sparse_count:
            return
        status = lib().mtg_apply_sparse(
            addr(indices_raw),
            indices_raw.size,
            addr(values_raw),
            values_raw.size,
            addr(result),
            result.size,
            sparse_count,
            accessor["count"],
            components,
            index_type,
            accessor["componentType"],
            _TYPE_CODE[accessor["type"]],
            index_offset,
            value_offset,
            int(bool(accessor.get("normalized", False))),
        )
        if status:
            raise GltfError(f"invalid sparse accessor data ({status})")

    def local_matrices(self) -> np.ndarray:
        nodes = self.document.get("nodes", [])
        count = len(nodes)
        if not count:
            return np.empty((0, 4, 4), dtype=np.float64)
        trs = np.empty((count, 10), dtype=np.float64)
        trs[:, :3] = (0.0, 0.0, 0.0)
        trs[:, 3:7] = (0.0, 0.0, 0.0, 1.0)
        trs[:, 7:] = (1.0, 1.0, 1.0)
        matrices = np.zeros((count, 16), dtype=np.float64)
        has_matrix = np.zeros(count, dtype=np.uint8)
        for i, node in enumerate(nodes):
            if "matrix" in node:
                value = np.asarray(node["matrix"], dtype=np.float64)
                if value.shape != (16,):
                    raise GltfError(f"node {i} matrix must have 16 values")
                matrices[i] = value
                has_matrix[i] = 1
                continue
            for key, start, length in (
                ("translation", 0, 3),
                ("rotation", 3, 4),
                ("scale", 7, 3),
            ):
                if key in node:
                    value = np.asarray(node[key], dtype=np.float64)
                    if value.shape != (length,):
                        raise GltfError(
                            f"node {i} {key} must have {length} values"
                        )
                    trs[i, start : start + length] = value
        result = np.empty((count, 4, 4), dtype=np.float64)
        status = lib().mtg_local_matrices(
            addr(trs),
            trs.size,
            addr(matrices),
            matrices.size,
            addr(has_matrix),
            has_matrix.size,
            addr(result),
            result.size,
            count,
        )
        if status:
            raise GltfError(f"Mojo local transform failed ({status})")
        return result

    def world_matrices(self) -> np.ndarray:
        nodes = self.document.get("nodes", [])
        count = len(nodes)
        if not count:
            return np.empty((0, 4, 4), dtype=np.float64)
        parents = np.full(count, -1, dtype=np.int64)
        children: list[list[int]] = [[] for _ in range(count)]
        for parent, node in enumerate(nodes):
            for child in node.get("children", []):
                if (
                    not isinstance(child, int)
                    or isinstance(child, bool)
                    or child < 0
                    or child >= count
                ):
                    raise GltfError(f"node {parent} has invalid child")
                if parents[child] != -1:
                    raise GltfError(f"node {child} has multiple parents")
                parents[child] = parent
                children[parent].append(child)
        queue = deque(int(i) for i in np.flatnonzero(parents == -1))
        order: list[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            queue.extend(children[node])
        if len(order) != count:
            raise GltfError("node hierarchy contains a cycle")
        local = self.local_matrices()
        traversal = np.asarray(order, dtype=np.int64)
        result = np.empty_like(local)
        status = lib().mtg_resolve_world(
            addr(local),
            local.size,
            addr(parents),
            parents.size,
            addr(traversal),
            traversal.size,
            addr(result),
            result.size,
            count,
        )
        if status:
            raise GltfError(f"Mojo transform resolution failed ({status})")
        return result

    def to_json(self, *, pretty: bool = True, embed_buffers: bool = True) -> str:
        document = copy.deepcopy(self.document)
        descriptions = document.get("buffers", [])
        for i, buffer in enumerate(self.buffers):
            descriptions[i]["byteLength"] = len(buffer)
            if embed_buffers:
                descriptions[i]["uri"] = (
                    "data:application/octet-stream;base64,"
                    + base64.b64encode(buffer).decode("ascii")
                )
        return json.dumps(
            document,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
        )

    # tinygltf: tiny_gltf.h WriteBinaryGltfStream
    def to_glb(self) -> bytes:
        document = copy.deepcopy(self.document)
        binary = b""
        if self.buffers and not document["buffers"][0].get("uri"):
            binary = bytes(self.buffers[0])
            document["buffers"][0]["byteLength"] = len(binary)
            document["buffers"][0].pop("uri", None)
        for i, buffer in enumerate(self.buffers):
            document["buffers"][i]["byteLength"] = len(buffer)
            if i != 0 or not binary:
                document["buffers"][i]["uri"] = (
                    "data:application/octet-stream;base64,"
                    + base64.b64encode(buffer).decode("ascii")
                )
        content = json.dumps(
            document, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        content += b" " * (-len(content) % 4)
        binary += b"\0" * (-len(binary) % 4)
        total = 12 + 8 + len(content) + (8 + len(binary) if binary else 0)
        result = bytearray(struct.pack("<4sII", b"glTF", 2, total))
        result += struct.pack("<II", len(content), 0x4E4F534A)
        result += content
        if binary:
            result += struct.pack("<II", len(binary), 0x004E4942)
            result += binary
        return bytes(result)

    def save(
        self,
        path: str | Path,
        *,
        binary: bool | None = None,
        pretty: bool = True,
        embed_buffers: bool = False,
    ) -> None:
        path = Path(path)
        as_binary = path.suffix.lower() == ".glb" if binary is None else binary
        if as_binary:
            path.write_bytes(self.to_glb())
        else:
            if embed_buffers:
                content = self.to_json(pretty=pretty, embed_buffers=True)
            else:
                document = copy.deepcopy(self.document)
                used: set[str] = set()
                for i, buffer in enumerate(self.buffers):
                    description = document["buffers"][i]
                    description["byteLength"] = len(buffer)
                    uri = description.get("uri", "")
                    if not uri or uri.startswith("data:"):
                        suffix = "" if i == 0 else str(i)
                        uri = f"{path.stem}{suffix}.bin"
                        while uri in used:
                            suffix += "_"
                            uri = f"{path.stem}{suffix}.bin"
                        description["uri"] = uri
                    used.add(uri)
                    target = path.parent / unquote_to_bytes(uri).decode("utf-8")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(buffer)
                content = json.dumps(
                    document,
                    indent=2 if pretty else None,
                    separators=None if pretty else (",", ":"),
                    ensure_ascii=False,
                )
            path.write_text(
                content,
                encoding="utf-8",
            )


def loads(
    data: str | bytes,
    *,
    base_dir: str | Path | None = None,
    binary: bool | None = None,
) -> Model:
    binary_chunk: bytes | None = None
    is_binary = (
        isinstance(data, bytes) and data.startswith(b"glTF")
        if binary is None
        else binary
    )
    if is_binary:
        if not isinstance(data, bytes):
            raise GltfError("binary glTF input must be bytes")
        document, binary_chunk = _parse_glb(data)
    else:
        try:
            text = data.decode("utf-8") if isinstance(data, bytes) else data
            document = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GltfError(f"invalid glTF JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise GltfError("Root element is not a JSON object")
    for i, view in enumerate(document.get("bufferViews", [])):
        if not isinstance(view, dict):
            raise GltfError(f"bufferView {i} is not an object")
        stride = view.get("byteStride", 0)
        if (
            not isinstance(stride, int)
            or isinstance(stride, bool)
            or stride < 0
            or stride > 252
            or stride % 4
        ):
            raise GltfError(
                "Invalid `byteStride' value. `byteStride' must be the multiple of 4"
            )
    root = Path(base_dir) if base_dir is not None else None
    buffers: list[bytes] = []
    for i, description in enumerate(document.get("buffers", [])):
        if not isinstance(description, dict):
            raise GltfError(f"buffer {i} is not an object")
        byte_length = description.get("byteLength")
        if (
            not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length < 0
        ):
            raise GltfError(f"buffer {i} has invalid byteLength")
        uri = description.get("uri")
        if isinstance(uri, str) and uri.startswith("data:"):
            value = _data_uri(uri)
        elif isinstance(uri, str) and uri:
            if root is None:
                raise GltfError(f"buffer {i} needs a base directory")
            try:
                value = (root / unquote_to_bytes(uri).decode("utf-8")).read_bytes()
            except OSError as exc:
                raise GltfError(f"failed to read buffer {uri}") from exc
        elif is_binary and i == 0:
            value = binary_chunk or b""
        else:
            raise GltfError("'uri' is missing from non binary glTF file buffer.")
        if len(value) < byte_length:
            raise GltfError(
                f"buffer {i} byteLength {byte_length} exceeds {len(value)} bytes"
            )
        buffers.append(value[:byte_length])
    return Model(document, buffers, base_dir=root, binary=is_binary)


def load(path: str | Path) -> Model:
    path = Path(path)
    return loads(
        path.read_bytes(),
        base_dir=path.parent,
        binary=True if path.suffix.lower() == ".glb" else None,
    )

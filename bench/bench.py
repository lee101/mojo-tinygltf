"""Benchmarks against NumPy references derived from tinygltf data paths."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_tinygltf import Model  # noqa: E402


def best(fn, repeat=5):
    fn()
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return min(times), result


def accessor_model(raw, component_type, count, accessor_type, normalized=False):
    return Model(
        {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(raw)}],
            "bufferViews": [{"buffer": 0, "byteLength": len(raw)}],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": component_type,
                    "count": count,
                    "type": accessor_type,
                    "normalized": normalized,
                }
            ],
        },
        [raw],
    )


def main():
    available_cpus = None
    if hasattr(os, "sched_getaffinity"):
        available_cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {available_cpus[0]})
    rows = []

    count = 2_000_000
    float_source = np.linspace(-10, 10, count * 3, dtype=np.float32)
    float_model = accessor_model(
        float_source.tobytes(), 5126, count, "VEC3"
    )
    mojo, got = best(lambda: float_model.accessor(0))
    reference, expected = best(
        lambda: np.frombuffer(float_model.buffers[0], dtype="<f4")
        .reshape(count, 3)
        .astype(np.float64)
    )
    np.testing.assert_array_equal(got, expected)
    rows.append(("float32 VEC3 decode, 2M", mojo, reference, "NumPy"))

    count = 2_000_000
    int_source = np.arange(count * 4, dtype=np.int16)
    int_model = accessor_model(
        int_source.tobytes(), 5122, count, "VEC4", True
    )
    mojo, got = best(lambda: int_model.accessor(0))
    reference, expected = best(
        lambda: np.maximum(
            np.frombuffer(int_model.buffers[0], dtype="<i2")
            .reshape(count, 4)
            .astype(np.float64)
            / 32767.0,
            -1.0,
        )
    )
    np.testing.assert_array_equal(got, expected)
    rows.append(("normalized int16 VEC4, 2M", mojo, reference, "NumPy"))

    count = 1_000_000
    sparse_count = 100_000
    indices = np.arange(0, count, count // sparse_count, dtype=np.uint32)
    values = np.linspace(-1, 1, sparse_count * 3, dtype=np.float32)
    buffers = [indices.tobytes(), values.tobytes()]
    sparse_model = Model(
        {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(x)} for x in buffers],
            "bufferViews": [
                {"buffer": 0, "byteLength": len(buffers[0])},
                {"buffer": 1, "byteLength": len(buffers[1])},
            ],
            "accessors": [
                {
                    "componentType": 5126,
                    "count": count,
                    "type": "VEC3",
                    "sparse": {
                        "count": sparse_count,
                        "indices": {"bufferView": 0, "componentType": 5125},
                        "values": {"bufferView": 1},
                    },
                }
            ],
        },
        buffers,
    )

    def sparse_reference():
        result = np.zeros((count, 3), dtype=np.float64)
        result[indices] = values.reshape(-1, 3)
        return result

    mojo, got = best(lambda: sparse_model.accessor(0))
    reference, expected = best(sparse_reference)
    np.testing.assert_array_equal(got, expected)
    rows.append(("sparse VEC3 patch, 1M/100k", mojo, reference, "NumPy"))

    node_count = 20_000
    nodes = [
        {
            "translation": [0.001, 0.002, 0.003],
            **({"children": [i + 1]} if i + 1 < node_count else {}),
        }
        for i in range(node_count)
    ]
    hierarchy = Model({"asset": {"version": "2.0"}, "nodes": nodes})

    def world_reference():
        local = np.repeat(np.eye(4)[None, :, :], node_count, axis=0)
        local[:, :3, 3] = (0.001, 0.002, 0.003)
        result = np.empty_like(local)
        result[0] = local[0]
        for i in range(1, node_count):
            result[i] = result[i - 1] @ local[i]
        return result

    mojo, got = best(hierarchy.world_matrices, repeat=3)
    reference, expected = best(world_reference, repeat=3)
    np.testing.assert_allclose(got, expected, rtol=0, atol=2e-12)
    rows.append(("world transforms, 20k chain", mojo, reference, "NumPy loop"))

    if available_cpus is not None and len(available_cpus) > 1:
        cpu_count = min(8, len(available_cpus))
        os.sched_setaffinity(0, set(available_cpus[:cpu_count]))
        count = 2_000_000
        mojo, got = best(lambda: float_model.accessor(0))
        reference, expected = best(
            lambda: np.frombuffer(float_model.buffers[0], dtype="<f4")
            .reshape(count, 3)
            .astype(np.float64)
        )
        np.testing.assert_array_equal(got, expected)
        rows.append(
            (
                f"float32 VEC3 decode, 2M/{cpu_count} CPUs",
                mojo,
                reference,
                "NumPy",
            )
        )

        mojo, got = best(lambda: int_model.accessor(0))
        reference, expected = best(
            lambda: np.maximum(
                np.frombuffer(int_model.buffers[0], dtype="<i2")
                .reshape(count, 4)
                .astype(np.float64)
                / 32767.0,
                -1.0,
            )
        )
        np.testing.assert_array_equal(got, expected)
        rows.append(
            (
                f"normalized int16 VEC4, 2M/{cpu_count} CPUs",
                mojo,
                reference,
                "NumPy",
            )
        )

    print(f"Machine: {platform.processor() or platform.machine()} ({platform.system()})")
    print()
    print("| Kernel | Mojo | Reference | Speedup | Reference |")
    print("|---|---:|---:|---:|---|")
    for name, mojo, reference, ref_name in rows:
        print(
            f"| {name} | {mojo * 1000:.2f} ms | {reference * 1000:.2f} ms "
            f"| {reference / mojo:.2f}x | {ref_name} |"
        )


if __name__ == "__main__":
    main()

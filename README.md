# mojo-tinygltf

`mojo-tinygltf` is a standalone Mojo/Python port of a compute-oriented subset of
[tinygltf](https://github.com/syoyo/tinygltf). It reads and writes glTF 2.0
JSON and GLB, resolves external and embedded buffers, decodes typed accessors,
applies sparse accessors, and resolves node transforms. The JSON document
remains lossless, so unknown extensions and `extras` survive a round trip.

This is a derived work of tinygltf under the
[MIT License](https://github.com/syoyo/tinygltf/blob/release/LICENSE). The
ported revision and source functions are listed in [NOTICE](NOTICE).
It is not a drop-in replacement for tinygltf's C++ API.

## Coverage

Implemented:

- glTF 2.0 JSON and GLB v2 parsing and serialization
- external file, percent-encoded, base64, and non-base64 data URI buffers
- `bufferView` slicing and bounds validation
- scalar, vector, and matrix accessor metadata and all component constants
  accepted by tinygltf (`BYTE` through `DOUBLE`)
- interleaved accessors, accessor and buffer-view offsets, integer
  normalization, and sparse accessors with all three legal index widths
- local node matrices from either glTF column-major matrices or `T * R * S`,
  plus parent-first world-transform resolution
- lossless preservation of materials, meshes, skins, animations, cameras,
  extensions, and extras as JSON values

Not implemented:

- image decompression or encoding
- Draco, meshopt, BasisU, or other extension payload decoding
- animation sampling and skin deformation
- remote HTTP buffers
- tinygltf's typed C++ structs for every material and extension property
- full glTF schema validation beyond buffer, accessor, sparse, hierarchy, and
  GLB safety checks

There is no maintained Python binding to the current tinygltf revision used
here. Parity tests therefore use tinygltf's own published regression
expectations for GLB and serialization, cross-load generated GLB with
`trimesh`, and compare numeric kernels with direct NumPy translations of the
upstream code. Degenerate fixtures cover empty geometry, a single triangle,
duplicate vertices, zero-area faces, a non-manifold edge, and unreferenced
vertices.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` emits `dist/libmojo-tinygltf.so`. The Python package is made
available by the pixi environment's `PYTHONPATH`.

## Usage

```python
import struct

from mojo_tinygltf import Model, loads

positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
model = Model(
    {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(positions)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(positions)}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
            }
        ],
        "nodes": [{"translation": [1, 2, 3]}],
    },
    [positions],
)

print(model.accessor(0))
print(model.world_matrices()[0])

encoded = model.to_glb()
round_tripped = loads(encoded)
round_tripped.save("triangle.gltf")  # writes triangle.gltf and triangle.bin
```

Run it inside the environment with `pixi run python example.py`.

## How it works

Python handles JSON, paths, data URIs, and the lossless document model. Before
calling Mojo it resolves every index and proves every byte range is inside its
owning buffer view. The ABI also receives explicit input and output capacities
and checks them before reconstructing unsafe pointers.

The hot path uses one shared library and five allocation-free C ABI exports.
NumPy owns the input and output memory; 64-bit integer addresses cross the ABI
and are reconstructed as mutable `UnsafePointer` values in Mojo. Accessors are
decoded into C-contiguous row-major `float64` arrays. Sparse values overwrite
those arrays directly. Node matrices are stored as flat 16-value row-major
blocks for hierarchy multiplication, while explicit glTF matrices are
transposed once from their on-disk column-major order. Contiguous float32 and
int16 accessors use target-width SIMD with scalar remainder loops. Decodes over
one million components use the MAX thread pool with up to eight
affinity-available CPUs; smaller inputs stay serial. Packed float32 sparse VEC3
values use a two-lane SIMD conversion and a scalar tail for each scattered row.

No GPU path is included. Accessor conversion and sparse patching are
bandwidth-bound at under 0.2 flop per byte moved. A 4x4 world multiplication is
also below 2 flops per byte after matrix traffic is counted, and parent-child
dependencies prevent parallel evaluation of the benchmark's chain. GPU
transfer and launch overhead therefore cannot be justified for these kernels.

## Benchmarks

Measured on this machine (`x86_64`, Linux) by the command below. Times are the
best of five warm runs except the hierarchy benchmark, which uses three. The
benchmark asserts output parity before printing each result. The first four
rows pin the process to one CPU for comparison with the previous table. The
final two rows allow eight CPUs and exercise the thresholded parallel decoder.
Other factory services remain active on the host, so absolute times can vary
between runs.

| Kernel | Mojo | Reference | Speedup | Reference |
|---|---:|---:|---:|---|
| float32 VEC3 decode, 2M | 17.45 ms | 19.02 ms | 1.09x | NumPy |
| normalized int16 VEC4, 2M | 30.95 ms | 66.86 ms | 2.16x | NumPy |
| sparse VEC3 patch, 1M/100k | 4.69 ms | 6.25 ms | 1.33x | NumPy |
| world transforms, 20k chain | 50.32 ms | 55.76 ms | 1.11x | NumPy loop |
| float32 VEC3 decode, 2M/8 CPUs | 4.79 ms | 18.41 ms | 3.84x | NumPy |
| normalized int16 VEC4, 2M/8 CPUs | 6.63 ms | 71.06 ms | 10.72x | NumPy |

In this run, Mojo was faster than its reference in every row. The real
thread-pool path produced the largest gains on the multi-CPU dense decoders.

Reproduce the table only through the locked task:

```bash
pixi run bench
```

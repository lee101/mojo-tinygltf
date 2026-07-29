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
one million components use up to eight affinity-available CPUs; smaller inputs
stay serial.

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
| float32 VEC3 decode, 2M | 31.74 ms | 33.33 ms | 1.05x | NumPy |
| normalized int16 VEC4, 2M | 51.41 ms | 117.01 ms | 2.28x | NumPy |
| sparse VEC3 patch, 1M/100k | 12.94 ms | 18.58 ms | 1.44x | NumPy |
| world transforms, 20k chain | 72.12 ms | 61.45 ms | 0.85x | NumPy loop |
| float32 VEC3 decode, 2M/8 CPUs | 16.08 ms | 35.00 ms | 2.18x | NumPy |
| normalized int16 VEC4, 2M/8 CPUs | 13.63 ms | 110.83 ms | 8.13x | NumPy |

In this run, Mojo improved both dense conversions and the sparse patch. The
hierarchy kernel was slower than its reference. No GPU path is included.

Reproduce the table only through the locked task:

```bash
pixi run bench
```

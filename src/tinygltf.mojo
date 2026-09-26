"""Compute kernels for decoded glTF data.

The ABI is deliberately allocation-free. Python owns every buffer and passes
its address as an Int, which keeps all exported functions non-parametric.
"""

from std.sys import simd_width_of


comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime DPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime MATRIX_WIDTH = 4
comptime PARALLEL_MIN_ELEMENTS = 1_000_000


@always_inline
def component_size(component_type: Int) -> Int:
    if component_type == 5120 or component_type == 5121:
        return 1
    if component_type == 5122 or component_type == 5123:
        return 2
    if (
        component_type == 5124
        or component_type == 5125
        or component_type == 5126
    ):
        return 4
    if component_type == 5130:
        return 8
    return -1


@always_inline
def component_count(accessor_type: Int) -> Int:
    if accessor_type == 65:
        return 1
    if accessor_type == 2:
        return 2
    if accessor_type == 3:
        return 3
    if accessor_type == 4:
        return 4
    if accessor_type == 34:
        return 4
    if accessor_type == 35:
        return 9
    if accessor_type == 36:
        return 16
    return -1


@always_inline
def component_offset(
    component_index: Int, component_bytes: Int, accessor_type: Int
) -> Int:
    if accessor_type >= 34 and accessor_type <= 36:
        var rows = accessor_type - 32
        var column_bytes = rows * component_bytes
        var column_stride = ((column_bytes + 3) // 4) * 4
        return (
            component_index // rows * column_stride
            + component_index % rows * component_bytes
        )
    return component_index * component_bytes


@always_inline
def element_size(component_bytes: Int, accessor_type: Int) -> Int:
    if accessor_type >= 34 and accessor_type <= 36:
        var rows = accessor_type - 32
        var column_stride = ((rows * component_bytes + 3) // 4) * 4
        return rows * column_stride
    return component_count(accessor_type) * component_bytes


# tinygltf: tiny_gltf.h GetComponentSizeInBytes, GetNumComponentsInType,
# Accessor::ByteStride
@export("mtg_byte_stride")
def mtg_byte_stride(
    component_type: Int, accessor_type: Int, buffer_view_stride: Int
) abi("C") -> Int:
    var size = component_size(component_type)
    if size <= 0:
        return -1
    if buffer_view_stride == 0:
        return element_size(size, accessor_type)
    if buffer_view_stride % size != 0:
        return -1
    return buffer_view_stride


@always_inline
def read_component(
    src: BPtr, offset: Int, component_type: Int, normalized: Bool
) -> Float64:
    var value = 0.0
    if component_type == 5120:
        var u = Int(src[offset])
        if u >= 128:
            u -= 256
        value = Float64(u)
        if normalized:
            value = max(value / 127.0, -1.0)
    elif component_type == 5121:
        value = Float64(src[offset])
        if normalized:
            value /= 255.0
    elif component_type == 5122:
        var u = Int((src + offset).bitcast[UInt16]().load[alignment=1]())
        if u >= 32768:
            u -= 65536
        value = Float64(u)
        if normalized:
            value = max(value / 32767.0, -1.0)
    elif component_type == 5123:
        value = Float64((src + offset).bitcast[UInt16]().load[alignment=1]())
        if normalized:
            value /= 65535.0
    elif component_type == 5124:
        var u = Int64((src + offset).bitcast[UInt32]().load[alignment=1]())
        if u >= 2147483648:
            u -= 4294967296
        value = Float64(u)
        if normalized:
            value = max(value / 2147483647.0, -1.0)
    elif component_type == 5125:
        value = Float64((src + offset).bitcast[UInt32]().load[alignment=1]())
        if normalized:
            value /= 4294967295.0
    elif component_type == 5126:
        value = Float64((src + offset).bitcast[Float32]().load[alignment=1]())
    elif component_type == 5130:
        value = (src + offset).bitcast[Float64]().load[alignment=1]()
    return value


# tinygltf: examples/raytrace/gltf-loader.cc arrayAdapter and
# tiny_gltf.h Accessor::ByteStride
@export("mtg_decode_accessor")
def mtg_decode_accessor(
    src_addr: Int,
    src_bytes: Int,
    dst_addr: Int,
    dst_elements: Int,
    count: Int,
    components: Int,
    component_type: Int,
    accessor_type: Int,
    byte_offset: Int,
    byte_stride: Int,
    normalized: Int,
    workers: Int,
) abi("C") -> Int:
    if (
        count < 0
        or components <= 0
        or src_addr == 0
        or dst_addr == 0
        or src_bytes < 0
        or dst_elements < 0
        or byte_offset < 0
    ):
        return -1
    var size = component_size(component_type)
    var expected_components = component_count(accessor_type)
    var packed_size = element_size(size, accessor_type)
    if (
        size <= 0
        or expected_components != components
        or byte_stride < packed_size
        or byte_stride % size != 0
    ):
        return -2
    if count > dst_elements // components:
        return -3
    if count > 0:
        if byte_offset > src_bytes or packed_size > src_bytes - byte_offset:
            return -4
        if count - 1 > (src_bytes - byte_offset - packed_size) // byte_stride:
            return -4
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = DPtr(unsafe_from_address=dst_addr)
    if (
        component_type == 5126
        and accessor_type < 32
        and byte_stride == components * 4
    ):
        var packed = (src + byte_offset).bitcast[Float32]()
        var total = count * components
        if total > PARALLEL_MIN_ELEMENTS and workers > 1:
            var task_count = min(
                workers,
                (total + PARALLEL_MIN_ELEMENTS - 1) // PARALLEL_MIN_ELEMENTS,
            )
            var block = (
                ((total + task_count - 1) // task_count + W - 1) // W * W
            )

            for task in range(task_count):
                var start = task * block
                var end = min(start + block, total)
                var k = start
                while k + W <= end:
                    dst.store(
                        k,
                        packed.load[width=W, alignment=1](k).cast[
                            DType.float64
                        ](),
                    )
                    k += W
                while k < end:
                    dst[k] = Float64(packed.load[alignment=1](k))
                    k += 1
            return 0
        var k = 0
        while k + W <= total:
            dst.store(
                k,
                packed.load[width=W, alignment=1](k).cast[DType.float64](),
            )
            k += W
        while k < total:
            dst[k] = Float64(packed.load[alignment=1](k))
            k += 1
        return 0
    if (
        component_type == 5122
        and accessor_type < 32
        and byte_stride == components * 2
    ):
        var packed = (src + byte_offset).bitcast[Int16]()
        var total = count * components
        if total > PARALLEL_MIN_ELEMENTS and workers > 1:
            var task_count = min(
                workers,
                (total + PARALLEL_MIN_ELEMENTS - 1) // PARALLEL_MIN_ELEMENTS,
            )
            var block = (
                ((total + task_count - 1) // task_count + W - 1) // W * W
            )

            for task in range(task_count):
                var start = task * block
                var end = min(start + block, total)
                var k = start
                while k + W <= end:
                    var values = packed.load[width=W, alignment=1](k).cast[
                        DType.float64
                    ]()
                    if normalized != 0:
                        values = max(values / 32767.0, -1.0)
                    dst.store(k, values)
                    k += W
                while k < end:
                    dst[k] = read_component(
                        src,
                        byte_offset + k * 2,
                        component_type,
                        normalized != 0,
                    )
                    k += 1
            return 0
        var k = 0
        while k + W <= total:
            var values = packed.load[width=W, alignment=1](k).cast[
                DType.float64
            ]()
            if normalized != 0:
                values = max(values / 32767.0, -1.0)
            dst.store(k, values)
            k += W
        while k < total:
            dst[k] = read_component(
                src, byte_offset + k * 2, component_type, normalized != 0
            )
            k += 1
        return 0
    for i in range(count):
        var base = byte_offset + i * byte_stride
        for j in range(components):
            dst[i * components + j] = read_component(
                src,
                base + component_offset(j, size, accessor_type),
                component_type,
                normalized != 0,
            )
    return 0


@always_inline
def read_sparse_index(src: BPtr, offset: Int, component_type: Int) -> Int64:
    if component_type == 5121:
        return Int64(src[offset])
    if component_type == 5123:
        return Int64((src + offset).bitcast[UInt16]().load[alignment=1]())
    if component_type == 5125:
        return Int64((src + offset).bitcast[UInt32]().load[alignment=1]())
    return -1


# tinygltf: examples/glview/glview.cc SetupMeshState sparse accessor patch loop
@export("mtg_apply_sparse")
def mtg_apply_sparse(
    indices_addr: Int,
    indices_bytes: Int,
    values_addr: Int,
    values_bytes: Int,
    dst_addr: Int,
    dst_elements: Int,
    sparse_count: Int,
    accessor_count: Int,
    components: Int,
    index_component_type: Int,
    value_component_type: Int,
    accessor_type: Int,
    indices_offset: Int,
    values_offset: Int,
    normalized: Int,
) abi("C") -> Int:
    if (
        sparse_count < 0
        or components <= 0
        or indices_addr == 0
        or values_addr == 0
        or dst_addr == 0
        or indices_bytes < 0
        or values_bytes < 0
        or dst_elements < 0
        or indices_offset < 0
        or values_offset < 0
    ):
        return -1
    var index_size = component_size(index_component_type)
    var value_size = component_size(value_component_type)
    if (
        (index_component_type != 5121)
        and (index_component_type != 5123)
        and (index_component_type != 5125)
    ):
        return -2
    if (
        value_size <= 0
        or component_count(accessor_type) != components
        or accessor_count < 0
        or sparse_count > accessor_count
        or accessor_count > dst_elements // components
    ):
        return -2
    var sparse_value_size = element_size(value_size, accessor_type)
    if sparse_count > 0:
        if (
            indices_offset > indices_bytes
            or index_size > indices_bytes - indices_offset
            or sparse_count - 1
            > (indices_bytes - indices_offset - index_size) // index_size
        ):
            return -4
        if (
            values_offset > values_bytes
            or sparse_value_size > values_bytes - values_offset
            or sparse_count - 1
            > (values_bytes - values_offset - sparse_value_size)
            // sparse_value_size
        ):
            return -4
    var indices = BPtr(unsafe_from_address=indices_addr)
    var values = BPtr(unsafe_from_address=values_addr)
    var dst = DPtr(unsafe_from_address=dst_addr)
    var previous_index = Int64(-1)
    if value_component_type == 5126 and accessor_type == 3:
        comptime VEC3_SIMD_WIDTH = 2
        comptime VEC3_WIDTH = 3
        var packed = (values + values_offset).bitcast[Float32]()
        for i in range(sparse_count):
            var index = read_sparse_index(
                indices, indices_offset + i * index_size, index_component_type
            )
            if index < 0 or index >= Int64(accessor_count):
                return -3
            if index <= previous_index:
                return -5
            previous_index = index
            dst.store(
                Int(index) * VEC3_WIDTH,
                packed.load[width=VEC3_SIMD_WIDTH, alignment=1](
                    i * VEC3_WIDTH
                ).cast[DType.float64](),
            )
            dst[Int(index) * VEC3_WIDTH + VEC3_SIMD_WIDTH] = Float64(
                packed.load[alignment=1](
                    i * VEC3_WIDTH + VEC3_SIMD_WIDTH
                )
            )
        return 0
    for i in range(sparse_count):
        var index = read_sparse_index(
            indices, indices_offset + i * index_size, index_component_type
        )
        if index < 0 or index >= Int64(accessor_count):
            return -3
        if index <= previous_index:
            return -5
        previous_index = index
        var value_base = values_offset + i * element_size(
            value_size, accessor_type
        )
        for j in range(components):
            dst[Int(index) * components + j] = read_component(
                values,
                value_base + component_offset(j, value_size, accessor_type),
                value_component_type,
                normalized != 0,
            )
    return 0


# tinygltf: tiny_gltf.h ParseNode; glTF node transform is M = T * R * S
@export("mtg_local_matrices")
def mtg_local_matrices(
    trs_addr: Int,
    trs_elements: Int,
    matrix_addr: Int,
    matrix_elements: Int,
    has_matrix_addr: Int,
    has_matrix_bytes: Int,
    dst_addr: Int,
    dst_elements: Int,
    count: Int,
) abi("C") -> Int:
    if (
        count < 0
        or trs_addr == 0
        or matrix_addr == 0
        or has_matrix_addr == 0
        or dst_addr == 0
        or count > trs_elements // 10
        or count > matrix_elements // 16
        or count > has_matrix_bytes
        or count > dst_elements // 16
    ):
        return -1
    var trs = DPtr(unsafe_from_address=trs_addr)
    var matrices = DPtr(unsafe_from_address=matrix_addr)
    var has_matrix = BPtr(unsafe_from_address=has_matrix_addr)
    var dst = DPtr(unsafe_from_address=dst_addr)
    for i in range(count):
        var d = i * 16
        if has_matrix[i] != 0:
            for row in range(4):
                for col in range(4):
                    dst[d + row * 4 + col] = matrices[d + col * 4 + row]
            continue
        var t = i * 10
        var x = trs[t + 3]
        var y = trs[t + 4]
        var z = trs[t + 5]
        var w = trs[t + 6]
        var sx = trs[t + 7]
        var sy = trs[t + 8]
        var sz = trs[t + 9]
        dst[d] = (1.0 - 2.0 * (y * y + z * z)) * sx
        dst[d + 1] = (2.0 * (x * y - z * w)) * sy
        dst[d + 2] = (2.0 * (x * z + y * w)) * sz
        dst[d + 3] = trs[t]
        dst[d + 4] = (2.0 * (x * y + z * w)) * sx
        dst[d + 5] = (1.0 - 2.0 * (x * x + z * z)) * sy
        dst[d + 6] = (2.0 * (y * z - x * w)) * sz
        dst[d + 7] = trs[t + 1]
        dst[d + 8] = (2.0 * (x * z - y * w)) * sx
        dst[d + 9] = (2.0 * (y * z + x * w)) * sy
        dst[d + 10] = (1.0 - 2.0 * (x * x + y * y)) * sz
        dst[d + 11] = trs[t + 2]
        dst[d + 12] = 0.0
        dst[d + 13] = 0.0
        dst[d + 14] = 0.0
        dst[d + 15] = 1.0
    return 0


# tinygltf: examples/common/matrix.cc Matrix::Mult
@export("mtg_resolve_world")
def mtg_resolve_world(
    local_addr: Int,
    local_elements: Int,
    parents_addr: Int,
    parents_elements: Int,
    order_addr: Int,
    order_elements: Int,
    dst_addr: Int,
    dst_elements: Int,
    count: Int,
) abi("C") -> Int:
    if (
        count < 0
        or local_addr == 0
        or parents_addr == 0
        or order_addr == 0
        or dst_addr == 0
        or count > local_elements // 16
        or count > parents_elements
        or count > order_elements
        or count > dst_elements // 16
    ):
        return -1
    var local = DPtr(unsafe_from_address=local_addr)
    var parents = IPtr(unsafe_from_address=parents_addr)
    var order = IPtr(unsafe_from_address=order_addr)
    var dst = DPtr(unsafe_from_address=dst_addr)
    for pos in range(count):
        var node = Int(order[pos])
        if node < 0 or node >= count:
            return -2
        var parent = Int(parents[node])
        var d = node * 16
        if parent < 0:
            for row in range(4):
                var offset = d + row * MATRIX_WIDTH
                dst.store(
                    offset,
                    local.load[width=MATRIX_WIDTH, alignment=1](offset),
                )
        else:
            if parent >= count:
                return -3
            var p = parent * 16
            var local_row_0 = local.load[width=MATRIX_WIDTH, alignment=1](d)
            var local_row_1 = local.load[width=MATRIX_WIDTH, alignment=1](
                d + MATRIX_WIDTH
            )
            var local_row_2 = local.load[width=MATRIX_WIDTH, alignment=1](
                d + 2 * MATRIX_WIDTH
            )
            var local_row_3 = local.load[width=MATRIX_WIDTH, alignment=1](
                d + 3 * MATRIX_WIDTH
            )
            for row in range(4):
                var parent_row = p + row * MATRIX_WIDTH
                var values = (
                    local_row_0 * dst[parent_row]
                    + local_row_1 * dst[parent_row + 1]
                    + local_row_2 * dst[parent_row + 2]
                    + local_row_3 * dst[parent_row + 3]
                )
                dst.store(d + row * MATRIX_WIDTH, values)
    return 0

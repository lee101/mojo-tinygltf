import numpy as np
import pytest

from helpers import triangle_model


@pytest.mark.parametrize(
    "name,positions,indices",
    [
        ("empty", [], []),
        ("single triangle", [0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2]),
        ("duplicate vertices", [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 2]),
        ("zero area face", [0, 0, 0, 1, 0, 0, 2, 0, 0], [0, 1, 2]),
        (
            "non-manifold edge",
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1],
            [0, 1, 2, 1, 0, 3, 0, 1, 4],
        ),
        (
            "unreferenced vertex",
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 9, 9, 9],
            [0, 1, 2],
        ),
    ],
)
def test_mesh_storage_is_lossless_for_degenerate_inputs(name, positions, indices):
    source = triangle_model(positions, indices)
    parsed = type(source).from_dict(source.document, source.buffers)
    np.testing.assert_array_equal(
        parsed.accessor(0), np.asarray(positions, dtype=float).reshape(-1, 3)
    )
    np.testing.assert_array_equal(parsed.accessor(1), indices)

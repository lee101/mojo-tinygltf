import math

import numpy as np
import pytest

from mojo_tinygltf import GltfError, Model


def model(nodes):
    return Model({"asset": {"version": "2.0"}, "nodes": nodes})


def test_default_node_is_identity():
    np.testing.assert_array_equal(model([{}]).local_matrices()[0], np.eye(4))


def test_trs_matches_numpy_reference():
    angle = math.pi / 2
    node = {
        "translation": [1, 2, 3],
        "rotation": [0, 0, math.sin(angle / 2), math.cos(angle / 2)],
        "scale": [2, 3, 4],
    }
    expected = np.array(
        [[0, -3, 0, 1], [2, 0, 0, 2], [0, 0, 4, 3], [0, 0, 0, 1]],
        dtype=float,
    )
    np.testing.assert_allclose(model([node]).local_matrices()[0], expected, atol=1e-15)


def test_explicit_matrix_is_column_major_and_wins_over_trs():
    row_major = np.arange(16, dtype=float).reshape(4, 4)
    node = {
        "matrix": row_major.T.ravel().tolist(),
        "translation": [100, 200, 300],
    }
    np.testing.assert_array_equal(model([node]).local_matrices()[0], row_major)


def test_world_transform_parent_times_local():
    nodes = [
        {"translation": [1, 0, 0], "children": [1]},
        {"translation": [0, 2, 0], "children": [2]},
        {"translation": [0, 0, 3]},
    ]
    world = model(nodes).world_matrices()
    np.testing.assert_array_equal(world[:, :3, 3], [[1, 0, 0], [1, 2, 0], [1, 2, 3]])


def test_forest_roots_are_independent():
    world = model(
        [{"translation": [1, 0, 0]}, {"translation": [0, 2, 0]}]
    ).world_matrices()
    np.testing.assert_array_equal(world[:, :3, 3], [[1, 0, 0], [0, 2, 0]])


def test_empty_nodes():
    value = model([]).world_matrices()
    assert value.shape == (0, 4, 4)


def test_cycle_rejected():
    with pytest.raises(GltfError, match="cycle"):
        model([{"children": [1]}, {"children": [0]}]).world_matrices()


def test_multiple_parents_rejected():
    with pytest.raises(GltfError, match="multiple parents"):
        model([{"children": [2]}, {"children": [2]}, {}]).world_matrices()


def test_invalid_child_rejected():
    with pytest.raises(GltfError, match="invalid child"):
        model([{"children": [1]}]).world_matrices()


def test_invalid_trs_length_rejected():
    with pytest.raises(GltfError, match="must have 4"):
        model([{"rotation": [0, 0, 1]}]).local_matrices()

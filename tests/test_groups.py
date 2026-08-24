import numpy as np

import mefikit as mf


def test_create_group_from_selection(umesh2):
    umesh2.groups["all"] = mf.sel.rect([-1.0, -1.0], [4.0, 1.0])
    assert "all" in umesh2.groups
    assert len(umesh2.groups["all"]) == 196


def test_create_group_filters_correctly(umesh2):
    umesh2.groups["left"] = mf.sel.rect([-1.0, -1.0], [1.0, 1.0])
    assert "left" in umesh2.groups
    assert len(umesh2.groups["left"]) == 49


def test_group_add_with_selection(umesh2):
    umesh2.groups["a"] = mf.sel.rect([-1.0, -1.0], [1.0, 1.0])
    assert len(umesh2.groups["a"]) == 49
    umesh2.groups["a"].add(mf.sel.rect([1.0, -1.0], [2.0, 1.0]))
    assert len(umesh2.groups["a"]) == 98


def test_group_add_with_ids(umesh2):
    umesh2.groups["z"] = {"QUAD4": np.array([0, 1])}
    ids = dict(umesh2.groups["z"].ids())
    assert set(ids) == {"QUAD4"}
    assert np.array_equal(ids["QUAD4"], [0, 1])


def test_group_remove_with_selection(umesh2):
    umesh2.groups["zone"] = mf.sel.rect([-1.0, -1.0], [4.0, 1.0])
    assert len(umesh2.groups["zone"]) == 196
    umesh2.groups["zone"].remove(mf.sel.rect([3.0, -1.0], [4.0, 1.0]))
    assert len(umesh2.groups["zone"]) == 147


def test_group_remove_with_ids(umesh2):
    umesh2.groups["zone"] = mf.sel.rect([-1.0, -1.0], [4.0, 1.0])
    n_before = len(umesh2.groups["zone"])
    umesh2.groups["zone"].remove({"QUAD4": np.array([0])})
    assert len(umesh2.groups["zone"]) == n_before - 1


def test_delete_group(umesh2):
    umesh2.groups["tmp"] = mf.sel.rect([-1.0, -1.0], [4.0, 1.0])
    assert "tmp" in umesh2.groups
    del umesh2.groups["tmp"]
    assert "tmp" not in umesh2.groups


def test_rename_group(umesh2):
    umesh2.groups["old"] = mf.sel.rect([-1.0, -1.0], [4.0, 1.0])
    ids_before = dict(umesh2.groups["old"].ids())
    umesh2.groups.rename("old", "new")
    assert "old" not in umesh2.groups
    assert "new" in umesh2.groups
    ids_after = dict(umesh2.groups["new"].ids())
    assert set(ids_after) == set(ids_before)
    for et in ids_before:
        assert np.array_equal(ids_after[et], ids_before[et])


def test_groups_mapping(umesh2):
    umesh2.groups["g1"] = {"QUAD4": np.array([0])}
    umesh2.groups["g2"] = {"QUAD4": np.array([1, 2])}
    assert set(umesh2.groups.keys()) == {"g1", "g2"}
    assert [name for name, _ in umesh2.groups.items()] == ["g1", "g2"]
    assert len(umesh2.groups) == 2


def test_groups_empty():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("TRI3", np.array([[0, 1, 2]], dtype=np.uint))
    assert len(mesh.groups) == 0


def test_sel_group_selection(umesh2):
    umesh2.groups["wall"] = mf.sel.rect([-1.0, -1.0], [1.0, 1.0])
    wall_sel = umesh2.select(mf.sel.group("wall"))
    assert len(wall_sel) == 49
    wall_mesh = wall_sel.to_mesh()
    assert wall_mesh.num_elements() == 49


def test_sel_exclude_group(umesh2):
    umesh2.groups["wall"] = mf.sel.rect([-1.0, -1.0], [1.0, 1.0])
    non_wall = umesh2.select(mf.sel.exclude_group("wall"))
    assert len(non_wall) == 147


def test_sel_types():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("TRI3", np.array([[0, 1, 2]], dtype=np.uint))
    mesh.add_regular_block("SEG2", np.array([[0, 1]], dtype=np.uint))
    result = mesh.select(mf.sel.types(["TRI3"]))
    assert len(result) == 1


def test_group_composition(umesh2):
    umesh2.groups["left"] = mf.sel.rect([-1.0, -1.0], [1.0, 1.0])
    umesh2.groups["bottom"] = mf.sel.rect([-1.0, -1.0], [4.0, 0.5])
    combined = umesh2.select(mf.sel.group("left") & mf.sel.group("bottom"))
    assert len(combined) == 24

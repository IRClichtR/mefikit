import numpy as np

import mefikit as mf


def test_select_to_group(umesh2):
    umesh2.select_to_group("all", mf.sel.rect([-1.0, -1.0], [4.0, 1.0]))
    assert umesh2.has_group("all")
    assert "all" in umesh2.group_names()


def test_select_to_group_filters_correctly(umesh2):
    umesh2.select_to_group("left", mf.sel.rect([-1.0, -1.0], [1.0, 1.0]))
    assert umesh2.has_group("left")
    names = umesh2.group_names()
    assert "left" in names


def test_add_to_group_with_selection(umesh2):
    umesh2.select_to_group("a", mf.sel.rect([-1.0, -1.0], [1.0, 1.0]))
    umesh2.add_to_group("a", mf.sel.rect([1.0, -1.0], [2.0, 1.0]))
    assert umesh2.has_group("a")


def test_add_to_group_with_ids(umesh2):
    umesh2.add_to_group("z", {"QUAD4": np.array([0, 1])})
    assert umesh2.has_group("z")


def test_remove_from_group_with_selection(umesh2):
    umesh2.select_to_group("zone", mf.sel.rect([-1.0, -1.0], [4.0, 1.0]))
    umesh2.remove_from_group("zone", mf.sel.rect([3.0, -1.0], [4.0, 1.0]))
    assert umesh2.has_group("zone")


def test_remove_from_group_with_ids(umesh2):
    umesh2.select_to_group("zone", mf.sel.rect([-1.0, -1.0], [4.0, 1.0]))
    umesh2.remove_from_group("zone", {"QUAD4": np.array([0])})
    assert umesh2.has_group("zone")


def test_delete_group(umesh2):
    umesh2.select_to_group("tmp", mf.sel.rect([-1.0, -1.0], [4.0, 1.0]))
    assert umesh2.has_group("tmp")
    umesh2.delete_group("tmp")
    assert not umesh2.has_group("tmp")


def test_rename_group(umesh2):
    umesh2.select_to_group("old", mf.sel.rect([-1.0, -1.0], [4.0, 1.0]))
    umesh2.rename_group("old", "new")
    assert not umesh2.has_group("old")
    assert umesh2.has_group("new")


def test_set_groups(umesh2):
    umesh2.set_groups(
        {"g1": {"QUAD4": np.array([0])}, "g2": {"QUAD4": np.array([1, 2])}}
    )
    assert umesh2.has_group("g1")
    assert umesh2.has_group("g2")
    assert set(umesh2.group_names()) == {"g1", "g2"}


def test_group_names_empty():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("TRI3", np.array([[0, 1, 2]], dtype=np.uint))
    assert mesh.group_names() == []


def test_sel_group_selection(umesh2):
    umesh2.select_to_group("wall", mf.sel.rect([-1.0, -1.0], [1.0, 1.0]))
    wall_mesh = umesh2.select(mf.sel.group("wall"))
    assert wall_mesh.num_elements() > 0


def test_sel_exclude_group(umesh2):
    umesh2.select_to_group("wall", mf.sel.rect([-1.0, -1.0], [1.0, 1.0]))
    non_wall = umesh2.select(mf.sel.exclude_group("wall"))
    assert non_wall.num_elements() > 0


def test_sel_types():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("TRI3", np.array([[0, 1, 2]], dtype=np.uint))
    mesh.add_regular_block("SEG2", np.array([[0, 1]], dtype=np.uint))
    result = mesh.select(mf.sel.types(["TRI3"]))
    assert result.num_elements() == 1


def test_group_composition(umesh2):
    umesh2.select_to_group("left", mf.sel.rect([-1.0, -1.0], [1.0, 1.0]))
    umesh2.select_to_group("bottom", mf.sel.rect([-1.0, -1.0], [4.0, 0.5]))
    combined = umesh2.select(mf.sel.group("left") & mf.sel.group("bottom"))
    assert combined.num_elements() > 0

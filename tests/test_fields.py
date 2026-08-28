import numpy as np
import pytest

import mefikit as mf

COORDS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
)
BLOCK = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.uint)


@pytest.fixture()
def quad2():
    mesh = mf.UMesh(COORDS.copy())
    mesh.add_regular_block("QUAD4", BLOCK.copy())
    mesh.set_field("T", {"QUAD4": np.array([[10.0], [20.0]])})
    return mesh


# --- creation ---


def test_create_field_from_float(quad2):
    quad2.fields["Z"] = 0.5
    vals = quad2.fields["Z"].values()["QUAD4"]
    assert vals.shape == (2,)
    assert np.allclose(vals, 0.5)


def test_create_field_from_array_1d(quad2):
    quad2.fields["Z"] = np.array([1.0, 2.0])
    assert np.allclose(quad2.fields["Z"].values()["QUAD4"], [1.0, 2.0])


def test_create_field_from_array_2d(quad2):
    quad2.fields["Z"] = np.array([[1.0], [2.0]])
    assert np.allclose(quad2.fields["Z"].values()["QUAD4"], [[1.0], [2.0]])


def test_create_field_from_per_block_dict(quad2):
    quad2.fields["Z"] = {"QUAD4": np.array([[3.0], [4.0]])}
    assert np.allclose(quad2.fields["Z"].values()["QUAD4"], [[3.0], [4.0]])


def test_create_field_from_field_expr(quad2):
    quad2.fields["T2"] = mf.Field("T") * 2
    assert np.allclose(quad2.fields["T2"].values()["QUAD4"].ravel(), [20.0, 40.0])


def test_create_field_from_field_name(quad2):
    quad2.fields["copy"] = "T"
    assert np.allclose(quad2.fields["copy"].values()["QUAD4"].ravel(), [10.0, 20.0])


def test_create_field_from_unknown_name_raises(quad2):
    with pytest.raises(ValueError):
        quad2.fields["X"] = "nope"


# --- mapping protocol ---


def test_fields_mapping_protocol(quad2):
    assert quad2.fields.keys() == ["T"]
    assert [name for name, _ in quad2.fields.items()] == ["T"]
    assert list(iter(quad2.fields)) == ["T"]
    assert len(quad2.fields) == 1
    assert "T" in quad2.fields
    assert "nope" not in quad2.fields


def test_fields_to_dict_matches_values(quad2):
    exported = quad2.fields.to_dict()
    assert set(exported) == {"T"}
    ref = quad2.fields["T"]
    for etype, arr in ref.values().items():
        assert np.array_equal(exported["T"][etype], arr)


def test_rename_missing_raises(quad2):
    with pytest.raises(KeyError):
        quad2.fields.rename("nope", "X")


def test_rename_collision_raises(quad2):
    with pytest.raises(ValueError):
        quad2.fields.rename("T", "T")


def test_delete_missing_raises(quad2):
    with pytest.raises(KeyError):
        del quad2.fields["nope"]


# --- field handle ---


def test_ref_metadata(quad2):
    ref = quad2.fields["T"]
    assert tuple(ref.shape) == (1,)
    assert ref.dimension() == 2
    assert len(ref) == 2


def test_ref_values_and_numpy_agree(quad2):
    ref = quad2.fields["T"]
    assert np.array_equal(np.asarray(ref.numpy()), ref.values()["QUAD4"])


def test_getitem_gathers_rows(quad2):
    got = quad2.fields["T"][{"QUAD4": [1]}]
    assert np.allclose(got["QUAD4"], [[20.0]])


def test_setitem_wildcards(quad2):
    quad2.fields["T"][...] = 7.0
    assert np.allclose(quad2.fields["T"].values()["QUAD4"], [[7.0], [7.0]])
    quad2.fields["T"][:] = 9.0
    assert np.allclose(quad2.fields["T"].values()["QUAD4"], [[9.0], [9.0]])


def test_setitem_with_selection(quad2):
    quad2.fields["T"][mf.sel.ids({"QUAD4": [0]})] = 42.0
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [42.0, 20.0])


def test_setitem_with_field_expr_rhs(quad2):
    quad2.fields["T"][...] = mf.Field("T") + 1
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [11.0, 21.0])


def test_setitem_with_field_name_rhs(quad2):
    quad2.fields["T"][...] = "T"
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [10.0, 20.0])
    quad2.fields["T"][{"QUAD4": [0]}] = "T"
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [10.0, 20.0])


def test_setitem_unknown_name_raises(quad2):
    with pytest.raises(ValueError):
        quad2.fields["T"][...] = "nope"


def test_setitem_row_order_regression(quad2):
    quad2.fields["T"][{"QUAD4": [1, 0]}] = np.array([[30.0], [40.0]])
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [40.0, 30.0])


def test_setitem_per_block_rhs(quad2):
    quad2.fields["T"][{"QUAD4": [0]}] = {"QUAD4": np.array([[5.0]])}
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [5.0, 20.0])


def test_ref_reductions(quad2):
    ref = quad2.fields["T"]
    v = np.array([10.0, 20.0])
    assert np.allclose(ref.min(), v.min())
    assert np.allclose(ref.max(), v.max())
    assert np.allclose(ref.sum(), v.sum())
    assert np.allclose(ref.mean(), v.mean())
    assert np.allclose(ref.var(), v.var())
    assert np.allclose(ref.var(ddof=1), v.var(ddof=1))
    assert np.allclose(ref.std(ddof=1), v.std(ddof=1))


def test_ref_integral(quad2):
    # both quads have measure 1.0 -> integral(T) = 10 * 1 + 20 * 1
    assert np.allclose(quad2.fields["T"].integral(), 30.0)


# --- lazy selections ---


def test_select_ids_len_repr(quad2):
    result = quad2.select(mf.sel.rect([0.5, -1.0], [3.0, 2.0]))
    ids = dict(result.ids())
    assert ids["QUAD4"].tolist() == [0, 1]
    assert len(result) == 2
    assert repr(result)


def test_select_reductions(quad2):
    result = quad2.select(mf.sel.rect([0.5, -1.0], [3.0, 2.0]))
    assert np.allclose(result.min("T"), [10.0])
    assert np.allclose(result.mean(mf.Field("T") * 2), [30.0])
    assert np.allclose(result.sum("T"), [30.0])
    assert np.allclose(result.var(mf.Field("T"), 1), [50.0])
    assert np.allclose(result.std(mf.Field("T"), 1), [np.sqrt(50.0)])


def test_select_unknown_field_raises(quad2):
    result = quad2.select(mf.sel.all())
    with pytest.raises(ValueError):
        result.mean("nope")
    with pytest.raises(ValueError):
        result.integral("nope")


def test_select_integral(quad2):
    result = quad2.select(mf.sel.ids({"QUAD4": [1]}))
    assert np.allclose(result.integral("T"), 20.0)


def test_select_to_mesh_fields(quad2):
    result = quad2.select(mf.sel.rect([0.5, -1.0], [3.0, 2.0]))
    bare = result.to_mesh(with_fields=False)
    assert bare.num_elements() == 2
    assert bare.fields.keys() == []
    carried = result.to_mesh()
    assert carried.fields.keys() == ["T"]
    assert np.allclose(carried.fields["T"].values()["QUAD4"].ravel(), [10.0, 20.0])


def test_sel_all(quad2):
    assert len(quad2.select(mf.sel.all())) == 2

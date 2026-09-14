import numpy as np
import pytest

from boonza._columns import ColumnTable


def make():
    t = ColumnTable(builtin=["x"])
    t.add_column("x", "int", default=-1)
    t.add_column("name", "str")
    t.add_column("pos", "float", shape=(3,))
    return t


def test_append_grows_and_fills_defaults():
    t = make()
    for k in range(100):
        t.append(1, {"x": k})
    assert len(t) == 100
    assert t.column("x").tolist() == list(range(100))
    assert t.column("name").tolist() == [""] * 100
    assert t.column("pos").shape == (100, 3)


def test_failed_append_rolls_back():
    t = make()
    t.append(2, {"x": [1, 2]})
    with pytest.raises(KeyError):
        t.append(1, {"x": 3, "missing": 1})
    assert len(t) == 2


def test_keep_resets_tail_to_default():
    t = make()
    t.append(4, {"x": [0, 1, 2, 3], "name": ["a", "b", "c", "d"]})
    t.keep([3, 1])
    assert t.column("x").tolist() == [3, 1]
    assert t.column("name").tolist() == ["d", "b"]
    t.append(1)
    assert t.column("x")[-1] == -1
    assert t.column("name")[-1] == ""


def test_take_and_extend_union_columns():
    a = make()
    a.append(2, {"x": [5, 6]})
    b = make()
    b.add_column("extra", float)
    b.append(1, {"x": 7, "extra": 1.5})
    a.extend(b)
    assert a.column("x").tolist() == [5, 6, 7]
    assert a.column("extra").tolist() == [0.0, 0.0, 1.5]
    c = a.take(np.array([False, True, True]))
    assert c.column("x").tolist() == [6, 7]


def test_kind_conflict_rejected():
    t = make()
    with pytest.raises(ValueError):
        t.add_column("name", float)
    with pytest.raises(ValueError):
        t.del_column("x")

import sqlite3

import numpy as np
import pytest
from canonical import assert_same, canon
from conftest import msys_file

import boonza
from boonza.io import dms
from boonza.io._sqlite_scan import SqliteFile


def _root(con, table):
    return con.execute("select rootpage from sqlite_master where name=?", (table,)).fetchone()[0]


def test_scan_matches_sqlite3(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.execute("create table t (id integer primary key, a integer, b float, c text)")
    rows = [
        (i, (-1) ** i * (i * 7919) ** 2, i * 0.5 if i % 3 else float(i), f"α{i % 17} ")
        for i in range(30000)
    ]
    rows += [(30000, None, None, None), (30001, -(2**63), 1e300, ""), (30002, 2**63 - 1, -0.0, "x")]
    con.executemany("insert into t values (?, ?, ?, ?)", rows)
    con.commit()
    ids, a, b, c = SqliteFile(np.fromfile(path, np.uint8)).read(
        _root(con, "t"), ["int", "int", "float", "str"], rowid_col=0
    )
    ref = con.execute("select id, a, b, c from t").fetchall()
    assert ids.tolist() == [r[0] for r in ref]
    assert a.tolist() == [r[1] or 0 for r in ref]
    assert b.tolist() == [r[2] or 0.0 for r in ref]
    assert c.tolist() == [r[3] or "" for r in ref]


def test_scan_declines_overflow_records(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.execute("create table t (s text)")
    con.executemany("insert into t values (?)", [("x" * 10000,)] * 3)
    con.commit()
    assert SqliteFile(np.fromfile(path, np.uint8)).read(_root(con, "t"), ["str"]) is None


@pytest.mark.parametrize("name", ["3.dms", "ww.dms", "methane-pdff.dms", "2f4k.dms"])
def test_fast_and_sqlite3_paths_agree(name, monkeypatch):
    path = msys_file(name)
    monkeypatch.setattr(dms, "FAST_MIN_ROWS", 0)
    fast = canon(boonza.load(path))
    monkeypatch.setattr(dms, "FAST_MIN_ROWS", float("inf"))
    slow = canon(boonza.load(path))
    assert_same(fast, slow)


def test_fast_path_is_used(monkeypatch):
    calls = []
    original = SqliteFile.read

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(SqliteFile, "read", spy)
    monkeypatch.setattr(dms, "FAST_MIN_ROWS", 0)
    boonza.load(msys_file("ww.dms"))
    assert calls and all(calls)


def test_scan_skipping_the_key_column(tmp_path):
    # regression: the rowid used to be written even when its column was skipped
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.execute("create table t (id integer primary key, name text)")
    con.executemany("insert into t values (?, ?)", [(i, f"n{i}") for i in range(5000)])
    con.commit()
    scan = SqliteFile(np.fromfile(path, np.uint8))
    for _ in range(20):
        key, name = scan.read(_root(con, "t"), ["skip", "str"], rowid_col=0)
        assert key is None and name[4999] == "n4999"


def test_scan_declines_truncated_file(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.execute("create table t (a integer, b text)")
    con.executemany("insert into t values (?, ?)", [(i, "x" * 50) for i in range(20000)])
    con.commit()
    root = _root(con, "t")
    data = np.fromfile(path, np.uint8)
    assert SqliteFile(data).read(root, ["int", "str"]) is not None
    assert SqliteFile(data[: len(data) // 2]).read(root, ["int", "str"]) is None

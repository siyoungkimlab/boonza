import sqlite3

import numpy as np

import boonza
from boonza._columns import STR
from boonza.io.dms import _factorize


def test_factorize_groups_non_adjacent_rows():
    chain = np.array(["A", "A", "B", "A", "B", "A"], dtype=STR)
    resid = np.array([1, 1, 1, 2, 1, 1])
    codes, first = _factorize(chain, resid)
    assert codes.tolist() == [0, 0, 1, 2, 1, 0]
    assert first.tolist() == [0, 2, 3]


def test_split_residue_is_one_residue(tmp_path):
    # atoms of one residue separated by another residue, as msys allows
    path = tmp_path / "split.dms"
    con = sqlite3.connect(path)
    con.execute(
        "create table particle (id integer primary key, anum integer, name text, "
        "resname text, resid integer, chain text, segid text)"
    )
    con.executemany(
        "insert into particle values (?, ?, ?, ?, ?, ?, ?)",
        [
            (0, 8, "O", "HOH", 1, "W", ""),
            (1, 6, "C", "LIG", 5, "L", ""),
            (2, 1, "H1", "HOH", 1, "W", ""),
        ],
    )
    con.commit()
    s = boonza.load(path)
    assert s.nresidues == 2 and s.nchains == 2
    assert s.residue_atoms(0).tolist() == [0, 2]

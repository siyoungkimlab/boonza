import numpy as np
import pytest

from boonza.io._maeparse import MaeError, parse_mae

TEXT = """
{
  s_m_m2io_version
  :::
  2.0.0
}
f_m_ct {
  s_m_title
  r_chorus_box_ax
  i_m_count
  b_m_flag
  :::
  "a \\"quoted\\" title"
  12.5
  <>
  1
  m_atom[3] {
    # First column is atom index #
    i_m_residue_number
    r_m_x_coord
    s_m_pdb_atom_name
    :::
    1 7 1.5 " N  "
    2 1.0 -2 CA # a comment #
    3 <> 3e1 "C#"
    :::
  }
}
"""


def test_parse_values_arrays_nulls_and_fallback():
    (ct,) = parse_mae(TEXT)
    assert ct["__name__"] == "f_m_ct"
    assert ct["m_title"] == 'a "quoted" title'
    assert ct["chorus_box_ax"] == 12.5
    assert ct["m_count"] is None
    assert ct["m_flag"] is True
    atoms = ct["m_atom"]
    assert atoms.size == 3
    assert atoms["m_residue_number"].tolist() == [7, 1, 0]  # "1.0" read as atoi does
    assert atoms.nulls["m_residue_number"].tolist() == [False, False, True]
    np.testing.assert_array_equal(atoms["m_x_coord"], [1.5, -2.0, 30.0])
    assert atoms["m_pdb_atom_name"].tolist() == [" N  ", "CA", "C#"]


@pytest.mark.parametrize(
    "bad",
    [
        "f_m_ct { s_m_title ::: }",
        "f_m_ct { x_bad ::: 1 }",
        "f_m_ct { ::: m_atom[1] { i_m_a ::: 1 2 3 ::: } }",
        "f_m_ct { ::: m_atom[1] { i_m_a ::: 1 2 }",
    ],
)
def test_malformed(bad):
    with pytest.raises(MaeError):
        parse_mae(bad)

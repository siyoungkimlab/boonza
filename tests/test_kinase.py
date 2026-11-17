"""Kinase conformational states (boonza.kinase) against what Modi & Dunbrack report.

The three structures are chain A of ABL1 entries the 2019 paper names: dasatinib in
BLAminus (2GQG), imatinib in BBAminus (2HYY), and bosutinib in a DFGout or DFGinter
conformation that fits no cluster (3UE4).
"""

import json
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza import kinase
from boonza.cli import main
from boonza.kinase import CENTROIDS, assign_cluster, find_motifs, kinase_state, kinase_states

DATA = Path(__file__).parent / "data"
# ABL1 229-500, the kinase domain
ABL = (
    "SPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPP"
    "FYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDT"
    "YTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRA"
    "CWQWNPSDRPSFAEIHQAFETMFQE"
)


@pytest.fixture(scope="module")
def abl():
    return {name: boonza.load(DATA / f"{name}_A.pdb.gz") for name in ("2GQG", "2HYY", "3UE4")}


def test_the_conserved_residues_are_found_from_the_sequence():
    (m,) = find_motifs(ABL)
    number = lambda k: k + 229  # noqa: E731
    assert number(m["hrd"]) == 361 and ABL[m["hrd"] : m["hrd"] + 3] == "HRD"
    assert number(m["dfg"]) == 381 and ABL[m["dfg"] : m["dfg"] + 3] == "DFG"
    assert (271, 286) in [(number(k), number(e)) for k, e in m["pairs"]]
    assert find_motifs("ACDEFGHIKLMNPQRSTVWY" * 20) == []


def test_spacings_are_counted_in_residue_numbers_so_gaps_do_not_shorten_them():
    numbers = np.arange(229, 229 + len(ABL))
    (whole,) = find_motifs(ABL, numbers)
    assert (numbers[whole["hrd"]], numbers[whole["dfg"]]) == (361, 381)
    # the structure lacks 276-283, between the Lys and the Glu, and 300-340
    keep = ~(((numbers >= 276) & (numbers <= 283)) | ((numbers >= 300) & (numbers <= 340)))
    seq = "".join(np.array(list(ABL))[keep])
    (cut,) = find_motifs(seq, numbers[keep])
    assert numbers[keep][cut["dfg"]] == 381
    assert (271, 286) in [(numbers[keep][k], numbers[keep][e]) for k, e in cut["pairs"]]


def test_a_lys_far_from_the_dfg_is_not_taken_for_the_one_of_beta3(abl):
    # the K271M of a kinase-dead construct: the other A/C-x-K of the sequence are elsewhere
    s = abl["2GQG"]
    names = np.array(s.residues["name"])
    names[np.asarray(s.residues["resid"]) == 271] = "MET"
    dead = s.clone()
    dead.residues["name"] = names
    assert kinase_states(dead) == []
    given = kinase_state(dead, lys=271, glu=286)
    assert given.label == "DFGin-BLAminus" and "β3 Lys is M here" in given.notes


def test_every_centroid_is_its_own_cluster():
    rotamers = {"minus": -65.0, "plus": 55.0, "trans": 180.0}
    for group, clusters in CENTROIDS.items():
        for name, angles in clusters.items():
            chi1 = next(v for k, v in rotamers.items() if name.endswith(k))
            got, distance, nearest = assign_cluster(group, np.array(angles, float), chi1)
            assert (got, nearest) == (name, name) and distance == pytest.approx(0.0, abs=1e-12)
            # only among its own spatial group and rotamer
            other = "DFGout" if group != "DFGout" else "DFGin"
            assert assign_cluster(other, np.array(angles, float), 180.0 - chi1)[0] != name


def test_angles_wrap_and_far_ones_have_no_cluster():
    blaminus = np.array(CENTROIDS["DFGin"]["BLAminus"], float)
    assert assign_cluster("DFGin", blaminus + 360.0, -71.0)[0] == "BLAminus"
    assert kinase.dihedral_distance([179.0], [-179.0]) == pytest.approx(
        2 * (1 - np.cos(np.radians(2.0)))
    )
    # 40 degrees off on every angle is the limit Kincore gives for its 0.45
    assert assign_cluster("DFGin", blaminus + 35.0, -71.0)[0] == "BLAminus"
    assert assign_cluster("DFGin", blaminus + 45.0, -71.0)[0] is None
    # a trans rotamer has no DFGout cluster to be compared with
    assert assign_cluster("DFGout", np.array(CENTROIDS["DFGout"]["BBAminus"], float), 180.0) == (
        None, pytest.approx(np.nan, nan_ok=True), None,
    )  # fmt: skip


def test_the_spatial_groups_and_rotamers_are_the_published_ranges():
    assert kinase.spatial_group(6.0, 14.0) == "DFGin"
    assert kinase.spatial_group(13.0, 10.0) == "DFGout"
    assert kinase.spatial_group(8.0, 9.0) == "DFGinter"
    assert kinase.spatial_group(13.0, 15.0) is None
    assert kinase.spatial_group(np.nan, 10.0) is None
    assert [kinase.rotamer(x) for x in (-60.0, 300.0, 60.0, 180.0, -150.0, 120.0)] == [
        "minus", "minus", "plus", "trans", "trans", "plus",
    ]  # fmt: skip


def test_abl_with_dasatinib_is_active_and_with_imatinib_is_the_type_two_state(abl):
    active = kinase_state(abl["2GQG"])
    assert active.label == "DFGin-BLAminus" and active.active and active.spine
    assert (active.lys, active.glu, active.hrd, active.dfg) == (271, 286, 361, 381)
    assert (active.hrd_motif, active.xdfg_motif) == ("HRD", "ADFG")
    assert active.chelix == "in" and active.loop == "extended"
    assert active.d1 < 11.0 < active.d2 and active.rotamer == "minus"
    assert active.cluster_distance < 0.1 and not active.notes

    out = kinase_state(abl["2HYY"])
    assert out.label == "DFGout-BBAminus" and not out.active and not out.spine
    assert out.d1 > 11.0 and out.d2 < 14.0
    assert out.chelix == "in"  # type II inhibitors leave the C-helix in
    assert out.loop == "folded" and out.spine_his_phe > 6.0


def test_a_structure_far_from_every_centroid_keeps_only_its_spatial_group(abl):
    st = kinase_state(abl["3UE4"])
    assert st.spatial == "DFGout" and st.cluster is None and st.label == "DFGout-None"
    assert st.rotamer == "plus" and st.nearest is None  # no DFGout cluster has a plus rotamer


def test_residue_numbers_and_frames_can_be_given(abl):
    s = abl["2HYY"]
    same = kinase_state(s, chain="A", lys=271, glu=286, dfg=381)
    assert same.label == "DFGout-BBAminus" and same.d1 == kinase_state(s).d1
    moved = kinase_state(s, positions=np.asarray(s.positions) + 5.0)
    assert moved.d1 == pytest.approx(same.d1) and moved.dihedrals == pytest.approx(same.dihedrals)
    with pytest.raises(ValueError, match="dfg=382"):
        kinase_state(s, dfg=382)
    other = kinase_state(s, lys=272, glu=286)  # a residue that is not a Lys is said, not refused
    assert other.lys == 272 and any("β3 Lys is" in n for n in other.notes)
    with pytest.raises(ValueError, match="together"):
        kinase_state(s, lys=271)
    with pytest.raises(ValueError, match="one frame"):
        kinase_state(s, positions=np.zeros((2, s.natoms, 3)))


def test_missing_atoms_are_reported_not_guessed(abl):
    s = abl["2GQG"]
    cut = s.clone(atoms=s.select("not (resid 382 and name CG CD1 CD2 CE1 CE2 CZ)").ids)
    st = kinase_state(cut)
    assert np.isnan(st.d1) and st.spatial is None and st.cluster is None
    assert st.label == "None-None" and any("missing" in n for n in st.notes)
    assert st.chelix == "in"  # what does not need the ring is still measured


def test_nothing_is_measured_across_a_break_in_the_chain(abl):
    s = abl["2GQG"]
    # residues 288-289 missing: the residue four after Glu286 is no longer four along
    cut = s.clone(atoms=s.select("not resid 288 289").ids)
    st = kinase_state(cut, lys=271, glu=286)
    assert np.isnan(st.d1) and st.spatial is None and any("four after" in n for n in st.notes)
    assert st.d2 == pytest.approx(kinase_state(s).d2)
    # the residue before the DFG missing: its dihedrals and the Asp phi cannot be had
    cut = s.clone(atoms=s.select("not resid 380").ids)
    st = kinase_state(cut)
    assert st.spatial == "DFGin" and st.cluster is None and np.isnan(st.dihedrals[:3]).all()
    assert not np.isnan(st.dihedrals[3:]).any()


def test_the_tip_atom_of_a_dfg_that_is_not_phe_is_the_one_kincore_takes():
    assert kinase.TIP == {"F": "CZ", "Y": "OH", "L": "CG", "M": "CE", "W": "CZ3"}
    (m,) = find_motifs(ABL.replace("VADFGLS", "VADLGLS"))
    assert ABL[m["dfg"] : m["dfg"] + 3] == "DFG"  # same place, found as D-L-G


def test_a_protein_that_is_not_a_kinase():
    lysozyme = boonza.load(DATA / "1LYZ.pdb")
    assert kinase_states(lysozyme) == []
    with pytest.raises(ValueError, match="no kinase domain"):
        kinase_state(lysozyme)


def test_command_line(capsys):
    files = [str(DATA / f"{name}_A.pdb.gz") for name in ("2GQG", "2HYY")]
    assert main(["kinase", *files]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:3] == ["file", "chain", "state"]
    assert lines[1].split()[:3] == ["2GQG_A.pdb.gz", "A", "DFGin-BLAminus"]
    assert lines[2].split()[:3] == ["2HYY_A.pdb.gz", "A", "DFGout-BBAminus"]

    assert main(["kinase", files[1], "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["label"] == "DFGout-BBAminus" and row["dfg"] == 381 and row["active"] is False
    assert len(row["dihedrals"]) == 6

    assert main(["kinase", str(DATA / "1LYZ.pdb")]) == 1
    assert "no kinase domain" in capsys.readouterr().err

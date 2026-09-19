"""Feature maps: what a pocket asks for, and where."""

import numpy as np
import pytest

import boonza
from boonza.pharmacophore import CODES, FAMILIES


@pytest.fixture(scope="module")
def screened():
    """A peptide and three unlike molecules that all put their acceptor in one
    place: phenol, benzoic acid and benzaldehyde, each in its own copy."""
    rng = np.random.default_rng(0)
    s = boonza.peptide("AAAAAAAAAA")
    s.positions = s.positions - s.positions.mean(0)
    for smiles in ("c1ccccc1O", "c1ccccc1C(=O)O", "c1ccccc1C=O"):
        s.append(boonza.from_smiles(smiles))
    s.cell = np.diag([40.0, 40.0, 40.0])
    lig = s.select("not polymer and noh").ids
    frag = np.unique(np.asarray(s.fragids)[lig])
    copies = [s.select(f"fragid {int(f)}").ids for f in frag]
    base = s.positions.copy()
    for c in copies:
        base[c] -= base[c].mean(0)
    frames = []
    for _ in range(40):
        x = base.copy()
        for c in copies:  # the three sit around one place, jittered
            x[c] = base[c] + np.array([5.0, 0.0, 0.0]) + rng.normal(scale=0.4, size=3)
        frames.append(x)
    return s, np.array(frames), [int(f) for f in frag]


def test_a_molecule_is_typed_not_just_counted(screened):
    s, _frames, frag = screened
    per_copy = boonza.ligand_features(s, "not polymer")
    assert len(per_copy) == 3
    families = {fam for copy in per_copy for fam, _atoms in copy}
    assert {"Donor", "Acceptor", "Aromatic", "Hydrophobe"} & families
    assert "ZnBinder" not in families  # left out on purpose
    for copy in per_copy:
        for fam, atoms in copy:
            assert fam in FAMILIES and len(atoms)


def test_the_maps_say_what_is_wanted_where(screened):
    s, frames, _frag = screened
    maps = boonza.feature_maps(s, frames, ligand="not polymer")
    assert set(maps) == set(FAMILIES)
    acceptor = maps["Acceptor"]
    assert len(acceptor.places) > 0 and acceptor.enrichment.max() > 20
    assert acceptor.counts.shape == maps["Hydrophobe"].counts.shape  # one grid for all
    assert np.allclose(acceptor.origin, maps["Aromatic"].origin)

    spots = boonza.hotspots(maps, enrichment=10.0)
    assert spots and spots[0].ligands >= 2  # unlike molecules agreeing is the point
    best = max(spots, key=lambda h: h.ligands)
    assert np.linalg.norm(best.center - np.array([5.0, 0.0, 0.0])) < 4.0
    assert best.radius > 0 and best.volume >= 3.0
    assert boonza.wanted(spots, [5.0, 0.0, 0.0], within=5.0)
    assert not boonza.wanted(spots, [50.0, 0.0, 0.0], within=5.0)


def test_hotspots_are_written_where_a_viewer_can_read_them(tmp_path, screened):
    s, frames, _frag = screened
    spots = boonza.hotspots(boonza.feature_maps(s, frames, ligand="not polymer"), enrichment=10.0)
    path = tmp_path / "hotspots.pdb"
    boonza.write_hotspots(path, spots)
    lines = [x for x in path.read_text().splitlines() if x.startswith("HETATM")]
    assert len(lines) == len(spots)
    for line, h in zip(lines, spots, strict=True):
        assert line[17:20].strip() == CODES[h.family]
        assert float(line[30:38]) == pytest.approx(h.center[0], abs=5e-4)
        assert float(line[54:60]) == pytest.approx(min(h.radius, 9.99), abs=5e-3)
        assert float(line[60:66]) == pytest.approx(min(h.enrichment, 999.99), abs=5e-3)
    back = boonza.load(path)  # and it is a structure file, not just text
    assert back.natoms == len(spots)

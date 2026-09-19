"""What a ligand touches, as a number per residue: comparable between molecules.

    a = boonza.interaction_fingerprints(s, frames, ligand="fragid 3 and noh")
    b = boonza.interaction_fingerprints(s, frames, ligand="fragid 7 and noh")
    boonza.similarity(a.mean(), b.mean())   # do these two bind the same way?

:func:`boonza.poses` compares a molecule with itself, frame by frame, so it
cannot say whether *two different* molecules sit the same way: there is no
atom of one that answers to an atom of the other.  A fingerprint describes a
frame by what the ligand is near rather than where its atoms are, so it has
the same length and the same meaning for every ligand, whatever its size or
chemistry.

Each residue contributes how close the ligand comes to it, softened:

    s = 1 / (1 + exp((d - center) / width))

so a residue at the ``center`` distance counts a half, nearer counts more,
and further counts less.  A hard cutoff would make a residue flicker in and
out as the ligand breathes; this does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pbc import capped_distances
from .symmetry import DEFAULT_LIGAND, _boxed_blocks, _ids


@dataclass
class Interactions:
    """A fingerprint per frame and copy, and which residues its columns are."""

    values: np.ndarray  # (rows, nresidues), 0 to 1
    residues: np.ndarray  # residue index in the system of every column
    where: np.ndarray  # (rows, 2): frame, copy
    center: float
    width: float

    def __len__(self) -> int:
        return len(self.values)

    def mean(self) -> np.ndarray:
        """The average fingerprint: what this ligand touches, over the frames given."""
        return self.values.mean(0) if len(self.values) else np.zeros(len(self.residues))

    def names(self, system) -> list[str]:
        """``['TYR34', 'LEU58', ...]``: what the columns are."""
        return [f"{system.residues['name'][r]}{system.residues['resid'][r]}"
                for r in self.residues]  # fmt: skip

    def table(self, system):
        """The fingerprints as a table: a row per frame or ligand, a column per residue."""
        import pandas as pd

        index = pd.MultiIndex.from_arrays(self.where.T, names=("frame", "copy"))
        return pd.DataFrame(self.values, index=index, columns=self.names(system))

    def write_structure(self, system, path, values=None) -> None:
        """Write the structure with the fingerprint in its B-factor column.

        Every atom of a residue carries that residue's number, so opening the
        file and colouring by B-factor shows what the ligand touches, on the
        structure rather than in a table.  ``values`` picks what to paint --
        one row, or a mean of your own -- and defaults to the mean of all.
        """
        from .io import save

        v = self.mean() if values is None else np.asarray(values, float).reshape(-1)
        if v.shape != (len(self.residues),):
            raise ValueError(f"values has {v.shape}, not one per residue {(len(self.residues),)}")
        out = system.clone()
        paint = np.zeros(out.nresidues)
        paint[self.residues] = v
        out.atoms["bfactor"] = paint[out.atoms["residue"]]
        save(out, path)

    def touched(self, system, share: float = 0.5) -> list[str]:
        """The residues held above ``share`` on average, as ``TYR34`` and so on."""
        keep = self.mean() >= share
        return [n for n, k in zip(self.names(system), keep, strict=True) if k]


# one hue, light to dark: the value is a magnitude, so the colour is a magnitude
SEQUENTIAL = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")  # fmt: skip


def _like_beside_like(values) -> np.ndarray:
    """Row order from average linkage: fingerprints that agree end up adjacent."""
    from .poses import _nn_chain

    n = len(values)
    if n < 3:
        return np.arange(n)
    d = 1.0 - similarity_matrix(values)
    np.fill_diagonal(d, 0.0)
    merges = _nn_chain(np.ascontiguousarray(d))
    groups = {i: [i] for i in range(n)}
    for a, b, _height in merges[np.argsort(merges[:, 2], kind="stable")].tolist():
        groups[int(a)] = groups.pop(int(a)) + groups.pop(int(b))
    return np.array(next(iter(groups.values())))


def plot_interactions(fingerprints, system, path, share: float = 0.2, labels=None,
                      order: bool = True) -> bool:  # fmt: skip
    """Draw what each ligand touches as a heatmap; False without matplotlib.

    Rows are the fingerprints, columns the residues any of them comes near.
    The colour is one hue from light to dark because the value is a magnitude:
    a rainbow would invent boundaries where the data has none.  With ``order``
    the rows are arranged so that ligands which agree sit together, which is
    what makes two ways of binding one site visible as two blocks.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return False
    values = np.asarray(fingerprints.values, float)
    keep = np.flatnonzero(values.max(0) >= share)
    if not len(keep):
        return False
    values = values[:, keep]
    names = [fingerprints.names(system)[i] for i in keep]
    rows = list(labels) if labels is not None else [
        f"{r}:{c}" for r, c in fingerprints.where.tolist()
    ]  # fmt: skip
    if order:
        at = _like_beside_like(fingerprints.values)
        values, rows = values[at], [rows[i] for i in at]
    cmap = LinearSegmentedColormap.from_list("touch", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(0.32 * len(names) + 2.5, 0.28 * len(rows) + 1.8))
    mesh = ax.pcolormesh(values, cmap=cmap, vmin=0.0, vmax=1.0, edgecolors="white",
                         linewidth=0.5)  # fmt: skip
    ax.set_xticks(np.arange(len(names)) + 0.5, names, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(rows)) + 0.5, rows, fontsize=7)
    ax.invert_yaxis()
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label("how close the ligand comes", fontsize=8)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def similarity(a, b) -> float:
    """How alike two fingerprints are: 1 the same residues to the same degree, 0 none.

    The cosine of the two, which asks about the pattern rather than how
    deeply either sits -- a small fragment in a pocket and a large one
    reaching further into it are alike here if they touch the same residues.
    """
    a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"fingerprints of {a.shape} and {b.shape}: they must share residues")
    size = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / size) if size else 0.0


def interaction_fingerprints(system, positions=None, ligand: str = DEFAULT_LIGAND,
                             protein: str = "protein and noh", center: float = 4.0,
                             width: float = 1.0, residues=None, periodic: bool = True,
                             ) -> Interactions:  # fmt: skip
    """How near the ligand comes to each residue, per frame, softened to 0-1.

    Every copy of the ligand -- one per molecule of the selection -- gives one
    fingerprint per frame.  ``residues`` fixes the columns, which is what lets
    fingerprints from different ligands, or different systems with the same
    protein, be compared: pass the ``residues`` of an earlier result.
    """
    lig = _ids(system, ligand)
    lig = lig[system.atoms["anum"][lig] > 1]
    if not len(lig):
        raise ValueError(f"ligand {ligand!r} selects no heavy atoms")
    prot = np.setdiff1d(_ids(system, protein), lig)
    if not len(prot):
        raise ValueError(f"protein {protein!r} selects no atoms outside the ligand")
    of_residue = system.atoms["residue"]
    columns = np.unique(of_residue[prot]) if residues is None else np.asarray(residues)
    rank = {int(r): i for i, r in enumerate(columns.tolist())}
    where_column = np.array([rank.get(int(r), -1) for r in of_residue[prot]])
    prot = prot[where_column >= 0]
    where_column = where_column[where_column >= 0]

    frag = np.asarray(system.fragids)[lig]
    copies = [lig[frag == f] for f in np.unique(frag)]
    need = np.union1d(prot, lig)
    at = np.searchsorted(need, prot)
    copy_at = [np.searchsorted(need, c) for c in copies]
    far = center + 6.0 * width  # beyond this the sigmoid is under 0.3%

    rows, place, frame = [], [], 0
    for xyz, boxes in _boxed_blocks(system, positions, need)[0]:
        for X, box in zip(xyz, boxes, strict=True):
            box = box if periodic and np.asarray(box).any() else None
            for c, atoms in enumerate(copy_at):
                near = np.full(len(columns), np.inf)
                i, _j, d = capped_distances(X[at], X[atoms], far, box)
                if len(i):
                    np.minimum.at(near, where_column[i], d)
                rows.append(1.0 / (1.0 + np.exp((near - center) / width)))
                place.append((frame, c))
            frame += 1
    if not rows:
        raise ValueError("no frames")
    return Interactions(values=np.array(rows), residues=columns, where=np.array(place, np.int64),
                        center=float(center), width=float(width))  # fmt: skip


def similarity_matrix(values) -> np.ndarray:
    """Every fingerprint against every other, as cosines."""
    v = np.asarray(values, float)
    norm = np.linalg.norm(v, axis=1)
    norm[norm == 0] = 1.0
    out = (v / norm[:, None]) @ (v / norm[:, None]).T
    return np.clip(out, -1.0, 1.0)


def site_interactions(system, runs, found, site: int, ligand: str = DEFAULT_LIGAND,
                      protein: str = "protein and noh", center: float = 4.0,
                      width: float = 1.0, periodic: bool = True) -> Interactions:  # fmt: skip
    """One fingerprint per ligand that visits ``site``: what each of them touches there.

    Rows are ``(run, copy)`` rather than frames, each the average over that
    copy's frames in the site.  The columns are the same residues for every
    row, so two rows can be compared however unlike the two molecules are --
    which is the question a pose cannot answer.
    """
    from .sites import _pairs

    pairs = _pairs(system, runs)
    rows, place, columns = [], [], None
    for r, (own, run) in enumerate(pairs):
        mine = found.frames(site)
        mine = mine[mine[:, 0] == r]
        if not len(mine):
            continue
        wanted = np.unique(mine[:, 2])
        seen = interaction_fingerprints(own, run[wanted], ligand=ligand, protein=protein,
                                        center=center, width=width, residues=columns,
                                        periodic=periodic)  # fmt: skip
        columns = seen.residues if columns is None else columns
        at = {int(f): i for i, f in enumerate(wanted.tolist())}
        for copy in np.unique(mine[:, 1]).tolist():
            frames = {at[int(f)] for f in mine[mine[:, 1] == copy][:, 2]}
            keep = np.array([i for i, (f, c) in enumerate(seen.where.tolist())
                             if c == copy and f in frames])  # fmt: skip
            if len(keep):
                rows.append(seen.values[keep].mean(0))
                place.append((r, copy))
    if not rows:
        raise ValueError(f"site {site} has no frames")
    return Interactions(values=np.array(rows), residues=columns, where=np.array(place, np.int64),
                        center=float(center), width=float(width))  # fmt: skip

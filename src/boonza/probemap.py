"""What each probe touches: the fingerprint a coarse-grained swim is for.

Probes in Martini come and go rather than settle, so what a run says is not
"a ligand sits here" but "this part of the surface likes this chemistry".
:func:`probe_contacts` counts, for every residue and every probe, the frames
in which they touch, and :meth:`ProbeMap.side_chains` pools the probes that
share a side chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CUTOFF = 6.0  # Å between beads for a contact, about a bead's diameter


@dataclass
class ProbeMap:
    """Contacts as a fraction of frames, residues down and probes across."""

    residues: list  # (chain, resid, name) of each row
    probes: list  # the probe of each column
    contacts: np.ndarray  # (residues, probes)
    frames: int

    def side_chains(self) -> tuple[list, np.ndarray]:
        """The same map with the probes pooled by side chain: a probe counts
        for both of its residues (XX counts once)."""
        letters = sorted({c for p in self.probes for c in p})
        out = np.zeros((len(self.residues), len(letters)))
        for k, letter in enumerate(letters):
            columns = [j for j, p in enumerate(self.probes) if letter in p]
            out[:, k] = self.contacts[:, columns].max(axis=1) if columns else 0.0
        return letters, out

    def top(self, n: int = 10, by: str = "probe"):
        """The (label, residue, fraction) a probe or side chain touches most."""
        labels, matrix = (self.probes, self.contacts) if by == "probe" else self.side_chains()
        rows = []
        for k, label in enumerate(labels):
            order = np.argsort(-matrix[:, k])[:n]
            rows += [(label, self.residues[r], float(matrix[r, k])) for r in order
                     if matrix[r, k] > 0]  # fmt: skip
        return rows

    def to_csv(self, path) -> None:
        import csv

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["chain", "resid", "residue", *self.probes])
            for k, (chain, resid, name) in enumerate(self.residues):
                w.writerow([chain, resid, name, *(f"{v:.4f}" for v in self.contacts[k])])


def probe_contacts(system, runs, probes, cutoff: float = CUTOFF, stride: int = 1,
                   periodic: bool = True) -> ProbeMap:  # fmt: skip
    """How often each residue of ``system`` touches each of ``probes``.

    ``runs`` is one trajectory or several (each with its own system, as
    :func:`boonza.sites` takes them); ``probes`` are the residue names the
    probes carry (``EK``, ``LL``, ...).  A residue and a probe touch in a
    frame when any of their beads are within ``cutoff`` Å, minimum image if
    ``periodic``.  Runs that hold different probes pool: each column counts
    only the frames of the runs that carried that probe.
    """
    from .spatial import pairs_within

    if not isinstance(runs, (list, tuple)):
        runs = [runs]
    pairs = [(system, r) if not isinstance(r, tuple) else r for r in runs]
    probes = list(probes)
    index = {p: k for k, p in enumerate(probes)}
    counts, seen, labels = None, np.zeros(len(probes)), None
    for own, run in pairs:
        res = np.asarray(own.atoms["residue"])
        names = np.array([str(x) for x in np.asarray(own.residues["name"])])
        chains = np.array([str(x) for x in np.asarray(own.chains["name"])])
        resid = np.asarray(own.residues["resid"])
        chain_of = np.asarray(own.residues["chain"])
        per_atom = names[res]
        is_probe = np.isin(per_atom, probes)
        is_target = ~is_probe & ~np.isin(per_atom, ["W", "ION", "NA", "CL", "HOH"])
        target_res = np.unique(res[is_target])
        if labels is None:
            labels = [(chains[chain_of[r]], int(resid[r]), names[r]) for r in target_res]
            counts = np.zeros((len(target_res), len(probes)))
        row_of = {int(r): k for k, r in enumerate(target_res)}
        column_of = {int(r): index[per_atom[res == r][0]] for r in np.unique(res[is_probe])}
        here = sorted({index[p] for p in np.unique(per_atom[is_probe])})
        pick = np.flatnonzero(is_target | is_probe)
        kind = is_probe[pick]
        frames = 0
        for frame in run[::stride] if stride > 1 else run:
            # a Frame of a trajectory, or plain positions, as boonza.sites takes them
            xyz = np.asarray(getattr(frame, "positions", frame))
            cell = np.asarray(getattr(frame, "box", own.cell)) if periodic else None
            i, j, _ = pairs_within(xyz[pick], cutoff, cell=cell)
            across = kind[i] != kind[j]
            a, b = i[across], j[across]
            protein = np.where(kind[a], b, a)  # the one that is not a probe
            probe = np.where(kind[a], a, b)
            touching = {(row_of[int(res[pick[p]])], column_of[int(res[pick[q]])])
                        for p, q in zip(protein.tolist(), probe.tolist(), strict=True)}  # fmt: skip
            for r, c in touching:
                counts[r, c] += 1
            frames += 1
        seen[here] += frames
    with np.errstate(invalid="ignore"):
        fractions = np.where(seen > 0, counts / np.where(seen > 0, seen, 1), 0.0)
    return ProbeMap(labels or [], probes, fractions, int(seen.max()) if len(seen) else 0)

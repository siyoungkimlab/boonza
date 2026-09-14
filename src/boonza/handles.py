"""Lightweight handles (index + system) for atoms, bonds, residues, chains and cts.

Handles hold no data of their own.  Indices are always 0..N-1, so deleting
or reordering atoms renumbers them; a handle created before such an edit
raises ``StaleHandleError`` instead of silently pointing at another atom.
Adding atoms never renumbers, so it does not invalidate handles.
"""

from __future__ import annotations

import numpy as np

from ._columns import scalar
from .elements import symbol


class StaleHandleError(RuntimeError):
    """A handle was used after the indices it refers to were renumbered."""


class _Handle:
    __slots__ = ("_sys", "_id", "_epoch")
    _kind = ""
    _attr = ""

    def __init__(self, system, id):
        self._sys = system
        self._id = int(id)
        self._epoch = system._epochs[self._kind]

    @property
    def id(self) -> int:
        if self._epoch != self._sys._epochs[self._kind]:
            raise StaleHandleError(
                f"{type(self).__name__} {self._id} was renumbered by a deletion or reorder; "
                "get a fresh handle from the system"
            )
        return self._id

    @property
    def system(self):
        return self._sys

    def __index__(self) -> int:
        return self.id

    def __int__(self) -> int:
        return self.id

    def __eq__(self, other):
        return (
            type(other) is type(self)
            and other._sys is self._sys
            and other._id == self._id
            and other._epoch == self._epoch
        )

    def __hash__(self):
        return hash((type(self).__name__, id(self._sys), self._id, self._epoch))

    def _table(self):
        return getattr(self._sys, self._attr)

    def __getitem__(self, prop: str):
        v = self._table().column(prop)[self.id]
        return v.copy() if isinstance(v, np.ndarray) else scalar(v)

    def __setitem__(self, prop: str, value) -> None:
        self._sys._set_values(self._kind, prop, self.id, value)

    def keys(self) -> list[str]:
        return self._table().names


def _column(name: str, doc: str = ""):
    def fget(self):
        return scalar(self._table().column(name)[self.id])

    def fset(self, value):
        self._sys._set_values(self._kind, name, self.id, value)

    return property(fget, fset, doc=doc)


class Atom(_Handle):
    __slots__ = ()
    _kind = "atom"
    _attr = "_atoms"

    name = _column("name")
    anum = _column("anum", "atomic number (0 for pseudo particles)")
    atomic_number = anum
    mass = _column("mass")
    charge = _column("charge", "partial charge")
    formal_charge = _column("formal_charge")

    @property
    def element(self) -> str:
        return symbol(self.anum)

    @property
    def pos(self) -> np.ndarray:
        """Copy of the position; assign to change it."""
        return self._sys._atoms.column("pos")[self.id].copy()

    @pos.setter
    def pos(self, xyz) -> None:
        self._sys._atoms.set("pos", xyz, self.id)

    @property
    def vel(self) -> np.ndarray:
        return self._sys._atoms.column("vel")[self.id].copy()

    @vel.setter
    def vel(self, v) -> None:
        self._sys._atoms.set("vel", v, self.id)

    @property
    def residue(self) -> Residue:
        return Residue(self._sys, self._sys._atoms.column("residue")[self.id])

    @residue.setter
    def residue(self, res) -> None:
        self._sys._set_parent("atom", self.id, res)

    @property
    def chain(self) -> Chain:
        return self.residue.chain

    @property
    def ct(self) -> Ct:
        return self.residue.chain.ct

    @property
    def fragid(self) -> int:
        return int(self._sys.fragids[self.id])

    @property
    def fragment(self) -> AtomSel:
        """All atoms covalently connected to this one (the molecule)."""
        return AtomSel(self._sys, self._sys.fragment_atoms(self.fragid))

    molecule = fragment

    @property
    def bonds(self) -> list[Bond]:
        return [Bond(self._sys, b) for b in self._sys._atom_bonds(self.id)]

    @property
    def bonded_atoms(self) -> AtomSel:
        return AtomSel(self._sys, self._sys.bonded_atoms(self.id))

    @property
    def nbonds(self) -> int:
        return len(self._sys.bonded_atoms(self.id))

    def add_bond(self, other, order: int = 1) -> Bond:
        return self._sys.add_bond(self, other, order=order)

    @property
    def fullname(self) -> str:
        r = self.residue
        return f"{r.chain.name}:{r.name}{r.resid}{r.insertion}:{self.name}"

    def __repr__(self) -> str:
        r = self.residue
        return f"<Atom {self._id} {self.name} {r.name}{r.resid}{r.insertion}>"


class Bond(_Handle):
    __slots__ = ()
    _kind = "bond"
    _attr = "_bonds"

    order = _column("order")

    @property
    def first(self) -> Atom:
        return Atom(self._sys, self._sys._bonds.column("i")[self.id])

    @property
    def second(self) -> Atom:
        return Atom(self._sys, self._sys._bonds.column("j")[self.id])

    @property
    def atoms(self) -> tuple[Atom, Atom]:
        return self.first, self.second

    def other(self, atom) -> Atom:
        i = int(self._sys._bonds.column("i")[self.id])
        j = int(self._sys._bonds.column("j")[self.id])
        a = int(atom)
        if a not in (i, j):
            raise ValueError(f"atom {a} is not in bond {self.id}")
        return Atom(self._sys, j if a == i else i)

    def __repr__(self) -> str:
        return f"<Bond {self._id} {self.first.id}-{self.second.id} order={self.order}>"


class Residue(_Handle):
    __slots__ = ()
    _kind = "residue"
    _attr = "_residues"

    name = _column("name")
    resid = _column("resid")
    insertion = _column("insertion")

    @property
    def chain(self) -> Chain:
        return Chain(self._sys, self._sys._residues.column("chain")[self.id])

    @chain.setter
    def chain(self, chn) -> None:
        self._sys._set_parent("residue", self.id, chn)

    @property
    def ct(self) -> Ct:
        return self.chain.ct

    @property
    def atoms(self) -> AtomSel:
        return AtomSel(self._sys, self._sys.residue_atoms(self.id))

    @property
    def natoms(self) -> int:
        return len(self._sys.residue_atoms(self.id))

    def add_atom(self, **props) -> Atom:
        return self._sys.add_atom(self, **props)

    def select_atom(self, name: str) -> Atom | None:
        ids = self._sys.residue_atoms(self.id)
        hit = ids[self._sys._atoms.column("name")[ids] == name]
        return Atom(self._sys, hit[0]) if len(hit) else None

    def __repr__(self) -> str:
        return f"<Residue {self._id} {self.name}{self.resid}{self.insertion}>"


class Chain(_Handle):
    __slots__ = ()
    _kind = "chain"
    _attr = "_chains"

    name = _column("name")
    segid = _column("segid")

    @property
    def ct(self) -> Ct:
        return Ct(self._sys, self._sys._chains.column("ct")[self.id])

    @ct.setter
    def ct(self, ct) -> None:
        self._sys._set_parent("chain", self.id, ct)

    @property
    def residues(self) -> list[Residue]:
        return [Residue(self._sys, r) for r in self._sys.chain_residues(self.id)]

    @property
    def nresidues(self) -> int:
        return len(self._sys.chain_residues(self.id))

    @property
    def atoms(self) -> AtomSel:
        return AtomSel(self._sys, self._sys.chain_atoms(self.id))

    @property
    def natoms(self) -> int:
        return len(self._sys.chain_atoms(self.id))

    def add_residue(self, name: str = "", resid: int = 0, insertion: str = "") -> Residue:
        return self._sys.add_residue(self, name=name, resid=resid, insertion=insertion)

    def __repr__(self) -> str:
        return f"<Chain {self._id} name={self.name!r} segid={self.segid!r}>"


class Ct(_Handle):
    """A namespace of chains, plus free-form key/value properties."""

    __slots__ = ()
    _kind = "ct"

    @property
    def name(self) -> str:
        return self._sys._ct_names[self.id]

    @name.setter
    def name(self, value: str) -> None:
        self._sys._ct_names[self.id] = str(value)

    def keys(self) -> list[str]:
        return list(self._sys._ct_props[self.id])

    def __getitem__(self, key: str):
        return self._sys._ct_props[self.id][key]

    def __setitem__(self, key: str, value) -> None:
        self._sys._ct_props[self.id][key] = value

    def __delitem__(self, key: str) -> None:
        del self._sys._ct_props[self.id][key]

    def __contains__(self, key: str) -> bool:
        return key in self._sys._ct_props[self.id]

    @property
    def chains(self) -> list[Chain]:
        return [Chain(self._sys, c) for c in self._sys.ct_chains(self.id)]

    @property
    def atoms(self) -> AtomSel:
        return AtomSel(self._sys, self._sys.ct_atoms(self.id))

    def add_chain(self, name: str = "", segid: str = "") -> Chain:
        return self._sys.add_chain(self, name=name, segid=segid)

    def __repr__(self) -> str:
        return f"<Ct {self._id} {self.name!r}>"


_LEVELS = ("residue", "chain", "ct", "fragment", "molecule")


class AtomSel:
    """An ordered set of atoms from one system."""

    __slots__ = ("_sys", "_ids", "_epoch")

    def __init__(self, system, ids):
        ids = np.array(ids, dtype=np.int64).reshape(-1)
        ids.flags.writeable = False
        self._sys = system
        self._ids = ids
        self._epoch = system._epochs["atom"]

    @property
    def ids(self) -> np.ndarray:
        if self._epoch != self._sys._epochs["atom"]:
            raise StaleHandleError("atom selection was renumbered by a deletion or reorder")
        return self._ids

    @property
    def system(self):
        return self._sys

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self):
        return (Atom(self._sys, i) for i in self.ids)

    def __contains__(self, atom) -> bool:
        return int(atom) in set(self.ids.tolist())

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._sys._atoms.column(key)[self.ids]
        if isinstance(key, int | np.integer):
            return Atom(self._sys, self.ids[key])
        return AtomSel(self._sys, self.ids[key])

    def __setitem__(self, key: str, values) -> None:
        self._sys._set_values("atom", key, self.ids, values)

    @property
    def positions(self) -> np.ndarray:
        return self._sys._atoms.column("pos")[self.ids]

    @positions.setter
    def positions(self, xyz) -> None:
        self._sys._atoms.set("pos", xyz, self.ids)

    @property
    def residues(self) -> list[Residue]:
        from .graph import unique_in_order

        res = unique_in_order(self._sys._atoms.column("residue")[self.ids])
        return [Residue(self._sys, r) for r in res]

    @property
    def chains(self) -> list[Chain]:
        from .graph import unique_in_order

        res = self._sys._atoms.column("residue")[self.ids]
        chn = unique_in_order(self._sys._residues.column("chain")[res])
        return [Chain(self._sys, c) for c in chn]

    @property
    def fragids(self) -> np.ndarray:
        return self._sys.fragids[self.ids]

    @property
    def fragments(self) -> list[AtomSel]:
        from .graph import unique_in_order

        return [
            AtomSel(self._sys, self._sys.fragment_atoms(f)) for f in unique_in_order(self.fragids)
        ]

    def expand(self, level: str) -> AtomSel:
        """All atoms sharing a residue, chain, ct or fragment with this selection."""
        if level not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS}")
        key = self._sys._level_key(level)
        return AtomSel(self._sys, np.flatnonzero(np.isin(key, key[self.ids])))

    def _other(self, other) -> np.ndarray:
        if isinstance(other, AtomSel):
            if other._sys is not self._sys:
                raise ValueError("selections come from different systems")
            return other.ids
        return np.asarray([int(a) for a in other], dtype=np.int64)

    def __or__(self, other) -> AtomSel:
        return AtomSel(self._sys, np.union1d(self.ids, self._other(other)))

    def __and__(self, other) -> AtomSel:
        return AtomSel(self._sys, np.intersect1d(self.ids, self._other(other)))

    def __sub__(self, other) -> AtomSel:
        return AtomSel(self._sys, np.setdiff1d(self.ids, self._other(other)))

    def clone(self, **kwargs):
        return self._sys.clone(self.ids, **kwargs)

    def to_rdkit(self, **kwargs):
        """RDKit molecule of these atoms; see ``boonza.chem.to_rdkit``."""
        from .chem import to_rdkit

        return to_rdkit(self._sys, self.ids, **kwargs)

    def describe(self, **kwargs):
        """Force-field report of these atoms; see ``boonza.describe``."""
        from .describe import describe

        return describe(self._sys, self.ids, **kwargs)

    def __repr__(self) -> str:
        return f"<AtomSel {len(self)} atoms>"

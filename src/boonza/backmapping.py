"""Coarse-grained beads back to atoms, whichever model they are.

:func:`backmap` is the one entry point; what it does depends on the model the beads
belong to.  SIRAH is turned back here (:mod:`boonza.sirah.backmapping`).  Martini
is not yet.
"""

from __future__ import annotations

#: the models :func:`backmap` knows of; ``auto`` reads the beads
MODELS = ("sirah", "martini")


def detect_model(cg) -> str | None:
    """Which model ``cg``'s beads are: ``sirah`` (its backbone beads GN, GC, GO, or a
    nucleotide's PX), ``martini`` (a BB bead), or None when it is neither."""
    names = {str(x).strip() for x in set(cg.atoms["name"].tolist())}
    if {"GN", "GC", "GO"} <= names or "PX" in names:
        return "sirah"
    if "BB" in names:
        return "martini"
    return None


def backmap(cg, reference=None, positions=None, *, model: str = "auto", **options):
    """The all-atom structure of a coarse-grained frame.

    ``cg`` is the coarse-grained system and ``positions`` its beads' positions for
    the frame wanted (default: its own); ``reference`` the all-atom structure the
    run was built from, where the model can use one.  ``model``: ``sirah``,
    ``martini``, or ``auto`` to read it from the bead names
    (:func:`detect_model`).  The other ``options`` are the model's own:
    :func:`boonza.sirah.backmap` for SIRAH.  Martini is not turned back yet.
    """
    if model == "auto":
        model = detect_model(cg)
        if model is None:
            raise ValueError("these beads are neither SIRAH's nor Martini's by their names: "
                             "say which with model")  # fmt: skip
    if model == "sirah":
        from .sirah import backmap as sirah_backmap

        return sirah_backmap(cg, reference, positions, **options)
    if model == "martini":
        raise ValueError("Martini beads cannot be turned back to atoms yet: only SIRAH's "
                         "(model='sirah')")  # fmt: skip
    raise ValueError(f"model is one of {', '.join(('auto', *MODELS))}, not {model!r}")

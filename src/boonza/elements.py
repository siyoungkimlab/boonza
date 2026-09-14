"""Element symbols by atomic number (0 is a pseudo particle), and bonding radii."""

import numpy as np

SYMBOLS = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)  # fmt: skip

_ANUM = {s.lower(): i for i, s in enumerate(SYMBOLS) if s}


def symbol(anum: int) -> str:
    return SYMBOLS[anum] if 0 <= anum < len(SYMBOLS) else ""


def atomic_number(sym: str) -> int:
    return _ANUM.get(sym.strip().lower(), 0)


# Radii used for bond guessing, as in msys (Bondi 1964 with corrections from
# Rowland & Taylor 1996, Mantina 2009 and CHARMM36 ion radii); index = atomic
# number 0..111, 2.0 beyond.
_RADII = np.array(
    [0.00, 1.10, 1.40, 1.30, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54, 1.41, 1.18, 1.84,
     2.10, 1.80, 1.80, 2.27, 1.88, 1.76, 1.37]
    + [2.00] * 7
    + [1.09, 1.09, 1.09, 1.87, 2.11, 1.85, 1.90, 1.83, 2.02, 1.90, 2.49]
    + [2.00] * 7
    + [1.63, 1.72, 1.36, 1.93, 2.17, 2.06, 2.06, 1.98, 2.16, 2.10, 1.89]
    + [2.00] * 21
    + [1.72, 1.66, 1.55, 1.96, 2.02, 2.07, 1.97, 2.02, 2.20, 3.48, 2.83]
    + [2.00] * 23
)  # fmt: skip
_NMSYS = len(_RADII)  # msys knows elements 0..111
_FAST = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}
_BY_SYMBOL = {s: i for i, s in enumerate(SYMBOLS[:_NMSYS]) if s}


# Element masses as msys has them (elements.cxx), index = atomic number 0..111;
# used to guess elements from masses (Amber prmtop files carry no elements).
_MSYS_MASSES = np.array([
    0.000000, 1.007940, 4.002600, 6.941000, 9.012182, 10.811000, 12.010700, 14.006700,
    15.999400, 18.998403, 20.179700, 22.989770, 24.305000, 26.981538, 28.085500, 30.973761,
    32.065000, 35.453000, 39.948000, 39.098300, 40.078000, 44.955910, 47.867000, 50.941500,
    51.996100, 54.938049, 55.845000, 58.933200, 58.693400, 63.546000, 65.409000, 69.723000,
    72.640000, 74.921600, 78.960000, 79.904000, 83.798000, 85.467800, 87.620000, 88.905850,
    91.224000, 92.906380, 95.940000, 98.000000, 101.070000, 102.905500, 106.420000, 107.868200,
    112.411000, 114.818000, 118.710000, 121.760000, 127.600000, 126.904470, 131.293000, 132.905450,
    137.327000, 138.905500, 140.116000, 140.907650, 144.240000, 145.000000, 150.360000, 151.964000,
    157.250000, 158.925340, 162.500000, 164.930320, 167.259000, 168.934210, 173.040000, 174.967000,
    178.490000, 180.947900, 183.840000, 186.207000, 190.230000, 192.217000, 195.078000, 196.966550,
    200.590000, 204.383300, 207.200000, 208.980380, 209.000000, 210.000000, 222.000000, 223.000000,
    226.000000, 227.000000, 232.038100, 231.035880, 238.028910, 237.000000, 244.000000, 243.000000,
    247.000000, 247.000000, 251.000000, 252.000000, 257.000000, 258.000000, 259.000000, 262.000000,
    261.000000, 262.000000, 266.000000, 264.000000, 269.000000, 268.000000, 271.000000, 272.000000,
]
)  # fmt: skip
_BY_MASS = np.argsort(_MSYS_MASSES, kind="stable")


def guess_atomic_number(mass) -> np.ndarray:
    """Atomic numbers from masses, as msys GuessAtomicNumber: the element whose
    mass is closest (0 for masses that are not positive or beyond the table)."""
    m = np.atleast_1d(np.asarray(mass, dtype=np.float64))
    sorted_masses = _MSYS_MASSES[_BY_MASS]
    rhs = np.searchsorted(sorted_masses, m, side="left")  # first not less than mass
    inside = (rhs > 0) & (rhs < len(sorted_masses))
    rhs_c = np.clip(rhs, 1, len(sorted_masses) - 1)
    lhs = rhs_c - 1
    ldiff = m - sorted_masses[lhs]
    rdiff = sorted_masses[rhs_c] - m
    pick = np.where(ldiff < rdiff, _BY_MASS[lhs], _BY_MASS[rhs_c])
    return np.where(inside, pick, 0).astype(np.int64)


def radii(anum) -> np.ndarray:
    """Bond-guessing radius for each atomic number (0 for negative, 2.0 if unknown)."""
    anum = np.asarray(anum, dtype=np.int64)
    out = np.full(anum.shape, 2.0)
    known = (anum >= 0) & (anum < _NMSYS)
    out[known] = _RADII[anum[known]]
    out[anum < 0] = 0.0
    return out


def element_for_abbreviation(abbr: str) -> int:
    """Atomic number for an element symbol, matched like msys: case-insensitive,
    first two characters only, a non-letter second character is ignored."""
    if len(abbr) == 1 and abbr in _FAST:
        return _FAST[abbr]
    c0 = abbr[:1].upper()
    c1 = abbr[1:2]
    c1 = c1.lower() if c1.isascii() and c1.isalpha() else ""
    return _BY_SYMBOL.get(c0 + c1, 0)


def msys_symbol(anum: int) -> str:
    """Element symbol as msys writes it: "X" for 0, "" when unknown."""
    if anum == 0:
        return "X"
    return SYMBOLS[anum] if 0 < anum < _NMSYS else ""


# Element masses as MDAnalysis guesses them (MDAnalysis.guesser.tables.masses),
# used for formats without masses (GRO) so mass-weighted analyses agree.
_GUESSED_MASSES = {
    "Ac": 227.028, "Ag": 107.8682, "Al": 26.981539, "Am": 243.0, "Ar": 39.948, "As": 74.92159,
    "At": 210.0, "Au": 196.96654, "B": 10.811, "BR": 79.904, "Ba": 137.327, "Be": 9.012182,
    "Bh": 264.0, "Bi": 208.98037, "Bk": 247.0, "C": 12.011, "CA": 40.08, "CL": 35.45,
    "CS": 132.9, "CU": 63.546, "Cd": 112.411, "Ce": 140.116, "Cf": 251.0, "Cm": 247.0,
    "Co": 58.9332, "Cr": 51.9961, "Db": 262.0, "Dy": 162.5, "Er": 167.26,
    "Es": 252.0, "Eu": 151.965, "F": 18.998, "FE": 55.847, "Fm": 257.0, "Fr": 223.0,
    "Ga": 69.723, "Gd": 157.25, "Ge": 72.61, "H": 1.008, "HE": 4.0026, "Hf": 178.49,
    "Hg": 200.59, "Ho": 164.93032, "Hs": 265.0, "I": 126.9045, "In": 114.82, "Ir": 192.22,
    "K": 39.102, "Kr": 83.8, "La": 138.9055, "Li": 6.941, "Lr": 262.0, "Lu": 174.967,
    "MG": 24.305, "Md": 258.0, "Mn": 54.93805, "Mo": 95.94, "Mt": 266.0, "N": 14.007,
    "NA": 22.98977, "NE": 20.1797, "Nb": 92.90638, "Nd": 144.24,
    "Ni": 58.6934, "No": 259.0, "Np": 237.048, "O": 15.999, "Os": 190.2, "P": 30.974,
    "Pa": 231.0359, "Pb": 207.2, "Pd": 106.42, "Pm": 145.0, "Po": 209.0, "Pr": 140.90765,
    "Pt": 195.08, "Pu": 244.0, "RB": 85.4678, "Ra": 226.025, "Re": 186.207, "Rf": 261.0,
    "Rh": 102.9055, "Rn": 222.0, "Ru": 101.07, "S": 32.06, "Sb": 121.757, "Sc": 44.95591,
    "Se": 78.96, "Sg": 263.0, "Si": 28.0855, "Sm": 150.36, "Sn": 118.71, "Sr": 87.62,
    "Ta": 180.9479, "Tb": 158.92534, "Tc": 98.0, "Te": 127.6, "Th": 232.0381, "Ti": 47.88,
    "Tl": 204.3833, "Tm": 168.93421, "U": 238.0289, "V": 50.9415, "W": 183.85, "Xe": 131.29,
    "Y": 88.90585, "Yb": 173.04, "ZN": 65.37, "Zr": 91.224,
}  # fmt: skip


def guessed_masses(anum) -> np.ndarray:
    """Masses (amu) for atomic numbers, as MDAnalysis guesses them; 0 when unknown."""
    anum = np.asarray(anum, dtype=np.int64)
    table = {}
    for z in np.unique(anum).tolist():
        if z <= 0:
            table[z] = 0.0
            continue
        sym = symbol(z)
        table[z] = _GUESSED_MASSES.get(sym.upper(), _GUESSED_MASSES.get(sym, 0.0))
    return np.array([table[z] for z in anum.tolist()], dtype=np.float64)

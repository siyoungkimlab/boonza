#!/usr/bin/env python
"""How mobile was the ligand?  Two modes, picked from what you pass:

  boonza_md_analysis.py <workdir> [<workdir> ...]  one CSV row per boonza md run
  boonza_md_analysis.py <structure> <trajectory>   per-frame dRMSD for one run

The headline column is drmsd: the average, over the production frames, of
the RMS deviation of the pocket-CA / ligand-heavy distance matrix from its
value at the start of production.  Small means the ligand sat still in its
pocket, large means it moved (and, past a few A, that it left).

dRMSD is used rather than a fitted ligand RMSD because it needs no
superposition and no unwrapping: every distance is minimum-image, so a ligand
that diffuses across a periodic boundary is still measured correctly.

It is boonza.drmsd, symmetry-corrected: each frame pairs the ligand atoms in
whichever of the molecule's symmetric ways fits best, so a flipped ring or
swapped carboxylate oxygens do not count as motion.  drmsd_plain pairs them
in file order instead.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import sys
import tomllib

import numpy as np

import boonza

COLUMNS = [
    # identity
    "workdir", "folder", "config", "model",
    "pocket_chain", "pocket_resid", "pocket_resname",
    # the point of the exercise
    "drmsd", "drmsd_std", "drmsd_min", "drmsd_max",
    "drmsd_first", "drmsd_last", "drmsd_plain", "drmsd_final_pdb",
    "drmsd_reference", "n_pocket_atoms", "n_ligand_heavy_atoms",
    # how the run ended
    "outcome", "production_ns_target", "production_ns_reached",
    "production_step_reached", "n_frames", "frame_interval_ns",
    # what the ligand did, as boonza md's own monitor saw it
    "initial_contact_count", "final_contact_count",
    "contact_fraction_mean", "contact_fraction_final",
    "min_pocket_distance_nm_first", "min_pocket_distance_nm_final",
    "min_pocket_distance_nm_max",
    # boltz confidence for this diffusion sample
    "confidence_score", "ptm", "iptm", "ligand_iptm",
    "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    # how it was set up
    "input_structure", "yaml", "ligand_smiles",
    "forcefields", "ligandff", "temperature_K", "integration_fs",
    "hmr", "early_stop", "monitor_target", "dihedral_restraint",
    "padding_nm", "saltM", "n_atoms_system",
    # sanity checks
    "mean_temperature_K", "mean_density_g_mL", "ns_per_day", "error",
]

# Counts and step numbers.  Some reach us as numpy floats -- monitor.csv is
# parsed as floats throughout, and json gives back whatever status.json holds
# -- and "137.0" in a column of counts is just noise to read past.
INT_COLUMNS = {
    "model", "pocket_resid", "n_pocket_atoms", "n_ligand_heavy_atoms",
    "n_atoms_system", "n_frames", "production_step_reached",
    "initial_contact_count", "final_contact_count",
}

MONITOR_SELECTORS = ("monitor_chain", "monitor_ligand", "monitor_component",
                     "monitor_selection")

# Set per worker by init_worker, so the settings survive a spawned start method.
YAMLDIR = ""
POCKETSEL = "(protein and name CA) or name BB GC"
POCKET_CUTOFF = 10.0
LIGANDSEL = "chain L or resname LIG LIG1"


def init_worker(yamldir, pocketsel, pocket_cutoff, ligandsel, threads=None):
    global YAMLDIR, POCKETSEL, POCKET_CUTOFF, LIGANDSEL
    YAMLDIR, POCKETSEL, POCKET_CUTOFF = yamldir, pocketsel, pocket_cutoff
    LIGANDSEL = ligandsel
    if threads:
        # boonza's distance kernels are numba-parallel; one thread per worker
        # process, or N workers each start a thread per core.
        import numba
        numba.set_num_threads(threads)


def heavy_ligand(system, ligandsel):
    """Heavy-atom indices of the ligand selection."""
    ids = system.select("(%s) and not hydrogen" % ligandsel).ids
    if len(ids) == 0:
        raise RuntimeError("no ligand heavy atoms matched %r" % ligandsel)
    return ids


def snapshot(system, positions, box):
    """``system`` with other coordinates and box, as a reference or one frame."""
    out = system.clone()  # not structure_only: that drops pseudo particles
    out.positions = positions
    out.cell = box
    return out


def from_file(system, path):
    """``system`` with the coordinates and box of a structure file of it.

    The atom names have to line up, not only the count: the positions are
    pasted on by index, and a .mae puts pseudo particles at the end rather than
    where they sit (Desmond keeps them in their own block), so a Martini system
    with a tryptophan virtual site comes back in a different order.  A .dms or
    a .pdb of the same system keeps it.
    """
    other = boonza.load(path)
    if other.natoms != system.natoms:
        raise RuntimeError("%s has %d atoms, the system %d"
                           % (path, other.natoms, system.natoms))
    mine = [str(n) for n in system.atoms["name"]]
    theirs = [str(n) for n in other.atoms["name"]]
    if mine != theirs:
        first = next(i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b)
        raise RuntimeError("%s is in a different atom order (atom %d is %s there, %s in the "
                           "system), so its coordinates belong to other atoms"
                           % (path, first, theirs[first], mine[first]))
    return snapshot(system, other.positions, other.cell)


def measure(system, reference, ligand, positions=None):
    """boonza.drmsd with this script's pocket: POCKETSEL within POCKET_CUTOFF
    of the ligand in the reference, fixed from then on."""
    return boonza.drmsd(system, reference, ligand=ligand, protein=POCKETSEL,
                        cutoff=POCKET_CUTOFF, positions=positions)


def frame_times(traj):
    """Frame times in ps, reading one atom per frame."""
    times = [block.times for block in traj.chunks(256, atoms=[0])]
    return np.concatenate(times) if times else np.empty(0)


def read_toml(path):
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def describe_forcefields(value):
    """final.toml's forcefields: a list whose items are a name, or a list of a
    force field and the patches merged onto it."""
    if not value:
        return None
    if isinstance(value, str):
        return value
    return "; ".join(item if isinstance(item, str) else " + ".join(item)
                     for item in value)


def read_report_csv(path):
    """Return {column: [floats]} for a boonza md report CSV, or {}."""
    try:
        with open(path) as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return {}
    if len(rows) < 2:
        return {}
    header = [h.strip().lstrip("#").strip('"') for h in rows[0]]
    cols = {h: [] for h in header}
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        for h, v in zip(header, row):
            try:
                cols[h].append(float(v))
            except ValueError:
                cols[h].append(np.nan)
    return cols


def mean(values):
    """The mean of the numbers among ``values``, or None when there are none.

    A column can parse to all-NaN rather than to nothing: boonza leaves
    contact_fraction empty on every row of a run whose ligand had no pocket
    contacts to begin with (initial_contact_count 0, nothing to divide by), and
    np.nanmean of that warns "Mean of empty slice" and returns NaN.
    """
    vals = np.asarray(values, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else None


def number(value):
    """``value`` as a float, or None when it is not a number (NaN, a blank cell)."""
    return float(value) if np.isfinite(value) else None


_yaml_cache = {}


def read_yaml(config, folder):
    """SMILES and protein sequence out of the boltz input YAML: DIR/<config>.yaml
    with -y, else the copy boltz left in the prediction folder.

    Deliberately a regex and not pyyaml: two scalars off a fixed template are
    not worth a dependency the analysis would otherwise not need.  The first
    ligand is the one given first in the YAML.
    """
    key = (config, folder)
    if key in _yaml_cache:
        return _yaml_cache[key]
    info = {}
    path = ""
    if config:
        path = os.path.join(YAMLDIR or folder, config + ".yaml")
    if path and os.path.exists(path):
        text = open(path).read()
        # The ligand of interest is the pocket binder when the YAML names one.
        binder = re.search(r"binder:\s*(\S+)", text)
        smiles = re.findall(r"id:\s*(\S+)\s+smiles:\s*(\S+)", text)
        pick = [s for i, s in smiles if binder and i == binder.group(1)]
        if pick or smiles:
            info["smiles"] = (pick or [smiles[0][1]])[0]
        m = re.search(r"sequence:\s*(\S+)", text)
        if m:
            info["sequence"] = m.group(1)
        info["yaml"] = os.path.normpath(path)
    _yaml_cache[key] = info
    return info


def last(values, default=None):
    return values[-1] if len(values) else default


def analyse(workdir):
    workdir = os.path.normpath(workdir)
    folder_dir = os.path.dirname(workdir)
    row = {"workdir": workdir}
    folder = os.path.basename(workdir)
    row["folder"] = folder

    # step3_md.sh names the work directory md_<stem of the input .mae>, and
    # boltz and step2 name that <config>_model_<N>.prepped.  A directory named
    # otherwise leaves these blank rather than guessing.
    stem = folder[3:] if folder.startswith("md_") else folder
    m = re.search(r"^(.*?)_model_(\d+)", stem)
    config = ""
    if m:
        config, row["model"] = m.group(1), int(m.group(2))
        row["config"] = config
        # input_A_1: the pocket probe sat on chain A residue 1, and that is
        # the only thing that differs between the runs.
        m = re.search(r"_([A-Za-z0-9]+)_(\d+)$", config)
        if m:
            row["pocket_chain"], row["pocket_resid"] = m.group(1), int(m.group(2))

    info = read_yaml(config, folder_dir)
    row["yaml"] = info.get("yaml", "")
    row["ligand_smiles"] = info.get("smiles", "")
    seq = info.get("sequence", "")
    resid = row.get("pocket_resid")
    if seq and resid and 1 <= resid <= len(seq):
        row["pocket_resname"] = seq[resid - 1]

    errors = []

    toml_path = os.path.join(workdir, "final.toml")
    mae = stem + ".mae"
    report_ns = None
    if os.path.exists(toml_path):
        t = read_toml(toml_path)
        mae = os.path.basename(t.get("input_structure", "")) or mae
        row["forcefields"] = (describe_forcefields(t.get("forcefields"))
                              or t.get("proteinff"))
        row["ligandff"] = t.get("ligandff")
        row["temperature_K"] = t.get("temperature")
        row["integration_fs"] = t.get("integration_fs")
        row["hmr"] = t.get("hmr")
        row["early_stop"] = t.get("early_stop")
        row["monitor_target"] = next(
            ("%s %s" % (k[len("monitor_"):], t[k]) for k in MONITOR_SELECTORS if t.get(k)),
            None)
        row["dihedral_restraint"] = t.get("dihedral_restraint")
        row["padding_nm"] = t.get("padding_nm")
        row["saltM"] = t.get("saltM")
        row["production_ns_target"] = t.get("production_ns")
        report_ns = t.get("production_report_interval_ns")
    else:
        errors.append("no final.toml")

    # Only the name is taken from final.toml, never the path it records: that
    # is where the run happened, on whatever machine ran it, and it does not
    # survive the tree being copied or moved.  step3 creates the work
    # directory beside its input, so re-rooting the name there points at the
    # file as it sits now.  A run that died before writing final.toml falls
    # back to the name step3 derived the directory from.
    row["input_structure"] = os.path.join(folder_dir, mae)

    status_path = os.path.join(workdir, "status.json")
    if os.path.exists(status_path):
        s = json.load(open(status_path))
        row["outcome"] = s.get("outcome")
        row["production_ns_reached"] = s.get("final_production_time_ns")
        row["production_step_reached"] = s.get("final_production_step")
        row.setdefault("production_ns_target", s.get("target_production_ns"))
    else:
        errors.append("no status.json")
        row["outcome"] = "incomplete"

    # Boltz confidence for this diffusion sample, from the prediction folder
    # the work directory was created inside.
    conf_path = os.path.join(
        folder_dir, "confidence_%s_model_%s.json" % (config, row.get("model")))
    if config and os.path.exists(conf_path):
        c = json.load(open(conf_path))
        for key in ("confidence_score", "ptm", "iptm", "ligand_iptm",
                    "complex_plddt", "complex_iplddt", "complex_pde",
                    "complex_ipde"):
            row[key] = c.get(key)

    mon = read_report_csv(os.path.join(workdir, "monitor.csv"))
    if mon:
        frac = np.array(mon.get("contact_fraction", []), dtype=float)
        dist = np.array(mon.get("min_ligand_pocket_distance_nm", []), dtype=float)
        counts = np.array(mon.get("contact_count", []), dtype=float)
        init = np.array(mon.get("initial_contact_count", []), dtype=float)
        row["initial_contact_count"] = last(init)
        row["final_contact_count"] = last(counts)
        if frac.size:
            row["contact_fraction_mean"] = mean(frac)
            row["contact_fraction_final"] = number(frac[-1])
        if dist.size:
            row["min_pocket_distance_nm_first"] = number(dist[0])
            row["min_pocket_distance_nm_final"] = number(dist[-1])
            row["min_pocket_distance_nm_max"] = (
                float(np.nanmax(dist)) if np.isfinite(dist).any() else None)  # fmt: skip

    state = read_report_csv(os.path.join(workdir, "state.csv"))
    for src, dst in (("Temperature (K)", "mean_temperature_K"),
                     ("Density (g/mL)", "mean_density_g_mL")):
        row[dst] = mean(state.get(src, []))

    perf = read_report_csv(os.path.join(workdir, "performance.csv"))
    nsday = perf.get("ns_per_day", [])
    if nsday:
        row["ns_per_day"] = float(nsday[-1])

    # --- dRMSD -----------------------------------------------------------
    # solvated.dms rather than solvated.pdb: it carries the bonds the
    # symmetry correction pairs ligand atoms by.
    top = os.path.join(workdir, "solvated.dms")
    traj = os.path.join(workdir, "trajectory.dcd")
    if not os.path.exists(top):
        errors.append("no solvated.dms")
        row["error"] = "; ".join(errors)
        return row

    try:
        system = boonza.load(top)
        row["n_atoms_system"] = int(system.natoms)

        # Prefer the ligand atoms boonza md's monitor watched over a
        # selection: they are the target early stop was told to watch, and
        # cannot pick up a second copy of the molecule.
        pocket_json = os.path.join(workdir, "pocket.json")
        if os.path.exists(pocket_json):
            pk = json.load(open(pocket_json))
            lig_idx = np.asarray(pk["ligand_heavy_atom_indices"], dtype=np.int64)
        else:
            lig_idx = heavy_ligand(system, LIGANDSEL)
        row["n_ligand_heavy_atoms"] = int(len(lig_idx))

        # Production starts from equilibrated.pdb, and the first saved frame is
        # a whole report interval in, so the reference matrix comes from that
        # file rather than from frame 0 -- otherwise the first interval of
        # motion is silently excluded from the average.
        equil = os.path.join(workdir, "equilibrated.pdb")
        have_equil = os.path.exists(equil)
        ref = from_file(system, equil) if have_equil else system
        row["drmsd_reference"] = "equilibrated" if have_equil else "solvated"

        # final.pdb is written whenever the run stops, so a run that detached
        # before its first production report -- exactly the runs with the most
        # mobile ligands -- still gets a number here rather than a blank cell.
        final_pdb = os.path.join(workdir, "final.pdb")
        if os.path.exists(final_pdb):
            r = measure(from_file(system, final_pdb), ref, lig_idx)
            row["drmsd_final_pdb"] = float(r.drmsd)
            row["n_pocket_atoms"] = int(len(r.pocket))

        if not os.path.exists(traj):
            raise RuntimeError("no trajectory.dcd")
        if os.path.getsize(traj) == 0:
            row["n_frames"] = 0
            raise RuntimeError(
                "trajectory.dcd is empty: the run stopped before its first "
                "production report, so drmsd_final_pdb is the only mobility "
                "measure for it")

        frames = boonza.open_trajectory(traj, system)
        row["n_frames"] = int(len(frames))
        if len(frames) == 0:
            raise RuntimeError("trajectory.dcd has no frames")
        r = measure(system, ref, lig_idx, positions=frames)
        drmsd = np.atleast_1d(r.drmsd)
        row["n_pocket_atoms"] = int(len(r.pocket))
        row["drmsd"] = float(np.mean(drmsd))
        row["drmsd_std"] = float(np.std(drmsd))
        row["drmsd_min"] = float(np.min(drmsd))
        row["drmsd_max"] = float(np.max(drmsd))
        row["drmsd_first"] = float(drmsd[0])
        row["drmsd_last"] = float(drmsd[-1])
        if r.plain_drmsd is not None:
            row["drmsd_plain"] = float(np.mean(r.plain_drmsd))
        if len(frames) > 1:
            times = frame_times(frames)
            row["frame_interval_ns"] = float(times[1] - times[0]) / 1000.0
        elif report_ns is not None:
            row["frame_interval_ns"] = report_ns
    except Exception as exc:                                    # noqa: BLE001
        errors.append("%s: %s" % (type(exc).__name__, exc))

    row["error"] = "; ".join(errors)
    return row


def analyse_safe(workdir):
    """A run that blows up in an unforeseen way costs its row, not the harvest."""
    try:
        return analyse(workdir)
    except Exception as exc:                                    # noqa: BLE001
        return {"workdir": os.path.normpath(workdir),
                "error": "%s: %s" % (type(exc).__name__, exc)}


def csv_value(value, column):
    """Blank for a missing value, plain digits for an integral count."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if column in INT_COLUMNS and not isinstance(value, bool):
        if isinstance(value, (int, np.integer)):
            return int(value)
        # A count that is not whole means something is wrong with the input;
        # keep it verbatim rather than truncating the evidence away.
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return value


def write_csv(rows, path):
    rows.sort(key=lambda r: (r.get("config", ""), r.get("model") or 0,
                             r.get("workdir", "")))
    fields = COLUMNS + sorted({k for r in rows for k in r} - set(COLUMNS))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: csv_value(r.get(k), k) for k in fields})


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    """Raw epilog, plus the default appended to every option that has one.

    -o, -y and -r have no useful default to show -- what they do when left off
    is a behaviour, not a value, and is described in prose -- and a selection
    string reads better quoted than ArgumentDefaultsHelpFormatter renders it.
    """

    def _get_help_string(self, action):
        help_text = action.help or ""
        if (action.default is argparse.SUPPRESS
                or action.default in (None, "")
                or "%(default)" in help_text):
            return help_text
        if isinstance(action.default, str):
            return help_text + " (default: %(default)r)"
        return super()._get_help_string(action)


def run_trajectory(args):
    """structure + trajectory: print the average dRMSD, write it per frame."""
    structure, trajectories = args.input[0], args.input[1:]
    system = boonza.load(structure)
    try:
        lig_idx = heavy_ligand(system, args.ligandsel)
    except RuntimeError:
        sys.exit("no ligand heavy atoms matched %r in %s -- pass -l with the "
                 "right selection" % (args.ligandsel, structure))
    trajs = [boonza.open_trajectory(t, system) for t in trajectories]

    # Frame 0 is the reference unless -r names a snapshot to measure against,
    # which is what you want when the first saved frame is already a report
    # interval into production (boonza md's equilibrated.pdb).
    if args.ref:
        ref, ref_name = from_file(system, args.ref), args.ref
    elif trajs and len(trajs[0]):
        f = trajs[0][0]
        ref, ref_name = snapshot(system, f.positions, f.box), "frame 0"
    else:
        sys.exit("no frames to measure")

    drmsd, times = [], []
    try:
        if trajs:
            for t in trajs:
                if len(t) == 0:
                    continue
                r = measure(system, ref, lig_idx, positions=t)
                drmsd.append(np.atleast_1d(r.drmsd))
                times.append(frame_times(t))
        else:
            r = measure(system, ref, lig_idx)
            drmsd.append(np.atleast_1d(r.drmsd))
            times.append(np.zeros(1))
    except ValueError as exc:
        sys.exit(str(exc))
    if not drmsd:
        sys.exit("no frames to measure")
    drmsd, times = np.concatenate(drmsd), np.concatenate(times)

    out = args.out or "drmsd.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "time_ps", "time_ns", "drmsd"])
        for i, (t, d) in enumerate(zip(times, drmsd)):
            writer.writerow([i, t, t / 1000.0, d])

    print("%s%s" % (structure,
                    " + " + ", ".join(trajectories) if trajectories else ""))
    print("  ligand:    (%s), %d heavy atoms" % (args.ligandsel, len(lig_idx)))
    print("  pocket:    (%s) within %g A of it, %d atoms"
          % (args.pocketsel, args.cutoff, len(r.pocket)))
    print("  reference: %s" % ref_name)
    print("  frames:    %d" % drmsd.size)
    print()
    print("  average dRMSD  %.3f A" % drmsd.mean())
    print("  std %.3f   min %.3f   max %.3f   first %.3f   last %.3f"
          % (drmsd.std(), drmsd.min(), drmsd.max(), drmsd[0], drmsd[-1]))
    print()
    print("wrote %s" % out)


def run_batch(args, dirs):
    """work directories: one summary row per boonza md run."""
    jobs = max(1, min(args.jobs, len(dirs)))
    print("%d work directories, %d workers." % (len(dirs), jobs))
    print("  pocket: (%s) within %g A of the ligand at t=0"
          % (args.pocketsel, args.cutoff))

    initargs = (args.yamldir, args.pocketsel, args.cutoff, args.ligandsel)
    rows = []
    if jobs == 1:
        init_worker(*initargs)
        for i, d in enumerate(dirs, 1):
            rows.append(analyse_safe(d))
            print("  %d/%d" % (i, len(dirs)), end="\r", file=sys.stderr)
    else:
        with mp.Pool(jobs, initializer=init_worker, initargs=(*initargs, 1)) as pool:
            for i, row in enumerate(pool.imap_unordered(analyse_safe, dirs), 1):
                rows.append(row)
                print("  %d/%d" % (i, len(dirs)), end="\r", file=sys.stderr)
    print(" " * 30, end="\r", file=sys.stderr)

    out = args.out or "analysis.csv"
    write_csv(rows, out)
    print("%d rows, %d with a dRMSD, %d with an error"
          % (len(rows), sum(1 for r in rows if r.get("drmsd") is not None),
             sum(1 for r in rows if r.get("error"))))
    print("wrote " + out)

def report_top20(path):
    """Print the 20 runs with the smallest drmsd."""
    rows = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:

            if row.get("outcome") != "target_reached":
                continue

            try:
                drmsd = float(row["drmsd"])
            except (KeyError, TypeError, ValueError):
                continue

            workdir = row.get("workdir", "")
            final_pdb = os.path.join(workdir, "final.pdb")
            rows.append((drmsd, final_pdb))

    rows.sort(key=lambda x: x[0])

    print("structure\tdrmsd")
    for drmsd, final_pdb in rows[:20]:
        print("%s\t%.3f" % (final_pdb, drmsd))

def main():
    global POCKETSEL, POCKET_CUTOFF
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
        epilog="""\
Given directories, these are boonza md work directories -- the ones holding
solvated.dms and trajectory.dcd -- and the output is one summary row each,
in analysis.csv.  Let the shell do the matching rather than naming a parent:

  %(prog)s boltz_results_*/predictions/*/md_*/
  %(prog)s -j 8 -o mobility.csv */*/*/md_*/
  %(prog)s -y ../yamls boltz_results_input/predictions/*/md_*/

Runs still in flight, or that died before writing a trajectory, still get a
row -- with the identity columns filled in and an error column saying what was
missing -- so a partial harvest is visible rather than silently short.

Paths are reported as given, so a relative glob yields relative workdir cells.

config, model and pocket_resid are read off the work directory name, which
step3_md.sh derives from the input: md_<config>_model_<N>.prepped, with the
config ending in <chain>_<resid> of the pocket it was built with (input_A_1).
A directory named otherwise just leaves those cells blank.

Given files instead, they are a structure and its trajectory (several
trajectories are read in order, as one run), and the output is the average
dRMSD on stdout plus a per-frame drmsd column in drmsd.csv:

  %(prog)s md_run/solvated.dms md_run/trajectory.dcd
  %(prog)s -r md_run/equilibrated.pdb md_run/solvated.dms md_run/trajectory.dcd
  %(prog)s -l "resname UNL" -c 8 system.gro traj1.xtc traj2.xtc

Here the ligand comes from -l rather than from boonza md's pocket.json, and
the reference is frame 0 unless -r names a snapshot to measure against.  Give
a structure with bonds (DMS, MAE, or PDB with CONECT records) so the ligand's
symmetric atoms can be paired.
""")
    p.add_argument("input", nargs="+", metavar="PATH",
                   help="boonza md work directories, or a structure file "
                        "followed by its trajectory files")
    p.add_argument("-o", "--out", default=None,
                   help="output CSV; without it, analysis.csv for work "
                        "directories or drmsd.csv for a trajectory")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4,
                   help="worker processes")
    p.add_argument("-y", "--yamldir", default="",
                   help="directory of boltz input YAMLs, read as "
                        "DIR/<config>.yaml, for ligand_smiles and "
                        "pocket_resname; without it, the YAML boltz left in "
                        "each prediction folder")
    # drmsd_trj.py defaults to CA within 5 A of the ligand, which for these
    # systems is only ~3 atoms -- too few to average over -- so the cutoff is
    # opened up to 10 A (~36 CAs).  Pass a wider -s (e.g. "protein and not
    # hydrogen") for a finer-grained measure; the pocket size is reported
    # either way, as n_pocket_atoms or on stdout.
    p.add_argument("-c", "--cutoff", type=float, default=10.0,
                   help="pocket cutoff in A for the dRMSD reference")
    p.add_argument("-s", "--pocketsel", default=POCKETSEL,
                   help="pocket atom selection (boonza's, as msys and VMD); the default takes "
                        "one atom a residue in any model -- CA all-atom, BB under Martini, GC "
                        "under SIRAH, which its map puts on the alpha carbon.  'protein' matches "
                        "no beads, so it guards the CA half alone (a calcium ion is named CA "
                        "too)")  # fmt: skip
    p.add_argument("-l", "--ligandsel", default="chain L",
                   help="ligand selection, hydrogens dropped from it either "
                        "way; ignored for a work directory that has a "
                        "pocket.json")
    p.add_argument("-r", "--ref", default=None,
                   help="structure to measure against, for a trajectory; "
                        "without it, the trajectory's own frame 0")
    args = p.parse_args()

    missing = [x for x in args.input if not os.path.exists(x)]
    if missing:
        p.error("no such file or directory: " + ", ".join(missing[:5]))
    if args.ref and not os.path.exists(args.ref):
        p.error("no such file: " + args.ref)
    if args.yamldir and not os.path.isdir(args.yamldir):
        p.error("no YAML directory at " + args.yamldir)

    dirs = [x for x in args.input if os.path.isdir(x)]
    if dirs and len(dirs) != len(args.input):
        p.error("give either work directories or a structure plus its "
                "trajectory, not a mixture")

    if dirs:
        run_batch(args, sorted(set(dirs)))
    else:
        if args.ref is None and len(args.input) == 1:
            p.error("a lone structure has nothing to measure against; add its "
                    "trajectory, or -r a reference to compare it to")
        POCKETSEL, POCKET_CUTOFF = args.pocketsel, args.cutoff
        run_trajectory(args)

    report_top20(args.out or 'analysis.csv')


if __name__ == "__main__":
    main()

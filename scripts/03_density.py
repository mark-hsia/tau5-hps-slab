#!/usr/bin/env python3
"""Density profile rho(z) and coexistence densities from a slab trajectory.

Per frame: unwrap each chain across periodic images, locate the slab centre
via the mass-weighted circular mean of chain-COM z, shift that centre to the
box middle, then histogram mass along z. Recentring every frame matters --
the slab diffuses along z over 10 us, and averaging without it smears the
interface and biases both coexistence densities toward each other.

Emits the layout ../Tau5_slab/csat_extractor.py already reads, so that
script works on these results with only ROOT_DIR changed.

  python scripts/03_density.py --temperature 280
  python scripts/03_density.py --all
"""
import argparse
import glob
import json
import os
import re
import sys

import mdtraj as md
import numpy as np

META = 'inputs/build_meta.json'
BIN_NM = 0.5
EQUIL_FRAC = 0.5
N_BLOCKS = 5
CHUNK = 500

AMU_PER_NM3_TO_MG_ML = 1.66053906660


def unwrap_chains(xyz, box, n_mol, n_res):
    nf = xyz.shape[0]
    c = xyz.reshape(nf, n_mol, n_res, 3).copy()
    L = box[:, None, None, :]
    d = np.diff(c, axis=2)
    d -= L * np.round(d / L)
    c[:, :, 1:, :] = c[:, :, :1, :] + np.cumsum(d, axis=2)
    return c


def slab_centre_z(com_z, mass_per_mol, Lz):
    """Mass-weighted circular mean: z is periodic, so a slab straddling
    the boundary has no meaningful arithmetic mean."""
    theta = 2.0 * np.pi * com_z / Lz[:, None]
    w = mass_per_mol[None, :]
    s = (w * np.sin(theta)).sum(1)
    c = (w * np.cos(theta)).sum(1)
    return (np.arctan2(s, c) % (2.0 * np.pi)) * Lz / (2.0 * np.pi)


def slab_box(tag, run_root='run/slab', prep_root='run/prep'):
    """Box lengths (nm) for a slab run.

    OpenMM's DCD here carries no unit-cell block and the topology PDB has no
    CRYST1, so mdtraj reports unitcell_lengths as None. These are NVT runs
    with a rigid box, so reading it once from the run's state XML is exact.
    """
    import openmm as mm
    import openmm.unit as u
    for name in ('state_final.xml', 'state_partial.xml'):
        p = os.path.join(run_root, tag, name)
        if os.path.exists(p):
            with open(p) as f:
                st = mm.XmlSerializer.deserialize(f.read())
            bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
            box = np.array([float(bv[i, i]) for i in range(3)])
            print(f'box source : {p} -> {np.round(box, 3)} nm')
            return box
    p = f'{prep_root}/state_npt_lowT.xml'
    with open(p) as f:
        st = mm.XmlSerializer.deserialize(f.read())
    bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
    box = np.array([float(bv[i, i]) for i in range(3)])
    box[2] *= 5.0                      # Z_MULT in 02_slab.py
    print(f'box source : {p} x Z_MULT -> {np.round(box, 3)} nm')
    return box


def process(temperature, variant='', out_root='results'):
    sfx = f'_{variant}' if variant else ''
    run_root = f'run/slab{sfx}'
    prep_root = f'run/prep{sfx}'
    meta_path = f'inputs{sfx}/build_meta.json'
    with open(meta_path) as f:
        meta = json.load(f)
    n_mol, n_res = int(meta['n_mol']), int(meta['n_residues'])

    tag = f'T{int(round(temperature))}'
    dcd = f'{run_root}/{tag}/traj.dcd'
    if not os.path.exists(dcd):
        print(f'{dcd} not found, skipping')
        return None

    top = md.load_topology(meta['system_pdb'])
    # Bead masses are per amino acid, not per element. The CA-only PDB
    # declares element C, so a.element.mass would give 12.011 for every
    # bead and underestimate all densities by ~8.4x. These are openabc's
    # HPSParser defaults, i.e. the masses OpenMM used during the run.
    HPS_MASS = {'ALA': 71.08, 'ARG': 156.2, 'ASN': 114.1, 'ASP': 115.1,
                'CYS': 103.1, 'GLN': 128.1, 'GLU': 129.1, 'GLY': 57.05,
                'HIS': 137.1, 'ILE': 113.2, 'LEU': 113.2, 'LYS': 128.2,
                'MET': 131.2, 'PHE': 147.2, 'PRO': 97.12, 'SER': 87.08,
                'THR': 101.1, 'TRP': 186.2, 'TYR': 163.2, 'VAL': 99.07}
    missing = sorted({a.residue.name for a in top.atoms} - set(HPS_MASS))
    if missing:
        raise SystemExit(f'no HPS mass for residue(s): {missing}')
    mass = np.array([HPS_MASS[a.residue.name] for a in top.atoms], dtype=float)
    if mass.shape[0] != n_mol * n_res:
        raise SystemExit(f'topology has {mass.shape[0]} atoms, '
                         f'expected {n_mol * n_res}')
    mass_mol = mass.reshape(n_mol, n_res)
    mass_per_mol = mass_mol.sum(1)
    total_mass = mass.sum()

    print(f'\n=== {temperature} K ===')
    print(f'trajectory : {dcd}')
    print(f'chain mass : {mass_per_mol[0]:.1f} amu   system {total_mass:.0f} amu')

    fixed_box = slab_box(tag, run_root, prep_root)
    profiles, box_all = [], []
    for chunk in md.iterload(dcd, top=top, chunk=CHUNK):
        xyz = chunk.xyz.astype(np.float64)
        if chunk.unitcell_lengths is None:
            box = np.tile(fixed_box, (xyz.shape[0], 1))
        else:
            box = chunk.unitcell_lengths.astype(np.float64)
        Lx, Ly, Lz = box[:, 0], box[:, 1], box[:, 2]

        c = unwrap_chains(xyz, box, n_mol, n_res)
        com = (c * mass_mol[None, :, :, None]).sum(2) / mass_per_mol[None, :, None]
        centre = slab_centre_z(com[:, :, 2], mass_per_mol, Lz)

        z = c[:, :, :, 2].reshape(xyz.shape[0], -1)
        z = (z - centre[:, None] + Lz[:, None] / 2.0) % Lz[:, None]

        nbins = int(round(Lz[0] / BIN_NM))
        for i in range(xyz.shape[0]):
            h, _ = np.histogram(z[i], bins=nbins, range=(0.0, Lz[i]), weights=mass)
            vbin = Lx[i] * Ly[i] * (Lz[i] / nbins)
            profiles.append(h / vbin * AMU_PER_NM3_TO_MG_ML)
        box_all.append(box)
        print(f'  {len(profiles)} frames', end='\r', flush=True)

    prof = np.asarray(profiles)
    box_all = np.concatenate(box_all, 0)
    nf, nb = prof.shape
    Lz = float(box_all[:, 2].mean())
    edges = np.linspace(0.0, Lz, nb + 1)
    zc = 0.5 * (edges[:-1] + edges[1:])
    print(f'  {nf} frames, {nb} bins, Lz {Lz:.2f} nm            ')

    recovered = (prof.mean(0) / AMU_PER_NM3_TO_MG_ML
                 * box_all[:, 0].mean() * box_all[:, 1].mean() * (Lz / nb)).sum()
    print(f'  mass check : {recovered:.0f} vs {total_mass:.0f} amu '
          f'({100 * recovered / total_mass:.2f}%)')

    start = int(nf * EQUIL_FRAC)
    prod = prof[start:]
    print(f'  production : frames {start}-{nf}')

    dz = np.abs(zc - Lz / 2.0)
    dense_t = prod[:, dz < 0.10 * Lz].mean(1)
    dilute_t = prod[:, dz > 0.40 * Lz].mean(1)

    def blocked(x):
        m = np.array([b.mean() for b in np.array_split(x, N_BLOCKS)])
        return float(m.mean()), float(m.std(ddof=1) / np.sqrt(N_BLOCKS))

    dense, dense_e = blocked(dense_t)
    dilute, dilute_e = blocked(dilute_t)
    h = len(dilute_t) // 2
    drift = float(dilute_t[h:].mean() - dilute_t[:h].mean())

    print(f'  rho_dense  : {dense:8.2f} +/- {dense_e:.2f} mg/mL')
    print(f'  rho_dilute : {dilute:8.3f} +/- {dilute_e:.3f} mg/mL  (c_sat)')
    print(f'  dilute drift over production : {drift:+.3f} mg/mL')
    if dense > 0 and dilute / dense > 0.25:
        print('  NOTE: dilute/dense > 0.25 -- phases poorly separated; '
              'may be at or above T_c')

    sysname = (f'Tau5_hps{sfx}_rep_T{int(round(temperature))}'
               f'_Z{int(round(Lz))}')
    outdir = os.path.join(out_root, sysname)
    datadir = os.path.join(outdir, 'data')
    os.makedirs(datadir, exist_ok=True)

    np.save(os.path.join(datadir, f'{sysname}_profile.npy'), prof)
    np.save(os.path.join(datadir, f'{sysname}_dilute_array.npy'), dilute_t)
    np.savetxt(os.path.join(outdir, 'profile.csv'),
               np.column_stack([zc, prod.mean(0), prod.std(0)]),
               delimiter=',', header='z_nm,rho_mg_per_mL,sd_mg_per_mL', comments='')

    summary = {
        'temperature_K': float(temperature),
        'variant': variant or 'WT',
        'sequence': meta.get('sequence'),
        'substitutions': meta.get('substitutions', []),
        'mass_model': 'openabc HPSParser amino-acid masses',
        'chain_mass_amu': float(mass_per_mol[0]),
        'system_mass_amu': float(total_mass),
        'force_field': 'HPS-Urry (mu=1, delta=0.08)',
        'n_frames': int(nf), 'n_frames_production': int(len(prod)),
        'box_x_nm': float(box_all[:, 0].mean()),
        'box_y_nm': float(box_all[:, 1].mean()), 'box_z_nm': Lz,
        'bin_nm': BIN_NM,
        'rho_dense_mg_per_mL': dense, 'rho_dense_err': dense_e,
        'rho_dilute_mg_per_mL': dilute, 'rho_dilute_err': dilute_e,
        'csat_mg_per_mL': dilute, 'dilute_drift_mg_per_mL': drift,
        'mass_recovery_pct': float(100 * recovered / total_mass),
    }
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  wrote {outdir}/')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--temperature', type=float)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--variant', default='')
    a = ap.parse_args()
    sfx = f'_{a.variant}' if a.variant else ''

    if a.all:
        temps = sorted(int(re.search(r'T(\d+)', os.path.basename(d)).group(1))
                       for d in glob.glob(f'run/slab{sfx}/T*')
                       if os.path.exists(f'{d}/traj.dcd'))
        if not temps:
            raise SystemExit(f'no trajectories found under run/slab{sfx}/')
        print(f'variant {a.variant or "WT"}, temperatures found: {temps}')
        rows = [s for T in temps if (s := process(float(T), a.variant))]
    elif a.temperature is not None:
        rows = [s for s in [process(a.temperature, a.variant)] if s]
    else:
        raise SystemExit('pass --temperature T or --all')

    if len(rows) > 1:
        print('\n=== coexistence summary ===')
        print(f'{"T (K)":>7} {"rho_dense":>12} {"c_sat":>12} {"ratio":>8}')
        for s in sorted(rows, key=lambda r: r['temperature_K']):
            r = (s['rho_dilute_mg_per_mL'] / s['rho_dense_mg_per_mL']
                 if s['rho_dense_mg_per_mL'] else float('nan'))
            print(f'{s["temperature_K"]:7.0f} {s["rho_dense_mg_per_mL"]:12.2f} '
                  f'{s["rho_dilute_mg_per_mL"]:12.3f} {r:8.3f}')
        with open(f'results/coexistence{sfx}.json', 'w') as f:
            json.dump(sorted(rows, key=lambda r: r['temperature_K']), f, indent=2)
        print(f'\nwrote results/coexistence{sfx}.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Build HPS inputs for the TAU5* W68A/W104A double mutant.

Reuses the WT compressed configuration as the production seed: both
sequences are 119 residues and the HPS bond is harmonic with the same
r0 for every pair, so WT coordinates are geometrically valid for the
mutant. Identical starting coordinates and box also make the WT/mutant
comparison controlled.

The force field is NOT reusable -- lambda, sigma, mass and charge are
per residue, so system.xml is rebuilt from the mutant sequence.

Writes:
  inputs_W2A/init_TAU5s_W2A_CA.pdb        mutant monomer
  inputs_W2A/start_multi_TAU5s_W2A.pdb    50 chains, WT coordinates, mutant identities
  inputs_W2A/build_meta.json
  run/prep_W2A/system.xml                 mutant force field, no barostat
  run/prep_W2A/state_npt_lowT.xml         copy of the WT compressed state
"""
import json
import os
import shutil
import string

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from openabc.forcefields import HPSModel
from openabc.forcefields.parsers import HPSParser
from openabc.utils import parse_pdb, write_pdb
from openabc.utils.helper_functions import build_straight_CA_chain

WT_SEQ = ('AAGSSGTLELPSTLSLYKSGALDEAAAYQSRDYYNFPLALAGPPPPPPPPHPHARIKLEN'
          'PLDYGSAWAAAAAQCRYGDLASLHGAGAAGPGSGSPSAAASSSWHTLFTAEEGQLYGPC')
SEQ = ('AAGSSGTLELPSTLSLYKSGALDEAAAYQSRDYYNFPLALAGPPPPPPPPHPHARIKLEN'
       'PLDYGSAAAAAAAQCRYGDLASLHGAGAAGPGSGSPSAAASSSAHTLFTAEEGQLYGPC')
VARIANT = 'W2A'
N_MOL, R0 = 50, 0.38

WT_META = 'inputs/build_meta.json'
WT_SEED = 'run/prep/state_npt_lowT.xml'
OUTDIR = f'inputs_{VARIANT}'
RUNDIR = f'run/prep_{VARIANT}'
CA_PDB = f'{OUTDIR}/init_TAU5s_{VARIANT}_CA.pdb'
SYS_PDB = f'{OUTDIR}/start_multi_TAU5s_{VARIANT}.pdb'
META = f'{OUTDIR}/build_meta.json'

O2T = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
       'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
       'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
       'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
T2O = {v: k for k, v in O2T.items()}


def main():
    assert len(SEQ) == len(WT_SEQ) == 119
    subs = [(i + 1, a, b) for i, (a, b) in enumerate(zip(WT_SEQ, SEQ)) if a != b]
    print(f'variant {VARIANT}: ' + ', '.join(f'{a}{i}{b}' for i, a, b in subs))

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(RUNDIR, exist_ok=True)

    print(f'\nbuilding mutant monomer -> {CA_PDB}')
    write_pdb(build_straight_CA_chain(SEQ, r0=R0), CA_PDB, write_TER=True)

    # Relabel the WT 50-chain PDB with mutant residue identities. Only the
    # residue names change; coordinates are untouched.
    with open(WT_META) as f:
        wt_meta = json.load(f)
    df = parse_pdb(wt_meta['system_pdb'])
    if len(df) != N_MOL * len(SEQ):
        raise SystemExit(f'{wt_meta["system_pdb"]}: {len(df)} atoms, '
                         f'expected {N_MOL * len(SEQ)}')
    wt_read = ''.join(T2O.get(r, 'X') for r in df['resname'][:len(SEQ)])
    if wt_read != WT_SEQ:
        raise SystemExit('WT PDB sequence does not match WT_SEQ')

    ids = list(string.ascii_uppercase + string.ascii_lowercase)
    df = df.copy()
    df['resname'] = [O2T[c] for c in SEQ] * N_MOL
    df['chainID'] = [ids[i // len(SEQ)] for i in range(len(df))]
    df['resSeq'] = np.arange(len(df)) % len(SEQ) + 1
    df['serial'] = np.arange(len(df)) + 1
    write_pdb(df, SYS_PDB, write_TER=True)
    print(f'wrote {SYS_PDB} (WT coordinates, mutant identities)')

    top = app.PDBFile(SYS_PDB).getTopology()
    chains = list(top.chains())
    bad = [i for i, c in enumerate(chains)
           if ''.join(T2O.get(r.name, 'X') for r in c.residues()) != SEQ]
    print(f'  chains {len(chains)}, atoms {top.getNumAtoms()}, '
          f'sequence matches {len(chains) - len(bad)}/{len(chains)}')
    if bad:
        raise SystemExit(f'mismatched chains: {bad[:10]}')

    # Force field must be rebuilt: per-residue lambda, sigma, mass, charge.
    print('\nbuilding mutant system')
    parser = HPSParser(CA_PDB)
    model = HPSModel()
    for _ in range(N_MOL):
        model.append_mol(parser)
    box_l = float(wt_meta['box_l_nm'])
    model.create_system(top, box_a=box_l, box_b=box_l, box_c=box_l)
    model.add_protein_bonds(force_group=1)
    model.add_contacts('Urry', mu=1, delta=0.08, force_group=2)
    model.add_dh_elec(force_group=3)
    model.save_system(f'{RUNDIR}/system.xml')
    print(f'wrote {RUNDIR}/system.xml')

    masses = [model.system.getParticleMass(i).value_in_unit(unit.dalton)
              for i in range(model.system.getNumParticles())]
    chain_mass = float(np.sum(masses[:len(SEQ)]))
    print(f'  chain mass {chain_mass:.1f} amu, system {np.sum(masses):.0f} amu')
    for i in range(model.system.getNumForces()):
        if isinstance(model.system.getForce(i), mm.MonteCarloBarostat):
            raise SystemExit('system.xml has a barostat; slab runs are NVT')

    # Seed state: WT compressed configuration. Positions and box are what
    # 02_slab.py reads; velocities are redrawn at the target temperature.
    shutil.copy(WT_SEED, f'{RUNDIR}/state_npt_lowT.xml')
    with open(f'{RUNDIR}/state_npt_lowT.xml') as f:
        st = mm.XmlSerializer.deserialize(f.read())
    bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    npos = len(st.getPositions())
    print(f'\nseed state <- {WT_SEED}')
    print(f'  {npos} positions, box {np.round(np.diag(bv), 3)} nm')
    if npos != N_MOL * len(SEQ):
        raise SystemExit(f'seed has {npos} positions, expected {N_MOL * len(SEQ)}')

    with open(META, 'w') as f:
        json.dump({'variant': VARIANT, 'sequence': SEQ, 'wt_sequence': WT_SEQ,
                   'substitutions': [f'{a}{i}{b}' for i, a, b in subs],
                   'n_residues': len(SEQ), 'n_mol': N_MOL,
                   'box_l_nm': box_l, 'r0_nm': R0,
                   'ca_pdb': CA_PDB, 'system_pdb': SYS_PDB,
                   'chain_mass_amu': chain_mass,
                   'seed_state_from': WT_SEED,
                   'config_validated': True}, f, indent=2)
    print(f'wrote {META}')
    print('\nOK -- ready for production runs')


if __name__ == '__main__':
    main()

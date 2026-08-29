#!/usr/bin/env python3
"""Build the TAU5* monomer CA chain and a 50-chain cubic starting box.

Force-field independent: this configuration seeds the HPS-Urry pipeline
and nothing here depends on the hydropathy scale.
"""
import json
import os

import numpy as np
import openmm.app as app
import openmm.unit as unit

from openabc.utils.helper_functions import build_straight_CA_chain, write_pdb
from openabc.utils.insert import insert_molecules

# TAU5* sequence
SEQ = ('AAGSSGTLELPSTLSLYKSGALDEAAAYQSRDYYNFPLALAGPPPPPPPPHPHARIKLEN'
       'PLDYGSAWAAAAAQCRYGDLASLHGAGAAGPGSGSPSAAASSSWHTLFTAEEGQLYGPC')

N_MOL = 50
BOX_L = 50.0          # nm; MUST match box_a/b/c in 01_prep.py
R0 = 0.38             # nm, CA-CA spacing of the straight chain
INSERT_RADIUS = 0.5   # nm, min separation enforced during insertion

OUTDIR = 'inputs'
CA_PDB = os.path.join(OUTDIR, 'init_TAU5s_CA.pdb')
SYSTEM_PDB = os.path.join(OUTDIR, 'start_multi_TAU5s.pdb')
META = os.path.join(OUTDIR, 'build_meta.json')

THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def verify(pdb_path):
    """Check chain count, sequence fidelity, and spatial extent."""
    pdb = app.PDBFile(pdb_path)
    top = pdb.getTopology()
    chains = list(top.chains())
    coord = np.array(pdb.getPositions().value_in_unit(unit.nanometer))

    print(f'  chains            : {len(chains)} (expected {N_MOL})')
    print(f'  atoms             : {top.getNumAtoms()} '
          f'(expected {N_MOL * len(SEQ)})')
    print(f'  residues/chain    : {sorted({len(list(c.residues())) for c in chains})}')

    bad = []
    for i, c in enumerate(chains):
        s = ''.join(THREE2ONE.get(r.name, 'X') for r in c.residues())
        if s != SEQ:
            bad.append(i)
    print(f'  sequence matches  : {len(chains) - len(bad)}/{len(chains)}')
    if bad:
        print(f'  MISMATCHED CHAINS : {bad[:10]}')

    lo, hi = coord.min(0), coord.max(0)
    print(f'  coord min (nm)    : {np.round(lo, 2)}')
    print(f'  coord max (nm)     : {np.round(hi, 2)}')
    print(f'  extent (nm)       : {np.round(hi - lo, 2)}')
    if (hi > BOX_L).any() or (lo < 0).any():
        print(f'  WARNING: atoms outside the [0, {BOX_L}] nm box -- '
              f'BOX_L in 01_prep.py must cover this extent')


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print(f'sequence length     : {len(SEQ)}')
    print(f'chains to insert    : {N_MOL}')
    print(f'box length (nm)     : {BOX_L}')

    # Deterministic: same sequence always gives the same straight chain.
    print(f'\nbuilding monomer -> {CA_PDB}')
    ca_atoms = build_straight_CA_chain(SEQ, r0=R0)
    write_pdb(ca_atoms, CA_PDB, write_TER=True)

    # Random insertion: only run if absent, so an scp'd file or an
    # earlier build is never silently overwritten.
    if os.path.exists(SYSTEM_PDB):
        print(f'\n{SYSTEM_PDB} exists -- skipping insertion')
    else:
        print(f'\ninserting {N_MOL} chains -> {SYSTEM_PDB}')
        insert_molecules(CA_PDB, SYSTEM_PDB, N_MOL,
                         radius=INSERT_RADIUS,
                         box=[BOX_L, BOX_L, BOX_L],
                         max_n_attempts=int(4e5))

    print(f'\nverifying {SYSTEM_PDB}')
    verify(SYSTEM_PDB)

    with open(META, 'w') as f:
        json.dump({'sequence': SEQ, 'n_residues': len(SEQ), 'n_mol': N_MOL,
                   'box_l_nm': BOX_L, 'r0_nm': R0,
                   'insert_radius_nm': INSERT_RADIUS,
                   'ca_pdb': CA_PDB, 'system_pdb': SYSTEM_PDB}, f, indent=2)
    print(f'\nwrote {META}')


if __name__ == '__main__':
    main()

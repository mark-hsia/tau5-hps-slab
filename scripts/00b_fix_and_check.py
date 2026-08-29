#!/usr/bin/env python3
"""Assign per-molecule chain IDs to the inserted box, then validate it.

insert_molecules writes every molecule under one chain ID with no TER
records, so OpenMM reads the file as a single 5950-residue chain. The
force field is unaffected (bonds come from the parser), but per-chain
analysis needs real chains. Also checks non-overlap under PBC, which is
the property insert_molecules actually guarantees -- unwrapped
coordinates extending past the box edge are periodic images, not errors.
"""
import json
import string

import numpy as np
import openmm.app as app
import openmm.unit as unit
from scipy.spatial import cKDTree

from openabc.utils import parse_pdb, write_pdb

SEQ = ('AAGSSGTLELPSTLSLYKSGALDEAAAYQSRDYYNFPLALAGPPPPPPPPHPHARIKLEN'
       'PLDYGSAWAAAAAQCRYGDLASLHGAGAAGPGSGSPSAAASSSWHTLFTAEEGQLYGPC')
N_RES = len(SEQ)
N_MOL = 50
BOX_L = 50.0

IN_PDB = 'inputs/start_multi_TAU5s.pdb'
OUT_PDB = 'inputs/start_multi_TAU5s_50chains.pdb'
META = 'inputs/build_meta.json'

THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def relabel():
    df = parse_pdb(IN_PDB)
    print(f'parsed {IN_PDB}: {len(df)} atoms')
    print(f'  columns: {df.columns.tolist()}')
    for col in ('chainID', 'resSeq', 'serial'):
        if col not in df.columns:
            raise SystemExit(f'expected column {col!r} missing')
    if len(df) != N_RES * N_MOL:
        raise SystemExit(f'expected {N_RES * N_MOL} atoms, found {len(df)}')

    ids = list(string.ascii_uppercase + string.ascii_lowercase)
    if N_MOL > len(ids):
        raise SystemExit(f'{N_MOL} molecules exceeds {len(ids)} single-char IDs')

    mol = np.arange(len(df)) // N_RES
    df = df.copy()
    df['chainID'] = [ids[m] for m in mol]
    df['resSeq'] = np.arange(len(df)) % N_RES + 1
    df['serial'] = np.arange(len(df)) + 1

    write_pdb(df, OUT_PDB, write_TER=True)
    print(f'wrote {OUT_PDB} with {N_MOL} chains')
    return mol


def check(mol):
    pdb = app.PDBFile(OUT_PDB)
    top = pdb.getTopology()
    chains = list(top.chains())
    coord = np.array(pdb.getPositions().value_in_unit(unit.nanometer))

    print(f'\n  chains           : {len(chains)} (expected {N_MOL})')
    print(f'  atoms            : {top.getNumAtoms()} (expected {N_RES * N_MOL})')
    print(f'  residues/chain   : {sorted({len(list(c.residues())) for c in chains})}'
          f' (expected [{N_RES}])')

    bad = [i for i, c in enumerate(chains)
           if ''.join(THREE2ONE.get(r.name, 'X') for r in c.residues()) != SEQ]
    print(f'  sequence matches : {len(chains) - len(bad)}/{len(chains)}')
    if bad:
        print(f'  MISMATCHED       : {bad[:10]}')

    lo, hi = coord.min(0), coord.max(0)
    print(f'  extent (nm)      : {np.round(hi - lo, 2)}'
          f'  [a {N_RES}-mer straight chain is ~{round((N_RES - 1) * 0.38, 1)} nm long]')

    # The real test: no inter-molecular contacts under the minimum image
    # convention, in the same box that create_system will declare.
    wrapped = np.mod(coord, BOX_L)
    wrapped[wrapped >= BOX_L] = 0.0
    tree = cKDTree(wrapped, boxsize=BOX_L)
    pairs = tree.query_pairs(r=1.0, output_type='ndarray')
    if len(pairs):
        inter = pairs[mol[pairs[:, 0]] != mol[pairs[:, 1]]]
    else:
        inter = pairs
    print(f'\n  PBC check in a {BOX_L} nm box:')
    print(f'    inter-molecular pairs < 1.0 nm : {len(inter)} (expect 0)')
    if len(inter):
        d = np.linalg.norm(
            (wrapped[inter[:, 0]] - wrapped[inter[:, 1]] + BOX_L / 2) % BOX_L
            - BOX_L / 2, axis=1)
        print(f'    closest inter-molecular pair   : {d.min():.3f} nm')
    return len(chains) == N_MOL and not bad and not len(inter)


def main():
    mol = relabel()
    ok = check(mol)

    try:
        with open(META) as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}
    meta['system_pdb'] = OUT_PDB
    meta['system_pdb_raw'] = IN_PDB
    meta['n_chains'] = N_MOL
    meta['config_validated'] = bool(ok)
    with open(META, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{"PASS" if ok else "FAIL"} -- {OUT_PDB} '
          f'{"is ready for prep" if ok else "needs attention"}')


if __name__ == '__main__':
    main()

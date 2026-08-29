#!/usr/bin/env python3
"""HPS-Urry prep: NVT relaxation, then low-T NPT compression.

Force field: HPS, Urry hydropathy scale, mu=1, delta=0.08, epsilon at the
openabc default (0.8368 kJ/mol), Debye-Huckel at defaults (ldby=1 nm,
dielectric 80, cutoff 3.5 nm). Temperature independent, so this prep runs
once and seeds every slab temperature.

  A  nvt      300 K, 10 fs, gamma=0.01/ps, 100 ns  -- relax random insertion
  B  npt_lowT 150 K, 20 fs, 1 bar,        250 ns  -- compress to a dense cube

There is deliberately no room-temperature NPT stage. This is an implicit
solvent model: the condensate coexists with vacuum, so its equilibrium
pressure is ~0, and a 1 bar barostat at 300 K compresses without bound
until the box falls below twice the nonbonded cutoff. Stage B only works
because 150 K jams the chains into a glass that resists compression. The
slab run equilibrates the condensate density itself at fixed volume.

Never add or remove forces from a System after a Context exists -- each
stage deserializes a fresh System from system.xml.
"""
import argparse
import json
import os
import sys
import time as walltime

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from openabc.forcefields import HPSModel
from openabc.forcefields.parsers import HPSParser

META = 'inputs/build_meta.json'
CA_PDB = 'inputs/init_TAU5s_CA.pdb'
RUN = 'run/prep'
SYSTEM_XML = f'{RUN}/system.xml'     # barostat-free; consumed by 02_slab.py
PROGRESS = f'{RUN}/progress.json'

CHUNK = 100_000
REPORT = 100_000
DCD_EVERY = 1_000_000
PRESSURE = 1.0 * unit.bar
SEED = 20260814

# name, T (K), dt (fs), friction (1/ps), steps, barostat
STAGES = [
    ('nvt',      300.0, 10.0, 0.01, 10_000_000, False),
    ('npt_lowT', 150.0, 20.0, 0.50, 12_500_000, True),
]


def load_progress():
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return json.load(f)
    return {'stage': 0, 'steps_done': 0}


def save_progress(p):
    tmp = PROGRESS + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, PROGRESS)


def build_model():
    with open(META) as f:
        meta = json.load(f)
    box_l = float(meta['box_l_nm'])
    n_mol = int(meta['n_mol'])
    system_pdb = meta['system_pdb']
    if not meta.get('config_validated'):
        print('WARNING: build_meta reports the configuration was not validated')

    print(f'configuration : {system_pdb}')
    print(f'box (nm)      : {box_l} cubic')
    print(f'molecules     : {n_mol}')

    parser = HPSParser(CA_PDB)
    model = HPSModel()
    for _ in range(n_mol):
        model.append_mol(parser)

    pdb = app.PDBFile(system_pdb)
    top = pdb.getTopology()
    model.create_system(top, box_a=box_l, box_b=box_l, box_c=box_l)
    model.add_protein_bonds(force_group=1)
    model.add_contacts('Urry', mu=1, delta=0.08, force_group=2)
    model.add_dh_elec(force_group=3)

    os.makedirs(RUN, exist_ok=True)
    model.save_system(SYSTEM_XML)
    print(f'wrote {SYSTEM_XML} (no barostat)')
    return top, pdb.getPositions()


def fresh_system(barostat, temperature):
    with open(SYSTEM_XML) as f:
        system = mm.XmlSerializer.deserialize(f.read())
    if barostat:
        system.addForce(mm.MonteCarloBarostat(PRESSURE, temperature * unit.kelvin))
    return system


def make_sim(system, top, temperature, dt, friction, seed):
    integ = mm.LangevinMiddleIntegrator(temperature * unit.kelvin,
                                        friction / unit.picosecond,
                                        dt * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    platform = mm.Platform.getPlatformByName('CUDA')
    return app.Simulation(top, system, integ, platform, {'Precision': 'mixed'}), integ


def transfer_state(sim, path, temperature, seed, keep_velocities):
    """Copy box, positions and (optionally) velocities from a state XML.

    Box is rebuilt strictly diagonal so no non-reduced form propagates.
    """
    with open(path) as f:
        st = mm.XmlSerializer.deserialize(f.read())
    bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    off = float(np.abs(bv - np.diag(np.diag(bv))).max())
    if off > 1e-9:
        raise SystemExit(f'{path}: box not diagonal (max off-diagonal {off})')
    lx, ly, lz = (float(bv[i, i]) for i in range(3))
    sim.context.setPeriodicBoxVectors(mm.Vec3(lx, 0, 0) * unit.nanometer,
                                      mm.Vec3(0, ly, 0) * unit.nanometer,
                                      mm.Vec3(0, 0, lz) * unit.nanometer)
    sim.context.setPositions(st.getPositions())
    if keep_velocities:
        sim.context.setVelocities(st.getVelocities())
    else:
        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
    print(f'    box from state: {lx:.3f} x {ly:.3f} x {lz:.3f} nm')


def run(max_hours):
    t0 = walltime.time()
    budget = max_hours * 3600.0
    top, coord = build_model()
    prog = load_progress()
    print(f'\nresuming at stage {prog["stage"]}, {prog["steps_done"]} steps done')

    for idx, (name, temp, dt, fric, total, barostat) in enumerate(STAGES):
        if idx < prog['stage']:
            print(f'\n--- stage {idx} {name}: complete, skipping')
            continue

        state_xml = f'{RUN}/state_{name}.xml'
        partial = f'{RUN}/state_{name}_partial.xml'
        dcd = f'{RUN}/traj_{name}.dcd'
        done = prog['steps_done'] if idx == prog['stage'] else 0

        print(f'\n--- stage {idx} {name}: T={temp} K dt={dt} fs '
              f'gamma={fric}/ps barostat={barostat}')
        print(f'    {done}/{total} steps ({total * dt / 1e6:.0f} ns)')

        system = fresh_system(barostat, temp)
        sim, integ = make_sim(system, top, temp, dt, fric, SEED + idx)

        prev = f'{RUN}/state_{STAGES[idx - 1][0]}.xml' if idx else None
        if done and os.path.exists(partial):
            print(f'    resuming {partial}')
            transfer_state(sim, partial, temp, SEED + idx, keep_velocities=True)
        elif prev and os.path.exists(prev):
            print(f'    from previous stage {prev}')
            transfer_state(sim, prev, temp, SEED + idx, keep_velocities=False)
        else:
            print('    from the inserted configuration; minimizing')
            sim.context.setPositions(coord)
            sim.minimizeEnergy()
            sim.context.setVelocitiesToTemperature(temp * unit.kelvin, SEED)

        if barostat:
            # Bare float, not a Quantity -- setParameter rejects Quantity.
            sim.context.setParameter(mm.MonteCarloBarostat.Temperature(), float(temp))

        sim.context.setTime(0.0)
        sim.reporters.append(app.DCDReporter(dcd, DCD_EVERY, enforcePeriodicBox=True,
                                             append=os.path.exists(dcd) and done > 0))
        sim.reporters.append(app.StateDataReporter(
            sys.stdout, REPORT, step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, totalEnergy=True, temperature=True,
            volume=True, density=True, speed=True))

        while done < total:
            n = int(min(CHUNK, total - done))
            sim.step(n)
            done += n
            if done < total and walltime.time() - t0 > budget:
                sim.saveState(partial)
                save_progress({'stage': idx, 'steps_done': done})
                print(f'\nbudget reached at {done}/{total} steps of {name}')
                print('INCOMPLETE -- resubmit to continue')
                return 0

        sim.saveState(state_xml)
        st = sim.context.getState(getPositions=True)
        bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        print(f'\nstage {name} complete. box (nm):')
        print(np.round(bv, 4))
        save_progress({'stage': idx + 1, 'steps_done': 0})
        if os.path.exists(partial):
            os.remove(partial)
        sim.reporters.clear()
        del st, sim, integ, system

    print('\nPREP COMPLETE')
    print(f'{RUN}/state_npt_lowT.xml feeds 02_slab.py')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-hours', type=float, default=11.0)
    a = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    return run(a.max_hours)


if __name__ == '__main__':
    sys.exit(main())

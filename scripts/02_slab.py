#!/usr/bin/env python3
"""Build a slab from the compressed cube and run NVT at one temperature.

Elongates z while keeping xy from the prep box, so the dense phase becomes
a slab spanning xy with vacuum above and below. Pure NVT -- no barostat.
The condensate and dilute phases find their own densities at fixed volume,
which is what makes the coexistence densities measurable.

Chains are made whole across periodic images before centering. Mean
centering raw unwrapped coordinates (as a naive script does) scatters
molecules across images instead of producing a compact slab.

  python 02_slab.py --temperature 310 --steps 1000000000
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

META = 'inputs/build_meta.json'
PREP = 'run/prep'
SYSTEM_XML = f'{PREP}/system.xml'
SEED_STATE = f'{PREP}/state_npt_lowT.xml'

Z_MULT = 5.0        # z elongation; 10.127 -> ~50.6 nm, matching prior work
DT_FS = 10.0
FRICTION = 0.5      # 1/ps, as in the earlier Urry slab run
CHUNK = 100_000
SAVE_EVERY = 100_000     # aligned with DCD_EVERY: at most one duplicate frame per restart
REPORT = 100_000
DCD_EVERY = 100_000   # 1 ns per frame at 10 fs


def make_whole_and_center(pos, box_l, n_mol, n_res, new_c):
    """Unwrap each chain, wrap its COM into the cell, center the condensate.

    box_l : (3,) prep box lengths
    new_c : z length of the elongated box
    """
    pos = np.asarray(pos, dtype=float)
    out = np.empty_like(pos)
    for m in range(n_mol):
        s = slice(m * n_res, (m + 1) * n_res)
        c = pos[s].copy()
        for i in range(1, len(c)):          # sequential minimum image
            d = c[i] - c[i - 1]
            c[i] = c[i - 1] + (d + box_l / 2.0) % box_l - box_l / 2.0
        out[s] = c - np.floor(c.mean(0) / box_l) * box_l

    target = np.array([box_l[0] / 2.0, box_l[1] / 2.0, new_c / 2.0])
    return out + (target - out.mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--temperature', type=float, required=True)
    ap.add_argument('--steps', type=int, default=1_000_000_000)  # 10 us at 10 fs
    ap.add_argument('--max-hours', type=float, default=46.0)
    ap.add_argument('--variant', default='',
                    help='sequence variant tag, e.g. W2A; empty means WT')
    a = ap.parse_args()

    T = a.temperature
    tag = f'T{int(round(T))}'
    # A variant reads its own inputs/ and prep/ and writes its own slab dir.
    # Run parameters are untouched, so WT and variant differ only in sequence.
    sfx = f'_{a.variant}' if a.variant else ''
    meta_path = f'inputs{sfx}/build_meta.json'
    system_xml = f'run/prep{sfx}/system.xml'
    seed_path = f'run/prep{sfx}/state_npt_lowT.xml'
    outdir = f'run/slab{sfx}/{tag}'
    print(f'variant            : {a.variant or "WT"}')
    os.makedirs(outdir, exist_ok=True)
    progress_path = f'{outdir}/progress.json'
    partial = f'{outdir}/state_partial.xml'
    final_xml = f'{outdir}/state_final.xml'
    final_pdb = f'{outdir}/last_frame.pdb'
    dcd = f'{outdir}/traj.dcd'

    with open(meta_path) as f:
        meta = json.load(f)
    n_mol, n_res = int(meta['n_mol']), int(meta['n_residues'])
    top = app.PDBFile(meta['system_pdb']).getTopology()

    with open(system_xml) as f:
        system = mm.XmlSerializer.deserialize(f.read())
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), mm.MonteCarloBarostat):
            raise SystemExit('system.xml contains a barostat; slab runs are NVT')

    done = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done = int(json.load(f).get('steps_done', 0))

    with open(seed_path) as f:
        seed_state = mm.XmlSerializer.deserialize(f.read())
    bv = seed_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    box_l = np.array([float(bv[i, i]) for i in range(3)])
    new_c = box_l[2] * Z_MULT

    cutoff = 3.5
    if min(box_l[0], box_l[1]) < 2 * cutoff:
        raise SystemExit(f'xy = {box_l[:2]} nm is below twice the {cutoff} nm cutoff')

    print(f'temperature        : {T} K')
    print(f'prep box (nm)      : {np.round(box_l, 3)}')
    print(f'slab box (nm)      : {box_l[0]:.3f} x {box_l[1]:.3f} x {new_c:.3f}'
          f'  (z x{Z_MULT})')
    print(f'total steps        : {a.steps} ({a.steps * DT_FS / 1e6:.0f} ns)'
          f'  already done {done}')

    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_l[0], 0, 0) * unit.nanometer,
        mm.Vec3(0, box_l[1], 0) * unit.nanometer,
        mm.Vec3(0, 0, new_c) * unit.nanometer)

    integ = mm.LangevinMiddleIntegrator(T * unit.kelvin,
                                        FRICTION / unit.picosecond,
                                        DT_FS * unit.femtosecond)
    integ.setRandomNumberSeed(int(round(T)) * 1000 + 7)
    sim = app.Simulation(top, system, integ,
                         mm.Platform.getPlatformByName('CUDA'),
                         {'Precision': 'mixed'})

    resumable = False
    if done and os.path.exists(partial):
        try:
            with open(partial) as f:
                st = mm.XmlSerializer.deserialize(f.read())
            resumable = True
        except Exception as exc:
            print(f'checkpoint {partial} unreadable ({exc}); restarting stage')
            done = 0
    if resumable:
        print(f'resuming from {partial} at {done} steps')
        pbv = st.getPeriodicBoxVectors()
        sim.context.setPeriodicBoxVectors(*pbv)
        sim.context.setPositions(st.getPositions())
        sim.context.setVelocities(st.getVelocities())
    else:
        pos = np.array(seed_state.getPositions().value_in_unit(unit.nanometer))
        coord = make_whole_and_center(pos, box_l, n_mol, n_res, new_c)
        span = coord.max(0) - coord.min(0)
        print(f'condensate span    : {np.round(span, 2)} nm')
        print(f'z occupancy        : {span[2] / new_c * 100:.0f}% of the box')
        sim.context.setPeriodicBoxVectors(
            mm.Vec3(box_l[0], 0, 0) * unit.nanometer,
            mm.Vec3(0, box_l[1], 0) * unit.nanometer,
            mm.Vec3(0, 0, new_c) * unit.nanometer)
        sim.context.setPositions(coord * unit.nanometer)
        print('minimizing')
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(T * unit.kelvin,
                                               int(round(T)) * 1000 + 11)

    sim.context.setTime(0.0)
    sim.reporters.append(app.DCDReporter(dcd, DCD_EVERY, enforcePeriodicBox=True,
                                         append=os.path.exists(dcd) and done > 0))
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, REPORT, step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, totalEnergy=True, temperature=True, speed=True))

    def checkpoint(steps_done):
        """Write state and step counter atomically.

        Preemption can arrive mid-write, so both files go to a temp path
        and are renamed into place -- a truncated XML on disk would make
        the run unresumable.
        """
        sim.saveState(partial + '.tmp')
        os.replace(partial + '.tmp', partial)
        with open(progress_path + '.tmp', 'w') as fh:
            json.dump({'steps_done': steps_done, 'temperature': T}, fh, indent=2)
        os.replace(progress_path + '.tmp', progress_path)

    t0 = walltime.time()
    budget = a.max_hours * 3600.0
    last_save = done
    while done < a.steps:
        n = int(min(CHUNK, a.steps - done))
        sim.step(n)
        done += n
        if done - last_save >= SAVE_EVERY:
            checkpoint(done)
            last_save = done
            print(f'    checkpoint at {done} steps '
                  f'({done * DT_FS / 1e6:.0f} ns)', flush=True)
        if done < a.steps and walltime.time() - t0 > budget:
            checkpoint(done)
            print(f'\nbudget reached at {done}/{a.steps} steps')
            print('INCOMPLETE -- resubmit to continue')
            return 0

    sim.saveState(final_xml)
    with open(progress_path, 'w') as f:
        json.dump({'steps_done': done, 'temperature': T, 'complete': True},
                  f, indent=2)
    st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(final_pdb, 'w') as f:
        app.PDBFile.writeFile(sim.topology, st.getPositions(), f)
    print(f'\nSLAB COMPLETE at {T} K')
    print(f'  {dcd}\n  {final_xml}\n  {final_pdb}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

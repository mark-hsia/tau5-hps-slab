#!/usr/bin/env python
# coding: utf-8


# load packages
import numpy as np
import pandas as pd
import sys
import os
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit

import mdtraj as md

# try:
#     import nglview
# except ImportError:
#     print('Please install nglview to visualize molecules in the jupyter notebooks.')

from openabc.forcefields.parsers import HPSParser
from openabc.forcefields import HPSModel
from openabc.utils.helper_functions import build_straight_CA_chain, write_pdb
from openabc.utils.insert import insert_molecules

# set simulation platform
platform_name = 'CUDA'
properties = {'Precision': 'mixed'}
platform = mm.Platform.getPlatformByName(platform_name)


# Parse a single FUS.

sequence = 'AAGSSGTLELPSTLSLYKSGALDEAAAYQSRDYYNFPLALAGPPPPPPPPHPHARIKLENPLDYGSAWAAAAAQCRYGDLASLHGAGAAGPGSGSPSAAASSSWHTLFTAEEGQLYGPC'
system_pdb = './50mer/start_multi_TAU5s.hps.pdb'
system_xml = './50mer/system.xml'


npt2_checkpoint = './50mer/checkpoint.NPT2.xml'

nvt_slab_checkpoint = './50mer/checkpoint-slab.NVT.xml'
nvt_slab_dcd = './50mer/output_slab_TAU5s.NVT.hps.dcd'
nvt_slab_lastframe_pdb = './50mer/output_slab_TAU5s.NVT.hps.pdb'

top = app.PDBFile(system_pdb).getTopology()
prev_state = npt2_checkpoint
with open(prev_state, 'r') as f:
    state_prev = mm.XmlSerializer.deserialize(f.read())
init_coord = np.array(state_prev.getPositions().value_in_unit(unit.nanometer))
init_box_vectors = state_prev.getPeriodicBoxVectors()

with open(system_xml) as f:
    system = mm.XmlSerializer.deserialize(f.read())

a, b, c = init_box_vectors
c *= 4

init_coord -= np.mean(init_coord, 0)
init_coord += np.array([a.x,b.y,c.z])/2.0
system.setDefaultPeriodicBoxVectors(a, b, c)


# Use the Urry scale optimal parameter ($\mu=1$ and $\Delta=0.08$) and run the simulation. 
timestep = 10*unit.femtosecond
pressure = 1*unit.bar
temperature = 300*unit.kelvin
friction_coeff = 0.5/unit.picosecond 
output_interval = 100000
time = 5e9*unit.femtosecond 

integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)

simulation = app.Simulation(top, system, integrator, platform, properties)
simulation.context.setPositions(init_coord)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)


dcd_reporter = app.DCDReporter(nvt_slab_dcd, output_interval, enforcePeriodicBox=True)
state_data_reporter = app.StateDataReporter(sys.stdout, output_interval, step=True, time=True, potentialEnergy=True, 
                                            kineticEnergy=True, totalEnergy=True, temperature=True, speed=True)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(state_data_reporter)
simulation.reporters.append(app.CheckpointReporter(nvt_slab_checkpoint, 100*output_interval, writeState=True))
simulation.step(time/timestep)

state = simulation.context.getState(True,True,True,True,True,True,True,True)
with open(slab_checkpoint, 'w') as f:
    f.write(mm.XmlSerializer.serialize(state))

box_vec = state.getPeriodicBoxVectors(asNumpy=True)
print('Final box vectors:')
print(box_vec)

# Get the current positions from the simulation context
positions = simulation.context.getState(getPositions=True).getPositions()

# Write to a PDB file
with open(nvt_slab_lastframe_pdb, "w") as f:
    app.PDBFile.writeFile(simulation.topology, positions, f)

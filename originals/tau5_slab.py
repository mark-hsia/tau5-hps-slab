#!/usr/bin/env python3

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

from openabc.forcefields.parsers import HPSParser
from openabc.forcefields import HPSModel
from openabc.utils.helper_functions import build_straight_CA_chain, write_pdb
from openabc.utils.insert import insert_molecules

# set simulation platform
platform_name = 'CUDA'
platform = mm.Platform.getPlatformByName(platform_name)
properties = {'Precision': 'mixed'}

# Parse a single FUS.

sequence = 'AAGSSGTLELPSTLSLYKSGALDEAAAYQSRDYYNFPLALAGPPPPPPPPHPHARIKLENPLDYGSAWAAAAAQCRYGDLASLHGAGAAGPGSGSPSAAASSSWHTLFTAEEGQLYGPC'
ca_pdb = 'init_TAU5s_CA.pdb'
system_pdb = './50mer/start_multi_TAU5s.hps.pdb'
nvt_dcd = './50mer/output_multi_TAU5s.NVT.hps.dcd'
system_xml = '50mer/system.xml'
nvt_checkpoint = './50mer/checkpoint.NVT.xml'
npt_dcd = '50mer/output_multi_TAU5s.NPT.hps.dcd'
npt_checkpoint = './50mer/checkpoint.NPT.xml'
npt2_dcd = '50mer/output_multi_TAU5s.NPT2.hps.dcd'
npt2_checkpoint = './50mer/checkpoint.NPT2.xml'

output_interval = 100000
nvt_dcd_reporter = app.DCDReporter(nvt_dcd, 10*output_interval, enforcePeriodicBox=True)
nvt_state_data_reporter = app.StateDataReporter(sys.stdout, output_interval, step=True, time=True, potentialEnergy=True, 
                                            kineticEnergy=True, totalEnergy=True, temperature=True, speed=True)

npt_dcd_reporter = app.DCDReporter(npt_dcd, 10*output_interval, enforcePeriodicBox=True)
npt2_dcd_reporter = app.DCDReporter(npt2_dcd, 10*output_interval, enforcePeriodicBox=True)
npt_state_data_reporter = app.StateDataReporter(sys.stdout, output_interval, step=True, time=True, potentialEnergy=True, 
                                            kineticEnergy=True, totalEnergy=True, temperature=True, speed=True, volume=True)

# Initialize a monomer and get a populated topology object
# This will be used as the base topology for each monomer
# and will be appended multiple times, equivalent to the number of 
# monomers in our multimeric system
protein_parser = HPSParser(ca_pdb)

box_l = 27.5

# Create a box with n_mol copies of our monomer by 
# inserting molecules into the simulation box randomly
# there are 50 total monomers, n_mol
n_mol = 50

system = HPSModel()
for i in range(n_mol):
    system.append_mol(protein_parser)

# Generate a topology from the multimeric CA containing PDB
# and fetch the initial positions
top = app.PDBFile(system_pdb).getTopology()
init_coord = app.PDBFile(system_pdb).getPositions()

# Putting it all together by populating the "protein" class object
# create a system with the topology "top" and cubic box of length
# box_l
system.create_system(top, box_a=box_l, box_b=box_l, box_c=box_l)

# Generate bond graph, we have beads so force_group=1
system.add_protein_bonds(force_group=1)

# HPS-Urry is the model that proves to be effective at producing 
# a healthy condensate
system.add_contacts('Urry', mu=1, delta=0.08, force_group=2)

# Add Debye-Huckel electrostatic interactions.
system.add_dh_elec(force_group=3)

# Save full system, coords, topology to xml
system.save_system(system_xml)

# Set simulation parameters such as temperature
# Note the requirement for units
temperature = 300*unit.kelvin
friction_coeff = 0.01/unit.picosecond # use smaller friction coefficient to accelerate dynamics
timestep = 10*unit.femtosecond
time = 1e5*unit.picosecond

# initialize the LangevinMiddleIntegrator
integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)

# setup the simulation engine for relaxation with the desired 
# topology, system, integrator, selecting GPU as the platform, and 
# platform properties
simulation = app.Simulation(topology=top, system=system.system, integrator=integrator, platform=platform, platformProperties=properties)

# initialize coordinates of each CA 
simulation.context.setPositions(init_coord)

# before anything we minimize the system Energy to avoid out-of-bounds error
# due to high energy
simulation.minimizeEnergy()

# thermalize the system to the desired temperature 
simulation.context.setVelocitiesToTemperature(temperature)

# Add reporters... log output, trajectory output, and checkpoint files
simulation.reporters.append(nvt_dcd_reporter)
simulation.reporters.append(nvt_state_data_reporter)
simulation.reporters.append(app.CheckpointReporter(nvt_checkpoint, 100*output_interval, writeState=True))

# Run simulation to relax system
simulation.step(time/timestep)

# write final state to xml state file, contains all information, pos, vel, topo, box geometry
state = simulation.context.getState(True,True,True,True,True,True,True,True)
with open(nvt_checkpoint, 'w') as f:
    f.write(mm.XmlSerializer.serialize(state))

# Use the Urry scale optimal parameter ($\mu=1$ and $\Delta=0.08$) and run the simulation. 
timestep = 20*unit.femtosecond
pressure = 1*unit.bar
temperature = 150*unit.kelvin
friction_coeff = 0.5/unit.picosecond 
time = 2.5e5*unit.picosecond 

state_data_reporter = app.StateDataReporter(sys.stdout, output_interval, step=True, time=True, potentialEnergy=True, 
                                            kineticEnergy=True, totalEnergy=True, temperature=True, speed=True, volume=True, density=True)

# Add Barostat to switch system to NPT simulation 
system.system.addForce(mm.MonteCarloBarostat(pressure, temperature))
integrator.setTemperature(temperature)
integrator.setFriction(friction_coeff)
integrator.setStepSize(timestep)

# reinitialize system to reset simulation time and preserve prior state
# box size, coordinates and velocities
simulation.context.reinitialize(preserveState=True)
simulation.context.setTime(0.0)

# clear old reporters and initialize two
simulation.reporters.clear()
simulation.reporters.append(npt_dcd_reporter)
simulation.reporters.append(npt_state_data_reporter)

# run NPT simulation at low temp to compress the box 
# efficiently
simulation.step(time/timestep)

# Save final state of NPT at low temp
state = simulation.context.getState(True,True,True,True,True,True,True,True)
with open(npt_checkpoint, 'w') as f:
    f.write(mm.XmlSerializer.serialize(state))

# report the box geometry to the log
box_vec = state.getPeriodicBoxVectors(asNumpy=True)
print('Final box vectors (NPT low Temp):')
print(box_vec)


# set the parameters for the NPT at 300K and 1bar
timestep = 10*unit.femtosecond
pressure = 1*unit.bar
temperature = 300*unit.kelvin
friction_coeff = 0.5/unit.picosecond 
output_interval = 100000
time = 1e7*unit.picosecond # 10 microseconds happens to converge the box size and pressure well

# set the temperature of the barostat
simulation.context.setParameter(mm.MonteCarloBarostat.Temperature(),temperature)

# Update the temperature to 300K
integrator.setTemperature(temperature)

# Update the friction to something more appropriate
integrator.setFriction(friction_coeff)
integrator.setStepSize(timestep)

# reset the time counter but preserve all other information
simulation.context.reinitialize(preserveState=True)
simulation.context.setTime(0.0)

# clear reporters and add the new ones
simulation.reporters.clear()
simulation.reporters.append(npt2_dcd_reporter)
simulation.reporters.append(npt_state_data_reporter)

# Run NPT at 300 K and 1 bar
simulation.step(time/timestep)

# Save final state
state = simulation.context.getState(True,True,True,True,True,True,True,True)
with open(npt2_checkpoint, 'w') as f:
    f.write(mm.XmlSerializer.serialize(state))

# report final box vectors
box_vec = state.getPeriodicBoxVectors(asNumpy=True)
print('Final box vectors (NPT rt):')
print(box_vec)

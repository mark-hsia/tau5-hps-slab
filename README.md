# TAU5* slab simulations, HPS-Urry

Coexistence densities and saturation concentrations for the TAU5* IDR and a
W68A/W104A double mutant, using the HPS model with the Urry hydropathy scale
(openabc + OpenMM).

## Pipeline

| script | role |
|---|---|
| `scripts/00_build_initial.py` | straight-CA monomer, 50-chain insertion into a 50 nm cube |
| `scripts/00b_fix_and_check.py` | assign per-molecule chain IDs, validate sequence and PBC non-overlap |
| `scripts/01_prep.py` | stage A: 100 ns NVT at 300 K; stage B: 250 ns NPT at 150 K, 1 bar |
| `scripts/02_slab.py` | elongate z 5x, run NVT production at one temperature |
| `scripts/03_density.py` | rho(z) with per-frame slab recentring; rho_dense, c_sat |
| `scripts/10_build_mutant.py` | mutant force field, reusing the WT compressed configuration |

`jobs/*.slurm` are the SLURM wrappers. `originals/` holds the pre-existing
scripts this was derived from.

## Force field

HPS, Urry hydropathy scale, `mu=1`, `delta=0.08`, epsilon at the openabc
default 0.8368 kJ/mol; Debye-Huckel with `ldby=1 nm`, dielectric 80, cutoff
3.5 nm. Temperature independent, so one prep seeds every temperature.

## Design notes

**No room-temperature NPT stage.** This is an implicit-solvent model: the
condensate coexists with vacuum, so its equilibrium pressure is ~0. A 1 bar
barostat at 300 K compresses without bound until the box drops below twice the
nonbonded cutoff. Stage B works only because 150 K jams the chains into a
glass that resists compression. The slab run equilibrates the condensate
density itself at fixed volume.

**Never mutate a System after a Context exists.** Adding or removing forces
from a `System` that a `Context` was built from corrupts it, surfacing later
as a spurious "periodic box vector must be parallel to x" error. Each stage
deserializes a fresh `System` from `system.xml`.

**Bead masses are per amino acid, not per element.** The CA-only PDB declares
element C, so `mdtraj`'s `atom.element.mass` returns 12.011 for every bead and
underestimates densities by ~8.4x. `03_density.py` uses openabc's `HPSParser`
mass table.

**The DCD carries no unit cell.** OpenMM's DCD here has no unit-cell block and
the topology PDB has no `CRYST1`, so `mdtraj` reports `unitcell_lengths` as
`None`. These are NVT runs with a rigid box, so the box is read once from the
run's state XML.

**Per-frame recentring.** The slab diffuses along z over 10 us. The centre is
the mass-weighted *circular* mean of chain-COM z, since z is periodic and a
slab straddling the boundary has no meaningful arithmetic mean.

**Mutant seeding.** Both sequences are 119 residues and the HPS bond uses the
same r0 for every pair, so WT coordinates are geometrically valid for the
mutant. Identical starting coordinates, box, seeds and run parameters mean the
two datasets differ only in sequence. `system.xml` is rebuilt, since lambda,
sigma, mass and charge are per residue.

## Results (WT, 50 chains, 10 us per temperature)

| T (K) | rho_dense (mg/mL) | c_sat (mg/mL) | ratio |
|---|---|---|---|
| 280 | 570.4 +/- 0.4 | 1.30 +/- 0.87 | 0.002 |
| 310 | 431.1 +/- 3.2 | 8.57 +/- 2.12 | 0.020 |
| 350 | 240.6 +/- 5.6 | 140.3 +/- 3.8 | 0.583 |

Critical scaling with beta = 0.325 fixed gives T_c ~ 352 K, rho_c ~ 181 mg/mL.

**Caveat on c_sat.** The dilute analysis region is 1039 nm^3, so one chain
registers as 19.3 mg/mL. Mean occupancy is 0.07 chains at 280 K, 0.44 at
310 K, 7.3 at 350 K. Only the 350 K c_sat is quantitative; at 280 K three of
four production quarters contain exactly zero chains, so that value is an
upper bound. Dilute sampling scales with chain count, not trajectory length.

## Environment

`openmm 8.3.1`, `cuda-version 12.6`, `openabc 1.0.9`, `mdtraj`, `python 3.11`,
all conda-forge. Simulation data is not tracked: trajectories live on scratch
and are regenerable from these scripts.

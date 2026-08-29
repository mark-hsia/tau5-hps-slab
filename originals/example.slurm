#!/bin/bash
#SBATCH --account=robustelli 
#SBATCH --partition=standard
#SBATCH --mail-user=youremailhere@whatever.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000mb
#SBATCH --time=4-00:00:00 
#SBATCH --gpus=1 
# SBATCH --nodelist=amp02 # use this to specify a node you want to use but only do this as a last resort we have access to other nodes not in the group and using preempt_robust for partition will kick others off, even members of our group


### Here -n openmm selects 'your' openmm conda environment, assuming that is what you named it
### tau5_slab.py generates a starting condensed phase of the multimeric simulation which will be used 
### by the last piece the slab_simulation.py
conda run -n openmm python tau5_slab.py

### we need to wait for all prior conda threads to close by issuing the 
### wait command 
wait

### Same thing again but for the slab simulation which takes the final configuration from the condensed phase 
### glob of protein in a cubic box and transforms the box into a slab geometry by extending the z-axis
### it is up to you to decide the multiple the z-axis should be relative to the x/y axes--to small and a single monomer
### interact with both sides of the slab, too large and you will be in the dilute phase region and the stability
### of the condensate will vanish. 
conda run -n openmm python slab_simulation.py

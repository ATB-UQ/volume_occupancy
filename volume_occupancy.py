import numpy as np
import pathlib
import MDAnalysis as mda
import math

from scipy.spatial.distance import cdist
from MDAnalysis.tests.datafiles import GRO, TPR, XTC

def dist(X, Y):
    return cdist(X, Y, metric='euclidean')

# Paths of the gro and xtc files to be analysed
gro_path = "MTRD_ZIT.gro"
xtc_path = "MTRD_ZIT.xtc"

# selection masks for the ligand and binding pocket atoms according to the rules described here: https://docs.mdanalysis.org/stable/documentation_pages/selections.html#selection-commands
ligand_mask = ["resname ZIT"]
pocket_mask = ["bynum 11222:11229" , "bynum 11241:11257" , "bynum 11631:11639" , "bynum 11640:11656" , "bynum 12533:12542" , "bynum 13055:13072" , "bynum 15467:15475" , "bynum 15802:15809" , "bynum 15832:15839" , "bynum 15848:15864" , "bynum 15943:15959"]

distance_threshold = 5.0 # modify this value to change the distance beyond the vdw radii of both atoms at which they will be considered close enough to be bound

universe = mda.Universe(gro_path, xtc_path, in_memory=True) # set to true to keep entire trajectory in memory (faster if there is enough memory to allow it)

# Name of output file
output_file_name = "distal_pocket_new_script.txt"

output_file = open(output_file_name, "w")  

for f in range(universe.trajectory.n_frames):
    frame = universe.trajectory._read_frame(f)
    ligand = universe.select_atoms(*ligand_mask) 
    pocket = universe.select_atoms(*pocket_mask)

    ligand_atom_vdw_radii = np.array([mda.topology.tables.vdwradii[t] for t in ligand.types])
    pocket_atom_vdw_radii = np.array([mda.topology.tables.vdwradii[t] for t in pocket.types])
    ligand_atom_to_pocket_atom_distances = dist(ligand.positions, pocket.positions)
    ligand_atom_to_pocket_atom_distances -= pocket_atom_vdw_radii
    ligand_atom_to_pocket_atom_distances = (ligand_atom_to_pocket_atom_distances.T - ligand_atom_vdw_radii).T 

    ligand_atom_indices_in_pocket = np.empty(ligand.positions.shape[0], dtype=bool)

    for ligand_atom_index, _ in enumerate(ligand.positions):
        if np.min(ligand_atom_to_pocket_atom_distances[ligand_atom_index] > distance_threshold): 
            ligand_atom_indices_in_pocket[ligand_atom_index] = False
        else: 
            ligand_atom_indices_in_pocket[ligand_atom_index] = True

    percentage = round((ligand_atom_indices_in_pocket.sum() / ligand_atom_indices_in_pocket.shape[0]) * 100, 2) 
    print(f"Frame {f}: {percentage}% of ligand atoms are in pocket")
    output_file.write(f"Frame {f}: {percentage}% of ligand atoms are in pocket\n")

output_file.close()

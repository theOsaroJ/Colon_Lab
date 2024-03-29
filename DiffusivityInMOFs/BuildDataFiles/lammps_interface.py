#!/usr/bin/env python
#coding: utf-8

#import lammps_interface
from lammps_interface.lammps_main import LammpsSimulation
from lammps_interface.structure_data import from_CIF, write_CIF
class Parameters:
    def __init__(self, cif):
        # File options
        self.cif_file = cif
        self.output_cif = False
        self.output_raspa = False

        # Force field options
        self.force_field = 'UFF'
        self.mol_ff = None
        self.h_bonding = False
        self.dreid_bond_type = 'harmonic'
        self.fix_metal = False
        
        # Simulation options
        self.minimize = False
        self.bulk_moduli = False
        self.thermal_scaling = False
        self.npt = False
        self.nvt = False
        self.cutoff = 12.5
        self.replication = None
        self.orthogonalize = False
        self.random_vel = False
        self.dump_dcd = 0
        self.dump_xyz = 0
        self.dump_lammpstrj = 0
        self.restart = False
        
        # Parameter options
        self.tol = 0.4
        self.neighbour_size = 5
        self.iter_count = 10
        self.max_dev = 0.01
        self.temp = 298.0
        self.pressure = 1.0
        self.nprodstp = 200000
        self.neqstp = 200000
# Molecule insertion options
        self.insert_molecule = ""
        self.deposit = 0
        
    def show(self):
        for v in vars(self):
            print('%-15s: %s' % (v, getattr(self, v)))
par = Parameters('/afs/crc.nd.edu/user/e/eosaro/MDClass/DiffusivityProject/Lammps_Interface/lammps_interface-master/IRMOF-1.cif')
#par.show()
sim = LammpsSimulation(par)
cell, graph = from_CIF(par.cif_file)
sim.set_cell(cell)
sim.set_graph(graph)
sim.split_graph()
sim.assign_force_fields()
sim.compute_simulation_size()
sim.merge_graphs()
if par.output_cif:
    print("CIF file requested. Exiting...")
    write_CIF(graph, cell)
    sys.exit()

sim.write_lammps_files()

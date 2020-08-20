#!/usr/bin/env python
# coding: utf-8

import numpy as np
import ase
from ase.build import graphene_nanoribbon
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.visualize import view
import matplotlib.pyplot as plt
import datetime
import sys
import argparse

parser = argparse.ArgumentParser(description='Script to generate and oxidize graphene sheet/nanoribbon')
parser.add_argument('-b','--border_type', type=str.lower, choices=['a','z'], help='border type, a/A for armchair or z/Z for zigzag, default: a', default='a')
parser.add_argument('-c','--cell', type=str.lower, help='simulation cell format, ortho or something else (only relevant for zigzag border type), default: ortho', default='ortho')
parser.add_argument('-flk','--flake', type=bool, help='generate flake or infinite structure with PBC, default: False', default=False, choices=[True,False])
parser.add_argument('-sat','--saturated_edge', type=bool, help='If flake is True, choose to saturate edge or not with hydrogen, default: True', default=True, choices=[True,False])
parser.add_argument('-nx','--nx', type=int, help='integer number of repetitions along x direction, default: 5', default=5)
parser.add_argument('-ny','--ny', type=int, help='integer number of repetitions along y direction, default: 5', default=5)
parser.add_argument('-vac','--vacuum', type=float, help='size of vacuum layer in Angstrom, default: 20', default=20)
parser.add_argument('-NN','--allow_NN', type=bool, help='allow NN oxidation. This is respected until reaching an oxidation threshold, then it is set to True, default: False', default=False, choices=[True,False])
parser.add_argument('-v','--verbose', type=bool, help='code verbose, default: True', default=True, choices=[True,False])
parser.add_argument('-show','--show_atoms', type=bool, help='whether or not to show atoms object before and after oxidation using ASE\'s visualization GUI, default: False', default=False, choices=[True,False])
parser.add_argument('-bCC','--bond_CC', type=float, help='C-C bond length, default: 1.42', default=1.42)
parser.add_argument('-bCOH','--bond_COH', type=float, help='C-OH bond length, default: 1.48', default=1.48)
parser.add_argument('-bOH','--bond_OH', type=float, help='O-H bond length, default: 0.98', default=0.98)
parser.add_argument('-bCOC','--bond_COC', type=float, help='height of oxigen from C-C bond, default: 1.22', default=1.22)
parser.add_argument('-OH','--OH', type=float, help='percentage of hydroxyl functions, default: 12.5', default=12.5)
parser.add_argument('-COC','--COC', type=float, help='percentage of epoxy functions, default: 12.5', default=12.5)
parser.add_argument('-ket','--ketone', type=float, help='percentage of ketone functions, default: 0.0', default=0.0)
parser.add_argument('-ketH','--ketone_H', type=float, help='percentage of hydrogenated ketone functions, default: 0.0', default=0.0)

# Read arguments from the command line
args = parser.parse_args()

border_type = args.border_type
cell = args.cell
flake = args.flake
saturated_edge = args.saturated_edge
nx = args.nx
ny = args.ny
vacuum = args.vacuum
allow_NN = args.allow_NN
verbose = args.verbose
show_atoms = args.show_atoms
bond_CC = args.bond_CC
bond_COH = args.bond_COH
bond_OH = args.bond_OH
bond_COC = args.bond_COC

frac_OH = args.OH
frac_COC = args.COC
frac_ket = args.ketone
frac_ketH = args.ketone_H

fraction = {'OH': frac_OH,'COC': frac_COC}

# ### 1. Build graphene sheet

if verbose: print('Creating graphene geometry ...')
print('Setting up graphene')

if border_type == 'z':
    if cell == 'ortho' and not flake:
        gr = graphene_nanoribbon(nx,ny,type='zigzag',saturated=False,C_C=bond_CC,vacuum=vacuum,magnetic=False,sheet=True,main_element='C')
        xyz = np.array(gr.get_positions())
        xyz = xyz[:,[0,2,1]]
        gr = Atoms(gr.symbols,positions=xyz,cell=[[gr.cell[0,0],0.,0.],[0.,gr.cell[2,2],0.],[0.,0.,gr.cell[1,1]]],pbc=[True,True,False])
    elif not flake:
        a = bond_CC * (3 ** .5)
        c = vacuum
        gr_uc = Atoms('C2',
                     scaled_positions=[[0, 0, 1/2],[2/3,1/3,1/2]],
                     cell=[[a,0,0], [- a * .5,a * 3 ** .5 / 2,0], [0,0,c]],
                     pbc=[1, 1, 0])
        gr = gr_uc.repeat((nx,ny,1))
else:
    gr = graphene_nanoribbon(nx,ny,type='armchair',saturated=False,C_C=bond_CC,vacuum=vacuum,magnetic=False,sheet=True,main_element='C')
    xyz = np.array(gr.get_positions())
    xyz = xyz[:,[0,2,1]]
    gr = Atoms(gr.symbols,positions=xyz,cell=[[gr.cell[0,0],0.,0.],[0.,gr.cell[2,2],0.],[0.,0.,gr.cell[1,1]]],pbc=[True,True,False])
natoms = gr.get_global_number_of_atoms()

gr.center(axis=2)

if show_atoms: view(gr)

if verbose:
    print(f'Generated graphene sheet with {natoms} atoms')
    print(f'Cell dimensions are {np.linalg.norm(gr.cell[0]):.2f} x {np.linalg.norm(gr.cell[1]):.2f} x {np.linalg.norm(gr.cell[2]):.2f} ang')


# ### 2. Build first neighbours list for each atom

if verbose: print('Building first neighbours list')

NNlist = neighbor_list('j', a=gr, cutoff=bond_CC+.2, self_interaction=False)

NNlist = NNlist.reshape(natoms,-1).astype(int)

NN_dict = {}
for atom in range(natoms):
    NN_dict.update({atom: NNlist[atom]})


# ### 3. Oxide geometry

if verbose: print('Initiating oxidation')

global_noxygens = 0
function_noxygens = {}
for func in fraction.keys():
    global_noxygens += int(fraction[func] / 100 * natoms)
    function_noxygens.update({func: int(fraction[func] / 100 * natoms)})

gr_ox = gr.copy()
ox_xyz = gr_ox.get_positions()

not_allowed = []

oxy_carbons = []

max_trials_hard = 1e4
max_trials_soft = 1e3

trials = 0

for group in function_noxygens.keys():
    added = 0.
    while added < function_noxygens[group] and trials <= max_trials_hard:
        
        atom = np.random.choice(np.arange(0,natoms))
        
        if atom not in not_allowed and atom not in oxy_carbons:
            
            side = (-1) ** added
            
            if group == 'OH':
                
                ase.build.add_adsorbate(gr_ox,adsorbate='O',height=side * bond_COH, position=ox_xyz[atom][:2])
                ase.build.add_adsorbate(gr_ox,adsorbate='H',height=side * (bond_COH+bond_OH), position=ox_xyz[atom][:2])
                added += 1
                
                oxy_carbons.append(atom)
                not_allowed.append(atom)
                if not allow_NN:
                    for nn in NN_dict[atom]:
                        not_allowed.append(nn)
                trials = 0
                
            elif group == 'COC':
                xyz_first = ox_xyz[atom]
                second_atom = np.random.choice(NN_dict[atom])
                
                if second_atom not in not_allowed and atom not in oxy_carbons:
                    
                    xyz_second = ox_xyz[second_atom]
                    COC_position = ( xyz_first + xyz_second ) / 2
                    ase.build.add_adsorbate(gr_ox,adsorbate='O',height= side * bond_COC, position=COC_position[:2])
                    added += 1
                    
                    oxy_carbons.append(atom)
                    oxy_carbons.append(second_atom)
                    not_allowed.append(atom)
                    not_allowed.append(second_atom)
                    
                    if not allow_NN:
                        for nn in NN_dict[atom]:
                            not_allowed.append(nn)
                        for nn in NN_dict[second_atom]:
                            not_allowed.append(nn)
                        
                    trials = 0
                    
                else: trials +=1
        
        else:
            trials += 1
            if trials == max_trials_soft:
                not_allowed = oxy_carbons
    
    if verbose: print(f'added {added:.0f} functions {group}: {added/natoms * 100:.2f} %')        
    if show_atoms: view(gr_ox)

if verbose: print(f'Final number of atoms: {gr_ox.get_global_number_of_atoms()}')

# ### 4. Write to file

carbons = (gr_ox.get_atomic_numbers() == 6).nonzero()[0]
oxygens = (gr_ox.get_atomic_numbers() == 8).nonzero()[0]
hydrogens = (gr_ox.get_atomic_numbers() == 1).nonzero()[0]

xyz = gr_ox.get_positions()

new_order = np.append(carbons,oxygens)
new_order = np.append(new_order,hydrogens)

new_gr_ox = Atoms(f'C{carbons.shape[0]:.0f}O{oxygens.shape[0]:.0f}H{hydrogens.shape[0]:.0f}',positions=xyz[new_order,:],cell=gr_ox.cell,pbc=[True,True,False])

timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

if verbose: print(f'Writing to file graphene_oxide_{timestamp}.vasp')

kwargs = {'vasp5': True}
new_gr_ox.write(f'graphene_oxide_{timestamp}.vasp',**kwargs)


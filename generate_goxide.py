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

def border_type_from_first(first_letter):
	if first_letter == 'a': return 'armchair'
	elif first_letter == 'z': return 'zigzag'

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
parser.add_argument('-bCOH','--bond_COH', type=float, help='C-OH bond length, default: 1.48', default=1.41)
parser.add_argument('-bOH','--bond_OH', type=float, help='O-H bond length, default: 0.98', default=0.98)
parser.add_argument('-bCOC','--bond_COC', type=float, help='height of oxigen from C-C bond, default: 1.275', default=1.275)
parser.add_argument('-bket','--bond_ketone', type=float, help='height of oxigen from C atom in ketone function, default: 1.22', default=1.22)
parser.add_argument('-OH','--OH', type=float, help='percentage of hydroxyl functions, default: 12.5', default=12.5)
parser.add_argument('-COC','--COC', type=float, help='percentage of epoxy functions, default: 12.5', default=12.5)
parser.add_argument('-ket','--ketone', type=float, help='percentage of ketone functions, default: 0.0', default=0.0)
parser.add_argument('-ketH','--ketone_H', type=float, help='percentage of hydrogenated ketone functions, default: 0.0', default=0.0)

# Read arguments from the command line
args = parser.parse_args()

border_type = border_type_from_first(args.border_type)
flake = args.flake
cell_type = args.cell

if not flake: saturated_edge = False
else: saturated_edge = args.saturated_edge

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
bond_ketone = args.bond_ketone

frac_OH = args.OH
frac_COC = args.COC
frac_ket = args.ketone
frac_ketH = args.ketone_H

fraction = {'OH': frac_OH,'COC': frac_COC, 'ket': frac_ket, 'ket+H': frac_ketH}

# ### 1. Build graphene sheet

if verbose:
	print('Creating graphene geometry ...')
	print('Setting up graphene')

if cell_type == 'ortho':
	gr = graphene_nanoribbon(nx,ny,type=border_type,saturated=False,C_C=bond_CC,vacuum=vacuum,magnetic=False,sheet=True,main_element='C')
	xyz = np.array(gr.get_positions())
	xyz = xyz[:,[0,2,1]]
	gr = Atoms(gr.symbols,positions=xyz,cell=[[gr.cell[0,0],0.,0.],[0.,gr.cell[2,2],0.],[0.,0.,gr.cell[1,1]]],pbc=[True,True,False])
else:
	a = bond_CC * (3 ** .5)
	c = vacuum
	gr_uc = Atoms('C2',
				  scaled_positions=[[0, 0, 1/2],[2/3,1/3,1/2]],
				  cell=[[a,0,0], [- a * .5,a * 3 ** .5 / 2,0], [0,0,c]],
				  pbc=[1, 1, 0])
	
	if flake:
		gr_uc.translate([-bond_CC * (3 ** .5)*0.5,0,0]) # this is done so the edge is nicely terminated
		gr_uc.wrap()
	
	gr = gr_uc.repeat((nx,ny,1))
	
if flake:
	if cell_type == 'ortho':
		a1_vacumm = [vacuum,0,0]
		a2_vacumm = [0,vacuum,0]
		
		gr_uc = graphene_nanoribbon(1,1,type=border_type,saturated=False,C_C=bond_CC,vacuum=vacuum,magnetic=False,sheet=True,main_element='C')
		xyz = np.array(gr_uc.get_positions())
		xyz = xyz[:,[0,2,1]]
		gr_uc = Atoms(gr_uc.symbols,positions=xyz,cell=[[gr_uc.cell[0,0],0.,0.],[0.,gr_uc.cell[2,2],0.],[0.,0.,gr_uc.cell[1,1]]],pbc=[True,True,False])
			
		if border_type == 'armchair':
			gr_uc.translate([bond_CC * (3 ** .5) * 0.5,0,0])
			gr_uc.wrap()
			gr = gr_uc.repeat((nx,ny,1))
			
		
	elif cell_type != 'ortho':
		a1_vacumm = [vacuum,0,0]
		a2_vacumm = [- vacuum * .5,vacuum * 3 ** .5 / 2,0]
		
	cell = gr.cell
	cell[0] += a1_vacumm
	cell[1] += a2_vacumm

	gr.cell = cell
	
	A_pos = np.array(gr.get_positions()[0,:])
		
	gr.center(axis=[0,1])
		
	A_pos_center = np.array(gr.get_positions()[0,:])
	one_to_zero = np.array(gr.get_positions()[0,:]) - np.array(gr.get_positions()[1,:])

	if saturated_edge:
	
		left_sat = gr_uc.copy()
		right_sat = gr_uc.copy()
		up_sat = gr_uc.copy()
		down_sat = gr_uc.copy()

		if cell_type != 'ortho':
			left_sat.pop(0)
			right_sat.pop(1)
			down_sat.pop(0)
			up_sat.pop(1)
			
		elif border_type == 'armchair':
			left_sat.pop(1)
			left_sat.pop(1)
			right_sat.pop(3)
			right_sat.pop(0)
			down_sat.pop(0)
			down_sat.pop(0)
			down_sat.pop(0)
			up_sat.pop(3)
			up_sat.pop(2)
			up_sat.pop(1)
		
		elif border_type == 'zigzag':
			left_sat.pop(1)
			right_sat = left_sat.copy()
			down_sat.pop(1)
			up_sat = down_sat.copy()

		left_sat.set_chemical_symbols(len(left_sat)*['H'])
		right_sat.set_chemical_symbols(len(right_sat)*['H'])
		down_sat.set_chemical_symbols(len(down_sat)*['H'])
		up_sat.set_chemical_symbols(len(up_sat)*['H'])

		left_sat = left_sat * (1,ny,1)
		right_sat = right_sat * (1,ny,1)
		down_sat = down_sat * (nx,1,1)
		up_sat = up_sat * (nx,1,1)

		left_cell = left_sat.cell
		right_cell = right_sat.cell
		down_cell = down_sat.cell
		up_cell = up_sat.cell

		left_cell[0] += a1_vacumm
		left_cell[1] += a2_vacumm
		left_sat.cell = left_cell

		right_cell[0] += a1_vacumm
		right_cell[1] += a2_vacumm
		right_sat.cell = right_cell

		down_cell[0] += a1_vacumm
		down_cell[1] += a2_vacumm
		down_sat.cell = down_cell

		up_cell[0] += a1_vacumm
		up_cell[1] += a2_vacumm
		up_sat.cell = up_cell

		bond_CH = 1.09

		left_sat.translate(A_pos_center-A_pos)
		down_sat.translate(A_pos_center-A_pos)
		
		if cell_type != 'ortho':
			down_sat.translate(one_to_zero)
		elif cell_type == 'ortho' and border_type == 'zigzag':
			down_sat.translate([bond_CC * 0.25,0,0])
		elif cell_type == 'ortho' and border_type == 'armchair':
			down_sat.translate([0,- 2 * bond_CC ,0])
			left_sat.translate([- bond_CC * (3 ** .5) * 0.5 ,0,0])

		if cell_type != 'ortho':
			left_sat.translate([- bond_CH * 3 ** 0.5 * 0.5, -bond_CH * 0.5, 0.])
			down_sat.translate([0,-bond_CH,0])
		else:
			left_sat.translate([-bond_CH,0, 0.])
			down_sat.translate([0,-bond_CH,0])

		right_sat.translate(A_pos_center-A_pos)
		up_sat.translate(A_pos_center-A_pos)

		right_shift = gr.get_positions()[np.argmax(gr.get_positions()[:,0])] - gr.get_positions()[0] 
		up_shift = gr.get_positions()[np.argmax(gr.get_positions()[:,1])] - gr.get_positions()[1]

		right_sat.translate(right_shift)			
		right_sat.translate([bond_CH,0,0])
		
		if cell_type != 'ortho':
			up_sat.translate(-one_to_zero)
		elif cell_type == 'ortho' and border_type == 'zigzag':
			up_sat.translate([0,-one_to_zero[1],0])
			up_sat.translate([bond_CC * 0.25,0,0])
		elif cell_type == 'ortho' and border_type == 'armchair':
			up_sat.translate(-one_to_zero)
			right_sat.translate([bond_CC * (3 ** .5) * 0.5 ,0,0])
			
		up_sat.translate(up_shift)
		up_sat.translate([0,bond_CH,0])

		gr = gr + down_sat + up_sat + left_sat + right_sat

gr.center(axis=2)

natoms = len(gr)
ncarbons = len((gr.get_atomic_numbers() == 6).nonzero()[0])

if show_atoms: view(gr)

if verbose:
    print(f'Generated graphene sheet with {natoms} atoms')
    print(f'Cell dimensions are {np.linalg.norm(gr.cell[0]):.2f} x {np.linalg.norm(gr.cell[1]):.2f} x {np.linalg.norm(gr.cell[2]):.2f} ang')


# ### 2. Build first neighbours list for each atom

if verbose: print('Building first neighbours list')

NNlist_i,NNlist_j = neighbor_list('ij', a=gr, cutoff={('C','C'): bond_CC+.2}, self_interaction=False)

carbon_atoms = np.unique(NNlist_i)

NN_dict = {}
for atom_i in NNlist_i:
    NN_dict.update({atom_i: []})

for atom_i,atom_j in zip(NNlist_i,NNlist_j):
	atom_list = NN_dict[atom_i]
	atom_list.append(atom_j)
	NN_dict.update({atom_i: atom_list})


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
        
        atom = np.random.choice(carbon_atoms)
        
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

            elif group == 'ket':
                ase.build.add_adsorbate(gr_ox,adsorbate='O',height=side * bond_ketone, position=ox_xyz[atom][:2])
                added += 1

                oxy_carbons.append(atom)
                not_allowed.append(atom)
                if not allow_NN:
                    for nn in NN_dict[atom]:
                        not_allowed.append(nn)
                trials = 0

            elif group == 'ket+H':
                ase.build.add_adsorbate(gr_ox,adsorbate='O',height=side * bond_ketone, position=ox_xyz[atom][:2])
                ase.build.add_adsorbate(gr_ox,adsorbate='H',height=side * (bond_ketone+bond_OH), position=ox_xyz[atom][:2])
                added += 1

                oxy_carbons.append(atom)
                not_allowed.append(atom)
                if not allow_NN:
                    for nn in NN_dict[atom]:
                        not_allowed.append(nn)
                trials = 0

        else:
            trials += 1
            if trials == max_trials_soft:
                not_allowed = oxy_carbons
    
    if verbose and added != 0: print(f'added {added:.0f} functions {group}: {added/natoms * 100:.2f} %')        
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


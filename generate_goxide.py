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

border_type = 'a' # z or a (zigzag or armchair)
border_type = border_type.lower()[0]
cell = 'ortho'
flake = False
n,m = 7,10
vacuum = 45
C_C = 1.42

fraction = {'OH': 12.5,'COC': 12.5}
allow_NN = False

verbose=True
show_atoms=False

bond_COH = 1.48
bond_OH = 0.98
bond_COC = 1.22

# ### 1. Build graphene sheet

if verbose: print('Creating graphene geometry ...')
print('Setting up graphene')
print('\n')

if border_type == 'z':
    if cell == 'ortho' and not flake:
        gr = graphene_nanoribbon(n,m,type='zigzag',saturated=False,C_C=C_C,vacuum=vacuum,magnetic=False,sheet=True,main_element='C')
        xyz = np.array(gr.get_positions())
        xyz = xyz[:,[0,2,1]]
        gr = Atoms(gr.symbols,positions=xyz,cell=[[gr.cell[0,0],0.,0.],[0.,gr.cell[2,2],0.],[0.,0.,gr.cell[1,1]]],pbc=[True,True,False])
    elif not flake:
        a = C_C * (3 ** .5)
        c = vacuum
        gr_uc = Atoms('C2',
                     scaled_positions=[[0, 0, 1/2],[2/3,1/3,1/2]],
                     cell=[[a,0,0], [- a * .5,a * 3 ** .5 / 2,0], [0,0,c]],
                     pbc=[1, 1, 0])
        gr = gr_uc.repeat((n,m,1))
else:
    gr = graphene_nanoribbon(n,m,type='armchair',saturated=False,C_C=C_C,vacuum=vacuum,magnetic=False,sheet=True,main_element='C')
    xyz = np.array(gr.get_positions())
    xyz = xyz[:,[0,2,1]]
    gr = Atoms(gr.symbols,positions=xyz,cell=[[gr.cell[0,0],0.,0.],[0.,gr.cell[2,2],0.],[0.,0.,gr.cell[1,1]]],pbc=[True,True,False])
natoms = gr.get_global_number_of_atoms()

if show_atoms: view(gr)

if verbose:
    print(f'Generated graphene sheet with {natoms} atoms')
    print(f'Cell dimensions are {np.linalg.norm(gr.cell[0]):.2f} x {np.linalg.norm(gr.cell[1]):.2f} x {np.linalg.norm(gr.cell[2]):.2f} ang')


# ### 2. Build first neighbours list for each atom

if verbose: print('Building first neighbours list')

NNlist = neighbor_list('j', a=gr, cutoff=C_C+.2, self_interaction=False)

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

new_gr_ox.write(f'graphene_oxide_{timestamp}.vasp')

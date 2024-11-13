from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
import glob
import qcelemental
import h5py
from qcelemental.models.procedures import TorsionDriveResult
import pandas as pd
import pathlib
from openff.toolkit import Molecule
from rdkit.Chem import rdMolTransforms
from openff.nagl.label.dataset import LabelledDataset



outputfile = h5py.File('xBiaryl-SPICE.hdf5', 'w')


def calc_s(delta_ij, c_ij):
    '''given ∆_(ij) returns a vector [cos(∆_(ij)), sin(∆_(ij))], s'''
    # turn into a 9X2 array
    s_ij = np.vstack((np.cos(delta_ij), np.sin(delta_ij))).T 
    
    cs_ij = c_ij[:, np.newaxis] * s_ij
    #print(f'c*s = {s_ij}')
    s = np.sum(cs_ij, axis=0)
    return s

def calc_alpha(s):
    '''given s returns a torsion angle, alpha'''
    r = np.sqrt(s[0]**2 + s[1]**2)
    #alpha = np.arctan2(r*np.cos(s[0]/r), r*np.sin(s[1]/r)) # probably wrong
    alpha = np.arctan2(s[0] / r, s[1] / r)
    
    return alpha
    
def parse_qcjson(filename):
    td_result = TorsionDriveResult.parse_file(filename)
    smiles = td_result.initial_molecule[0].extras["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
    name = smiles.replace("/", "")
    # unused as of yet
    group = outputfile.create_group(name)
    group.create_dataset("smiles", data=[smiles], dtype=h5py.string_dtype())
    group.create_dataset("atomic_numbers", data=td_result.initial_molecule[0].atomic_numbers, dtype=np.int16)
    energies, gradients, conformations = [], [], []
    for angle, molecule in td_result.final_molecules.items():
        energies.append(td_result.final_energies[angle])
        conformations.append(molecule.geometry)
        gradients.append(td_result.optimization_history[angle][0].trajectory[0].properties.return_gradient)
    if "conformations" in group:
        del group["conformations"]  # delete existing dataset if it exists
    conformations = group.create_dataset("conformations", data=np.array(conformations), dtype=np.float32)
    conformations.attrs["units"] = "bohr"
    if "dft total energy" in group:
        del group["dft total energy"]
    ds = group.create_dataset("dft total energy", data=np.array(energies), dtype=np.float64)
    ds.attrs["units"] = "hartree"
    if "dft total gradient" in group:
        del group["dft total gradient"]
    ds = group.create_dataset('dft total gradient', data=np.array(gradients), dtype=np.float32)
    ds.attrs["units"] = "hartree/bohr"
    
    return td_result

def qcmol_to_rdkit(qcmol):
    mol = Molecule.from_qcschema(qcmol)
    rdkit_mol = mol.to_rdkit()
    return rdkit_mol

def gen_mol_dict(td_result):
    mol_dict = {}
    dihedral_idxs = td_result.keywords.dihedrals[0]
    smiles = td_result.initial_molecule[0].extras["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
    for angle, qcmol in td_result.final_molecules.items():
        energy = td_result.final_energies[angle]
        mol = qcmol_to_rdkit(qcmol)
        for i,idx in enumerate(dihedral_idxs):
            mol.SetProp(f'dihedral_{i}', str(idx))
        mol.SetProp('energy', str(energy))
        mol.SetProp('smiles', str(smiles))
        mol_dict[(float(angle)/360) / (2*np.pi)] = Chem.Mol(mol)
    return mol_dict

def get_dihedrals(mol, return_features=True):
        # get list of sets of atom indexes
        # every atom connected to begin atom
        # with every atom connected to end atom
        # [X, begin, end, Y] e.g. ethane has 3*3, so list of lists, len 9
        
        begin_atom = int(mol.GetProp('dihedral_1'))
        end_atom = int(mol.GetProp('dihedral_2'))
        conformer=mol.GetConformer(0)
        #print(f'begin atom: {begin_atom}; end atom: {end_atom}')
    
        # get neighbours of the begin atom
        begin_neighbours = mol.GetAtomWithIdx(begin_atom).GetNeighbors()
        begin_nbr_idxs = [neighbour.GetIdx() for neighbour in begin_neighbours]
        #print("Begin atom neighbors:", begin_nbr_idxs)
        
        # and for end
        end_neighbours = mol.GetAtomWithIdx(end_atom).GetNeighbors()
        end_nbr_idxs = [neighbour.GetIdx() for neighbour in end_neighbours]
        #print("End atom neighbors:", end_nbr_idxs)
        
        # get indices for dihedrals by looping through bond atom neighbours
        # first remove neighbour that is in the bond
        begin_nbr_idxs.remove(end_atom)
        end_nbr_idxs.remove(begin_atom)
    
        dihedral_indices = []
        for nbr_b in begin_nbr_idxs:
            for nbr_e in end_nbr_idxs:
                dihedral = [nbr_b] + [begin_atom, end_atom] + [nbr_e]
                dihedral_indices.append(dihedral)
        dihedral_atoms = [mol.GetAtomWithIdx(idx[0]) for idx in dihedral_indices]

        if return_features:
            # generating features as dummy atomic number
            c = [atm.GetAtomicNum() for atm in dihedral_atoms]

        # calc dihedral angle for each
        dihedral_angles = []
        conf = mol.GetConformer(0)
        for indices in dihedral_indices:
            #print(indices)
            dihedral = rdMolTransforms.GetDihedralRad(conf, *indices)
            dihedral_angles.append(dihedral)
            #print(dihedral)
        #print(dihedral_indices)
        #print(dihedral_angles)
        return np.array(dihedral_angles), np.array(c)



files = glob.glob('biaryl_minimal/result_biaryl_*.json')
dat_dfs = []

for file in files[:3]:
    print(file)
    td_result = parse_qcjson(file)
    td_result.final_energies
    
    mol_dict = gen_mol_dict(td_result)
    for mol in mol_dict.values():
        print(mol.GetProp('energy'))
        
    output_sdf = 'output_conformers.sdf'
    sdf_writer = Chem.SDWriter(output_sdf)
    
    for mol in mol_dict.values():
        for conf_id in range(mol.GetNumConformers()):
            mol.SetProp('conformer', str(conf_id))
            sdf_writer.write(mol)
    
    sdf_writer.close()
    
    
    dihedral_list = []
    alphas = []
    final_energies = []
    
    for angle, mol in mol_dict.items():
        dihedrals, c = get_dihedrals(mol)
        s = calc_s(dihedrals, c)
        alpha = calc_alpha(s)
        energy = float(mol.GetProp('energy'))
        smiles = mol.GetProp('smiles')
        alphas.append(alpha)
        final_energies.append(energy)
    data_dict = {'smiles': smiles, 'alpha': alphas, 'energy': final_energies}
    df = pd.DataFrame.from_dict(data_dict)
    dat_dfs.append(df)
    
final_df = pd.concat(dat_dfs)
pd.DataFrame.to_parquet(final_df, 'e_v_alpha.parquet')


dataset = LabelledDataset("/home/cree/code/nagl_torsion/e_v_alpha")

dataset.to_pandas()

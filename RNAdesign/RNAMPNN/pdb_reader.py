import torch
import Bio.PDB
from Bio.PDB import PDBParser
from constants import *
import warnings
atom_names = {
    'A': ["C4'", "C1'", 'N9', "O4'", 'P'],
    'G': ["C4'", "C1'", 'N9', "O4'", 'P'],
    'U': ["C4'", "C1'", 'N1', "O4'", 'P'],
    'C': ["C4'", "C1'", 'N1', "O4'", 'P'],
}
warnings.filterwarnings('ignore')
def pdbreader(filepath):
    p = PDBParser()
    s = p.get_structure('input',filepath)
    bb_coord = torch.zeros(0,5,3)
    seq = ''
    for model in s:
        for chain in model:
            for residue in chain:
                res_coord = torch.zeros(5,3)
                seq += residue.resname
                for atom in residue:
                    if atom.name in atom_names[residue.resname]:
                        res_coord[atom_names[residue.resname].index(atom.name)] = torch.tensor(atom.coord)
                bb_coord = torch.cat((bb_coord, res_coord.unsqueeze(0)), dim=0)
    return bb_coord,seq

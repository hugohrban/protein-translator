import sys
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser

if __name__ == "__main__":
    filename = sys.argv[1]
    parser = (
        PDBParser(QUIET=True) if filename.endswith(".pdb") else MMCIFParser(QUIET=True)
    )
    structure = parser.get_structure("protein", filename)
    plddt = []
    for atom in structure.get_atoms():
        if atom.get_id() == "CA":
            # pLDDT is stored in the b-factor field of the CA atom
            plddt_value = atom.bfactor
            plddt.append(plddt_value)
    # print(plddt)
    plddt = np.array(plddt) * 100
    print(round(np.mean(plddt), 2), ",")


from Bio.PDB import PDBParser

def list_chains(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    print(f"Structure: {pdb_file}")
    for model in structure:
        print(f"Model {model.id}")
        for chain in model:
            # Count residues
            res_count = len([r for r in chain if r.id[0] == ' '])
            print(f"  Chain {chain.id}: {res_count} residues")

if __name__ == "__main__":
    list_chains("1IVO.pdb")

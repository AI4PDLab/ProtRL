
import Bio.PDB
from Bio.PDB import PDBParser

def get_sequence_manual(chain):
    aa_codes = {
        'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G',
        'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L', 'MET':'M', 'ASN':'N',
        'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T', 'VAL':'V',
        'TRP':'W', 'TYR':'Y'
    }
    seq = []
    res_list = []
    for res in chain:
        if Bio.PDB.is_aa(res):
            code = aa_codes.get(res.get_resname(), 'X')
            seq.append(code)
            res_list.append(res)
    return "".join(seq), res_list

def main():
    parser = PDBParser(QUIET=True)
    # Check 1IVO C
    print("Checking 1IVO Chain C...")
    s1 = parser.get_structure("ref", "1IVO.pdb")
    c1 = s1[0]['C']
    seq1, res1 = get_sequence_manual(c1)
    print(f"Ref Seq: {seq1}")

    # Check modified_color A
    print("Checking modified_color Chain A...")
    s2 = parser.get_structure("mob", "modified_color.pdb")
    c2 = s2[0]['A']
    seq2, res2 = get_sequence_manual(c2)
    print(f"Mob Seq: {seq2}")
    
    # Check PPBuilder on Mob
    ppb = Bio.PDB.PPBuilder()
    pps = ppb.build_peptides(c2)
    print(f"PPBuilder Mob found {len(pps)} polypeptides")
    for pp in pps:
        print(f"  PP Seq: {pp.get_sequence()}")

if __name__ == "__main__":
    main()

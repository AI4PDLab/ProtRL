
import Bio.PDB
from Bio import Align
from Bio.PDB import PDBParser, Superimposer, NeighborSearch
import numpy as np
import argparse
import sys

def get_structure(pdb_file, model_id=0):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("struct", pdb_file)[model_id]

def get_sequence_and_residues(chain):
    """
    Returns (sequence_string, list_of_residues)
    Only considers standard amino acids.
    """
    ppb = Bio.PDB.PPBuilder()
    # PPBuilder returns a list of Polypeptides
    pps = ppb.build_peptides(chain)
    
    # If gaps/hetero residues break the chain, we might get multiple PPs
    # For simplicity, let's just iterate residues manually to ensure strict mapping
    
    aa_codes = {
        'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G',
        'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L', 'MET':'M', 'ASN':'N',
        'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T', 'VAL':'V',
        'TRP':'W', 'TYR':'Y'
    }
    
    seq = []
    res_list = []
    
    for res in chain:
        if Bio.PDB.is_aa(res) and res.get_resname() in aa_codes:
            seq.append(aa_codes[res.get_resname()])
            res_list.append(res)
            
    return "".join(seq), res_list

def align_structures_seq(ref_chain, mob_chain):
    """
    Align sequences and return a mapping of Ref_Residue -> Mob_Residue.
    Uses LOCAL alignment to handle length differences (e.g. domain vs full protein).
    """
    seq_ref, res_ref = get_sequence_and_residues(ref_chain)
    seq_mob, res_mob = get_sequence_and_residues(mob_chain)
    
    print(f"Ref Seq ({len(seq_ref)}): {seq_ref[:20]}...")
    print(f"Mob Seq ({len(seq_mob)}): {seq_mob[:20]}...")

    # Pairwise aligner - GLOBAL mode for similar length
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    
    alignments = aligner.align(seq_ref, seq_mob)
    
    if not alignments:
        print("No alignment found.")
        return {}, [], []
        
    alignment = alignments[0] # Best alignment
    print(f"Alignment Score: {alignment.score}")
    
    # Robust iteration using formatted alignment strings
    # format(alignment) returns a representation. 
    # Let's use the explicit target/query strings if possible, or build them.
    # In recent Bio, alignment can be iterated as columns?
    # Let's use the coordinates to extract aligned segments safely.
    
    # Use robust string iteration
    # alignment[0] is target (Ref) with gaps
    # alignment[1] is query (Mob) with gaps
    
    # In Bio 1.86, alignment[i] might be a sequence object? Convert to str.
    
    # Get the aligned strings properly
    # If alignment is sliceable:
    try:
        seq_str_ref = alignment[0]
        seq_str_mob = alignment[1]
    except:
        # Fallback for some versions
        lines = format(alignment, "fasta").splitlines()
        # >target
        # SEQ...
        # >query
        # SEQ...
        # Just assume alignment object supports array access for components
        seq_str_ref = ""
        seq_str_mob = ""
    
    # Ensure they are strings
    s1 = str(seq_str_ref)
    s2 = str(seq_str_mob)
    
    print(f"Aligned Ref: {s1}")
    print(f"Aligned Mob: {s2}")
    
    # Build mapping
    r_idx = 0
    m_idx = 0
    
    mapping = {}
    ref_aligned = []
    mob_aligned = []
    
    for c1, c2 in zip(s1, s2):
        is_ref_res = (c1 != '-')
        is_mob_res = (c2 != '-')
        
        if is_ref_res and is_mob_res:
            # Match/Mismatch - Map them
            if r_idx < len(res_ref) and m_idx < len(res_mob):
                r_obj = res_ref[r_idx]
                m_obj = res_mob[m_idx]
                mapping[r_obj] = m_obj
                ref_aligned.append(r_obj)
                mob_aligned.append(m_obj)
                
        # Advance counters
        if is_ref_res: r_idx += 1
        if is_mob_res: m_idx += 1
        
    return mapping, ref_aligned, mob_aligned

def calculate_charge(chain):
    # D, E: -1; K, R: +1; H: +0.1
    # Check numbering/residues
    net = 0.0
    aa_charge = {'ASP': -1, 'GLU': -1, 'LYS': 1, 'ARG': 1, 'HIS': 0.1}
    for res in chain:
        if Bio.PDB.is_aa(res):
            net += aa_charge.get(res.get_resname(), 0)
    return net

def find_contacts(residues, distance=3.5):
    # Return set of (resA_idx, resA_name, resB_idx, resB_name) for contacts
    # Optimized: flatten atoms
    atoms = []
    for r in residues:
        atoms.extend(list(r.get_atoms()))
    
    ns = NeighborSearch(atoms)
    contacts = set()
    
    # Filter only heavy polar atoms or just generic contacts?
    # User asked for H-bonds, VdW, Clashes.
    # Let's do generic "Heavy Atom Contacts < 3.5A" as proxy for interactions.
    
    polar = {"N", "O", "S"}
    
    for atom1 in atoms:
        # Optimization: search only near atoms
        # Neighbors
        center = atom1.get_coord()
        neighbors = ns.search(center, distance)
        
        for atom2 in neighbors:
            if atom1 == atom2: continue
            r1 = atom1.get_parent()
            r2 = atom2.get_parent()
            if r1 == r2: continue
            
            # Peptide bond
            if abs(r1.id[1] - r2.id[1]) <= 1: continue
            
            # Sort pair
            pair = tuple(sorted([(r1.id[1], r1.get_resname(), atom1.name), (r2.id[1], r2.get_resname(), atom2.name)]))
            contacts.add(pair)
            
    return contacts

def find_contacts_obj(residues):
    atoms = []
    for r in residues: atoms.extend(r.get_atoms())
    ns = NeighborSearch(atoms)
    c = set()
    for a1 in atoms:
        for a2 in ns.search(a1.get_coord(), 3.5):
            r1, r2 = a1.get_parent(), a2.get_parent()
            if r1==r2 or abs(r1.id[1]-r2.id[1])<=1: continue
            c.add(tuple(sorted((r1, r2), key=lambda r: r.id[1])))
    return c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_pdb")
    parser.add_argument("mob_pdb")
    parser.add_argument("--ref_chain", default="C", help="Chain ID for reference (e.g. C for EGF)")
    parser.add_argument("--mob_chain", default="A", help="Chain ID for mobile (e.g. A)")
    args = parser.parse_args()

    s_ref = get_structure(args.ref_pdb)
    s_mob = get_structure(args.mob_pdb)
    
    # helper
    def get_chain(structure, chain_id):
        if chain_id in structure: return structure[chain_id]
        c = list(structure.get_chains())[0]
        print(f"Chain {chain_id} not found, using {c.id}")
        return c

    c_ref = get_chain(s_ref, args.ref_chain)
    c_mob = get_chain(s_mob, args.mob_chain)

    # 1. Align & Map
    print(f"Aligning {c_ref} and {c_mob}...")
    mapping, aligned_ref_res, aligned_mob_res = align_structures_seq(c_ref, c_mob)
    print(f"Mapped {len(mapping)} residues.")
    
    if len(mapping) < 3:
        print("Error: Too few aligned residues to superimpose.")
        sys.exit(1)
    
    # 2. Superimpose
    sup = Superimposer()
    # Select CA atoms using mapping
    atoms_ref = []
    atoms_mob = []
    
    for r_ref, r_mob in mapping.items():
        if 'CA' in r_ref and 'CA' in r_mob:
            atoms_ref.append(r_ref['CA'])
            atoms_mob.append(r_mob['CA'])

    print(f"Superimposing using {len(atoms_ref)} atoms...")
    sup.set_atoms(atoms_ref, atoms_mob)
    sup.apply(s_mob.get_atoms()) # Move mobile structure
    
    print(f"Superimposed RMSD: {sup.rms:.3f} A")
    
    # 3. Save superimposed
    io = Bio.PDB.PDBIO()
    io.set_structure(s_mob)
    io.save("superimposed_mob.pdb")
    
    # 4. Analysis
    # Charge
    q_ref = calculate_charge(c_ref)
    q_mob = calculate_charge(c_mob)
    print(f"Charge: Ref {q_ref:.1f} -> Mob {q_mob:.1f} (Delta {q_mob - q_ref:.1f})")
    
    print("Calculating contacts...")
    c_obj_ref = find_contacts_obj(c_ref)
    c_obj_mob = find_contacts_obj(c_mob)
    
    # Project Mob contacts to Ref
    rev_map = {v: k for k, v in mapping.items()}
    
    matched_interactions = []
    novel_interactions = []
    
    for m1, m2 in c_obj_mob:
        if m1 in rev_map and m2 in rev_map:
            r1, r2 = rev_map[m1], rev_map[m2]
            # Check if r1-r2 connected in Ref
            pair_ref = tuple(sorted((r1, r2), key=lambda r: r.id[1]))
            if pair_ref in c_obj_ref:
                matched_interactions.append((m1, m2))
            else:
                novel_interactions.append((m1, m2))
        else:
            novel_interactions.append((m1, m2))
            
    print(f"Conserved Interactions: {len(matched_interactions)}")
    print(f"New Interactions: {len(novel_interactions)}")
    
    print("\nTop 10 New Interactions (Internal):")
    # Sort by residue ID
    sorted_novel = sorted(novel_interactions, key=lambda x: x[0].id[1])
    for r1, r2 in sorted_novel[:10]:
        print(f"  {r1.get_resname()}{r1.id[1]} - {r2.get_resname()}{r2.id[1]}")
    if len(sorted_novel) > 10: print("  ...")
        
    # Clashes
    def count_clashes(chain):
        atoms = [a for a in chain.get_atoms()]
        ns = NeighborSearch(atoms)
        clash = 0
        for a1 in atoms:
            for a2 in ns.search(a1.get_coord(), 2.2):
                if a1==a2: continue
                r1, r2 = a1.get_parent(), a2.get_parent()
                if r1==r2 or abs(r1.id[1]-r2.id[1])<=1: continue
                if a1.element=='S' and a2.element=='S' and (a1-a2)>1.9: continue
                clash+=1
        return clash // 2

    cl_ref = count_clashes(c_ref)
    cl_mob = count_clashes(c_mob)
    print(f"Clashes: Ref {cl_ref} vs Mob {cl_mob}")



if __name__ == "__main__":
    main()

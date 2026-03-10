import pandas as pd
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio import PDB
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from pymsaviz import MsaViz

def frequency(align):
    # Initialize with zeros
    amino_acids = ['-']
    # Use the length of the *filtered* alignment
    cols = align.get_alignment_length()
    data = np.zeros((cols, len(amino_acids)))
    df = pd.DataFrame(columns=amino_acids, data=data)

    # Iterate through the alignment and count amino acids at each position
    for a in range(cols):
        column = align[:, a]
        for x in column:
            if x not in df.columns:
                df[x] = 0.0
            df.at[a, x] += 1
    
    # Changes the index in order that the first aa is actually 1
    df.index = df.index + 1
    return df

def main():
    # File paths
    aln_path = "/users/nferruz/fstocco/Desktop/EGFR_analysis/EGFR_finalresults_december/alignment_analysis/lcl_Query_11129174 and 100 other sequences.aln"
    pdb_path = "/users/nferruz/fstocco/Desktop/EGFR_analysis/EGFR_finalresults_december/alignment_analysis/1_3_3_18_373_0_round_33_115a7_unrelaxed_alphafold2_multimer_v3_model_5_seed_001.pdb"
    output_pdb = "modified_color.pdb"

    print(f"Reading alignment from {aln_path}...")
    try:
        align = AlignIO.read(aln_path, "fasta")
    except Exception as e:
        print(f"Error reading alignment: {e}")
        sys.exit(1)

    # Filter alignment to include only columns where the query (first sequence) is not a gap
    query_seq = align[0]
    print(f"Query sequence ID: {query_seq.id}")
    
    # Get indices where query is not '-'
    non_gap_indices = [i for i, char in enumerate(query_seq) if char != '-']
    
    print(f"Filtering alignment from {align.get_alignment_length()} columns to {len(non_gap_indices)} columns (query length).")
    
    # Create new records containing only the relevant columns
    filtered_records = []
    for record in align:
        # Extract chars at non_gap_indices. String join is efficient.
        # record.seq acts like a string in many BioPython versions, but explicit string conversion is safe
        seq_str = str(record.seq)
        new_seq_str = "".join([seq_str[i] for i in non_gap_indices])
        filtered_records.append(SeqRecord(Seq(new_seq_str), id=record.id, description=record.description))
    
    # Create new alignment object
    filtered_align = MultipleSeqAlignment(filtered_records)
    
    # Calculate frequencies on the filtered alignment
    frequency_table = frequency(filtered_align)
    
    # Calculate stats
    most_frequent_amino_acids = []
    identity = []

    for index, row in frequency_table.iterrows():
        most_frequent_amino_acid = row.idxmax()
        most_frequent_count = row.max()
        total_count = row.sum()
        
        if total_count == 0:
            identity_percentage = 0
        elif most_frequent_amino_acid == "-":
            identity_percentage = 0
        else:
            identity_percentage = (most_frequent_count / total_count) * 100
        
        most_frequent_amino_acids.append(most_frequent_amino_acid)
        identity.append(identity_percentage)
        
    frequency_table['MostFrequentAminoAcid'] = most_frequent_amino_acids
    frequency_table['FrequencyPercentage'] = identity

    print(frequency_table.head())

    # Load PDB
    print(f"Reading PDB from {pdb_path}...")
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("my_protein", pdb_path)
    
    # Map to B-factors (Chain A only)
    print("Mapping frequencies to Chain A...")
    i = 0
    mapped_residues = 0
    
    for model in structure:
        for chain in model:
            if chain.id != 'A':
                continue
            
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                
                if i < len(identity):
                    for atom in residue:
                        atom.set_bfactor(identity[i])
                    mapped_residues += 1
                    i += 1
                else:
                    print(f"Warning: Reached end of alignment at residue {residue.id}")

    print(f"Mapped {mapped_residues} residues to Chain A.")

    # Save PDB
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"Saved {output_pdb}")

    # Plot alignment (using the filtered alignment)
    print("Generating MSA plot...")
    try:
        mv = MsaViz(filtered_align, wrap_length=60, show_grid=True, show_consensus=True)
        fig = mv.plotfig()
        plt.savefig("alignment_visualization.png")
        print("Saved alignment_visualization.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
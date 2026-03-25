import os
import math
import random
import argparse
import statistics
from more_itertools import chunked

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import json
import tempfile
import glob
from glob import glob
import csv
from collections import defaultdict

from functions import *

import pdbfixer
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import io

import freesasa

from moleculekit.molecule import Molecule
from moleculekit.writers import PDBwrite

from Bio.PDB import PDBParser
import itertools
from scipy.spatial.distance import cdist

# ---------------------------
# Utility Functions
# ---------------------------
def seed_everything(seed=2003):
    """
    Sets random seed for reproducibility across libraries.
    """
    np.random.seed(seed)
    random.seed(seed)

def append_to_csv(**kwargs):
    
    output_file = kwargs.pop("output_file")
    fieldnames = [  
      "name","pMPNN","d_rmsd","shape_complimentary","uns_hydrogens",
      "hydrophobicity","REU","interface_dSASA",
      "i_plddt", "i_ptm","b_pae","ipsae","helicity","lenght","ptm",
      "t_pae","delta_delta_interaction","binder_sasa", "b_plddt",
      "sequence","i_pae", "lis" ,"d0res", "d0chn",
      "d0dom", "iptm_d0chn", "pdockq2","pdockq", "contact_probs", "iteration_num", "numbers_clusters", "seq_identity"
    ]
    file_exists = os.path.exists(output_file) and os.stat(output_file).st_size > 0

    print(" ➜ About to write:", {k: kwargs[k] for k in fieldnames})

    with open(output_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(kwargs)

def get_interface_atoms(pdbfile, fetch_atoms, cutoff=6.0):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure('complex', pdbfile)

    coords, residues, atom_residue_map, residue_atom_map, atom_number = [], [], {}, {}, []
    
    atom_idx = 0
    for model in structure:
        for residue in structure.get_residues():
          res_no = residue.get_id()[1]
          atom_number = []
          for atom in residue:
              atom_residue_map[atom_idx] = res_no
              if atom.get_name() in "CA":
                  coords.append(atom.get_coord())
                  residues.append((atom_idx, res_no))
              atom_number.append(atom_idx)
              atom_idx += 1
          residue_atom_map[res_no] = atom_number
    coords = np.array(coords)
    print(coords)

    residues = np.array(residues)

    res_nums = residues[:, 1].astype(int)
    chain1_residues = len(set(structure[0]["B"].get_residues()))
    
    chain1_idx = np.where(res_nums <= chain1_residues)[0]
    chain2_idx = np.where(res_nums > chain1_residues)[0]
    print(f"chainres {chain2_idx}")
    # Compute pairwise distances 
    dists = cdist(coords[chain1_idx], coords[chain2_idx])
    print(f"distances {dists}")
    # Identify residues at the interface
    interface_pairs = np.where(dists < cutoff)
    interface_residues = set(res_nums[chain1_idx[interface_pairs[0]]]) | set(res_nums[chain2_idx[interface_pairs[1]]])

    return atom_residue_map, np.array(sorted(interface_residues)), residue_atom_map


def boltz_metrics(name, path, binder_length):
    
    name = name.lower()
    metrics_file = f"confidence_{name}_model_0.json"
    pae_file = f"pae_{name}_model_0.npz"
    plddt_file = f"plddt_{name}_model_0.npz"
    pdb_file = f"{name}_model_0.pdb"
    
    with open(os.path.join(path, metrics_file), "r") as f:
        metrics_summary = json.load(f)
    
    atom_residue_map, interface_residues, residue_atom_map = get_interface_atoms(os.path.join(path,pdb_file), "CA")
    interface_atoms = [residue_atom_map[x] for x in interface_residues.tolist()]

    print(interface_atoms)


    ptm = metrics_summary['chains_ptm']["1"]
    i_ptm = metrics_summary['iptm']
    
    pae = np.load(os.path.join(path, pae_file))
    pae = pae["pae"]
    b_pae = pae[binder_length:, : binder_length].mean()
    t_pae = pae[: binder_length, binder_length:].mean()
    i_pae = np.array([pae[x] for x in interface_residues.tolist()]).mean().item()

    plddt = np.load(os.path.join(path, plddt_file))
    plddt = plddt["plddt"]
    print(len(plddt))
    print(interface_residues)
    t_plddt = np.array(plddt[:binder_length]).mean().item()  
    b_plddt = np.array(plddt[binder_length:]).mean().item()
    i_plddt = np.array([plddt[x] for x in interface_residues.tolist()]).mean().item()

    return i_ptm , pae , b_plddt, i_plddt, ptm, b_pae, t_pae, i_pae

def split_structures(file1, output_file):
    
    mol1 = Molecule(file1)
    mol1.filter(np.where(mol1.chain == 'B')[0])
    
    PDBwrite(mol1, output_file, mode='pdb')
    print(f"Splitted structure saved to {output_file}")

def get_ipsae(pae_path, pdb_path,d_path, arg1=10, arg2=10):
    # run ipsae.py

    print("#####"+pae_path)
    
    command = ["python", f"{d_path}scripts/functions/ipsae.py", pae_path, pdb_path, str(arg1), str(arg2)]
    out = subprocess.run(command, capture_output=True, text=True)
    print(out)
    output = pdb_path.replace(".pdb",f"_{arg1}_{arg2}.txt")
    print(output)
    with open(output) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # header is the second line in your file
    header = lines[0].split()

    # the two asymmetric rows are the 3rd and 2nd lines from the end
    row_ab = dict(zip(header, lines[-3].split()))
    row_ba = dict(zip(header, lines[-2].split()))

    # choose the asymmetric row with the *minimum* ipSAE
    ipSAE_min = min([row_ab, row_ba], key=lambda r: float(r["ipSAE"]))
    ipSAE_max = max([row_ab, row_ba], key=lambda r: float(r["ipSAE"]))

    ipsae_min =  ipSAE_min["ipSAE"]
    ipsae_max = ipSAE_max["ipSAE"]
    iptm_af = ipSAE_max["ipTM_af"]
    lis = ipSAE_max["LIS"]
    d0res = ipSAE_max["d0res"]
    d0chn = ipSAE_max["d0chn"]
    d0dom = ipSAE_max["d0dom"]
    iptm_d0chn = ipSAE_max["ipTM_d0chn"]
    pdockq2 = ipSAE_max["pDockQ2"]
    pdockq = ipSAE_max["pDockQ"]

    return ipsae_min, iptm_af, lis, d0res, d0chn, d0dom, iptm_d0chn, pdockq2, pdockq


def get_chain_indices(chain_ids):

    chain_map = defaultdict(list)
    for i, c in enumerate(chain_ids):
        chain_map[c].append(i)
    return dict(chain_map)

def compute_sasa(pdb_path):

    myOptions = { 'separate-chains': True, 'separate-models': True}
    structureArray = freesasa.structureArray(pdb_path, myOptions)

    for model in structureArray:
        if model.chainLabel(1)=="B":

            result = freesasa.calc(model)

    return float(result.totalArea())




def ratio_contacted_key_residues(name, hotspot_residues, interface_residues_pdb_ids_target_str):

    interface_residues_list = interface_residues_pdb_ids_target_str.replace("A", "").split(",")
    interface_residues_list = [int(x) for x in interface_residues_list if x.strip()]  # include a check for empty strings
    hotspot_residues = [int(x) for x in hotspot_residues]
    common_elements = set(hotspot_residues) & set(interface_residues_list)
    
    contact_prob = len(common_elements) / len(hotspot_residues)
    print(len(common_elements), len(hotspot_residues))
    
    return contact_prob

def get_pMPNN(pdb_file):

    binder_pdb = pdb_file.replace(".pdb","") + "_binder.pdb"
    split_structures(pdb_file, binder_pdb)

    with tempfile.TemporaryDirectory() as output_dir:
       
            command_line_arguments = [
                "python",
                "/gpfs/scratch/crg77/models/ProteinMPNN/protein_mpnn_run.py",
                "--pdb_path", binder_pdb,
                "--score_only", "1",
                "--save_score", "1",
                "--out_folder", output_dir,
                "--batch_size", "1"
            ]

            proc = subprocess.run(command_line_arguments, stdout=subprocess.PIPE, check=True)
            output = proc.stdout.decode('utf-8')
            for x in output.split('\n'):
                if x.startswith('Score for'):
                                name = x.split(',')[0][10:-9]
                                mean =x.split(',')[1].split(':')[1]
    return float(mean)



def run_foldx_(pdb_path):
    old_dir = os.getcwd()
    pdb_dir = os.path.dirname(pdb_path)
    os.chdir(pdb_dir)
    pdb_file = os.path.basename(pdb_path)
    command = ["foldx", "-c", "RepairPDB", f"--pdb={pdb_file}"]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    base_name = pdb_file.replace(".pdb", "")
    pdb_repaired = base_name + "_Repair.pdb"
    command = ["foldx", "-c", "AnalyseComplex", f"--pdb={pdb_repaired}", "--analyseComplexChains=A,B", "--output-file=AC"]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    file_name = "Interface_Residues_AC_AC.fxout"
    with open(file_name, "r") as f:
        data = f.readlines()
    interface = data[-1].split()
    list_mutations = ""
    for residue in interface:
        if residue.isupper():
            list_mutations += residue + "A" + ","
    print(list_mutations[:-1] + ";")
    with open("individual_list.txt", "w") as f:
        f.write(list_mutations[:-1] + ";")
    print(pdb_repaired)
    command = ["foldx", "-c", "BuildModel", f"--pdb={pdb_repaired}", "--mutant-file=individual_list.txt"]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    mutant_pdb = f"{base_name}_Repair_1.pdb"
    command = ["foldx", "-c", "AnalyseComplex", f"--pdb={mutant_pdb}", "--analyseComplexChains=A,B", "--output-file=AC"]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    WT_pdb = f"WT_{base_name}_Repair_1.pdb"
    command = ["foldx", "-c", "AnalyseComplex", f"--pdb={WT_pdb}", "--analyseComplexChains=A,B", "--output-file=AC"]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    summary_file = "Summary_AC_AC.fxout"
    dg_values = []
    with open(summary_file, "r") as f:
        for line in f:
            print(line)
            if line.startswith("./") and ("Repair_1") in line:
                parts = line.split()
                dg_values.append(float(parts[5]))
    os.chdir(old_dir)
    
    return dg_values[0] - dg_values[1] # ddG(mut - wt), positive the better

def get_seq_identity(name, d_path):

    seq_idenity_file = f"{d_path}algn/algn_results.txt"

    df = pd.read_csv(seq_idenity_file, sep="\t", header=None)

    filtered_df = df[df[0] == name]
    
    try:
        return max(filtered_df[2].to_list())
    except:  
        return 0
 

def get_numbers_clusters(d_path):
    path = f"{d_path}algn/clusterRes.txt_rep_seq.fasta"

    with open (path) as f:
        clusters = f.readlines()
    
    return len(clusters)/2

def calc_ss_percentage(pdb_file, path):
    # Parse the structure

    chain_id="B"
    atom_distance_cutoff=4.0

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=f"{path}scripts/functions/dssp")

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues, _ = hotspot_residues(pdb_file, chain_id, atom_distance_cutoff)
    interacting_residues = set(interacting_residues.keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

def split_structures(file1, output_file):
    
    mol1 = Molecule(file1)
    mol1.filter(np.where(mol1.chain == 'B')[0])
    
    PDBwrite(mol1, output_file, mode='pdb')
    print(f"Splitted structure saved to {output_file}")

def py_ros_score_interface(pdb_file, path):
    
    with open(f"{path}scripts/functions/default_4stage_multimer.json", "r") as f:
        advanced_settings = json.load(f)

    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball "{path}scripts/functions/DAlphaBall.gcc" -corrections::beta_nov16 true -relax:default_repeats 1')
    pr_relax(pdb_file, pdb_file)
    print(f"Scoring interface of {pdb_file}")
    interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str = score_interface(pdb_file, binder_chain="B")
    print(interface_scores)
    print(f"Target interface residues: {interface_residues_pdb_ids_target_str}")
    return interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str
 

# ---------------------------
# Dataset Generation
# ---------------------------
def generate_dataset(iteration_num, label, array, d_path, num_arrays):
    data = dict()
    data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        }

    seq_lenght = []
    with open(f"{d_path}seq_gen_{label}_iteration{iteration_num}.fasta", "r") as f:
        rep_seq = f.readlines()

    sequences_rep = {}
    for line in rep_seq:
        if ">" in line:
            name = line.split("\t")[0].replace(">", "").strip()
        else:
            sequences_rep[name] = {"sequence": line.strip()}
    
    
    hotspot_residues = [18,43,44,46,49,50,53,61,62,64,65,66,68,69,70,72,73,75,76,77,80]
    
    numbers_clusters = get_numbers_clusters(d_path)

 
    for entry in sequences_rep:
        try:
                print(entry)
                name = entry
                sequence = sequences_rep[str(name)]['sequence']

                seq_identity = 0

                pdb_path = f"{d_path}boltz_output_iteration{iteration_num}/{name.lower()}"
                pdb_file = pdb_path + f"/{name.lower()}_model_0.pdb"

                i_ptm, pae , b_plddt, i_plddt, ptm, b_pae, t_pae, i_pae = boltz_metrics(name.lower(), pdb_path, len(sequence))
                
                ipsae, iptm_af, lis, d0res, d0chn, d0dom, iptm_d0chn, pdockq2, pdockq = get_ipsae(f"{pdb_path}/pae_{name.lower()}_model_0.npz", pdb_file, d_path)

                binder_sasa = compute_sasa(pdb_file)

                delta_delta_interaction=0
                #delta_delta_interaction = run_foldx_(pdb_file)

                interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str = py_ros_score_interface(pdb_file, d_path)
                
                helicity, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, i_plddt, trajectory_ss_plddt = calc_ss_percentage(pdb_file, d_path)
                                        
                pMPNN = get_pMPNN(pdb_file)
                
                contact_probs = ratio_contacted_key_residues(name, hotspot_residues, interface_residues_pdb_ids_target_str)

                shape_complimentary = interface_scores["interface_sc"]
                uns_hydrogens = interface_scores["interface_delta_unsat_hbonds"]
                hydrophobicity = interface_scores["surface_hydrophobicity"]
                REU = interface_scores["binder_score"]/len(sequence)
                interface_dSASA = interface_scores["interface_dSASA"]

                d_rmsd = target_pdb_rmsd(pdb_file, "/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/templates/nipahvirus_template.pdb", "A")
                lenght = len(sequence)

                output_file = d_path + f"logs_output.csv"
                append_to_csv(
                        name                = name,
                        pMPNN               = -pMPNN,
                        d_rmsd              = d_rmsd.item(),
                        shape_complimentary = shape_complimentary,
                        uns_hydrogens       = uns_hydrogens,
                        hydrophobicity      = hydrophobicity,
                        REU                 = REU,
                        interface_dSASA     = interface_dSASA,   # now guaranteed to go in the right slot
                        i_plddt             = i_plddt,
                        i_ptm               = i_ptm,
                        b_pae               = b_pae,
                        ipsae               = ipsae,
                        helicity            = helicity,
                        lenght              = lenght,
                        ptm                 = ptm,
                        t_pae               = t_pae,
                        delta_delta_interaction = delta_delta_interaction,
                        binder_sasa         = binder_sasa,
                        sequence            = sequence,
                        i_pae               = i_pae,
                        b_plddt             = b_plddt,
                        output_file         = output_file, 
                        lis                 = lis,
                        d0res               = d0res ,
                        d0chn               = d0chn,
                        d0dom               = d0dom,
                        iptm_d0chn          = iptm_d0chn,
                        pdockq2             = pdockq2,
                        pdockq              = pdockq, 
                        contact_probs       = contact_probs, 
                        iteration_num       = iteration_num,
                        numbers_clusters    = numbers_clusters,
                        seq_identity        = seq_identity,
                    )
        except:
            print("ERROR: ",name)
# ---------------------------
#     MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--array", type=int, required=True)
    parser.add_argument("--d_path", type=str, required=False)
    parser.add_argument("--num_arrays", type=int, required=True)



    args = parser.parse_args()
    d_path = args.d_path
    seed_everything(42)

    dataset = generate_dataset(args.iteration_num, args.label.strip(), args.array, args.d_path, args.num_arrays)
   

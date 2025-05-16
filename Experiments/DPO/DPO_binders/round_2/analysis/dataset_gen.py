import os
import math
import random
import argparse
import statistics
from more_itertools import chunked

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict
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



# ---------------------------
# Utility Functions
# ---------------------------
def seed_everything(seed=2003):
    """
    Sets random seed for reproducibility across libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def append_to_csv(name, pMPNN, d_rmsd, pAE_bt, shape_complimentary, uns_hydrogens, hydrophobicity,
                    binder_score, interface_dSASA, iteration_num, plddt, i_pTM,
                    pAE_b, ipsae, helicity, lenght, pae, ptm, has_clash, pAE_t, pae_2,contact_probs, 
                    delta_delta_interaction, binder_sasa, score, sequence, output_file):

    file_exists = os.path.exists(output_file) and os.stat(output_file).st_size > 0
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "name",
            "pMPNN",
            "d_rmsd",
            "pAE_bt",
            "shape_complimentary",
            "uns_hydrogens",
            "hydrophobicity",
            "binder_score",
            "interface_dSASA",
            "iteration_num",
            "plddt",
            "i_pTM",
            "pAE_b",
            "ipsae",
            "helicity",
            "lenght",
            "ptm",
            "has_clash",
            "pAE_t",
            "pae_2",
            "delta_delta_interaction", 
            "binder_sasa",
            "score",
            "sequence",
            "contact_probs",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writerow({
            "name": name,
            "pMPNN": pMPNN,
            "d_rmsd": d_rmsd,
            "pAE_bt": pAE_bt,
            "shape_complimentary": shape_complimentary,
            "uns_hydrogens": uns_hydrogens,
            "hydrophobicity": hydrophobicity,
            "binder_score": binder_score,
            "interface_dSASA": interface_dSASA,
            "iteration_num": iteration_num,
            "plddt": plddt,
            "i_pTM": i_pTM,
            "pAE_b": pAE_b,
            "ipsae": ipsae,
            "helicity": helicity,
            "lenght": lenght,
            "ptm": ptm,
            "has_clash": has_clash,
            "pAE_t": pAE_t,
            "pae_2": pae_2,
            "contact_probs": contact_probs,
            "delta_delta_interaction": delta_delta_interaction,
            "score":score,
            "sequence":sequence,
            "binder_sasa": binder_sasa,
        })


def formatting_sequence(sequence, ec_label):
    """
    Formats correctly the sequence as in the ZymCTRL trainset.
    """
    return f"{ec_label}<sep><start>{sequence}<end><|endoftext|>"

def af_metrics(name, path):
    name = name.lower()
    metrics_file = f"{name}_summary_confidences.json"
   
    with open(os.path.join(path, metrics_file), "r") as f:
        metrics_summary = json.load(f)

    with open(os.path.join(path, metrics_file.replace("summary_","")), "r") as f:
        metrics = json.load(f)

    pae = np.array(metrics["pae"])
    ptm = metrics_summary['ptm']
    iptm = metrics_summary['iptm']
    has_clash = metrics_summary['has_clash']
    
    #ipsae = get_ipsae(path, arg1=10, arg2=10)
    ipsae = 1
    
    chain_ids = metrics["token_chain_ids"]  
    atom_ids = metrics["atom_chain_ids"]
    plddt = metrics['atom_plddts']

    chain_ids_binder = [x for x in chain_ids if x == "B"]
    atom_ids_binder = [x for x in atom_ids if x == "B"]
    
    plddt = np.array(plddt[:len(atom_ids_binder)]).mean()

    pae = np.array(metrics["pae"])
    b_pae = pae[len(chain_ids_binder):, :len(chain_ids_binder)].mean()
    t_pae = pae[:len(chain_ids_binder), len(chain_ids_binder):].mean()

    pae_2 = (b_pae.mean() + t_pae.mean()) / 2

    return iptm, pae , plddt, ptm, has_clash, ipsae, b_pae, t_pae, pae_2

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


def load_alphafold_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_contact_points(sub_probs, chain_ids):
    contact = []
    for i in range(len(chain_ids)):
        contact.append(sum(sub_probs[:,i]))
        
    return np.array(contact)

def ratio_contacted_key_residues(name, hotspot_residues, interface_residues_pdb_ids_target_str):

    interface_residues_list = interface_residues_pdb_ids_target_str.replace("A", "").split(",")
    interface_residues_list = [int(x) for x in interface_residues_list if x.strip()]  # include a check for empty strings
    hotspot_residues = [int(x) for x in hotspot_residues]
    common_elements = set(hotspot_residues) & set(interface_residues_list)
    
    contact_prob = len(common_elements) / len(hotspot_residues)
    print(len(common_elements), len(hotspot_residues))
    
    return contact_prob

def get_pMPNN(pdb_file):
    

    with tempfile.TemporaryDirectory() as output_dir:
       
            command_line_arguments = [
                "python",
                "/home/woody/b114cb/b114cb23/ProteinMPNN/protein_mpnn_run.py",
                "--pdb_path", pdb_file,
                "--pdb_path_chains", "B",
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

def convert_cif_to_pdb(cif_file):
    """
    Converts a CIF file to PDB format and returns the PDB string.
    """
    fixer = PDBFixer(cif_file)

    # Handle missing atoms/residues if needed
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    # Store PDB data in a string buffer
    pdb_file = cif_file.replace("cif","pdb")
    with open(pdb_file, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

def get_ipsae(json_path, arg1=10, arg2=10):
    # run ipsae.py
    command = ["python", "/home/woody/b114cb/b114cb23/ProtWrap/scripts/functions/ipsae.py", json_path, str(arg1), str(arg2)]
    subprocess.run(command)
    output_path=""
    with open(output_path, "r") as f:
        ipsae_data = f.read()
    ipsae = 1 
    #Process data

    return ipsae


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


def calc_ss_percentage(pdb_file):
    # Parse the structure

    chain_id="B"
    atom_distance_cutoff=4.0

    with open("/home/woody/b114cb/b114cb23/ProtWrap/scripts/functions/default_4stage_multimer.json", "r") as f:
        advanced_settings = json.load(f)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

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

def py_ros_score_interface(pdb_file):
    
    with open("/home/woody/b114cb/b114cb23/ProtWrap/scripts/functions/default_4stage_multimer.json", "r") as f:
        advanced_settings = json.load(f)

    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball "/home/woody/b114cb/b114cb23/ProtWrap/scripts/functions/DAlphaBall.gcc" -corrections::beta_nov16 true -relax:default_repeats 1')
    
    print(f"Scoring interface of {pdb_file}")
    interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str = score_interface(pdb_file, binder_chain="B")
    print(f"Target interface residues: {interface_residues_pdb_ids_target_str}")
    return interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str
 

# ---------------------------
# Dataset Generation
# ---------------------------
def generate_dataset(iteration_num, label, mode, array):
    data = dict()
    data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        }

    seq_lenght = []
    with open(f"best1000.fasta", "r") as f:
        rep_seq = f.readlines()

    sequences_rep = {}
    for line in rep_seq:
        if ">" in line:
            name = line.split("\t")[0].replace(">", "").strip()
        else:
            sequences_rep[name] = {"sequence": line.strip()}
    
    
    hotspot_residues = [18,43,44,46,49,50,53,61,62,64,65,66,68,69,70,72,73,75,76,77,80]
    
    arrays  = 20
    items = list(sequences_rep.items())
    size  = math.ceil(len(items) / arrays)
    chunks = [dict(chunk) for chunk in chunked(items, size)]

    print(f"array number {array}")
    print(len(chunks))
    print(chunks[array])
    old = pd.read_csv("logs_output.csv")
    done = set(old["name"].to_list())

    for entry in chunks[array]:
            if entry not in done:
                print(entry)
                name = entry
                sequence = sequences_rep[str(name)]['sequence']
                formatted = formatting_sequence(sequence, label)

                path = f"./alphafold_output/{name.lower()}"
                i_pTM, pae , plddt, ptm, has_clash, ipsae, pAE_b, pAE_t, pae_2 = af_metrics(name.lower(), path)
                
                pAE_bt = (pAE_b + pAE_t)/2

                cif_file = path + f"/{name.lower()}_model.cif"
                convert_cif_to_pdb(cif_file)
                pdb_file = path + f"/{name.lower()}_model.pdb"

                binder_sasa = compute_sasa(pdb_file)
                delta_delta_interaction = run_foldx_(pdb_file)
                interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str = py_ros_score_interface(pdb_file)
                
                print(interface_residues_pdb_ids_target_str)

                helicity, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, i_plddt, trajectory_ss_plddt = calc_ss_percentage(pdb_file)
                
                pMPNN = get_pMPNN(pdb_file)
                contact_probs = ratio_contacted_key_residues(name, hotspot_residues, interface_residues_pdb_ids_target_str)
                print(interface_residues_pdb_ids_target_str)
                print(f"Contact probs: {contact_probs}")
                shape_complimentary = interface_scores["interface_sc"]
                uns_hydrogens = interface_scores["interface_delta_unsat_hbonds"]
                hydrophobicity = interface_scores["surface_hydrophobicity"]
                binder_score = interface_scores["binder_score"]
                interface_dSASA = interface_scores["interface_dSASA"]
                d_rmsd = target_pdb_rmsd(pdb_file, "/home/woody/b114cb/b114cb23/Filippo/DPO_EGFR_/DPO_EGFR_exp/round2/analysis/5wb7.pdb", "A")
                lenght = len(sequence)
                print(f"Sequence: {name}, plddt: {plddt}, i_pTM: {i_pTM}, pAE_b: {pAE_b}, ipsae: {ipsae}, pAE_bt: {pAE_bt}, helicity: {helicity}, pMPNN: {pMPNN}, lenght: {lenght}")

                score = lenght*(plddt + i_pTM - 0.5*pAE_b - 0.5*pAE_bt - pMPNN + 0.03*shape_complimentary + 0.1*hydrophobicity + 0.3*binder_score + 0.003*interface_dSASA - uns_hydrogens + 5*contact_probs + 0.5*d_rmsd + (0.1 * binder_sasa) + delta_delta_interaction)
                output_file = f"logs_output.csv"
                append_to_csv(name, -pMPNN, d_rmsd, pAE_bt, shape_complimentary, uns_hydrogens, hydrophobicity,
                                binder_score, interface_dSASA, iteration_num, plddt, i_pTM,
                                pAE_b, ipsae, helicity, lenght, pae, ptm, has_clash, pAE_t, pae_2,contact_probs, delta_delta_interaction, binder_sasa, score,formatting_sequence(sequence, label),  output_file)
    #except: 
    #    print("out of range")
# ---------------------------
#     MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--array", type=int, required=True)


    args = parser.parse_args()

    seed_everything(42)

    dataset = generate_dataset(args.iteration_num, args.label.strip(), args.mode, args.array)
   

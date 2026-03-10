import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import argparse
import os


##### Load the module ESM ######
tokenizer_esm = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/esm_fold") # Download tokenizer
model_esm = EsmForProteinFolding.from_pretrained("/home/woody/b114cb/b114cb23/esm_fold")  # Download model
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
model_esm = model_esm.to(device)

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int)
parser.add_argument("--label", type=str)
args = parser.parse_args()
iteration_num = args.iteration_num
ec_label = args.label
ec_label = ec_label.strip()
    
    

model_esm.eval()



count = 0
sequences = {}
current_name = None

with open(f"database_{ec_label}.fasta", "r") as f:
    data = f.readlines()
# Process each line
for line in data:
    if line.startswith('>'):  # Check if it's a header line
        current_name = line.strip()  # Save the header as the current sequence name
        sequences[f'{current_name}_{count}'] = ""  # Initialize sequence for this header
        count += 1  # Increment the count for unique keys
    else:
        if current_name is not None:  # Add sequence lines to the current header
            sequences[f'{current_name}_{count - 1}'] += line.strip()

print(sequences)


count = 0
error = 0

for name, sequence in sequences.items():
    count += 1  
    with torch.no_grad():
      output = model_esm.infer_pdb(sequence)
      torch.cuda.empty_cache()
      name = name[1:]
      os.makedirs(f"PDB_{ec_label}", exist_ok=True)
      with open(f"PDB_{ec_label}/{name}.pdb", "w") as f:
            f.write(output)
    torch.cuda.empty_cache()
del model_esm


import json
import torch
import biotite.sequence as biotite_seq
from biotite.sequence.align import SubstitutionMatrix, align_optimal, get_sequence_identity
import numpy as np
import argparse
from Bio.PDB.MMCIFParser import MMCIFParser
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import math
from more_itertools import chunked


def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)

def prepare_af3_json(name, seq, iteration_num, array):

    directory_name = f"alphafold3_input_iteration_{iteration_num}/alphafold3_input_iteration_{iteration_num}_array{array}"
    os.makedirs(directory_name, exist_ok=True)
    with open('json_template_CD80.json', 'r') as file:
      json_template = json.load(file)
    
    json_template["name"] = name
    json_template["sequences"][0]["protein"]["sequence"] = seq
    
    doc_name = f"{directory_name}/{name}.json"

    with open(doc_name, "w") as f:
        json.dump(json_template, f, indent=2)
        
def main(label, model,special_tokens,device,tokenizer):

    
    # Generating sequences
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
        top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=20) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")

    # Compute perplexity for every generated sequence in the batch
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) for output in new_outputs ]

    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) # duplicated sequences?

    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()
    iteration_num = args.iteration_num
    label = args.label

    device = torch.device("cuda")  # Replace with 'cpu' if you don't have a GPU
    print('Reading pretrained model and tokenizer')

    model_name = f"output_iteration{iteration_num}"
    if iteration_num == 0:
        model_name = args.model_dir

    print(f'Model {model_name} has been loaded')

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir) 
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']
    
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Set of canonical amino acids

    all_sequences = []
    for i in range(10):
        sequences = main(label, model, special_tokens, device, tokenizer)
        for key, value in sequences.items():
            for index, val in enumerate(value):
                if all(char in canonical_amino_acids for char in val[0]):
                    name = label.replace(".", "_")
                    seq_info = {
                        'label': label,
                        'batch': i,
                        'index': index,
                        'pepr': float(val[1]),
                        'fasta': f">{name}_{i}_{index}_iteration{iteration_num}\t{val[1]}\n{val[0]}\n",
                        'name': f"{name}_{i}_{index}_iteration{iteration_num}",
                        'seq': val[0]
                    }
                    all_sequences.append(seq_info)
    #all_sequences.sort(key=lambda x: x['pepr'])
    #top_sequences = all_sequences[:20] #get the top 20
    fasta_content = ''.join(seq['fasta'] for seq in all_sequences)
    
    output_filename = f"seq_gen_{label}_iteration{iteration_num}.fasta"
    print(fasta_content)
    with open(output_filename, "w") as fn:
        fn.write(fasta_content)

    arrays  = 20
    total = len(all_sequences)
    chunk_size = math.ceil(total / arrays)

    for part_idx, chunk in enumerate(chunked(all_sequences, chunk_size)):
        for info in chunk:
            prepare_af3_json(info['name'], info['seq'], iteration_num, part_idx)

    
    
import json
import yaml
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


def calculateloglikelihood(input_ids, model, ref_model, tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs_model = model(input_ids, labels=input_ids)
        outputs_ref_model = ref_model(input_ids, labels=input_ids)

    loss, logits = outputs_model[:2]
    ref_loss, logits=outputs_ref_model[:2]
    i_reward = -(loss - ref_loss)
    return i_reward
    

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
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer), calculateloglikelihood(output, model, ref_model, tokenizer)) for output in new_outputs ]
    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) # duplicated sequences?

    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1], x[2]) for x in ppls]
    return sequences

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--d_path", type=str, required=False)
    args = parser.parse_args()

    label = args.label

    device = torch.device("cuda")  # Replace with 'cpu' if you don't have a GPU
    print('Reading pretrained model and tokenizer')

    
    model_name = args.model_dir

    print(f'Model {model_name} has been loaded')

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir) 
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model = GPT2LMHeadModel.from_pretrained(args.model_dir).to(device)

    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']
    
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Set of canonical amino acids

    all_sequences = []
    for i in range(500):
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
                        'fasta': f">{name}_{i}_{index}\t{val[1]}\t{val[2]}\n{val[0]}\n",
                        'name': f"{name}_{i}_{index}",
                        'seq': val[0]
                    }
                    all_sequences.append(seq_info)
    #all_sequences.sort(key=lambda x: x['pepr'])
    #top_sequences = all_sequences[:20] #get the top 20
    fasta_content = ''.join(seq['fasta'] for seq in all_sequences)
    
    output_filename = args.d_path + f"/fasta_out_10k.fasta"
    print(fasta_content)
    with open(output_filename, "w") as fn:
        fn.write(fasta_content)

    
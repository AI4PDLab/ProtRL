
    
import json
import yaml
import torch
import numpy as np
import argparse
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, InfNanRemoveLogitsProcessor

import os
from tqdm import tqdm
import math
from more_itertools import chunked
from peft import LoraConfig, get_peft_model, PeftModel


def remove_characters(seq, char_list):
    "This function removes special tokens used during training."
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids, model, tokenizer):
    """ Compute average perplexity across entire batch """
    # Set pad token to -100 in labels (default value ignored by torch CSE)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    labels = input_ids.clone()  # CLONE!!!
    
    # Mask padding tokens in labels
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)  # Use labels, not input_ids
    
    print(outputs.loss)
    return math.exp(outputs.loss.item())  # Add .item() to convert tensor to Python number

def calculateloglikelihood(input_ids, model, ref_model, tokenizer):
    """ Compute log-likelihood difference between model and reference model """
    # Set pad token to -100 in labels (default value ignored by torch CEL)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    labels = input_ids.clone()  # CLONE!!!
    labels[labels == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)  # Use labels, not input_ids
        out_ref = ref_model(input_ids, labels=labels)  # Use labels, not input_ids
        loss = outputs.loss
        ref_loss = out_ref.loss
    
    return float(-(loss - ref_loss).item())
    

def prepare_bolts_yaml(name, seq, iteration_num, array, d_path):
   
    data = {
        'sequences': [
            {
                'protein': {
                    'id': 'A',
                    'sequence': 'QNYTRSTDNQAVIKDALQGIQQQIKGLADKIGTEIGPKVSLIDTSSTITIPANIGLLGSKISQSTASINENVNEKCKFTLPPLKIHECNISCPNPLPFREYRPQTEGVSNLVGLPNNICLQKTSNQILKPKLISYTLPVVGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCT',
                    'msa': '/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/templates/nipah.a3m',
                }
            },
            {
                'protein': {
                    'id': 'B',
                    'sequence': seq,
                    'msa': 'empty',
                }
            }
        ],
        'templates': [
            {
                'cif': f"/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/templates/epirin_b_template_hit_0_chains_a.cif",
                'chain_id': "B"
            }, 
            {
                'cif': f"/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/templates/epirin_b_template_hit_1_chains_a.cif",
                'chain_id': "B"
            }, 
            {
                'cif': f"/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/templates/epirin_b_template_hit_2_chains_a.cif",
                'chain_id': "B"
            }, 
            {
                'cif': f"/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/templates/epirin_b_template_hit_3_chains_a.cif",
                'chain_id': "B"
            }
        ],
    }

    directory_name = d_path + f"boltz_input_iteration_{iteration_num}/boltz_input_iteration_{iteration_num}_array{array}"
    os.makedirs(directory_name, exist_ok=True)
    filename = f"{directory_name}/{name}.yaml"
    print(filename)
    with open(filename, 'w') as file:
        yaml.dump(data, file, sort_keys=False)

def main(label, model,device,tokenizer):

        # 1. Create a logits processor to handle numerical instability
    logits_processor = LogitsProcessorList([
        InfNanRemoveLogitsProcessor(),
    ])

    # Generating sequences
    input_ids = tokenizer.encode("",return_tensors='pt').to(device)

    outputs = model.generate(
            input_ids.to(device), 
            max_length=150,                
            num_return_sequences=10,
            do_sample=True,                
            top_k=15,                      
            temperature=1,               
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
        
    
    ppls = []
    # Compute perplexity for every generated sequence in the batch
    for output in outputs:
        # decode from CUDA tensor safely
            text = tokenizer.decode(output.tolist(), skip_special_tokens=True)
            text = text.replace(" ","")
            print(text)
            ppl = calculatePerplexity(input_ids, model, tokenizer)
            ll  = calculateloglikelihood(input_ids, model, ref_model, tokenizer)
            ppls.append((text, ppl, ll))

    ppls.sort(key=lambda i: i[1])  # lower perplexity is better
    print(ppls)
    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1], x[2]) for x in ppls]

    return sequences

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--d_path", type=str, required=True)
    args = parser.parse_args()

    iteration_num = args.iteration_num
    label = args.label

    device = torch.device("cuda")  # Replace with 'cpu' if you don't have a GPU
    print('Reading pretrained model and tokenizer')

    
    if args.iteration_num == 0:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    else: 
        model_path = f"{args.d_path}output_iteration{args.iteration_num}"
        print("Loading model and injecting peft layers")
        base = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, model_path, is_trainable=True).to(device)

        print(f'Model {model_path} has been loaded')

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,add_eos_token=False) 
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Set of canonical amino acids

    all_sequences = []
    for i in range(10):
        sequences = main(label, model, device, tokenizer)
        for key, value in sequences.items():
            for index, val in enumerate(value):
                if all(char in canonical_amino_acids for char in val[0]):
                    name = label.replace(".", "_")
                    seq_info = {
                        'label': label,
                        'batch': i,
                        'index': index,
                        'pepr': float(val[1]),
                        'fasta': f">{name}_{i}_{index}_iteration{iteration_num}\t{val[1]}\t{val[2]}\n{val[0]}\n",
                        'name': f"{name}_{i}_{index}_iteration{iteration_num}",
                        'seq': val[0]
                    }
                    all_sequences.append(seq_info)
    #all_sequences.sort(key=lambda x: x['pepr'])
    #top_sequences = all_sequences[:20] #get the top 20
    fasta_content = ''.join(seq['fasta'] for seq in all_sequences)
    
    output_filename = args.d_path + f"seq_gen_{label}_iteration{iteration_num}.fasta"
    print(fasta_content)
    with open(output_filename, "w") as fn:
        fn.write(fasta_content)

    
    for info in all_sequences:
            prepare_bolts_yaml(info['name'], info['seq'], iteration_num, 0, args.d_path)

    
 
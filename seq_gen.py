import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import math
import os
from tqdm import tqdm

def calculate_perplexity(input_ids, model):
    """
    Computes perplexity for the generated sequences.
    Perplexity = exp(loss)
    """
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return math.exp(loss)

def generate_sequences(label, model, tokenizer, device, num_sequences=20, max_aa_length=100):
    """
    Generates sequences using the model.

    ProtGPT3 conditions generation on a leading DIRECTION token:
        "1" = forward  (N->C, left-to-right)
        "2" = reverse  (C->N, right-to-left)
    and it was trained with a BOS token in front of that. The tokenizer is loaded with
    add_bos_token=True (see main), so encode(label) yields [BOS, <direction token>] - exactly
    the in-distribution prompt. Generating without BOS / without a direction token drives the
    model out-of-distribution and produces degenerate, repetitive sequences (e.g. "MMMKKK...").

    max_aa_length caps the number of amino acids; the tokenizer is character-level (one token
    per residue, no space tokens), so the token budget equals the amino-acid budget.

    Returns the generated ids and the prompt length so callers can strip the prompt
    ([BOS] + direction token) and keep only the generated protein.
    """
    input_ids = tokenizer.encode(label, return_tensors='pt').to(device)
    prompt_len = input_ids.shape[1]

    outputs = model.generate(
        input_ids,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
        max_new_tokens=max_aa_length,
        do_sample=True,
        num_return_sequences=num_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    return outputs, prompt_len

def main():
    parser = argparse.ArgumentParser(description="Generate sequences using a pretrained LLM.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument("--label", type=str, required=True, help="Prompt/Label for generation.")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of sequences to generate.")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum amino-acid length of generated sequences.")
    parser.add_argument("--iteration_num", type=int, default=0, help="Iteration number for model selection and output filename.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save results and look for models.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine model path based on iteration number
    iteration_num = args.iteration_num
    if iteration_num <= 1:
        model_name = args.model_dir
    else:
        # Check output_dir for the PREVIOUS iteration's model
        prev_iteration = iteration_num - 1
        prev_model_dir = os.path.join(args.output_dir, f'output_iteration{prev_iteration}')
        if not os.path.exists(prev_model_dir):
            raise FileNotFoundError(
                f"Expected a checkpoint from iteration {prev_iteration} at '{prev_model_dir}', "
                f"but it doesn't exist. This usually means iteration {prev_iteration} crashed "
                f"before it could save a model."
            )
        model_name = prev_model_dir

    print(f"Loading model and tokenizer from {model_name}...")
    # Tokenizer is usually loaded from the base model directory.
    # add_bos_token=True: auto-prepend BOS to the prompt (ProtGPT3 needs it).
    # add_eos_token=False + padding_side="left": standard settings for generation.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        add_eos_token=False,
        add_bos_token=True,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    print("Model loaded.")

    print(f"Generating {args.num_sequences} sequences for label: '{args.label}'")

    generated_ids, prompt_len = generate_sequences(args.label, model, tokenizer, device, args.num_sequences, args.max_length)

    results = []
    print("Computing metrics...")
    for i, output_ids in enumerate(tqdm(generated_ids)):
        # Strip the prompt ([BOS] + direction token) and keep only the generated protein,
        # then remove the spaces the character-level tokenizer inserts when decoding.
        completion_ids = output_ids[prompt_len:]
        sequence_text = tokenizer.decode(completion_ids, skip_special_tokens=True).replace(" ","")

        # Calculate metrics (perplexity on the full sequence including the prompt)
        output_ids_batch = output_ids.unsqueeze(0)

        ppl = calculate_perplexity(output_ids_batch, model)

        results.append({
            "sequence": sequence_text,
            "perplexity": ppl,
        })

    # Sort by perplexity (lower is better)
    results.sort(key=lambda x: x["perplexity"])

    output_filename = os.path.join(args.output_dir, f"seq_gen_{args.label}_iteration{iteration_num}.fasta")
    
    # Write to FASTA
    print(f"Writing results to {output_filename}")
    with open(output_filename, "w") as f:
        for i, res in enumerate(results):
            header = f">{args.label}_{i}\tppl={res['perplexity']:.4f}"
            f.write(f"{header}\n{res['sequence']}\n")

    print("Done.")

if __name__ == "__main__":
    main()

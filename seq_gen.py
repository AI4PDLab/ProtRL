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

def calculate_log_likelihood(input_ids, model, ref_model):
    """
    Computes implicit reward (log likelihood difference) between model and ref_model.
    Reward = ref_loss - model_loss
    """
    with torch.no_grad():
        outputs_model = model(input_ids, labels=input_ids)
        outputs_ref_model = ref_model(input_ids, labels=input_ids)

    loss = outputs_model.loss
    ref_loss = outputs_ref_model.loss
    implicit_reward = ref_loss - loss 
    return implicit_reward.item()

def generate_sequences(label, model, tokenizer, device, num_sequences=20, max_length=100):
    """
    Generates sequences using the model.
    """
    input_ids = tokenizer.encode(label, return_tensors='pt').to(device)
    
    outputs = model.generate(
        input_ids, 
        top_k=9, 
        repetition_penalty=1.2,
        max_length=max_length,    
        do_sample=True,
        num_return_sequences=num_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Generate sequences using a pretrained LLM.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument("--label", type=str, required=True, help="Prompt/Label for generation.")
    parser.add_argument("--num_sequences", type=int, default=20, help="Number of sequences to generate.")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated sequences.")
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
        # Check output_dir first for the PREVIOUS iteration's model
        prev_iteration = iteration_num - 1
        prev_model_dir = os.path.join(args.output_dir, f'output_iteration{prev_iteration}')
        if os.path.exists(prev_model_dir):
            model_name = prev_model_dir
        else:
            # Fallback
            model_name = f'./output_iteration{prev_iteration}'

    print(f"Loading model and tokenizer from {model_name}...")
    try:
        # Tokenizer is usually loaded from the base model directory
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        # Reference model is loaded from the base model directory
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device) 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded.")

    print(f"Generating {args.num_sequences} sequences for label: '{args.label}'")

    generated_ids = generate_sequences(args.label, model, tokenizer, device, args.num_sequences, args.max_length)
    
    results = []
    print("Computing metrics...")
    for i, output_ids in enumerate(tqdm(generated_ids)):
        # Decode sequence and remove spaces (assuming protein sequences)
        sequence_text = tokenizer.decode(output_ids, skip_special_tokens=True).replace(" ","")
        
        # Calculate metrics
        output_ids_batch = output_ids.unsqueeze(0)
        
        ppl = calculate_perplexity(output_ids_batch, model)
        reward = calculate_log_likelihood(output_ids_batch, model, ref_model)
        
        results.append({
            "sequence": sequence_text,
            "perplexity": ppl,
            "reward": reward
        })

    # Sort by perplexity (lower is better)
    results.sort(key=lambda x: x["perplexity"])

    output_filename = os.path.join(args.output_dir, f"seq_gen_{args.label}_iteration{iteration_num}.fasta")
    
    # Write to FASTA
    print(f"Writing results to {output_filename}")
    with open(output_filename, "w") as f:
        for i, res in enumerate(results):
            header = f">{args.label}_{i}\tppl={res['perplexity']:.4f}\treward={res['reward']:.4f}"
            f.write(f"{header}\n{res['sequence']}\n")

    print("Done.")

if __name__ == "__main__":
    main()

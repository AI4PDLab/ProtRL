import os
import json
import argparse
from tokenizers import Tokenizer, pre_tokenizers, models
from transformers import LlamaTokenizerFast, LlamaConfig, LlamaForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='test/tiny', help='Directory to save the model and tokenizer')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating tiny model in {output_dir}...")

    # ------ Build a Tokenizer form scratch ------
    # Define Tokenizer object
    my_tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    # Define how to split input sting
    my_tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")

    # Initialise the correct tokeinzer trainer
    trainer = my_tokenizer.model.get_trainer()

    # Add special tokens
    spec_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "[UNK]"]
    trainer.special_tokens = spec_tokens

    # Set vocabulary size. Include the ProtGPT3-style direction tokens "1" (forward) and
    # "2" (reverse) so the tiny test model can be prompted the same way as the real model.
    corpus = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2']
    trainer.vocab_size = len(corpus) + len(spec_tokens)

    # Build tokenizer from interator
    my_tokenizer.train_from_iterator(corpus, trainer=trainer)

    # ------- Covert Toknezier obj to FastTokenizer compatible with transformers --
    my_fast_tokenizer = LlamaTokenizerFast(tokenizer_object=my_tokenizer)
    
    my_fast_tokenizer.add_special_tokens({
        'pad_token': '<|pad|>',
        'bos_token': '<|bos|>',
        'eos_token': '<|eos|>'
    })

    # Save tokenizer
    my_fast_tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    # ------ Build Tiny Llama Model ------
    # Configuration for a tiny model (20M params approx, but reduced here for extreme speed)
    # Using the user's "tiny" config as a base but ensuring it matches the tokenizer vocab
    config = LlamaConfig(
        vocab_size=len(my_fast_tokenizer),
        hidden_size=64,           # Reduced for speed (was 512)
        intermediate_size=256,    # Reduced for speed (was 1024)
        num_hidden_layers=4,      # Reduced for speed (was 8)
        num_attention_heads=4,    # Reduced for speed (was 8)
        max_position_embeddings=128, # Reduced for speed
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=my_fast_tokenizer.pad_token_id,
        bos_token_id=my_fast_tokenizer.bos_token_id,
        eos_token_id=my_fast_tokenizer.eos_token_id,
        tie_word_embeddings=False
    )

    # Initialize model
    model = LlamaForCausalLM(config)
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Save config json explicitly if needed (save_pretrained does this, but user asked for specific json dump)
    # We can just rely on save_pretrained for the model config.
    
    print(f"Tiny model saved to {output_dir}")

if __name__ == "__main__":
    main()

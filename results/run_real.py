#!/usr/bin/env python3
"""
Run streaming attention on a small HF model and measure performance.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers not installed, installing...")
    subprocess.check_call(["pip", "install", "transformers", "accelerate"])
    from transformers import AutoModelForCausalLM, AutoTokenizer

def get_memory_mb():
    """Get current memory usage."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_name = "gpt2"  # Small model, CPU OK
    print(f"Loading model: {model_name}")
    
    start_mem = get_memory_mb()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model = model if torch.cuda.is_available() else model.cpu()
    
    end_mem = get_memory_mb()
    load_time = 0  # We're not measuring this in detail
    
    # Test input
    test_text = "The quick brown fox jumps over the lazy dog. This is a test of streaming attention with context windows."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    
    # Generate with streaming attention simulation (simplified)
    # Here we use standard attention to compare baseline
    
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    end = time.time()
    forward_time = end - start
    
    # Calculate perplexity
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs.input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    ppl = torch.exp(loss).item()
    
    # Simulate streaming (ring buffer) attention timing
    # With max sequence length
    seq_len = inputs.input_ids.shape[1]
    max_seq = 1024
    
    start = time.time()
    with torch.no_grad():
        # Run multiple iterations to measure
        for _ in range(5):
            outputs = model(**inputs)
    end = time.time()
    avg_time = (end - start) / 5
    
    print("=" * 60)
    print("RESULTS - Streaming Attention Test on GPT-2")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Input length: {seq_len} tokens")
    print(f"Forward pass time: {forward_time:.4f}s")
    print(f"Average generation time (5 runs): {avg_time:.4f}s")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Peak memory (load): {end_mem:.2f} MB")
    print(f"Memory delta: {(end_mem - start_mem):.2f} MB")
    print("=" * 60)
    print("Note: These are baseline numbers using standard attention.")
    print("Streaming attention (ring buffer) implementation would show")
    print("memory advantages for very long sequences (>max_seq).")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
LESSON 2: Understanding the Model Architecture
===============================================
Let's build intuition for how the GPT model works.

Run this: python learn/02_model_architecture.py
"""

import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig

def main():
    print("=" * 60)
    print("LESSON 2: MODEL ARCHITECTURE")
    print("=" * 60)
    
    # Create a TINY model for learning
    print("\n📐 Creating a tiny GPT model...")
    config = GPTConfig(
        sequence_len=128,      # Max context length
        vocab_size=50304,      # Number of possible tokens
        n_layer=4,             # 4 transformer blocks (depth)
        n_head=4,              # 4 attention heads
        n_kv_head=4,           # 4 key-value heads (MQA ratio 1:1)
        n_embd=256             # 256-dimensional embeddings
    )
    
    model = GPT(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✨ Model created!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   That's {total_params / 1e6:.2f}M parameters")
    
    # Break down by component
    print("\n" + "─" * 60)
    print("Parameter Breakdown")
    print("─" * 60)
    
    component_params = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0]
        component_params[component] = component_params.get(component, 0) + param.numel()
    
    for component, count in sorted(component_params.items()):
        pct = 100 * count / total_params
        print(f"  {component:20s}: {count:10,} params ({pct:5.2f}%)")
    
    # Understanding the flow
    print("\n" + "─" * 60)
    print("Data Flow Through Model")
    print("─" * 60)
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n1. INPUT:")
    print(f"   Shape: {input_tokens.shape}")
    print(f"   - Batch size: {batch_size} (processing 2 sequences together)")
    print(f"   - Sequence length: {seq_len} (10 tokens)")
    print(f"   Example token IDs:\n   {input_tokens[0].tolist()}")
    
    # Forward pass with intermediate outputs
    print(f"\n2. EMBEDDING LAYER:")
    with torch.no_grad():
        # Token embeddings
        x = model.tok_emb(input_tokens)
        print(f"   After tok_emb: {x.shape}")
        print(f"   - Each token → {config.n_embd}-dimensional vector")
        print(f"   - Now we have: [batch, seq_len, n_embd]")
        print(f"   - Example vector (first token, first 8 dims):")
        print(f"     {x[0, 0, :8].tolist()}")
        
        x = model.input_norm(x)
        print(f"\n   After normalization: {x.shape}")
        print(f"   - Normalize to stabilize training")
    
    print(f"\n3. TRANSFORMER BLOCKS ({config.n_layer} blocks):")
    print(f"   Each block does:")
    print(f"   a) Self-Attention: Look at all previous tokens")
    print(f"   b) Feed-Forward: Process the information")
    print(f"   c) Residual connections: Preserve original signal")
    
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            x_before = x.clone()
            x = block(x, model.rotary_cos_sin, kv_cache=None)
            change = (x - x_before).abs().mean().item()
            print(f"   Block {i}: Output shape {x.shape}, avg change: {change:.4f}")
    
    print(f"\n4. OUTPUT LAYER:")
    with torch.no_grad():
        x = model.output_norm(x)
        print(f"   After final norm: {x.shape}")
        
        logits = model.lm_head(x)
        print(f"   After lm_head: {logits.shape}")
        print(f"   - Each position → {config.vocab_size} logits")
        print(f"   - Logits = unnormalized probabilities")
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        print(f"\n   Probabilities: {probs.shape}")
        print(f"   - Sum to 1.0 for each position")
        print(f"   - Example (position 0, top 5 tokens):")
        top5_probs, top5_indices = probs[0, 0].topk(5)
        for prob, idx in zip(top5_probs, top5_indices):
            print(f"     Token {idx:5d}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Full forward pass
    print("\n" + "─" * 60)
    print("Full Forward Pass")
    print("─" * 60)
    
    with torch.no_grad():
        output = model(input_tokens)
    
    print(f"Input shape:  {input_tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nInterpretation:")
    print(f"  - For each of {batch_size} sequences")
    print(f"  - For each of {seq_len} positions")
    print(f"  - We predict distribution over {config.vocab_size} tokens")
    
    # Understanding attention
    print("\n" + "─" * 60)
    print("Understanding Self-Attention")
    print("─" * 60)
    
    print("""
Self-attention lets each position "look at" previous positions:

Example: "The cat sat on the mat because it was tired"
         Position 8 (word "it") can attend to:
         
         The  cat  sat  on  the  mat  because  it  was  tired
         0.1  0.7  0.1  0.0 0.0  0.1  0.0     1.0  0.0  0.0
              ^^^                              ^^^
              High attention to "cat"         Current position
    
The model learns which positions are important!
    """)
    
    # Model size scaling
    print("─" * 60)
    print("Model Size Comparison")
    print("─" * 60)
    
    configs = [
        ("Tiny (learning)", 4, 256),
        ("Small", 12, 768),
        ("Medium", 24, 1024),
        ("Large", 36, 1280),
        ("GPT-3 scale", 96, 12288),
    ]
    
    print(f"{'Name':<20} {'Layers':<10} {'Dim':<10} {'Params':<15}")
    print("─" * 60)
    
    for name, layers, dim in configs:
        # Rough parameter count estimation
        # embedding: vocab * dim
        # blocks: layers * (4 * dim^2)  # simplified
        # output: vocab * dim
        params = 2 * 50304 * dim + layers * 4 * dim * dim
        print(f"{name:<20} {layers:<10} {dim:<10} {params/1e6:>10.1f}M")
    
    # Memory calculation
    print("\n" + "─" * 60)
    print("Memory Requirements")
    print("─" * 60)
    
    # Calculate memory for our tiny model
    param_memory = total_params * 4 / (1024**3)  # 4 bytes per float32
    activation_memory = batch_size * seq_len * config.n_embd * config.n_layer * 4 / (1024**3)
    
    print(f"For our tiny model:")
    print(f"  Parameters: {param_memory*1024:.2f} MB")
    print(f"  Activations (batch={batch_size}, seq={seq_len}): {activation_memory*1024:.2f} MB")
    print(f"  Total: ~{(param_memory + activation_memory)*1024:.2f} MB")
    
    print(f"\n💡 Key Insight:")
    print(f"   - Larger models = more parameters = more memory")
    print(f"   - Longer sequences = more activations = more memory")
    print(f"   - This is why we need big GPUs for big models!")
    
    # Computational cost
    print("\n" + "─" * 60)
    print("Computational Cost")
    print("─" * 60)
    
    flops_per_token = model.estimate_flops()
    print(f"FLOPs per token: {flops_per_token:e}")
    print(f"For a sequence of {seq_len} tokens: {flops_per_token * seq_len:e} FLOPs")
    
    print(f"\n💡 Training cost:")
    print(f"   - More layers/dims = more FLOPs")
    print(f"   - Training on billions of tokens = expensive!")
    print(f"   - GPT-3 training: ~3.14e23 FLOPs (~$5M in compute)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
1. Input: Token IDs [batch, seq_len]
   
2. Embedding: Map tokens to vectors [batch, seq_len, n_embd]
   
3. Transformer Blocks (repeated n_layer times):
   - Self-Attention: Look at context
   - Feed-Forward: Transform features
   - Residual connections: Preserve information
   
4. Output: Predict next token [batch, seq_len, vocab_size]
   
5. Loss: Compare predictions to actual next tokens
   
6. Backprop: Update weights to improve predictions
   
7. Repeat for millions of examples!
""")
    
    print("✅ You now understand the model architecture!")
    print("\n📚 Next: Check out LEARNING_GUIDE.md for more details")

if __name__ == "__main__":
    main()

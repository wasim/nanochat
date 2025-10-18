# 🎓 Nanochat Learning Guide for Python Engineers

> A comprehensive guide to understanding language model training from scratch

## 📖 Table of Contents
1. [Core Concepts](#core-concepts)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [Training Pipeline](#training-pipeline)
5. [Hands-On Examples](#hands-on-examples)

---

## 🎯 Core Concepts

### What is a Language Model?

Think of it like **autocomplete on steroids**:
- Input: "The weather today is"
- Model predicts: "sunny" (or "rainy", "cold", etc.)

The model learns probability distributions over words/tokens.

### Key Terms for Python Engineers

| ML Term | Python Analogy | What It Means |
|---------|----------------|---------------|
| **Token** | `str.split()` | Text broken into pieces (words/subwords) |
| **Embedding** | `dict` lookup | Converting tokens to numbers |
| **Tensor** | `numpy.ndarray` | Multi-dimensional array |
| **Forward Pass** | `function(input)` | Running data through the model |
| **Backward Pass** | Automatic derivatives | Computing gradients |
| **Loss** | Error metric | How wrong the predictions are |
| **Optimizer** | Update rule | How to adjust weights |
| **Batch** | List of examples | Processing multiple inputs together |

### The Big Picture: Training Loop

```python
for epoch in range(num_epochs):
    # 1. Get data
    x, y = get_batch()  # x = input text, y = target (next token)
    
    # 2. Forward pass (prediction)
    logits = model(x)  # model predicts next token
    loss = compute_loss(logits, y)  # how wrong are we?
    
    # 3. Backward pass (learning)
    loss.backward()  # compute gradients
    optimizer.step()  # update weights
    
    # 4. Repeat until smart!
```

---

## 🏗️ Architecture Deep Dive

### 1. **Transformer Architecture** (the heart of GPT)

```
Input Text → Tokens → Embeddings → Transformer Blocks → Output Predictions
```

#### What's a Transformer Block?

```python
# Simplified pseudo-code
def transformer_block(x):
    # Self-attention: "look at all previous words"
    attended = self_attention(x)  # What context is relevant?
    x = x + attended  # Residual connection
    x = normalize(x)
    
    # Feed-forward: "process the information"
    transformed = feed_forward(x)  # Transform features
    x = x + transformed  # Residual connection
    x = normalize(x)
    
    return x
```

**Why this matters**: Each block learns to understand context better.

---

## 📂 File-by-File Breakdown

### **Core Files** (Start Here)

#### 1. `tokenizer.py` - Text → Numbers
```python
# What it does:
"Hello world" → [15496, 995]  # Convert text to token IDs
[15496, 995] → "Hello world"  # Convert back
```

**Key concepts**:
- **BPE (Byte Pair Encoding)**: Smart way to split text
- **Vocab**: Dictionary of all possible tokens (~50k tokens)
- **Special tokens**: `<|bos|>` (beginning), `<|eos|>` (end), `<|pad|>`

#### 2. `gpt.py` - The Model Brain
```python
class GPT(nn.Module):
    def __init__(self, config):
        # Token embedding: token_id → vector
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        
        # Transformer blocks (the magic!)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        
        # Output layer: vector → probabilities over vocab
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

**Architecture features**:
- **Rotary embeddings**: Better position encoding
- **QK normalization**: Stable training
- **ReLU² activation**: Faster, simpler
- **MQA (Multi-Query Attention)**: Efficient inference

#### 3. `dataloader.py` - Feeding Data
```python
# What it does:
# 1. Load raw text files
# 2. Tokenize them
# 3. Create batches of (input, target) pairs
# 4. Shuffle and distribute across GPUs

# Example batch:
# input:  [15496, 995, 318]  # "Hello world is"
# target: [995, 318, 257]    # "world is a"  (shifted by 1)
```

#### 4. `engine.py` - Generation/Inference
```python
# Given prompt, generate text
prompt = "The capital of France is"
tokens = tokenizer(prompt)
output = engine.generate(tokens, max_tokens=10)
# → "The capital of France is Paris and it has"
```

**Generation strategies**:
- **Greedy**: Pick highest probability
- **Sampling**: Sample from distribution
- **Temperature**: Control randomness

#### 5. `loss_eval.py` - Measuring Performance
```python
# Cross-entropy loss: how wrong are predictions?
loss = -log(probability_of_correct_token)

# Lower loss = better predictions
# Measured in "bits per byte" (bpb) or "perplexity"
```

### **Optimization Files**

#### 6. `muon.py` - Custom Optimizer
```python
# Muon = Momentum-based optimizer for matrices
# Used for transformer weights (most parameters)
# Faster convergence than Adam for large models
```

**Why special optimizer?**
- Different learning rates for different layers
- Matrix-specific optimization (not just vectors)

#### 7. `adamw.py` - Standard Optimizer
```python
# AdamW = Adam with Weight Decay
# Used for embeddings (token → vector mappings)
# Industry standard, well-tested
```

### **Training Infrastructure**

#### 8. `checkpoint_manager.py` - Saving/Loading
```python
# Save model weights every N steps
# Resume training from checkpoint
# Useful for long training runs (days/weeks)
```

#### 9. `common.py` - Utilities
```python
# DDP (Distributed Data Parallel) setup
# Multi-GPU training coordination
# Logging, device detection (CUDA/MPS/CPU)
```

#### 10. `configurator.py` - Config Management
```python
# Parse command-line arguments
# Override defaults: --depth=12 --batch_size=32
```

### **Evaluation Files**

#### 11. `core_eval.py` - Benchmark Tasks
```python
# Test on standard benchmarks:
# - MMLU: Multiple choice questions
# - GSM8K: Math problems
# - HumanEval: Code generation
# - ARC: Reading comprehension
```

#### 12. `dataset.py` - Task Datasets
```python
# Load and format evaluation datasets
# Convert to model-friendly format
```

---

## 🚂 Training Pipeline

### Step-by-Step: `scripts/base_train.py`

```python
# 1. SETUP
model = GPT(config)  # Create model
optimizer = setup_optimizers()  # How to update weights
train_loader = get_data_loader()  # Load training data

# 2. TRAINING LOOP
for step in range(num_iterations):
    # 2a. Get batch
    x, y = next(train_loader)  # x=input, y=target
    
    # 2b. Forward pass
    logits = model(x)  # Predictions
    loss = cross_entropy(logits, y)  # Error
    
    # 2c. Backward pass
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    
    # 2d. Evaluation (periodic)
    if step % eval_every == 0:
        val_loss = evaluate(model, val_loader)
        print(f"Step {step}, Loss: {val_loss}")
    
    # 2e. Checkpoint (periodic)
    if step % save_every == 0:
        save_checkpoint(model, optimizer, step)
```

### Key Hyperparameters

```python
# Model Size
depth = 20              # Number of transformer blocks
model_dim = 1280        # Hidden dimension (depth * 64)
num_heads = 10          # Attention heads

# Training
total_batch_size = 524288     # Tokens per update
device_batch_size = 32        # Per-GPU batch size
num_iterations = 5000         # Training steps
grad_accum_steps = 16         # Accumulate gradients

# Optimization
matrix_lr = 0.02        # Learning rate for Muon
embedding_lr = 0.2      # Learning rate for embeddings
weight_decay = 0.0      # L2 regularization

# Evaluation
eval_every = 250        # Validation frequency
eval_tokens = 10M       # Tokens for validation
```

---

## 💡 Hands-On Examples

### Example 1: Understanding Tokens

```python
from nanochat.tokenizer import get_tokenizer

# Initialize tokenizer
tokenizer = get_tokenizer()

# Tokenize text
text = "Hello, world!"
tokens = tokenizer(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Decode back
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")
```

### Example 2: Loading a Model

```python
from nanochat.gpt import GPT, GPTConfig
import torch

# Create model config
config = GPTConfig(
    sequence_len=1024,
    vocab_size=50304,
    n_layer=12,        # 12 transformer blocks
    n_head=6,          # 6 attention heads
    n_embd=768         # 768-dim embeddings
)

# Initialize model
model = GPT(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass
x = torch.randint(0, 50304, (1, 256))  # Batch of 1, seq len 256
logits = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {logits.shape}")  # [1, 256, 50304]
```

### Example 3: Text Generation

```python
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT

# Load model and tokenizer
tokenizer = get_tokenizer()
model = GPT.from_pretrained("path/to/checkpoint")
engine = Engine(model, tokenizer)

# Generate text
prompt = "Once upon a time"
tokens = tokenizer(prompt, prepend="<|bos|>")
output, _ = engine.generate_batch(
    tokens,
    num_samples=1,
    max_tokens=50,
    temperature=0.8  # Higher = more creative
)

print(tokenizer.decode(output[0]))
```

### Example 4: Training a Tiny Model

```bash
# Train a small model (good for learning)
python scripts/base_train.py \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=100 \
    --run=my_first_model
```

---

## 🔬 Deep Dive: How Attention Works

### Self-Attention in Plain English

Imagine reading: **"The cat sat on the mat because it was tired"**

Question: What does "it" refer to?

**Self-attention** lets the model look at all previous words:
- "it" attends to "cat" (high attention weight)
- "it" ignores "mat" (low attention weight)
- Model learns: "it" = "cat"

### In Code

```python
# Simplified attention mechanism
def attention(Q, K, V):
    # Q = Query: "What am I looking for?"
    # K = Key:   "What do I contain?"
    # V = Value: "What information do I have?"
    
    # 1. Compute similarity: Q · K^T
    scores = Q @ K.T / sqrt(d_k)  # [seq_len, seq_len]
    
    # 2. Softmax: convert to probabilities
    attn_weights = softmax(scores, dim=-1)
    
    # 3. Weighted sum: aggregate information
    output = attn_weights @ V
    
    return output
```

**Visualization**:
```
Word:    The  cat  sat  on   the  mat  because it   was  tired
Attn:    0.05 0.70 0.05 0.02 0.05 0.03 0.05   1.0  0.03 0.02
         └────┴────┴────┴────┴────┴────┴──────▲────┴────┴──────┘
                                               "it" looks back
```

---

## 📊 Understanding Loss & Metrics

### 1. **Cross-Entropy Loss**
```python
# How to interpret:
loss = 2.5   # Bad: model is very uncertain
loss = 1.0   # Good: model is getting confident
loss = 0.1   # Great: model is very confident
```

### 2. **Bits Per Byte (BPB)**
```python
# Lower = better compression = better understanding
bpb = 1.5  # Model understands text well
bpb = 2.5  # Model struggles with text
```

### 3. **Perplexity**
```python
perplexity = exp(loss)
# "On average, model is confused among N options"
perplexity = 10  # Confused among 10 words
perplexity = 2   # Almost certain (binary choice)
```

### 4. **CORE Metric**
```python
# Averaged score across multiple benchmarks
# Range: 0.0 (random guessing) to 1.0 (perfect)
core_metric = 0.65  # Good performance
```

---

## 🎮 Common Training Patterns

### Pattern 1: Learning Rate Schedule

```python
# Warmup → Stable → Cooldown
#     /‾‾‾‾‾‾‾‾\___
#    /            \
#   /              \__
# Start   Middle    End

def get_lr_multiplier(step):
    if step < warmup_steps:
        return step / warmup_steps  # Ramp up
    elif step < stable_steps:
        return 1.0  # Keep constant
    else:
        return decay_factor  # Ramp down
```

**Why?** Prevents early instability and helps final convergence.

### Pattern 2: Gradient Accumulation

```python
# Problem: GPU memory limited → small batches → unstable
# Solution: Accumulate gradients over multiple mini-batches

for micro_batch in range(grad_accum_steps):
    loss = model(x, y) / grad_accum_steps
    loss.backward()  # Accumulate gradients

optimizer.step()  # Update once with accumulated gradients
optimizer.zero_grad()
```

### Pattern 3: Distributed Training

```python
# Multiple GPUs working together
# DDP (Distributed Data Parallel):
# 1. Each GPU has full model copy
# 2. Each processes different data
# 3. Gradients synced across GPUs
# 4. All update identically

# Usage:
torchrun --nproc_per_node=8 base_train.py
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory (OOM)
```python
# Solution: Reduce batch size
--device_batch_size=16  # Instead of 32
```

### Issue 2: Loss Explodes (NaN)
```python
# Solutions:
1. Lower learning rate: --matrix_lr=0.01
2. Enable gradient clipping: --grad_clip=1.0
3. Use mixed precision training (bfloat16)
```

### Issue 3: Slow Training
```python
# Solutions:
1. Use multiple GPUs: torchrun --nproc_per_node=4
2. Increase batch size: --device_batch_size=64
3. Use torch.compile(): model = torch.compile(model)
```

### Issue 4: Poor Generation Quality
```python
# Check:
1. Training loss still decreasing? (not converged)
2. Validation loss similar to train? (not overfitting)
3. Try different temperature: temperature=0.7
```

---

## 🚀 Next Steps

### Beginner Path
1. ✅ **Understand tokenization**: Run `scripts/tok_eval.py`
2. ✅ **Train tiny model**: `base_train.py` with small params
3. ✅ **Generate text**: Use `scripts/chat_cli.py`
4. ✅ **Read loss curves**: Understand training dynamics

### Intermediate Path
1. 📊 **Experiment with hyperparameters**: depth, learning rate
2. 🔧 **Modify architecture**: Add layers, change attention
3. 📈 **Analyze attention patterns**: Visualize what model learns
4. 🎯 **Fine-tune on custom data**: Domain-specific models

### Advanced Path
1. 🧪 **Implement new optimizers**: Test alternatives to Muon
2. 🏗️ **Architecture search**: Find optimal model size
3. 📚 **Scaling laws**: Understand model size vs performance
4. 🤖 **RLHF**: Reinforcement learning from human feedback

---

## 📚 Key Papers & Resources

### Must-Read Papers
1. **"Attention Is All You Need"** (2017) - Original Transformer
2. **"GPT-3"** (2020) - Language model scaling
3. **"Chinchilla"** (2022) - Optimal training compute

### Code References
- `nanochat/gpt.py` - Model architecture
- `scripts/base_train.py` - Training loop
- `nanochat/engine.py` - Text generation

### Debugging Tips
```python
# Add breakpoint in VS Code
import pdb; pdb.set_trace()

# Or use VS Code's debugger (F5)
# Set breakpoints by clicking line numbers
```

---

## 🎯 Quick Reference

### Training a Model
```bash
# Small (for testing)
python scripts/base_train.py --depth=4 --num_iterations=100

# Medium (for real use)
python scripts/base_train.py --depth=12 --num_iterations=5000

# Large (multi-GPU)
torchrun --nproc_per_node=8 scripts/base_train.py --depth=20
```

### Evaluating a Model
```bash
# Core benchmarks
python scripts/base_eval.py --checkpoint=base_checkpoints/d12

# Chat evaluation
python scripts/chat_eval.py --checkpoint=chat_checkpoints/final
```

### Interactive Chat
```bash
# CLI
python scripts/chat_cli.py --checkpoint=chat_checkpoints/final

# Web UI
python scripts/chat_web.py --checkpoint=chat_checkpoints/final
```

---

## 💬 Questions to Understand Better

1. **"Why do we need embeddings?"**
   - Computers can't process words directly, only numbers
   - Embeddings map words to vectors that capture meaning
   - Similar words → similar vectors

2. **"What's the difference between training and inference?"**
   - Training: Learn from data (backward pass, update weights)
   - Inference: Use learned model (forward pass only, no updates)

3. **"Why so many hyperparameters?"**
   - Model architecture (depth, width)
   - Optimization (learning rates, batch sizes)
   - Training schedule (warmup, cooldown)
   - Each affects quality/speed tradeoff

4. **"How long does training take?"**
   - Small model (depth=4): Minutes to hours
   - Medium model (depth=12): Hours to days
   - Large model (depth=20): Days to weeks
   - GPT-4 scale: Months on thousands of GPUs

---

## 🎓 Learning Checkpoints

Mark your progress:
- [ ] Understand tokenization
- [ ] Grasp transformer architecture
- [ ] Run successful training
- [ ] Generate coherent text
- [ ] Evaluate on benchmarks
- [ ] Modify model architecture
- [ ] Train distributed (multi-GPU)
- [ ] Implement custom features

---

**Happy Learning! 🚀**

Remember: Everyone starts confused. ML is iterative learning - experiment, break things, and learn by doing!

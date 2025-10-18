# 🎯 Nanochat Quick Reference Card

> Keep this handy while learning!

## 🚀 Quick Commands

```bash
# Learning (start here!)
python learn/00_quick_overview.py       # Overview
python learn/01_tokenization_demo.py    # Lesson 1
python learn/02_model_architecture.py   # Lesson 2
python learn/03_training_demo.py        # Lesson 3

# Training
python scripts/base_train.py --depth=4 --num_iterations=100        # Tiny (learning)
python scripts/base_train.py --depth=12 --num_iterations=5000      # Small (real)
torchrun --nproc_per_node=8 scripts/base_train.py --depth=20       # Large (multi-GPU)

# Evaluation
python scripts/base_eval.py --checkpoint=base_checkpoints/d12
python scripts/chat_eval.py --checkpoint=chat_checkpoints/final

# Interactive
python scripts/chat_cli.py --checkpoint=chat_checkpoints/final
python scripts/chat_web.py --checkpoint=chat_checkpoints/final

# VS Code Debugging
# Just press F5 with any Python file open!
```

---

## 📐 Model Size Quick Reference

| Depth | Params | Time | Use Case |
|-------|--------|------|----------|
| 4 | ~10M | Minutes | Learning |
| 12 | ~100M | Hours | Small prod |
| 20 | ~400M | Days | Medium prod |
| 36+ | 1B+ | Weeks | Large scale |

---

## 🎛️ Key Hyperparameters

```python
# Model Architecture
depth = 12              # Number of layers
model_dim = depth * 64  # Hidden dimension
max_seq_len = 2048      # Context length

# Training
total_batch_size = 524288    # Tokens per update
device_batch_size = 32       # Per-GPU batch
num_iterations = 5000        # Training steps

# Optimization
matrix_lr = 0.02        # Muon learning rate
embedding_lr = 0.2      # Adam learning rate
weight_decay = 0.0      # Regularization
grad_clip = 1.0         # Gradient clipping

# Evaluation
eval_every = 250        # Validation frequency
core_metric_every = 2000 # Benchmark frequency
```

---

## 📊 Tensor Shapes Cheat Sheet

```python
# Common shapes (B=batch, T=seq_len, D=model_dim, V=vocab_size)
input_tokens:    [B, T]         # Token IDs
embeddings:      [B, T, D]      # Embedded vectors
after_attention: [B, T, D]      # Attended vectors
after_ffn:       [B, T, D]      # Transformed vectors
logits:          [B, T, V]      # Raw predictions
probs:           [B, T, V]      # After softmax

# Attention internals
Q: [B, num_heads, T, head_dim]
K: [B, num_kv_heads, T, head_dim]
V: [B, num_kv_heads, T, head_dim]
```

---

## 🧠 Core Concepts in 30 Seconds

**Token**: Piece of text (word/subword)  
**Embedding**: Token → vector of numbers  
**Attention**: "What context is relevant?"  
**Transformer**: Stack of attention + feed-forward  
**Loss**: How wrong are predictions?  
**Training**: Adjust weights to reduce loss  
**Generation**: Predict next token, repeat  

---

## 🔧 Common Operations

```python
# Model
model = GPT(config)
logits = model(input_tokens)        # Forward pass
loss = F.cross_entropy(logits, targets)
loss.backward()                     # Compute gradients
optimizer.step()                    # Update weights

# Tokenizer
tokens = tokenizer("Hello world")   # Encode
text = tokenizer.decode(tokens)     # Decode

# Generation
output = engine.generate_batch(
    tokens, 
    max_tokens=50, 
    temperature=0.8
)

# Device
x = x.to(device)                    # Move to GPU
with torch.no_grad():               # Disable gradients
    output = model(x)
```

---

## 📚 Essential Files

```
learn/
  00_quick_overview.py    → Start here
  01_tokenization_demo.py → Lesson 1
  02_model_architecture.py→ Lesson 2
  03_training_demo.py     → Lesson 3

nanochat/
  gpt.py          → Model definition
  tokenizer.py    → Text ↔ numbers
  engine.py       → Generation
  dataloader.py   → Data feeding
  muon.py         → Optimizer
  loss_eval.py    → Metrics

scripts/
  base_train.py   → Main training
  base_eval.py    → Evaluation
  chat_cli.py     → Interactive chat

Documentation/
  LEARNING_GUIDE.md           → Full guide
  ARCHITECTURE_DIAGRAMS.md    → Visual diagrams
  GLOSSARY.md                 → Term definitions
  LEARNING_RESOURCES_INDEX.md → Navigation
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of Memory** | Reduce `--device_batch_size` |
| **NaN Loss** | Lower `--matrix_lr` or `--embedding_lr` |
| **Slow Training** | Check GPU usage, increase batch size |
| **Import Error** | Check PYTHONPATH in VS Code launch.json |
| **Poor Generation** | Train longer, check validation loss |
| **Can't Resume** | Checkpoint saved every N steps |

---

## 📈 Training Monitoring

**Good Signs:**
- ✅ Loss steadily decreasing
- ✅ Validation ≈ training loss
- ✅ Stable gradients
- ✅ Tokens/sec consistent

**Bad Signs:**
- ❌ Loss not decreasing → LR too low
- ❌ Loss exploding → LR too high  
- ❌ Val >> train → Overfitting
- ❌ NaN values → Instability

---

## 🎯 Learning Checkpoints

- [ ] Understand tokenization
- [ ] Know transformer architecture
- [ ] Train a model successfully
- [ ] Generate coherent text
- [ ] Read and understand source code
- [ ] Modify hyperparameters
- [ ] Debug training issues
- [ ] Implement custom features

---

## 💡 Pro Tips

1. **Start small**: depth=4 for learning
2. **Use debugger**: F5 in VS Code
3. **Print shapes**: `print(tensor.shape)`
4. **Monitor loss**: Should go down!
5. **Save often**: Use checkpoints
6. **Experiment**: Try different configs
7. **Read code**: Best way to learn
8. **Be patient**: Training takes time

---

## 🔗 Quick Links

- **Overview**: `python learn/00_quick_overview.py`
- **Full Guide**: Open `LEARNING_GUIDE.md`
- **Glossary**: Open `GLOSSARY.md`
- **Diagrams**: Open `ARCHITECTURE_DIAGRAMS.md`

---

## 🆘 Emergency Commands

```bash
# Kill training
Ctrl+C (or Cmd+C on Mac)

# Check GPU usage
nvidia-smi  # Linux/Windows
# or check Activity Monitor on Mac

# Free memory
torch.cuda.empty_cache()  # In Python

# Debug mode
python -m pdb script.py  # Drop into debugger
```

---

**Print this and keep it next to your keyboard!** 📄🖨️

*Last updated: October 2025*

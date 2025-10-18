# 🎓 Complete Training Guide: From Base Model to ChatBot

> **The Full Journey**: Pre-training → Supervised Fine-Tuning (SFT) → Reinforcement Learning (RL)

This guide walks you through the **entire pipeline** of training a language model from scratch to a chatbot that can follow instructions.

---

## 📋 Table of Contents

1. [Overview: The 3-Stage Pipeline](#overview-the-3-stage-pipeline)
2. [Stage 0: Tokenizer Training](#stage-0-tokenizer-training)
3. [Stage 1: Base Pre-Training](#stage-1-base-pre-training)
4. [Stage 2: Supervised Fine-Tuning (SFT)](#stage-2-supervised-fine-tuning-sft)
5. [Stage 3: Reinforcement Learning (RL)](#stage-3-reinforcement-learning-rl)
6. [Evaluation & Testing](#evaluation--testing)
7. [Deployment & Usage](#deployment--usage)
8. [Complete Example Workflow](#complete-example-workflow)

---

## 🎯 Overview: The 3-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANGUAGE MODEL TRAINING                       │
└─────────────────────────────────────────────────────────────────┘

Stage 0: TOKENIZER TRAINING
   │
   ├─► Train custom tokenizer on your data
   │   Script: tok_train.py
   │   Input:  Raw text files
   │   Output: tokenizer.model
   │
   ▼

Stage 1: BASE PRE-TRAINING (Foundation)
   │
   ├─► Train on massive text data
   │   Script: base_train.py
   │   Goal:   Learn language patterns
   │   Output: Base model checkpoint
   │
   ▼

Stage 2: SUPERVISED FINE-TUNING (SFT)
   │
   ├─► Train on instruction-response pairs
   │   Script: chat_sft.py
   │   Goal:   Learn to follow instructions
   │   Output: SFT model checkpoint
   │
   ▼

Stage 3: REINFORCEMENT LEARNING (RL)
   │
   ├─► Optimize for human preferences
   │   Script: chat_rl.py
   │   Goal:   Align with human values
   │   Output: Final chatbot checkpoint
   │
   ▼

DEPLOYMENT
   │
   └─► Use for inference
       Scripts: chat_cli.py, chat_web.py
       Output: Interactive chatbot!
```

### What Each Stage Does:

| Stage | What It Learns | Example |
|-------|----------------|---------|
| **Base Training** | Language patterns, facts, reasoning | "The capital of France is Paris" |
| **SFT** | Follow instructions | User: "What's the capital of France?" → "The capital of France is Paris." |
| **RL** | Human preferences | User: "Explain quantum physics" → Clear, helpful explanation (not jargon) |

---

## 🔤 Stage 0: Tokenizer Training

### Purpose
Train a custom tokenizer optimized for your specific domain/language.

### When to Do This
- **Skip if**: Using general English text (use default tokenizer)
- **Do if**: Working with specialized domain, multiple languages, or code

### Script: `tok_train.py`

```bash
python scripts/tok_train.py \
    --data_dir=/path/to/training/data \
    --vocab_size=50304 \
    --output_dir=tokenizers/custom
```

### Key Parameters
```python
vocab_size = 50304      # Size of vocabulary (default is good)
data_dir = "data/"      # Directory with .txt files
output_dir = "custom/"  # Where to save tokenizer
```

### Evaluation: `tok_eval.py`

Test how well your tokenizer compresses text:

```bash
python scripts/tok_eval.py \
    --tokenizer_path=tokenizers/custom/tokenizer.model \
    --test_file=data/validation.txt
```

**Good compression ratio**: 3-4 characters per token for English

---

## 🧠 Stage 1: Base Pre-Training

### Purpose
Train the foundational model that understands language.

### What It Learns
- Grammar and syntax
- Facts and knowledge
- Reasoning patterns
- Next token prediction

### Script: `base_train.py`

#### Quick Start (Small Model for Learning)
```bash
python scripts/base_train.py \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=100 \
    --run=tiny_test
```

#### Production Training (Medium Model)
```bash
python scripts/base_train.py \
    --depth=12 \
    --max_seq_len=2048 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --num_iterations=10000 \
    --eval_every=250 \
    --core_metric_every=2000 \
    --run=base_12layer
```

#### Large Scale (Multi-GPU)
```bash
torchrun --nproc_per_node=8 scripts/base_train.py \
    --depth=20 \
    --max_seq_len=2048 \
    --device_batch_size=32 \
    --total_batch_size=4194304 \
    --num_iterations=50000 \
    --run=base_20layer_production
```

### Key Hyperparameters

```python
# Model Architecture
depth = 12                  # Number of transformer layers
max_seq_len = 2048          # Context window size
# Automatically derived:
# - model_dim = depth * 64
# - num_heads = model_dim // 128

# Training Horizon (use ONE of these)
num_iterations = 10000      # Explicit number of steps
target_flops = 1e20         # Train to reach target FLOPs
target_param_data_ratio = 20  # Chinchilla ratio (20:1)

# Optimization
total_batch_size = 524288   # Total tokens per update
device_batch_size = 32      # Per-GPU batch size
embedding_lr = 0.2          # Learning rate for embeddings
matrix_lr = 0.02            # Learning rate for transformer weights
weight_decay = 0.0          # L2 regularization
grad_clip = 1.0             # Gradient clipping

# Evaluation
eval_every = 250            # Validation frequency (steps)
eval_tokens = 10485760      # Tokens for validation
core_metric_every = 2000    # Benchmark eval frequency
sample_every = 2000         # Text generation frequency
```

### Training Data

By default, uses tokenized data from `~/.cache/nanochat/tokenized_data/`:
- `train/` - Training shards
- `val/` - Validation shards

### Monitoring Training

Watch these metrics:
```python
# Loss (should decrease)
train/loss: 2.5 → 1.8 → 1.2 → ...  # Lower is better
val/bpb: 1.5 → 1.2 → 1.0 → ...     # Bits per byte

# Speed
tok_per_sec: 50000   # Tokens processed per second
mfu: 45%             # Model FLOPs Utilization

# Quality (every 2000 steps)
core_metric: 0.65    # Average benchmark score (0-1)
```

### Output

Checkpoints saved to `base_checkpoints/d{depth}/`:
```
base_checkpoints/d12/
├── step_000000/
│   ├── model.pt          # Model weights
│   ├── optimizers.pt     # Optimizer states
│   └── metadata.json     # Training metadata
├── step_002000/
├── step_004000/
└── ...
```

### Evaluation: `base_eval.py`

Evaluate base model on benchmarks:

```bash
python scripts/base_eval.py \
    --checkpoint=base_checkpoints/d12/step_010000 \
    --max_per_task=500
```

**Benchmarks tested**:
- **MMLU**: Multiple choice questions (knowledge)
- **GSM8K**: Math word problems (reasoning)
- **HumanEval**: Code generation
- **ARC**: Reading comprehension
- **CORE**: Aggregated metric

### Loss Evaluation: `base_loss.py`

Calculate detailed validation loss:

```bash
python scripts/base_loss.py \
    --checkpoint=base_checkpoints/d12/step_010000 \
    --eval_tokens=20000000
```

---

## 💬 Stage 2: Supervised Fine-Tuning (SFT)

### Purpose
Teach the model to follow instructions and have conversations.

### What Changes
- **Base model**: "Paris is the capital" (completes text)
- **SFT model**: "The capital of France is Paris." (answers question)

### Script: `chat_sft.py`

#### Quick Start (Test SFT)
```bash
python scripts/chat_sft.py \
    --base_checkpoint=base_checkpoints/d12/step_010000 \
    --depth=12 \
    --num_iterations=1000 \
    --device_batch_size=8 \
    --total_batch_size=65536 \
    --run=sft_test
```

#### Production SFT
```bash
python scripts/chat_sft.py \
    --base_checkpoint=base_checkpoints/d12/step_010000 \
    --depth=12 \
    --max_seq_len=2048 \
    --num_iterations=5000 \
    --device_batch_size=16 \
    --total_batch_size=262144 \
    --matrix_lr=0.001 \
    --embedding_lr=0.01 \
    --eval_every=100 \
    --sample_every=500 \
    --run=sft_production
```

### Key Differences from Base Training

```python
# Load pre-trained weights
base_checkpoint = "base_checkpoints/d12/step_010000"

# Lower learning rates (fine-tuning, not training from scratch)
matrix_lr = 0.001        # 50x lower than base training
embedding_lr = 0.01      # 20x lower than base training

# Smaller batches (instruction data is more "dense")
total_batch_size = 262144  # vs 524288 for base training

# Fewer iterations (don't need as much data)
num_iterations = 5000      # vs 10000+ for base training
```

### Training Data Format

SFT expects conversation-formatted data:

```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

Stored in `~/.cache/nanochat/chat_data/`:
- `train/` - Training conversations
- `val/` - Validation conversations

### Special Tokens

```python
<|bos|>        # Beginning of sequence
<|user|>       # User message marker
<|assistant|>  # Assistant message marker
<|eos|>        # End of sequence
```

Example:
```
<|bos|><|user|>What is 2+2?<|assistant|>2+2 equals 4.<|eos|>
```

### Monitoring SFT

```python
# Loss (should be lower than base model)
train/loss: 1.5 → 1.2 → 0.9  # SFT loss typically lower
val/bpb: 0.8 → 0.7 → 0.6     # Better compression on chat data

# Quality samples (check every 500 steps)
User: "Explain photosynthesis"
Assistant: "Photosynthesis is the process by which..."
```

### Output

Checkpoints saved to `chat_checkpoints/sft_*/`:
```
chat_checkpoints/sft_production/
├── step_000000/
├── step_001000/
├── step_002000/
└── ...
```

### Evaluation: `chat_eval.py`

Evaluate chatbot quality:

```bash
python scripts/chat_eval.py \
    --checkpoint=chat_checkpoints/sft_production/step_005000 \
    --max_samples=500
```

**Evaluates**:
- **SmolTalk**: Conversational ability
- **Instruction following**: Task completion
- **Helpfulness**: Quality of responses

---

## 🎯 Stage 3: Reinforcement Learning (RL)

### Purpose
Align the model with human preferences using RLHF (Reinforcement Learning from Human Feedback).

### What It Improves
- **Helpfulness**: More useful responses
- **Harmlessness**: Avoid harmful content
- **Honesty**: More truthful, less hallucination
- **Formatting**: Better structure and clarity

### Script: `chat_rl.py`

#### RL Training
```bash
python scripts/chat_rl.py \
    --sft_checkpoint=chat_checkpoints/sft_production/step_005000 \
    --depth=12 \
    --num_iterations=2000 \
    --device_batch_size=4 \
    --total_batch_size=32768 \
    --rl_lr=0.0001 \
    --kl_coef=0.1 \
    --eval_every=50 \
    --run=rl_final
```

### Key RL Hyperparameters

```python
# Load SFT checkpoint
sft_checkpoint = "chat_checkpoints/sft_production/step_005000"

# RL-specific settings
rl_lr = 0.0001           # Learning rate (even lower than SFT)
kl_coef = 0.1            # KL divergence penalty (stay close to SFT)
gamma = 0.99             # Reward discount factor
clip_ratio = 0.2         # PPO clipping parameter

# Smaller batches (RL is sample-intensive)
total_batch_size = 32768

# Fewer iterations
num_iterations = 2000
```

### How RL Works

```
1. Generate Response
   Model generates answer to prompt

2. Score Response
   Reward model scores quality (0-1)

3. Compute Reward
   reward = score - kl_penalty
   (KL penalty prevents drift from SFT model)

4. Update Policy
   Use PPO to improve model toward higher rewards

5. Repeat
```

### Reward Model

The reward model learns to predict human preferences:
- Trained on comparison data: "Response A is better than B"
- Outputs score for each response
- Higher score = more aligned with human preferences

### Monitoring RL

```python
# Rewards (should increase)
avg_reward: 0.5 → 0.6 → 0.7 → ...

# KL divergence (shouldn't increase too much)
kl_div: 0.1 → 0.15 → 0.2  # If > 0.5, increase kl_coef

# Sample quality
User: "Write a poem about AI"
Assistant: [Check if creative, coherent, on-topic]
```

### Output

Final checkpoints in `chat_checkpoints/rl_*/`:
```
chat_checkpoints/rl_final/
├── step_000000/
├── step_000500/
├── step_001000/
└── step_002000/  ← Final model
```

---

## 📊 Evaluation & Testing

### Base Model Evaluation

```bash
# Full benchmark suite
python scripts/base_eval.py \
    --checkpoint=base_checkpoints/d12/step_010000

# Just calculate loss
python scripts/base_loss.py \
    --checkpoint=base_checkpoints/d12/step_010000 \
    --eval_tokens=50000000
```

### Chat Model Evaluation

```bash
# Conversation quality
python scripts/chat_eval.py \
    --checkpoint=chat_checkpoints/rl_final/step_002000 \
    --max_samples=1000
```

### Interactive Testing

```bash
# Command-line interface
python scripts/chat_cli.py \
    --checkpoint=chat_checkpoints/rl_final/step_002000 \
    --temperature=0.7

# Web interface
python scripts/chat_web.py \
    --checkpoint=chat_checkpoints/rl_final/step_002000 \
    --port=8000
```

---

## 🚀 Deployment & Usage

### Command-Line Chat

```bash
python scripts/chat_cli.py \
    --checkpoint=chat_checkpoints/rl_final/step_002000 \
    --temperature=0.7 \
    --max_tokens=512
```

**Parameters**:
```python
temperature = 0.7    # Randomness (0.0=deterministic, 1.0=creative)
top_p = 0.9          # Nucleus sampling threshold
max_tokens = 512     # Maximum response length
```

### Web Interface

```bash
python scripts/chat_web.py \
    --checkpoint=chat_checkpoints/rl_final/step_002000 \
    --host=0.0.0.0 \
    --port=8000
```

Access at: `http://localhost:8000`

### Programmatic Usage

```python
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT
import torch

# Load model
tokenizer = get_tokenizer()
model = GPT.from_checkpoint("chat_checkpoints/rl_final/step_002000")
model.eval()

# Create engine
engine = Engine(model, tokenizer)

# Generate response
prompt = "<|bos|><|user|>What is Python?<|assistant|>"
tokens = tokenizer(prompt)

with torch.no_grad():
    response_tokens, _ = engine.generate_batch(
        [tokens],
        num_samples=1,
        max_tokens=256,
        temperature=0.7
    )

response = tokenizer.decode(response_tokens[0])
print(response)
```

---

## 🎯 Complete Example Workflow

### Step-by-Step: Training a Small Chatbot

#### 1. **Base Training** (4-8 hours on GPU)

```bash
# Train base model
python scripts/base_train.py \
    --depth=12 \
    --num_iterations=10000 \
    --run=my_base_model

# Evaluate
python scripts/base_eval.py \
    --checkpoint=base_checkpoints/d12/step_010000
```

**Expected results**:
- Training loss: ~1.2
- Validation bpb: ~1.0
- CORE metric: ~0.60-0.65

#### 2. **Supervised Fine-Tuning** (1-2 hours on GPU)

```bash
# Fine-tune for chat
python scripts/chat_sft.py \
    --base_checkpoint=base_checkpoints/d12/step_010000 \
    --depth=12 \
    --num_iterations=5000 \
    --run=my_sft_model

# Test interactively
python scripts/chat_cli.py \
    --checkpoint=chat_checkpoints/my_sft_model/step_005000
```

**Expected results**:
- SFT loss: ~0.8
- Model follows instructions
- Responses in correct format

#### 3. **Reinforcement Learning** (1-2 hours on GPU)

```bash
# Apply RL alignment
python scripts/chat_rl.py \
    --sft_checkpoint=chat_checkpoints/my_sft_model/step_005000 \
    --depth=12 \
    --num_iterations=2000 \
    --run=my_final_chatbot

# Final test
python scripts/chat_cli.py \
    --checkpoint=chat_checkpoints/my_final_chatbot/step_002000
```

**Expected results**:
- Average reward: ~0.7
- More helpful responses
- Better alignment with preferences

#### 4. **Deploy**

```bash
# Launch web interface
python scripts/chat_web.py \
    --checkpoint=chat_checkpoints/my_final_chatbot/step_002000 \
    --port=8000
```

Visit `http://localhost:8000` and chat with your model!

---

## 📈 Training Timeline & Costs

### Small Model (depth=12, ~100M params)

| Stage | Time | GPU | Cost |
|-------|------|-----|------|
| Base Training | 8 hours | 1x A100 | ~$25 |
| SFT | 2 hours | 1x A100 | ~$6 |
| RL | 2 hours | 1x A100 | ~$6 |
| **Total** | **12 hours** | **1x A100** | **~$37** |

### Medium Model (depth=20, ~400M params)

| Stage | Time | GPU | Cost |
|-------|------|-----|------|
| Base Training | 3 days | 8x A100 | ~$500 |
| SFT | 8 hours | 8x A100 | ~$70 |
| RL | 8 hours | 8x A100 | ~$70 |
| **Total** | **3.5 days** | **8x A100** | **~$640** |

### Large Model (depth=36, ~2B params)

| Stage | Time | GPU | Cost |
|-------|------|-----|------|
| Base Training | 2 weeks | 64x A100 | ~$8,000 |
| SFT | 2 days | 64x A100 | ~$1,000 |
| RL | 2 days | 64x A100 | ~$1,000 |
| **Total** | **2.5 weeks** | **64x A100** | **~$10,000** |

---

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Reduce batch size
--device_batch_size=8  # instead of 32

# Reduce sequence length
--max_seq_len=1024  # instead of 2048
```

#### Loss Not Decreasing
```bash
# Check learning rate
--matrix_lr=0.01  # Try lower

# Check data quality
# Make sure tokenized_data/ has enough examples
```

#### SFT Model Not Following Instructions
```bash
# Train longer
--num_iterations=10000  # instead of 5000

# Check data format
# Ensure chat_data/ has correct message format
```

#### RL Rewards Not Improving
```bash
# Adjust KL coefficient
--kl_coef=0.05  # Lower allows more exploration

# Check reward model quality
# May need better preference data
```

---

## 📚 Additional Resources

### Configuration Files

All scripts support config files:
```bash
# Create config
cat > my_config.py << EOF
depth = 12
num_iterations = 10000
device_batch_size = 32
run = "my_experiment"
EOF

# Use config
python scripts/base_train.py --config=my_config.py
```

### Weights & Biases Logging

Enable wandb logging:
```bash
python scripts/base_train.py \
    --run=my_experiment  # Any name except "dummy"
```

### Mid-Training Checkpoints

For very long training runs, use `mid_train.py` to continue:
```bash
python scripts/mid_train.py \
    --checkpoint=base_checkpoints/d12/step_005000 \
    --num_iterations=5000  # Train 5000 more steps
```

---

## 🎓 Learning Path

### For Beginners
1. ✅ Complete interactive lessons in `learn/`
2. ✅ Train tiny base model (depth=4)
3. ✅ Run SFT on tiny model
4. ✅ Test in chat_cli

### For Practitioners
1. ✅ Train full base model (depth=12)
2. ✅ Apply SFT with production settings
3. ✅ Run RL alignment
4. ✅ Deploy web interface

### For Researchers
1. ✅ Experiment with architectures
2. ✅ Modify training algorithms
3. ✅ Implement custom evaluation
4. ✅ Contribute improvements

---

## 🎉 Next Steps

**You now have a complete guide to the entire pipeline!**

Start here:
1. Train a small base model
2. Fine-tune it for chat
3. Chat with your creation!

```bash
# The complete journey in 3 commands:
python scripts/base_train.py --depth=4 --num_iterations=1000
python scripts/chat_sft.py --base_checkpoint=base_checkpoints/d4/step_001000 --num_iterations=500
python scripts/chat_cli.py --checkpoint=chat_checkpoints/.../step_000500
```

**Happy training!** 🚀🤖✨

# 📖 Nanochat Glossary - ML Terms for Python Engineers

A comprehensive glossary translating machine learning concepts into terms familiar to Python engineers.

---

## A

**Activation Function**
- **What**: Function applied element-wise to introduce non-linearity
- **Python analogy**: `lambda x: max(0, x)` for ReLU
- **Why**: Without it, stacked layers would just be one linear transformation
- **In nanochat**: ReLU² (squared ReLU) in feed-forward layers

**AdamW**
- **What**: Adaptive Moment Estimation with Weight Decay
- **Python analogy**: Smart learning rate adjuster per parameter
- **Why**: Works well with most neural networks, requires less tuning
- **In nanochat**: Used for embeddings and output layer

**Attention**
- **What**: Mechanism to focus on relevant parts of input
- **Python analogy**: Weighted dictionary lookup where weights are learned
- **Why**: Lets model understand context (e.g., "it" refers to "cat")
- **In nanochat**: Core of transformer blocks

**Autoregressive**
- **What**: Predicting next item based on previous items
- **Python analogy**: `for i in range(len(seq)): next = predict(seq[:i])`
- **Why**: How language models generate text one token at a time
- **In nanochat**: Used in `engine.py` for generation

---

## B

**Backpropagation (Backward Pass)**
- **What**: Computing gradients by chain rule from output to input
- **Python analogy**: Automatic differentiation (like sympy but for tensors)
- **Why**: Tells us how to adjust weights to reduce loss
- **In nanochat**: `loss.backward()` - PyTorch handles it!

**Batch**
- **What**: Multiple examples processed together
- **Python analogy**: `list[examples]` processed in parallel
- **Why**: Faster on GPUs, more stable gradients
- **In nanochat**: `device_batch_size` parameter

**Batch Normalization / LayerNorm / RMSNorm**
- **What**: Normalize activations to stabilize training
- **Python analogy**: `(x - mean) / std` but learned
- **Why**: Prevents exploding/vanishing gradients
- **In nanochat**: RMSNorm used (simpler, no learnable params)

**Bits Per Byte (BPB)**
- **What**: Compression metric for language models
- **Python analogy**: How many bits needed to encode each byte
- **Why**: Lower = better text understanding
- **In nanochat**: Used in `loss_eval.py`

**BPE (Byte Pair Encoding)**
- **What**: Tokenization algorithm
- **Python analogy**: Iteratively merge frequent character pairs
- **Why**: Balances vocabulary size vs. sequence length
- **In nanochat**: Used in `tokenizer.py` via rustbpe

---

## C

**Causal Mask**
- **What**: Prevents attention to future tokens
- **Python analogy**: `if j > i: attention[i, j] = -inf`
- **Why**: Model shouldn't "cheat" by looking ahead
- **In nanochat**: Implicit in attention implementation

**Checkpoint**
- **What**: Saved model state
- **Python analogy**: `pickle.dump(model.state_dict())`
- **Why**: Resume training, save best model
- **In nanochat**: `checkpoint_manager.py`

**Context Length / Context Window**
- **What**: Maximum sequence length model can process
- **Python analogy**: `max_length` for a sliding window
- **Why**: Limits how much past text model remembers
- **In nanochat**: `sequence_len` parameter (e.g., 2048)

**Cross-Entropy Loss**
- **What**: Measures difference between predicted and true distributions
- **Python analogy**: `-log(probability_of_correct_answer)`
- **Why**: Standard loss for classification/prediction tasks
- **In nanochat**: Used in training loop

---

## D

**DDP (Distributed Data Parallel)**
- **What**: Multi-GPU training strategy
- **Python analogy**: `multiprocessing.Pool` but for GPUs
- **Why**: Faster training with multiple GPUs
- **In nanochat**: `common.py` handles setup

**Depth**
- **What**: Number of transformer blocks/layers
- **Python analogy**: `len(layers)` in a sequential pipeline
- **Why**: Deeper = more capacity to learn complex patterns
- **In nanochat**: `n_layer` parameter (e.g., depth=12 → 12 layers)

**Dtype (Data Type)**
- **What**: Precision of numbers (float32, float16, bfloat16)
- **Python analogy**: `float` vs `numpy.float32` vs `numpy.float16`
- **Why**: Lower precision = faster, less memory, slight accuracy loss
- **In nanochat**: Uses bfloat16 for training (mixed precision)

---

## E

**Embedding**
- **What**: Mapping discrete tokens to continuous vectors
- **Python analogy**: `dict[token_id] = vector` but learned
- **Why**: Neural networks need continuous inputs
- **In nanochat**: `tok_emb` in `gpt.py`

**Epoch**
- **What**: One pass through entire training dataset
- **Python analogy**: One iteration of outer loop over all data
- **Why**: Training typically runs multiple epochs
- **In nanochat**: Not explicitly tracked (uses step count instead)

---

## F

**FLOPs (Floating Point Operations)**
- **What**: Count of arithmetic operations
- **Python analogy**: Counting `+`, `*`, etc. operations
- **Why**: Measures computational cost
- **In nanochat**: `estimate_flops()` in model

**Feed-Forward Network (FFN)**
- **What**: Simple neural network (2 layers + activation)
- **Python analogy**: `f(x) = W2 @ relu(W1 @ x)`
- **Why**: Process information after attention
- **In nanochat**: Part of each transformer block

**Fine-tuning**
- **What**: Training pretrained model on specific task
- **Python analogy**: Continuing training from checkpoint
- **Why**: Adapt general model to specific use case
- **In nanochat**: SFT scripts (`chat_sft.py`)

**Forward Pass**
- **What**: Computing model output from input
- **Python analogy**: `output = model(input)`
- **Why**: Generate predictions
- **In nanochat**: `model(x)` in training loop

---

## G

**Gradient**
- **What**: Derivative of loss w.r.t. parameter
- **Python analogy**: Slope telling "which way and how much to move"
- **Why**: Tells optimizer how to update weights
- **In nanochat**: Computed by `loss.backward()`

**Gradient Accumulation**
- **What**: Sum gradients over multiple mini-batches before updating
- **Python analogy**: `total_grad += batch_grad` N times, then update
- **Why**: Simulate larger batch size with limited memory
- **In nanochat**: `grad_accum_steps` in training

**Gradient Clipping**
- **What**: Limit gradient magnitude
- **Python analogy**: `grad = min(grad, max_grad)`
- **Why**: Prevent exploding gradients
- **In nanochat**: `grad_clip` parameter

**GPT (Generative Pre-trained Transformer)**
- **What**: Decoder-only transformer architecture
- **Python analogy**: Stack of transformer blocks
- **Why**: Powerful for text generation
- **In nanochat**: Core model in `gpt.py`

---

## H

**Head Dimension**
- **What**: Dimension of each attention head
- **Python analogy**: `head_dim = n_embd // n_head`
- **Why**: Divides model capacity across attention heads
- **In nanochat**: Typically 128

**Hidden Dimension / Model Dimension**
- **What**: Size of internal representations
- **Python analogy**: Feature vector length
- **Why**: Larger = more capacity but more compute
- **In nanochat**: `n_embd` (e.g., 768, 1280)

**Hyperparameter**
- **What**: Configuration value set before training
- **Python analogy**: Function parameters but for training
- **Why**: Control model architecture and training behavior
- **In nanochat**: depth, learning rate, batch size, etc.

---

## I

**Inference**
- **What**: Using trained model to make predictions
- **Python analogy**: `result = trained_model(input)` (no training)
- **Why**: Deploy model for actual use
- **In nanochat**: `engine.py` generation

**Iteration / Step**
- **What**: One optimizer update
- **Python analogy**: One loop iteration
- **Why**: Basic unit of training progress
- **In nanochat**: `num_iterations` parameter

---

## K

**KV Cache**
- **What**: Cached key/value tensors for faster generation
- **Python analogy**: Memoization of previous computations
- **Why**: Don't recompute attention for previous tokens
- **In nanochat**: Used in `engine.py`

---

## L

**Layer**
- **What**: Transformation step in neural network
- **Python analogy**: Function in a pipeline
- **Why**: Stack layers to build complex functions
- **In nanochat**: Transformer blocks are layers

**Learning Rate**
- **What**: How much to adjust weights per update
- **Python analogy**: Step size in optimization
- **Why**: Too high = unstable, too low = slow
- **In nanochat**: `matrix_lr`, `embedding_lr`

**Logits**
- **What**: Raw (unnormalized) prediction scores
- **Python analogy**: Scores before `softmax()`
- **Why**: Easier to work with mathematically
- **In nanochat**: Output of model before softmax

**Loss / Loss Function**
- **What**: Measure of prediction error
- **Python analogy**: `error = metric(prediction, target)`
- **Why**: Objective to minimize during training
- **In nanochat**: Cross-entropy loss

---

## M

**Mixed Precision Training**
- **What**: Use lower precision (float16) for some operations
- **Python analogy**: Sometimes use `int` instead of `float` for speed
- **Why**: Faster training, less memory
- **In nanochat**: `autocast_ctx` with bfloat16

**Model Dimension**
- **What**: See "Hidden Dimension"

**Momentum**
- **What**: Weighted average of past gradients
- **Python analogy**: Moving average for smoothing
- **Why**: Stabilizes training, faster convergence
- **In nanochat**: Used in Muon optimizer

**MQA (Multi-Query Attention)**
- **What**: Share key/value heads across query heads
- **Python analogy**: Reuse same lookup table for multiple queries
- **Why**: Faster inference with minimal quality loss
- **In nanochat**: `n_kv_head` < `n_head`

**Muon Optimizer**
- **What**: Momentum-based optimizer for matrices
- **Python analogy**: Custom update rule for 2D tensors
- **Why**: Better for transformer weights
- **In nanochat**: Used for Linear layers in `muon.py`

---

## N

**Next Token Prediction**
- **What**: Predicting the next token in sequence
- **Python analogy**: `seq[i+1] = predict(seq[:i+1])`
- **Why**: Core training objective for language models
- **In nanochat**: Training task

**Normalization**
- **What**: Rescale values to have specific mean/variance
- **Python analogy**: Z-score normalization
- **Why**: Stabilizes training
- **In nanochat**: RMSNorm used

---

## O

**Optimizer**
- **What**: Algorithm for updating model weights
- **Python analogy**: Search strategy in optimization
- **Why**: How model learns from gradients
- **In nanochat**: Muon and AdamW

**Overfitting**
- **What**: Model memorizes training data, doesn't generalize
- **Python analogy**: `if x == training_example: return cached_answer`
- **Why**: Model performs poorly on new data
- **In nanochat**: Monitor via validation loss

---

## P

**Parameter**
- **What**: Learnable weight in model
- **Python analogy**: Variable being optimized
- **Why**: What the model learns
- **In nanochat**: All `nn.Linear` weights, embeddings, etc.

**Perplexity**
- **What**: Exponentiated cross-entropy loss
- **Python analogy**: `exp(loss)`
- **Why**: Interpretable as "effective vocabulary size"
- **In nanochat**: Alternative metric to loss

**Positional Encoding**
- **What**: Add position information to embeddings
- **Python analogy**: `embedding[i] += position_vector[i]`
- **Why**: Model needs to know word order
- **In nanochat**: Uses rotary embeddings (RoPE)

**Prompt**
- **What**: Input text to start generation
- **Python analogy**: Seed for text generation
- **Why**: Conditions model output
- **In nanochat**: Input to `engine.generate()`

---

## Q

**Quantization**
- **What**: Reduce precision of weights (e.g., 32-bit → 8-bit)
- **Python analogy**: `int(x * 255)` instead of `float(x)`
- **Why**: Smaller model size, faster inference
- **In nanochat**: Not implemented by default

**Query, Key, Value (QKV)**
- **What**: Three projections in attention mechanism
- **Python analogy**: Query = "what to look for", Key = "what I have", Value = "my content"
- **Why**: Flexible information routing
- **In nanochat**: `c_q`, `c_k`, `c_v` in attention

---

## R

**Regularization**
- **What**: Techniques to prevent overfitting
- **Python analogy**: Penalty for complex models
- **Why**: Better generalization
- **In nanochat**: Weight decay, dropout (minimal)

**Residual Connection / Skip Connection**
- **What**: Add input to output: `y = f(x) + x`
- **Python analogy**: `result = transform(data) + data`
- **Why**: Easier gradient flow, helps training deep networks
- **In nanochat**: Used in every transformer block

**RLHF (Reinforcement Learning from Human Feedback)**
- **What**: Fine-tune using human preferences
- **Python analogy**: Learning from user ratings
- **Why**: Align model with human values
- **In nanochat**: `chat_rl.py` script

**RMSNorm (Root Mean Square Normalization)**
- **What**: Simpler normalization: `x / rms(x)`
- **Python analogy**: `x / sqrt(mean(x**2))`
- **Why**: Works well, fewer parameters than LayerNorm
- **In nanochat**: Used throughout model

**RoPE (Rotary Position Embedding)**
- **What**: Encode position via rotation in complex plane
- **Python analogy**: Circular coordinate system
- **Why**: Better for long sequences, relative positions
- **In nanochat**: `apply_rotary_emb()` in attention

---

## S

**Sampling**
- **What**: Randomly select next token from distribution
- **Python analogy**: `random.choices(tokens, weights=probs)`
- **Why**: Generate diverse text (vs. greedy selection)
- **In nanochat**: Used in generation with temperature

**Scaling Laws**
- **What**: Relationships between model size, data, compute, and performance
- **Python analogy**: Big O notation for ML
- **Why**: Predict performance from resource investment
- **In nanochat**: Guides model size choices

**Sequence Length**
- **What**: Number of tokens in input/output
- **Python analogy**: `len(token_list)`
- **Why**: Longer = more context but more memory
- **In nanochat**: `max_seq_len` parameter

**Softmax**
- **What**: Convert scores to probabilities
- **Python analogy**: `exp(x) / sum(exp(x))`
- **Why**: Output valid probability distribution
- **In nanochat**: Applied to logits for predictions

**SFT (Supervised Fine-Tuning)**
- **What**: Train on instruction-response pairs
- **Python analogy**: Learning from examples
- **Why**: Make model follow instructions
- **In nanochat**: `chat_sft.py`

**State Dict**
- **What**: Dictionary of all model parameters
- **Python analogy**: `{param_name: tensor_value}`
- **Why**: Save/load model weights
- **In nanochat**: `model.state_dict()`

---

## T

**Temperature**
- **What**: Control randomness in sampling
- **Python analogy**: `probs = softmax(logits / temperature)`
- **Why**: High = creative, low = conservative
- **In nanochat**: Parameter in generation

**Tensor**
- **What**: Multi-dimensional array
- **Python analogy**: `numpy.ndarray` but on GPU with autograd
- **Why**: Fundamental data structure for neural networks
- **In nanochat**: Everything is tensors

**Token**
- **What**: Piece of text (word/subword)
- **Python analogy**: Element from `str.split()` but smarter
- **Why**: Discrete units for model to process
- **In nanochat**: See `tokenizer.py`

**Top-k Sampling**
- **What**: Sample from k most likely tokens
- **Python analogy**: `choices(sorted(tokens)[:k], weights=probs[:k])`
- **Why**: Balance diversity and quality
- **In nanochat**: Used in generation

**Top-p Sampling (Nucleus Sampling)**
- **What**: Sample from smallest set with cumulative probability p
- **Python analogy**: `while sum(probs) < p: add_token()`
- **Why**: Adaptive cutoff based on confidence
- **In nanochat**: Used in generation

**Transformer**
- **What**: Architecture based on attention mechanism
- **Python analogy**: Stack of attention + feed-forward blocks
- **Why**: State-of-the-art for sequence tasks
- **In nanochat**: Core architecture

**Training Loop**
- **What**: Repeated forward-backward-update cycle
- **Python analogy**: `while not converged: train_step()`
- **Why**: How model learns
- **In nanochat**: Main loop in `base_train.py`

---

## U

**Unembedding / LM Head**
- **What**: Convert hidden states back to vocabulary logits
- **Python analogy**: Reverse of embedding lookup
- **Why**: Generate token predictions
- **In nanochat**: `lm_head` in model

---

## V

**Validation Set**
- **What**: Data held out for evaluation
- **Python analogy**: Test data not used in training
- **Why**: Measure generalization, prevent overfitting
- **In nanochat**: Separate from training data

**Vocabulary / Vocab**
- **What**: Set of all possible tokens
- **Python analogy**: `set(all_possible_tokens)`
- **Why**: Defines model's "knowledge" of text pieces
- **In nanochat**: ~50k tokens

---

## W

**Warmup**
- **What**: Gradually increase learning rate at start
- **Python analogy**: Ramp from 0 to target value
- **Why**: Stabilize early training
- **In nanochat**: `warmup_ratio` parameter

**Weight**
- **What**: Learnable parameter (synonym for parameter)
- **Python analogy**: Variable in optimization
- **Why**: What gets updated during training
- **In nanochat**: All model parameters

**Weight Decay**
- **What**: L2 regularization on weights
- **Python analogy**: `loss += lambda * sum(w**2)`
- **Why**: Prevent large weights, improve generalization
- **In nanochat**: `weight_decay` parameter

---

## Z

**Zero Grad**
- **What**: Reset gradients to zero
- **Python analogy**: `gradients.clear()`
- **Why**: Don't accumulate from previous iterations
- **In nanochat**: `model.zero_grad()` in training loop

---

## Common Acronyms

| Acronym | Full Name | Quick Description |
|---------|-----------|-------------------|
| **AI** | Artificial Intelligence | Machines performing intelligent tasks |
| **BPB** | Bits Per Byte | Loss metric for language models |
| **BPE** | Byte Pair Encoding | Tokenization algorithm |
| **DDP** | Distributed Data Parallel | Multi-GPU training |
| **FFN** | Feed-Forward Network | Simple neural network layer |
| **FLOP** | Floating Point Operation | Unit of computation |
| **GPT** | Generative Pre-trained Transformer | Model architecture |
| **GPU** | Graphics Processing Unit | Hardware for training |
| **KV** | Key-Value | Cached attention components |
| **LLM** | Large Language Model | Big text generation model |
| **LR** | Learning Rate | Step size in optimization |
| **MLP** | Multi-Layer Perceptron | Feed-forward network |
| **MQA** | Multi-Query Attention | Efficient attention variant |
| **NLP** | Natural Language Processing | AI for text |
| **OOM** | Out Of Memory | GPU memory exhausted |
| **QKV** | Query, Key, Value | Attention components |
| **RLHF** | Reinforcement Learning from Human Feedback | Alignment method |
| **RMS** | Root Mean Square | Normalization method |
| **RoPE** | Rotary Position Embedding | Position encoding |
| **SFT** | Supervised Fine-Tuning | Training on examples |

---

## Python Engineer Translation Guide

| When You See | Think Of It As |
|--------------|----------------|
| `model(x)` | Forward pass (predict) |
| `loss.backward()` | Compute gradients (automatic) |
| `optimizer.step()` | Update weights (learn) |
| `model.train()` | Set mode to training |
| `model.eval()` | Set mode to inference |
| `torch.no_grad()` | Disable gradient computation |
| `tensor.shape` | Like `array.shape` in NumPy |
| `[B, T, D]` | Batch, Time/Sequence, Dimension |
| `@` | Matrix multiplication |
| `*` | Element-wise multiplication |
| `tensor.to(device)` | Move to GPU/CPU |
| `F.softmax(x, dim=-1)` | Normalize to probabilities |
| `nn.Module` | Base class for models |
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.Embedding(vocab, dim)` | Lookup table |

---

This glossary should help you navigate the ML terminology in nanochat!

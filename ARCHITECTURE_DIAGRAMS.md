# Nanochat Architecture Diagrams

## 1. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

   Raw Text                Tokenization              Model Input
┌────────────┐          ┌──────────────┐         ┌──────────────┐
│"Hello      │          │              │         │              │
│ world!"    │  ─────►  │  [15496,     │  ─────► │  Tensor:     │
│            │          │   995, ...]  │         │  [B, T]      │
└────────────┘          └──────────────┘         └──────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         GPT MODEL                                │
│                                                                  │
│  ┌────────────────┐                                             │
│  │  Token         │  Convert tokens to vectors                  │
│  │  Embedding     │  [B, T] → [B, T, D]                        │
│  └────────┬───────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────┐                                             │
│  │  Transformer   │  Process context                            │
│  │  Block 1       │  Self-Attention + Feed-Forward             │
│  └────────┬───────┘                                             │
│           │                                                      │
│  ┌────────────────┐                                             │
│  │  Transformer   │  Repeated N times                           │
│  │  Block 2       │  (N = depth/num_layers)                    │
│  └────────┬───────┘                                             │
│           │                                                      │
│          ...                                                     │
│           │                                                      │
│  ┌────────────────┐                                             │
│  │  Transformer   │                                             │
│  │  Block N       │                                             │
│  └────────┬───────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────┐                                             │
│  │  Output Layer  │  Predict next token                         │
│  │  (lm_head)     │  [B, T, D] → [B, T, V]                    │
│  └────────┬───────┘                                             │
└───────────┼─────────────────────────────────────────────────────┘
            │
            ▼
    ┌──────────────┐
    │ Predictions  │  Probability for each token in vocab
    │ [B, T, V]    │  V = vocab_size (~50k)
    └──────────────┘
```

Legend:
- B = Batch size
- T = Sequence length (time)
- D = Model dimension (n_embd)
- V = Vocabulary size

---

## 2. Transformer Block Detail

```
┌───────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK                       │
│                                                            │
│  Input: [B, T, D]                                         │
│      │                                                     │
│      ├──────────┐                                         │
│      │          │                                          │
│      │     ┌────▼────────────────┐                        │
│      │     │  Self-Attention     │  "What context matters?"│
│      │     │                     │                         │
│      │     │  Q = W_q × Input    │  Query: What to look for│
│      │     │  K = W_k × Input    │  Key: What I contain   │
│      │     │  V = W_v × Input    │  Value: My information │
│      │     │                     │                         │
│      │     │  Attention =        │                         │
│      │     │    softmax(QK'/√d)V │  Weighted combination  │
│      │     └────┬────────────────┘                        │
│      │          │                                          │
│      └────►(+)◄─┘    Residual Connection                  │
│             │                                              │
│        ┌────▼────┐                                         │
│        │  Norm   │   RMSNorm (stabilize)                  │
│        └────┬────┘                                         │
│             │                                              │
│      ├──────┼──────┐                                      │
│      │      │      │                                       │
│      │  ┌───▼──────────────┐                              │
│      │  │  Feed-Forward    │  "Process information"       │
│      │  │                  │                               │
│      │  │  FFN(x) =        │                               │
│      │  │    W2·ReLU²(W1·x)│  2 layers + activation       │
│      │  │                  │                               │
│      │  └───┬──────────────┘                              │
│      │      │                                              │
│      └────►(+)◄──┘    Residual Connection                 │
│             │                                              │
│        ┌────▼────┐                                         │
│        │  Norm   │   RMSNorm (stabilize)                  │
│        └────┬────┘                                         │
│             │                                              │
│      Output: [B, T, D]                                     │
└─────────────┼─────────────────────────────────────────────┘
              │
              ▼
        To next block
```

---

## 3. Attention Mechanism Visualization

```
Input sequence: "The cat sat on the mat"

Token positions:  0    1    2    3   4    5
                [The] [cat] [sat] [on] [the] [mat]
                  │    │     │    │    │     │
                  ▼    ▼     ▼    ▼    ▼     ▼
               ┌──────────────────────────────────┐
               │        Embedding Layer           │
               └──────────────────────────────────┘
                  │    │     │    │    │     │
                  ▼    ▼     ▼    ▼    ▼     ▼
               ┌──────────────────────────────────┐
               │      Self-Attention              │
               │                                  │
               │  Each position "looks at"        │
               │  all previous positions:         │
               │                                  │
               │  Position 0 (The):    [1.0]     │
               │  Position 1 (cat):    [0.5, 1.0]│
               │  Position 2 (sat):    [0.1, 0.7, 1.0]│
               │  Position 3 (on):     [0.05, 0.3, 0.6, 1.0]│
               │                                  │
               │  Higher values = more attention  │
               └──────────────────────────────────┘
                  │    │     │    │    │     │
                  ▼    ▼     ▼    ▼    ▼     ▼
               Output: Context-aware vectors
```

Example: Position 2 (word "sat") attends strongly to:
- Position 1 ("cat") - the subject
- Position 2 ("sat") - itself
This helps it understand "cat sat" as a unit.

---

## 4. Training Loop Flowchart

```
                    ┌──────────────┐
                    │   START      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Load Data    │
                    └──────┬───────┘
                           │
        ┌──────────────────┴──────────────────┐
        │         TRAINING LOOP               │
        │         (Repeat N times)            │
        │                                     │
        │  ┌────────────────────┐            │
        │  │ 1. Get Batch       │            │
        │  │    x, y = next()   │            │
        │  └────────┬───────────┘            │
        │           │                         │
        │           ▼                         │
        │  ┌────────────────────┐            │
        │  │ 2. Forward Pass    │            │
        │  │    logits = model(x)│           │
        │  └────────┬───────────┘            │
        │           │                         │
        │           ▼                         │
        │  ┌────────────────────┐            │
        │  │ 3. Compute Loss    │            │
        │  │    loss = CE(logits,y)│         │
        │  └────────┬───────────┘            │
        │           │                         │
        │           ▼                         │
        │  ┌────────────────────┐            │
        │  │ 4. Backward Pass   │            │
        │  │    loss.backward() │            │
        │  │    (Compute grads) │            │
        │  └────────┬───────────┘            │
        │           │                         │
        │           ▼                         │
        │  ┌────────────────────┐            │
        │  │ 5. Optimizer Step  │            │
        │  │    opt.step()      │            │
        │  │    (Update weights)│            │
        │  └────────┬───────────┘            │
        │           │                         │
        │           ▼                         │
        │  ┌────────────────────┐            │
        │  │ 6. Zero Gradients  │            │
        │  │    model.zero_grad()│           │
        │  └────────┬───────────┘            │
        │           │                         │
        │           ▼                         │
        │      Loss < threshold? ────No──┐   │
        │           │                     │   │
        │          Yes                    │   │
        └───────────┼─────────────────────┘   │
                    │                         │
                    ▼                         │
             ┌──────────────┐                │
             │ Save Model   │                │
             └──────┬───────┘                │
                    │                         │
                    ▼                         │
                 ┌────┐                       │
                 │END │                       │
                 └────┘                       │
                    ▲                         │
                    │                         │
                    └─────────────────────────┘
```

---

## 5. Memory Layout

```
GPU Memory Breakdown (example for depth=12, batch=32, seq=2048):

┌────────────────────────────────────────────────────┐
│                  GPU MEMORY                         │
│                                                     │
│  ┌─────────────────────────────────────────┐      │
│  │  Model Parameters                       │  2GB │
│  │  - Embeddings                           │      │
│  │  - Transformer weights                  │      │
│  │  - Output layer                         │      │
│  └─────────────────────────────────────────┘      │
│                                                     │
│  ┌─────────────────────────────────────────┐      │
│  │  Activations (Forward Pass)             │  4GB │
│  │  - Intermediate layer outputs           │      │
│  │  - Attention scores                     │      │
│  └─────────────────────────────────────────┘      │
│                                                     │
│  ┌─────────────────────────────────────────┐      │
│  │  Gradients (Backward Pass)              │  2GB │
│  │  - Gradient for each parameter          │      │
│  └─────────────────────────────────────────┘      │
│                                                     │
│  ┌─────────────────────────────────────────┐      │
│  │  Optimizer States                       │  4GB │
│  │  - Momentum buffers                     │      │
│  │  - Adam first/second moments            │      │
│  └─────────────────────────────────────────┘      │
│                                                     │
│  Total: ~12GB (for this example)                   │
└────────────────────────────────────────────────────┘
```

---

## 6. Distributed Training Architecture

```
Multi-GPU Training (DDP - Distributed Data Parallel):

┌─────────────────────────────────────────────────────────┐
│                    MASTER NODE                          │
│                                                         │
│  GPU 0               GPU 1               GPU 2          │
│  ┌────────┐         ┌────────┐         ┌────────┐     │
│  │ Model  │         │ Model  │         │ Model  │     │
│  │ Copy   │         │ Copy   │         │ Copy   │     │
│  └───┬────┘         └───┬────┘         └───┬────┘     │
│      │                  │                  │           │
│      │ Batch 0          │ Batch 1          │ Batch 2   │
│      │ [data]           │ [data]           │ [data]    │
│      │                  │                  │           │
│      ▼                  ▼                  ▼           │
│  ┌────────┐         ┌────────┐         ┌────────┐     │
│  │Forward │         │Forward │         │Forward │     │
│  │  Pass  │         │  Pass  │         │  Pass  │     │
│  └───┬────┘         └───┬────┘         └───┬────┘     │
│      │                  │                  │           │
│      ▼                  ▼                  ▼           │
│  ┌────────┐         ┌────────┐         ┌────────┐     │
│  │Backward│         │Backward│         │Backward│     │
│  │  Pass  │         │  Pass  │         │  Pass  │     │
│  └───┬────┘         └───┬────┘         └───┬────┘     │
│      │                  │                  │           │
│      │                  │                  │           │
│      └──────────┬───────┴──────────────────┘           │
│                 │                                       │
│                 ▼                                       │
│         ┌───────────────┐                              │
│         │ All-Reduce    │  Average gradients           │
│         │ Gradients     │  across all GPUs             │
│         └───────┬───────┘                              │
│                 │                                       │
│      ┏━━━━━━━━━━┻━━━━━━━━━━┓                          │
│      ▼          ▼           ▼                           │
│  ┌────────┐ ┌────────┐ ┌────────┐                     │
│  │ Update │ │ Update │ │ Update │  All GPUs update     │
│  │ Weights│ │ Weights│ │ Weights│  identically         │
│  └────────┘ └────────┘ └────────┘                     │
│                                                         │
└─────────────────────────────────────────────────────────┘

Result: 3x faster training!
```

---

## 7. Loss Curves (What Good Training Looks Like)

```
Loss over training:

  High │                                 
  Loss │  ●                              Training Loss
       │   ●●                            (should decrease)
       │     ●●                          
       │       ●●●                       
       │          ●●●                    
       │             ●●●●                Validation Loss
       │                ●●●●●            (should decrease, but
       │                     ●●●●        stay close to train)
       │                        ●●●○○○   
  Low  │                           ●●●○○○○
  Loss │─────────────────────────────────────────►
       0                          Steps          N

Good signs:
  ● Training loss decreasing steadily
  ○ Validation loss following training loss
  
Bad signs:
  ✗ Training loss not decreasing → Learning rate too low
  ✗ Training loss exploding → Learning rate too high
  ✗ Validation >> Training → Overfitting
```

---

## 8. Generation Process

```
Text Generation (Autoregressive):

Step 1: Input prompt
   "The capital of" → [15496, 4880, 286]
                            │
                            ▼
Step 2: Model predicts
   [15496, 4880, 286] → Model → [logits for position 3]
                                       │
                                       ▼
                                 Top predictions:
                                 - "France" (30%)
                                 - "Spain" (15%)
                                 - "Italy" (10%)
                                       │
Step 3: Sample/select                 ▼
   Pick "France" (token 4881)
   
Step 4: Append and repeat
   [15496, 4880, 286, 4881] → Model → [logits for position 4]
                                       │
                                       ▼
                                 Top predictions:
                                 - "is" (45%)
                                 - "," (20%)
                                       │
Step 5: Continue...                   ▼
   [15496, 4880, 286, 4881, 318] → Model → ...
   
   "The capital of France is" ...

Repeat until:
  - Max length reached
  - End token generated
  - Stop condition met
```

---

## 9. File Dependencies

```
Main Training Script Flow:

base_train.py
    │
    ├──► gpt.py (Model)
    │      └──► common.py (Utils)
    │
    ├──► dataloader.py (Data)
    │      ├──► tokenizer.py
    │      └──► dataset.py
    │
    ├──► muon.py (Optimizer)
    │
    ├──► adamw.py (Optimizer)
    │
    ├──► loss_eval.py (Validation)
    │      └──► tokenizer.py
    │
    ├──► core_eval.py (Benchmarks)
    │      └──► tasks/*.py
    │
    ├──► checkpoint_manager.py (Save/Load)
    │
    └──► engine.py (Generation)
         └──► tokenizer.py
```

---

## 10. Tensor Shape Transformations

```
Example: Batch size=2, Sequence length=4, Model dim=768, Vocab=50304

Input tokens:
   Shape: [2, 4]
   Example: [[101, 2054, 2003, 1996],
             [102, 2129, 2024, 2017]]
         │
         ▼
Token Embedding:
   Shape: [2, 4, 768]
   (Each token ID → 768-dim vector)
         │
         ▼
Transformer Block 1:
   Shape: [2, 4, 768]
   (Self-attention + FFN, shape unchanged)
         │
         ▼
Transformer Block 2:
   Shape: [2, 4, 768]
   ...
         │
         ▼
Transformer Block N:
   Shape: [2, 4, 768]
         │
         ▼
Output Layer (lm_head):
   Shape: [2, 4, 50304]
   (Each position → scores for all vocab)
         │
         ▼
Softmax:
   Shape: [2, 4, 50304]
   (Convert scores to probabilities)
         │
         ▼
Loss Calculation:
   Compare to targets [2, 4]
   Result: scalar loss value
```

---

These diagrams should help you visualize how everything fits together!
Refer back to these as you go through the learning materials.

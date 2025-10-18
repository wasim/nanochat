# 🧪 Interactive Learning Demos

Hands-on Python scripts to understand nanochat from the ground up.

---

## 🎯 What's Here

Four interactive demos that progressively teach you how nanochat works:

| File | Topic | Time | Level |
|------|-------|------|-------|
| `00_quick_overview.py` | Big picture | 5 min | Beginner |
| `01_tokenization_demo.py` | BPE tokenizer | 10 min | Beginner |
| `02_model_architecture.py` | GPT structure | 15 min | Intermediate |
| `03_training_demo.py` | Training loop | 20 min | Intermediate |

**Total learning time**: ~50 minutes

---

## 🚀 How to Use

### Quick Start
```bash
# Run them in order
python learn/00_quick_overview.py
python learn/01_tokenization_demo.py
python learn/02_model_architecture.py
python learn/03_training_demo.py
```

### With Conda Environment
```bash
# Activate your environment first
conda activate nanochat
python learn/00_quick_overview.py
```

---

## 📖 Demo Breakdown

### **00_quick_overview.py** (5 min)
**What it does**: Birds-eye view of the entire system  
**You'll learn**:
- What nanochat is
- Main components
- Training pipeline overview
- Quick examples

**No computation required** - just explanations

**Run it**: `python learn/00_quick_overview.py`

---

### **01_tokenization_demo.py** (10 min)
**What it does**: Shows how text becomes numbers  
**You'll learn**:
- What is BPE (Byte Pair Encoding)
- How tokenization works
- Vocabulary structure
- Token → text mapping

**Interactive mode**: Try your own text!

**Run it**: `python learn/01_tokenization_demo.py`

**Example output**:
```
Text: "Hello world!"
Tokens: [15496, 995, 0]
```

---

### **02_model_architecture.py** (15 min)
**What it does**: Explores GPT model structure  
**You'll learn**:
- Embeddings (token → vector)
- Transformer blocks
- Attention mechanism
- Parameter counting

**Creates a tiny model** to show shapes

**Run it**: `python learn/02_model_architecture.py`

**Shows**:
- Input: [batch, seq_len]
- Hidden: [batch, seq_len, embed_dim]
- Output: [batch, seq_len, vocab_size]

---

### **03_training_demo.py** (20 min)
**What it does**: Trains a tiny model in real-time  
**You'll learn**:
- Forward pass
- Loss calculation
- Backpropagation
- Gradient updates

**Actually trains for 10 steps** - watch loss go down!

**Run it**: `python learn/03_training_demo.py`

**Example output**:
```
Step 1: Loss = 10.52
Step 2: Loss = 9.84
Step 3: Loss = 9.12
...
Model is learning! 🎉
```

---

## 🎓 Learning Path

### Week 1: Foundations
**Day 1-2**: Run `00_quick_overview.py`
- Understand the big picture
- Know what each component does

**Day 3-4**: Run `01_tokenization_demo.py`
- Play with tokenization
- Try different texts
- Understand vocab

**Day 5-7**: Run `02_model_architecture.py`
- Explore model structure
- Understand tensor shapes
- Count parameters

### Week 2: Training
**Day 8-10**: Run `03_training_demo.py`
- Watch training happen
- Understand loss
- See gradients update

**Day 11-14**: Review all demos
- Run them again
- Experiment with code
- Modify parameters

---

## 🔧 Customization

All demos are **designed to be modified**!

### Experiment Ideas

**01_tokenization_demo.py**:
```python
# Try different texts
text = "Your custom text here"

# Use different tokenizer
tokenizer = Tokenizer("models/tok8192.bin")
```

**02_model_architecture.py**:
```python
# Make model bigger
config.n_layer = 6  # More layers
config.n_head = 8   # More attention heads
```

**03_training_demo.py**:
```python
# Train longer
for step in range(100):  # Instead of 10
    
# Different learning rate
optimizer.lr = 1e-4  # Smaller = slower but more stable
```

---

## 📊 What You'll Understand

After completing all demos:

✅ **Tokenization**: Text → numbers → text  
✅ **Architecture**: How GPT is structured  
✅ **Training**: How models learn from data  
✅ **Tensors**: Shapes and transformations  
✅ **Gradients**: How parameters update  
✅ **Loss**: Measuring model performance  

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'nanochat'"
**Fix**: Make sure you're in the project root:
```bash
cd /path/to/nanochat
python learn/01_tokenization_demo.py
```

Or set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/nanochat:$PYTHONPATH
```

### "FileNotFoundError: tokenizer not found"
**Fix**: Train tokenizer first:
```bash
python scripts/tok_train.py
```

Or use existing one from models/

### "RuntimeError: CUDA out of memory"
**Fix**: Demos use CPU by default, but if modified:
```python
device = "cpu"  # Force CPU
```

---

## � Next Steps

After completing these demos:

1. **Read docs**: Go to `../docs/LEARNING_GUIDE.md`
2. **Train models**: Follow `../docs/COMPLETE_TRAINING_GUIDE.md`
3. **Master plan**: Use `../FOLLOW_THROUGH_GUIDE.md`

---

## 💡 Learning Tips

1. **Run in order** - Each builds on previous
2. **Take notes** - Write down questions
3. **Experiment** - Modify the code!
4. **Repeat** - Run multiple times
5. **Ask questions** - Use docs/GLOSSARY.md

---

## 📦 File Sizes

```
00_quick_overview.py      ~11 KB  (Explanations)
01_tokenization_demo.py   ~5 KB   (BPE demo)
02_model_architecture.py  ~8 KB   (Model structure)
03_training_demo.py       ~9 KB   (Training loop)
```

**Total**: ~33 KB of interactive learning code

---

## 🎯 Learning Objectives

By the end of these demos, you should be able to:

- [ ] Explain what tokenization does
- [ ] Describe GPT architecture components
- [ ] Understand how training reduces loss
- [ ] Read tensor shape transformations
- [ ] Know what embeddings are
- [ ] Understand attention mechanism basics
- [ ] Explain forward vs backward pass
- [ ] Interpret loss curves

---

**These demos are your foundation!**

Everything else builds on these concepts. Take your time, experiment, and have fun! 🎉

Questions? Check:
- `../docs/GLOSSARY.md` - Term definitions
- `../docs/LEARNING_GUIDE.md` - Deeper concepts
- `../docs/ARCHITECTURE_DIAGRAMS.md` - Visual guides
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=100
```

### Medium Model (with GPU)
```bash
python scripts/base_train.py \
    --depth=12 \
    --num_iterations=5000
```

## 🎯 Learning Checklist

- [ ] Run Lesson 1 (Tokenization)
- [ ] Run Lesson 2 (Architecture)
- [ ] Run Lesson 3 (Training)
- [ ] Read LEARNING_GUIDE.md
- [ ] Train your first model
- [ ] Generate text from trained model
- [ ] Experiment with hyperparameters

## 💬 Key Concepts

| Concept | What It Means |
|---------|---------------|
| **Token** | Text split into pieces (like words) |
| **Embedding** | Convert token to vector of numbers |
| **Attention** | Look at context from previous words |
| **Loss** | How wrong the predictions are |
| **Training** | Adjusting weights to reduce loss |

## 🆘 Need Help?

- Check [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for detailed explanations
- Run interactive demos with examples
- Use VS Code debugger (F5) to step through code

Happy Learning! 🚀

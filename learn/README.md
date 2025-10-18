# 🎓 Nanochat Learning Path

Welcome! This guide will teach you language model training from scratch.

## 📚 Interactive Lessons

Run these Python scripts in order to build understanding:

### Lesson 1: Tokenization
```bash
python learn/01_tokenization_demo.py
```
**What you'll learn:**
- How text becomes numbers
- Why subword tokenization?
- Interactive tokenization playground

### Lesson 2: Model Architecture  
```bash
python learn/02_model_architecture.py
```
**What you'll learn:**
- How the GPT model works
- Understanding layers and attention
- Parameter counting and memory

### Lesson 3: Training Loop
```bash
python learn/03_training_demo.py
```
**What you'll learn:**
- The basic training algorithm
- Loss functions and gradients
- Watching a model learn in real-time

## 📖 Comprehensive Guide

See [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for:
- Complete conceptual explanations
- File-by-file breakdown
- Advanced topics
- Troubleshooting tips

## 🚀 Hands-On Training

After completing the lessons, try:

### Small Model (CPU/Mac friendly)
```bash
python scripts/base_train.py \
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

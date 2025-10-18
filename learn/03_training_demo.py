"""
LESSON 3: Understanding Training
=================================
Let's see how the model learns from data.

Run this: python learn/03_training_demo.py
"""

import torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

def main():
    print("=" * 60)
    print("LESSON 3: TRAINING LOOP")
    print("=" * 60)
    
    # Setup
    print("\n🔧 Setting up a tiny model for demo...")
    tokenizer = get_tokenizer()
    
    config = GPTConfig(
        sequence_len=64,
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128
    )
    
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {total_params:,} parameters")
    
    # Create training data
    print("\n" + "─" * 60)
    print("Training Data")
    print("─" * 60)
    
    # Simple training example
    training_text = "The capital of France is Paris. The capital of Germany is Berlin."
    tokens = tokenizer(training_text, prepend="<|bos|>")
    
    print(f"Text: '{training_text}'")
    print(f"Tokens: {tokens}")
    print(f"Length: {len(tokens)} tokens")
    
    # Create input/target pairs
    # Input:  [tok0, tok1, tok2, tok3]
    # Target: [tok1, tok2, tok3, tok4]  (shifted by 1)
    
    input_tokens = torch.tensor([tokens[:-1]])   # All but last
    target_tokens = torch.tensor([tokens[1:]])   # All but first
    
    print(f"\nInput shape:  {input_tokens.shape}")
    print(f"Target shape: {target_tokens.shape}")
    
    print("\nExample: Predicting next token")
    for i in range(min(3, len(tokens) - 1)):
        input_text = tokenizer.decode(tokens[:i+1])
        target_text = tokenizer.decode([tokens[i+1]])
        print(f"  Given: '{input_text}' → Predict: '{target_text}'")
    
    # Understanding Loss
    print("\n" + "─" * 60)
    print("Understanding Loss Function")
    print("─" * 60)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_tokens)  # [batch, seq_len, vocab_size]
        
    print(f"Logits shape: {logits.shape}")
    print(f"  - For each position, we have {config.vocab_size} scores")
    print(f"  - Higher score = model thinks token is more likely")
    
    # Calculate loss
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        target_tokens.view(-1)
    )
    
    print(f"\nInitial loss: {loss.item():.4f}")
    print(f"  - This measures how wrong the predictions are")
    print(f"  - Random guessing would give: {torch.log(torch.tensor(config.vocab_size)):.4f}")
    print(f"  - Lower is better!")
    
    # Show predictions vs targets
    print("\n" + "─" * 60)
    print("Predictions vs Reality (before training)")
    print("─" * 60)
    
    with torch.no_grad():
        probs = F.softmax(logits[0], dim=-1)
        _, predicted_tokens = probs.max(dim=-1)
    
    print("\nFirst 5 predictions:")
    for i in range(min(5, len(target_tokens[0]))):
        target_id = target_tokens[0, i].item()
        pred_id = predicted_tokens[i].item()
        target_text = tokenizer.decode([target_id])
        pred_text = tokenizer.decode([pred_id])
        prob = probs[i, target_id].item()
        
        match = "✓" if target_id == pred_id else "✗"
        print(f"  {match} Position {i}: Target '{target_text:10s}' | "
              f"Predicted '{pred_text:10s}' | "
              f"Prob: {prob:.4f}")
    
    # Training loop
    print("\n" + "─" * 60)
    print("Training Loop (Simplified)")
    print("─" * 60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    print("\n🏋️ Training for 10 steps...")
    print(f"{'Step':<6} {'Loss':<10} {'Accuracy':<10}")
    print("─" * 30)
    
    model.train()
    losses = []
    accuracies = []
    
    for step in range(10):
        # Forward pass
        logits = model(input_tokens)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            target_tokens.view(-1)
        )
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == target_tokens).sum().item()
            total = target_tokens.numel()
            accuracy = correct / total
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        
        print(f"{step:<6} {loss.item():<10.4f} {accuracy*100:<10.2f}%")
    
    # After training
    print("\n" + "─" * 60)
    print("After Training")
    print("─" * 60)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_tokens)
        probs = F.softmax(logits[0], dim=-1)
        predictions = probs.argmax(dim=-1)
    
    print("\nPredictions now:")
    for i in range(min(5, len(target_tokens[0]))):
        target_id = target_tokens[0, i].item()
        pred_id = predictions[i].item()
        target_text = tokenizer.decode([target_id])
        pred_text = tokenizer.decode([pred_id])
        prob = probs[i, target_id].item()
        
        match = "✓" if target_id == pred_id else "✗"
        print(f"  {match} Position {i}: Target '{target_text:10s}' | "
              f"Predicted '{pred_text:10s}' | "
              f"Prob: {prob:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Initial loss:  {losses[0]:.4f}")
    print(f"  Final loss:    {losses[-1]:.4f}")
    print(f"  Reduction:     {losses[0] - losses[-1]:.4f}")
    print(f"\n  Initial acc:   {accuracies[0]*100:.2f}%")
    print(f"  Final acc:     {accuracies[-1]*100:.2f}%")
    print(f"  Improvement:   +{(accuracies[-1] - accuracies[0])*100:.2f}%")
    
    # What happens in real training
    print("\n" + "─" * 60)
    print("Real Training at Scale")
    print("─" * 60)
    
    print("""
What we just did:
  ✓ 10 training steps
  ✓ 1 training example (~20 tokens)
  ✓ Tiny model (< 1M params)
  ✓ Training time: < 1 second

Real GPT training:
  • 100,000+ training steps
  • 10+ billion tokens of data
  • 1B - 175B+ parameters
  • Training time: weeks to months
  • Cost: $100K - $10M+
  
But the SAME basic algorithm:
  1. Forward pass (predict)
  2. Calculate loss (how wrong?)
  3. Backward pass (gradients)
  4. Update weights (learn)
  5. Repeat!
    """)
    
    # Understanding gradients
    print("\n" + "─" * 60)
    print("Understanding Gradients")
    print("─" * 60)
    
    print("""
Gradient = "How should I change this weight to reduce loss?"

Example:
  Weight = 0.5
  Loss = 2.0
  Gradient = -0.3
  
  Interpretation: "Decrease this weight by ~0.3 to reduce loss"
  
  Update: Weight = 0.5 - (learning_rate * 0.3)
                 = 0.5 - (0.001 * 0.3)
                 = 0.4997
  
PyTorch calculates gradients automatically (autograd)!
    """)
    
    # Inspect some gradients
    print("Example gradients from our model:")
    model.zero_grad()
    logits = model(input_tokens)
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        target_tokens.view(-1)
    )
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None and "blocks.0" in name:
            grad_norm = param.grad.norm().item()
            print(f"  {name:40s}: grad norm = {grad_norm:.6f}")
            if grad_norm == 0:
                print("    → No update needed")
            break
    
    # Training tips
    print("\n" + "─" * 60)
    print("Training Tips")
    print("─" * 60)
    
    print("""
1. Learning Rate:
   - Too high: Loss explodes 🔥
   - Too low: Learning too slow 🐌
   - Just right: Smooth improvement 📈

2. Batch Size:
   - Larger: More stable gradients, faster
   - Smaller: More noise, uses less memory
   - Trade-off based on GPU memory

3. Monitoring:
   - Watch training loss (should decrease)
   - Check validation loss (shouldn't increase)
   - If val > train: overfitting!

4. Checkpoints:
   - Save model every N steps
   - Can resume if training crashes
   - Keep best model based on validation

5. Patience:
   - Training takes time
   - Don't stop too early
   - But know when to stop (convergence)
    """)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("""
Training Loop in 4 Steps:

  1. FORWARD PASS
     input → model → predictions
     
  2. COMPUTE LOSS
     compare predictions to targets
     
  3. BACKWARD PASS
     calculate gradients (how to improve)
     
  4. UPDATE WEIGHTS
     adjust model parameters using gradients
     
  Repeat millions of times → Model learns!

The magic: PyTorch handles the hard math (backprop) for you!
    """)
    
    print("✅ You now understand training!")
    print("\n📚 Next steps:")
    print("   - Run a real training: python scripts/base_train.py --depth=4 --num_iterations=100")
    print("   - Monitor loss curves")
    print("   - Experiment with hyperparameters")

if __name__ == "__main__":
    main()

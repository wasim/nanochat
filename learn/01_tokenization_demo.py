"""
LESSON 1: Understanding Tokenization
=====================================
Tokens are how we convert text to numbers that the model can process.

Run this to understand: python learn/01_tokenization_demo.py
"""

from nanochat.tokenizer import get_tokenizer, get_token_bytes
import torch

def main():
    print("=" * 60)
    print("LESSON 1: TOKENIZATION")
    print("=" * 60)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device="cpu")
    
    print(f"\n📚 Vocab size: {tokenizer.get_vocab_size():,} tokens")
    print("   (This means the model knows ~50k different word pieces)")
    
    # Example 1: Simple text
    print("\n" + "─" * 60)
    print("Example 1: Simple English")
    print("─" * 60)
    text1 = "Hello, world!"
    tokens1 = tokenizer(text1)
    print(f"Original text: '{text1}'")
    print(f"Token IDs:     {tokens1}")
    print(f"Num tokens:    {len(tokens1)}")
    print(f"Decoded back:  '{tokenizer.decode(tokens1)}'")
    
    # Example 2: Longer text
    print("\n" + "─" * 60)
    print("Example 2: Full Sentence")
    print("─" * 60)
    text2 = "The quick brown fox jumps over the lazy dog."
    tokens2 = tokenizer(text2)
    print(f"Original text: '{text2}'")
    print(f"Token IDs:     {tokens2}")
    print(f"Num tokens:    {len(tokens2)}")
    
    # Show individual token -> text mappings
    print("\nBreakdown:")
    for i, token_id in enumerate(tokens2):
        text_piece = tokenizer.decode([token_id])
        print(f"  Token {i:2d}: ID {token_id:5d} → '{text_piece}'")
    
    # Example 3: Special tokens
    print("\n" + "─" * 60)
    print("Example 3: Special Tokens")
    print("─" * 60)
    text3 = "Hello"
    tokens3 = tokenizer(text3, prepend="<|bos|>", append="<|eos|>")
    print(f"Text: '{text3}'")
    print(f"With special tokens: {tokens3}")
    print("  <|bos|> = Beginning of sequence")
    print("  <|eos|> = End of sequence")
    print(f"Decoded: '{tokenizer.decode(tokens3)}'")
    
    # Example 4: Compare encoding efficiency
    print("\n" + "─" * 60)
    print("Example 4: Why Subword Tokenization?")
    print("─" * 60)
    examples = [
        "dog",
        "dogs",
        "doggy",
        "supercalifragilisticexpialidocious",
    ]
    for word in examples:
        tokens = tokenizer(word)
        print(f"'{word:40s}' → {len(tokens):2d} tokens: {tokens}")
    
    print("\n💡 Key Insight:")
    print("   - Common words = 1 token (efficient)")
    print("   - Rare/long words = multiple tokens")
    print("   - This is why it's called 'subword' tokenization")
    
    # Example 5: Byte-level information
    print("\n" + "─" * 60)
    print("Example 5: Bytes Per Token (for loss calculation)")
    print("─" * 60)
    text5 = "Python programming"
    tokens5 = tokenizer(text5)
    print(f"Text: '{text5}'")
    print(f"Tokens: {tokens5}")
    print(f"Byte lengths for each token:")
    for i, token_id in enumerate(tokens5):
        byte_len = token_bytes[token_id].item()
        text_piece = tokenizer.decode([token_id])
        print(f"  Token {i}: '{text_piece:15s}' = {byte_len} bytes")
    
    total_bytes = sum(token_bytes[t].item() for t in tokens5)
    print(f"\nTotal: {len(tokens5)} tokens = {total_bytes} bytes")
    print(f"Average: {total_bytes / len(tokens5):.2f} bytes per token")
    
    # Example 6: Model input format
    print("\n" + "─" * 60)
    print("Example 6: Model Input Format")
    print("─" * 60)
    text6 = "The capital of France is"
    tokens6 = tokenizer(text6, prepend="<|bos|>")
    
    # Convert to tensor (what the model actually receives)
    tensor = torch.tensor([tokens6])  # Add batch dimension
    print(f"Text: '{text6}'")
    print(f"Token list: {tokens6}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"  - Dimension 0 (batch): {tensor.shape[0]} (we have 1 example)")
    print(f"  - Dimension 1 (sequence): {tensor.shape[1]} (sequence length)")
    print(f"\nActual tensor:\n{tensor}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Try your own text! (Press Ctrl+C to exit)")
    
    try:
        while True:
            user_text = input("\nEnter text to tokenize: ")
            if not user_text.strip():
                continue
            
            user_tokens = tokenizer(user_text)
            print(f"Tokens: {user_tokens}")
            print(f"Count:  {len(user_tokens)}")
            print("Breakdown:")
            for i, token_id in enumerate(user_tokens):
                piece = tokenizer.decode([token_id])
                print(f"  [{i}] {token_id:5d} → '{piece}'")
    except KeyboardInterrupt:
        print("\n\n✅ Done! You now understand tokenization.")

if __name__ == "__main__":
    main()

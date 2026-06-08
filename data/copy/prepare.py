"""
Prepare the copy task dataset.
Special tokens <bos>, <in>, <out>, <eos>, <pad> are treated as single tokens.
Each line in input.txt is a complete training example.
Will save train_examples.pkl and val_examples.pkl containing lists of token sequences,
and meta.pkl containing the encoder and decoder.
"""
import os
import pickle

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

with open(input_file_path, 'r') as f:
    lines = f.read().strip().split('\n')

print(f"Total lines in dataset: {len(lines):,}")

# Get all unique characters (excluding special tokens which we handle separately)
# We'll treat <bos>, <in>, <out>, <eos>, <pad> as special tokens
special_tokens = ['<bos>', '<in>', '<out>', '<eos>', '<pad>']

# Extract all characters from the data, excluding special tokens
chars = set()
for line in lines:
    i = 0
    while i < len(line):
        # Check if we're at a special token
        found_special = False
        for token in special_tokens:
            if line[i:i+len(token)] == token:
                found_special = True
                i += len(token)
                break

        if not found_special:
            if line[i] != '\n':
                chars.add(line[i])
            i += 1

# Sort regular characters and add special tokens at the beginning
chars = sorted(list(chars))
all_tokens = special_tokens + chars
vocab_size = len(all_tokens)
print("special tokens:", special_tokens)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# Create mapping from tokens to integers
stoi = {token: i for i, token in enumerate(all_tokens)}
itos = {i: token for i, token in enumerate(all_tokens)}

print("stoi:", stoi)
print("itos:", itos)

def encode(s):
    """Encode a string to a list of token ids, treating special tokens as single tokens."""
    tokens = []
    i = 0
    while i < len(s):
        # Check if we're at a special token
        found_special = False
        for token in special_tokens:
            if s[i:i+len(token)] == token:
                tokens.append(stoi[token])
                i += len(token)
                found_special = True
                break

        if not found_special:
            # Skip newlines and other characters not in our vocabulary
            if s[i] in stoi:
                tokens.append(stoi[s[i]])
            i += 1

    return tokens

def decode(l):
    """Decode a list of token ids back to a string."""
    return ''.join([itos[i] for i in l])

# Encode each line as a separate example
examples = []
out_token_id = stoi['<out>']
eos_token_id = stoi['<eos>']

for line in lines:
    encoded_line = encode(line)

    # Validate the example:
    # 1. Must contain <out> token
    # 2. Must have at least one character after <out> (before <eos>)
    if out_token_id in encoded_line:
        out_idx = encoded_line.index(out_token_id)
        # Check if there's at least one token after <out> that's not <eos> or <pad>
        after_out = encoded_line[out_idx+1:]
        has_content = any(token not in [eos_token_id, stoi['<pad>']] for token in after_out)

        if has_content:
            examples.append(encoded_line)

print(f"Valid examples: {len(examples):,}")

# Split into train/val (90/10 split)
num_examples = len(examples)
split_idx = int(num_examples * 0.9)
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

print(f"train examples: {len(train_examples):,}")
print(f"val examples: {len(val_examples):,}")

# Save examples to pickle files
train_path = os.path.join(os.path.dirname(__file__), 'train_examples.pkl')
val_path = os.path.join(os.path.dirname(__file__), 'val_examples.pkl')

with open(train_path, 'wb') as f:
    pickle.dump(train_examples, f)

with open(val_path, 'wb') as f:
    pickle.dump(val_examples, f)

print(f"Saved {train_path}")
print(f"Saved {val_path}")

# Save the meta information as well
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Preparation complete!")

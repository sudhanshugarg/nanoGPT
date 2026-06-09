"""
Generate a copy task dataset with 2000 examples.
Each example has the format: <bos><in>STRING<out>STRING<eos>
where STRING is 1-20 random lowercase a-z characters, and input/output are identical.
"""

import os
import random
import string

# Set random seed for reproducibility
random.seed(42)

# Output file path
output_file = os.path.join(os.path.dirname(__file__), 'input.txt')

# Generate 2000 examples
examples = []
for _ in range(2000):
    # Generate random string of 1-20 lowercase letters
    length = random.randint(1, 20)
    random_string = ''.join(random.choices(string.ascii_lowercase, k=length))

    # Create example in the format: <bos><in>STRING<out>STRING<eos>
    example = f"<bos><in>{random_string}<out>{random_string}<eos>"
    examples.append(example)

# Write to input.txt
with open(output_file, 'w') as f:
    f.write('\n'.join(examples))

print(f"Generated {len(examples)} training examples")
print(f"Saved to {output_file}")
print(f"\nFirst 5 examples:")
for example in examples[:5]:
    print(f"  {example}")

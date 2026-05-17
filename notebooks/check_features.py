with open('outputs/selected_features.txt', 'r') as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")
for i, line in enumerate(lines):
    print(f"  [{i}] repr: {repr(line)}")

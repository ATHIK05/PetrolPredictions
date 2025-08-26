import sys

def remove_duplicates(filepath):
    unique_lines = set()
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line: # Only consider non-empty lines for uniqueness check
                    if stripped_line not in unique_lines:
                        unique_lines.add(stripped_line)
                        lines.append(line) # Keep the original line with its ending
                else:
                    lines.append(line) # Keep empty lines as they are

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Duplicates removed from {filepath}. Unique lines saved.")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_to_process = sys.argv[1]
        remove_duplicates(file_to_process)
    else:
        print("Usage: python remove_duplicates.py <filepath>")

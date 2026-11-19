import json

def inspect_nb(path, name):
    print("=" * 60)
    print(f"INSPECTING: {name}")
    print("=" * 60)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb.get('cells', []))}")
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        lines = source.strip().split('\n')
        first_line = lines[0] if lines and lines[0].strip() else "(empty)"
        print(f"Cell {i} [{cell_type}]: {first_line[:90]}")
    return nb

nb1 = inspect_nb(r"C:\Users\Dell\.gemini\antigravity\brain\63fde9ca-f083-42fb-8c6f-3b13ede6c873\.user_uploaded\media_1795079406955_2b232c44.ipynb", "Untitled0.ipynb (Notebook 1 - OCR)")
nb2 = inspect_nb(r"C:\Users\Dell\.gemini\antigravity\brain\63fde9ca-f083-42fb-8c6f-3b13ede6c873\.user_uploaded\media_1795113776136_a5773b26.ipynb", "IntelliPlace_Answer_Grading_Google_Colab.ipynb (Notebook 2 - Evaluation)")

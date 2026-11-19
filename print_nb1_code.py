import json

with open(r"C:\Users\Dell\.gemini\antigravity\brain\63fde9ca-f083-42fb-8c6f-3b13ede6c873\.user_uploaded\media_1795079406955_2b232c44.ipynb", "r", encoding="utf-8") as f:
    nb1 = json.load(f)

for i in [5, 6]:
    print("="*60)
    print(f"NOTEBOOK 1 - CELL {i}")
    print("="*60)
    source = "".join(nb1['cells'][i]['source'])
    print(source)

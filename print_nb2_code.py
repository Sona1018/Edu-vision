import json

with open(r"C:\Users\Dell\.gemini\antigravity\brain\63fde9ca-f083-42fb-8c6f-3b13ede6c873\.user_uploaded\media_1795113776136_a5773b26.ipynb", "r", encoding="utf-8") as f:
    nb2 = json.load(f)

for i in [3, 4, 6, 7]:
    print("="*60)
    print(f"NOTEBOOK 2 - CELL {i}")
    print("="*60)
    source = "".join(nb2['cells'][i]['source'])
    print(source[:1500])
    if len(source) > 1500:
        print("... [TRUNCATED] ...")

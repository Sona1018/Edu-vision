"""
Analyze all 15 uploaded student .txt files.
Extract: student name, enrollment, questions attempted, answer text per question.
"""
import re
import os
import json

upload_dir = r"C:\Users\Dell\.gemini\antigravity\brain\63fde9ca-f083-42fb-8c6f-3b13ede6c873\.user_uploaded"

file_map = {
    "1.txt": "media_1795111441483_913a592a.txt",
    "2.txt": "media_1795111441483_f5333143.txt",
    "3.txt": "media_1795111441483_dc3b4fea.txt",
    "4.txt": "media_1795111441483_4743f754.txt",
    "5.txt": "media_1795111441483_480a29a4.txt",
    "6.txt": "media_1795111441483_b9051155.txt",
    "7.txt": "media_1795111441483_77b5db29.txt",
    "8.txt": "media_1795111441483_84cf74b1.txt",
    "9.txt": "media_1795111441486_67d0bc78.txt",
    "10.txt": "media_1795111441488_4b7482ab.txt",
    "11.txt": "media_1795111441489_ef940a61.txt",
    "12.txt": "media_1795111441492_3a45ee43.txt",
    "13.txt": "media_1795111441490_9ec942ba.txt",
    "14.txt": "media_1795111769229_85c34315.txt",
    "15.txt": "media_1795111769311_3cf7b054.txt",
}

scratch_dir = r"C:\Users\Dell\.gemini\antigravity\brain\63fde9ca-f083-42fb-8c6f-3b13ede6c873\scratch"
os.makedirs(scratch_dir, exist_ok=True)

# Also copy files to workspace for easy access
work_dir = r"d:\IntelliPlace-AI-Answer-Grading-System-main\student_sheets"
os.makedirs(work_dir, exist_ok=True)

results = []
all_questions = set()

for label in sorted(file_map.keys(), key=lambda x: int(x.split('.')[0])):
    fname = file_map[label]
    src = os.path.join(upload_dir, fname)
    dst = os.path.join(work_dir, label)
    
    try:
        with open(src, 'r', encoding='utf-8') as f:
            text = f.read()
    except:
        try:
            with open(src, 'r', encoding='latin-1') as f:
                text = f.read()
        except:
            text = ""
    
    # Copy to workspace
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(text)
    
    if not text.strip():
        results.append({"file": label, "name": "EMPTY FILE", "enrollment": "", "questions": [], "word_count": 0})
        continue
    
    # Extract student info
    name_match = re.search(r'Name[:\s-]*([^\n\r,]+)', text, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else "Unknown"
    
    enroll_match = re.search(r'(?:Enroll(?:ment)?|Roll\s*No\.?)[:\s-]*(\d+)', text, re.IGNORECASE)
    enrollment = enroll_match.group(1).strip() if enroll_match else ""
    
    # If name not found via "Name:", check first few lines
    if name == "Unknown":
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and not re.match(r'^(Q\d|Ques|Assignment|Marketing|MM |UNIT|BTECH|CSE)', line, re.IGNORECASE):
                if re.search(r'[a-zA-Z]{3,}', line) and len(line) < 60:
                    # Could be a name line
                    # Check if it has enrollment number
                    parts = re.split(r'\s{2,}|\t', line)
                    for p in parts:
                        p = p.strip()
                        if re.match(r'^[A-Za-z\s]+$', p) and len(p) > 3:
                            name = p
                            break
                    if name != "Unknown":
                        break
    
    # Find questions
    q_pattern = r'(?:Q(?:ues)?\.?\s*[-:]?\s*(\d+)|(\d+)\s*[.)]\s*(?:What|Define|Explain|Differentiate|Discuss))'
    q_matches = re.finditer(q_pattern, text, re.IGNORECASE)
    questions_found = set()
    for m in q_matches:
        qnum = m.group(1) or m.group(2)
        if qnum:
            qn = int(qnum)
            if 1 <= qn <= 10:
                questions_found.add(qn)
                all_questions.add(qn)
    
    # Also check for "Q Define..." without number (treat as Q1)
    if not questions_found:
        if re.search(r'Q\s+Define', text, re.IGNORECASE):
            questions_found.add(1)
            all_questions.add(1)
    
    results.append({
        "file": label,
        "name": name,
        "enrollment": enrollment,
        "questions": sorted(questions_found),
        "word_count": len(text.split()),
        "line_count": len(text.split('\n'))
    })

# Print summary
print("=" * 70)
print("STUDENT SHEET ANALYSIS SUMMARY")
print("=" * 70)
print(f"\nTotal files: {len(results)}")
print(f"Unique questions across all sheets: {sorted(all_questions)}")
print(f"Total unique questions: {len(all_questions)}\n")

print(f"{'File':<10} {'Name':<25} {'Enrollment':<15} {'Questions':<20} {'Words':<8}")
print("-" * 78)
for r in results:
    qs = ','.join(map(str, r['questions'])) if r['questions'] else 'EMPTY'
    print(f"{r['file']:<10} {r['name'][:24]:<25} {r['enrollment']:<15} {qs:<20} {r.get('word_count',0):<8}")

# Save to JSON for later use
with open(os.path.join(scratch_dir, 'student_analysis.json'), 'w', encoding='utf-8') as f:
    json.dump({"students": results, "all_questions": sorted(all_questions)}, f, indent=2)

print(f"\nSaved analysis to {scratch_dir}/student_analysis.json")
print(f"Copied all files to {work_dir}/")

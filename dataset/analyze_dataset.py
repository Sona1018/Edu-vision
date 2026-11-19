import json
from collections import Counter

with open('intelliplace_dataset_v2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

issues = []
for q in data:
    scores = sorted([a['score'] for a in q['answers']], reverse=True)
    has_high = any(s >= 8 for s in scores)
    has_mid = any(5 <= s <= 7 for s in scores)
    has_low = any(s <= 3 for s in scores)
    
    if not has_high:
        issues.append(f'{q["question_id"]}: no high score (8+)')
    if not has_mid:
        issues.append(f'{q["question_id"]}: no mid range (5-7)')
    if not has_low:
        issues.append(f'{q["question_id"]}: no low score (<=3)')

print(f'Total questions: {len(data)}')
print(f'Issues found: {len(issues)}')
for i in issues:
    print(f'  {i}')

# Check answer count consistency
for q in data:
    if len(q['answers']) != 5:
        print(f'  {q["question_id"]}: has {len(q["answers"])} answers!')

# Per-subject score distributions
subjects = {}
for q in data:
    s = q['subject']
    if s not in subjects:
        subjects[s] = {'count': 0, 'scores': []}
    subjects[s]['count'] += 1
    for a in q['answers']:
        subjects[s]['scores'].append(a['score'])

for s, info in subjects.items():
    print(f'\n{s}: {info["count"]} questions')
    score_dist = Counter(info['scores'])
    for sc in range(1, 11):
        print(f'  Score {sc:2d}: {score_dist.get(sc, 0)} answers')

# Check first words diversity
starts = []
for q in data:
    for a in q['answers']:
        first_word = a['text'].split()[0].lower() if a['text'] else ''
        starts.append(first_word)
start_counts = Counter(starts).most_common(10)
print(f'\nMost common answer starting words:')
for word, count in start_counts:
    print(f'  "{word}": {count}')

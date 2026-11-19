import json

d = json.load(open('dataset/intelliplace_dataset.json', 'r', encoding='utf-8'))
print(f'Total questions: {len(d)}')
print(f'Total answers: {sum(len(q.get("answers", [])) for q in d)}')
subjects = set(q.get('subject', '') for q in d)
print(f'Subjects: {subjects}')
print('---')

for i, q in enumerate(d[:3]):
    print(f'\nQ[{i}]: {q["question_id"]}')
    print(f'  Subject: {q.get("subject", "N/A")}')
    print(f'  Rubric concepts: {len(q["rubric"])}')
    print(f'  Answers: {len(q["answers"])}')
    scores = [a["score"] for a in q["answers"]]
    print(f'  Score range: {min(scores)} - {max(scores)}')
    print(f'  Rubric concept names: {[c["concept_name"] for c in q["rubric"]]}')
    print(f'  Sample answer (first 100 chars): {q["answers"][0]["text"][:100]}...')

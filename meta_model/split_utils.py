import json
import random
from pathlib import Path


def load_question_split(split_file):
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    return data["train"], data["test"]


def generate_question_split(question_ids, split_file, seed=42, test_size=0.2):
    random.seed(seed)

    question_ids = list(set(question_ids))
    random.shuffle(question_ids)

    split_idx = int(len(question_ids) * (1 - test_size))

    train_ids = question_ids[:split_idx]
    test_ids = question_ids[split_idx:]

    split_data = {
        "train": train_ids,
        "test": test_ids
    }

    Path(split_file).parent.mkdir(parents=True, exist_ok=True)
    Path(split_file).write_text(json.dumps(split_data, indent=2), encoding="utf-8")

    print(f"[INFO] Generated new split file: {split_file}")
    print(f"[INFO] Train questions: {len(train_ids)} | Test questions: {len(test_ids)}")

    return train_ids, test_ids


def fixed_split_indices(df, split_file):
    question_ids = df["question_id"].unique().tolist()

    split_path = Path(split_file)

    # 🔥 AUTO FIX LOGIC
    if not split_path.exists():
        print("[WARNING] Split file not found. Generating new one...")
        train_ids, test_ids = generate_question_split(question_ids, split_file)

    else:
        train_ids, test_ids = load_question_split(split_file)

        # Check coverage
        missing = set(question_ids) - set(train_ids) - set(test_ids)

        if missing:
            print("[WARNING] Split file incomplete. Regenerating...")
            train_ids, test_ids = generate_question_split(question_ids, split_file)

    # Map to indices
    train_idx = df[df["question_id"].isin(train_ids)].index.tolist()
    test_idx = df[df["question_id"].isin(test_ids)].index.tolist()

    return train_idx, test_idx
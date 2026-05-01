import argparse
import json
import random
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_rows(rows):
    for i, row in enumerate(rows):
        if "tokens" not in row or "labels" not in row:
            raise ValueError(f"Row {i} must contain 'tokens' and 'labels'.")
        if len(row["tokens"]) != len(row["labels"]):
            raise ValueError(f"Row {i} has mismatched token and label lengths.")


def split_data(rows, train_ratio=0.8, dev_ratio=0.1, seed=42):
    random.seed(seed)
    rows = rows[:]
    random.shuffle(rows)

    n = len(rows)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    train_rows = rows[:train_end]
    dev_rows = rows[train_end:dev_end]
    test_rows = rows[dev_end:]
    return train_rows, dev_rows, test_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to annotated JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    validate_rows(rows)

    train_rows, dev_rows, test_rows = split_data(rows, seed=args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_rows, out_dir / "train.jsonl")
    save_jsonl(dev_rows, out_dir / "dev.jsonl")
    save_jsonl(test_rows, out_dir / "test.jsonl")

    print(f"Saved {len(train_rows)} train rows")
    print(f"Saved {len(dev_rows)} dev rows")
    print(f"Saved {len(test_rows)} test rows")


if __name__ == "__main__":
    main()

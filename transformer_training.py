import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "allenai/scibert_scivocab_uncased"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_label_list(rows):
    labels = sorted({label for row in rows for label in row["labels"]})
    return labels


def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256,
    )

    aligned_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label2id[labels[word_id]])
            else:
                label_ids.append(-100)
            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def compute_metrics_builder(id2label):
    def compute_metrics(prediction_output):
        logits, labels = prediction_output
        preds = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_row, label_row in zip(preds, labels):
            pred_labels = []
            gold_labels = []
            for pred_id, gold_id in zip(pred_row, label_row):
                if gold_id != -100:
                    pred_labels.append(id2label[pred_id])
                    gold_labels.append(id2label[gold_id])
            true_predictions.append(pred_labels)
            true_labels.append(gold_labels)

        print(classification_report(true_labels, true_predictions))
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def rows_to_dataset(rows):
    return Dataset.from_dict({
        "tokens": [row["tokens"] for row in rows],
        "labels": [row["labels"] for row in rows],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output_dir", default="outputs/scibert_quant_ie")
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    test_rows = load_jsonl(args.test)

    label_list = collect_label_list(train_rows + dev_rows + test_rows)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = rows_to_dataset(train_rows)
    dev_dataset = rows_to_dataset(dev_rows)
    test_dataset = rows_to_dataset(test_rows)

    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
    )
    dev_dataset = dev_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )


    training_args = TrainingArguments(
    output_dir="outputs/scibert_quant_ie",
    eval_strategy="epoch",
    save_strategy="epoch",

    # turn these OFF for now
    load_best_model_at_end=False,

    # remove these lines if they exist
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=8,
    learning_rate=2e-5,
    logging_strategy="epoch",
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    print(metrics)
    print("\nValidation results:")
    print(trainer.evaluate())

    print("\nTest results:")
    print(trainer.evaluate(eval_dataset=test_dataset)) 

    print("\nFinal test results")
    metrics = trainer.evaluate(test_dataset)
    print(metrics)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)


if __name__ == "__main__":
    main()

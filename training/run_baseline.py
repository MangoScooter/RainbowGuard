# training/run_baseline.py
# Usage (PowerShell):
#   # Train on Civil Comments (small CPU subset)
#   python training\run_baseline.py --data civil --epochs 1 --save-dir models\baseline_distilbert
#
#   # Train on prepared CSVs made by prepare_custom_dataset.py (columns: text,label)
#   python training\run_baseline.py --data prepared --prepared-dir data\prepared --epochs 2 --save-dir models\my_run
#
# Outputs:
#   - Model + tokenizer saved to --save-dir
#   - Metrics JSON saved to docs\baseline_metrics.json
#   - Classification report (threshold 0.5) saved to docs\baseline_clf_report.txt

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import IntervalStrategy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Baseline toxic classifier (DistilBERT)")
    p.add_argument("--model", default="distilbert-base-uncased", help="HF model name")
    p.add_argument("--data", choices=["civil", "prepared"], default="civil",
                   help="'civil' uses HF civil_comments subset; 'prepared' reads CSVs in --prepared-dir")
    p.add_argument("--prepared-dir", default="data/prepared",
                   help="Dir containing train.csv/validation.csv/test.csv (for --data prepared)")
    p.add_argument("--save-dir", default="models/baseline_distilbert", help="Where to save model/tokenizer")
    p.add_argument("--docs-dir", default="docs", help="Where to write metrics/report files")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--train-bs", type=int, default=8)
    p.add_argument("--eval-bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--civil-train", type=int, default=2000, help="Size of train subset for civil_comments")
    p.add_argument("--civil-val", type=int, default=1000, help="Size of validation subset for civil_comments")
    p.add_argument("--civil-test", type=int, default=1000, help="Size of test subset for civil_comments")
    return p.parse_args()


def load_civil_subset(train_n: int, val_n: int, test_n: int, seed: int) -> DatasetDict:
    ds = load_dataset("civil_comments")

    def to_label(ex):
        ex["label"] = int(ex["toxicity"] >= 0.5)
        return ex

    ds = ds.map(to_label)
    ds = ds.rename_column("text", "sentence")
    keep = ["sentence", "label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    small = {
        "train": ds["train"].shuffle(seed=seed).select(range(train_n)),
        "validation": ds["validation"].shuffle(seed=seed).select(range(val_n)),
        "test": ds["test"].shuffle(seed=seed).select(range(test_n)),
    }
    return DatasetDict(small)


def load_prepared(dir_path: str | Path) -> DatasetDict:
    d = Path(dir_path)
    req = ["train.csv", "validation.csv", "test.csv"]
    for f in req:
        if not (d / f).exists():
            raise FileNotFoundError(f"Missing {(d / f).as_posix()}. "
                                    f"Run prepare_custom_dataset.py first or check the path.")
    def read_csv(p: Path) -> Dataset:
        df = pd.read_csv(p)
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(f"{p} must have columns: text,label")
        # coerce
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df[df["label"].isin([0, 1])]
        df["label"] = df["label"].astype(int)
        return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    return DatasetDict({
        "train": read_csv(d / "train.csv"),
        "validation": read_csv(d / "validation.csv"),
        "test": read_csv(d / "test.csv"),
    })


def tokenize(ds: DatasetDict, model_name: str) -> tuple[DatasetDict, AutoTokenizer, DataCollatorWithPadding]:
    tok = AutoTokenizer.from_pretrained(model_name)

    def _tok(batch):
        # accept either 'sentence' or 'text'
        text_key = "sentence" if "sentence" in batch else "text"
        return tok(batch[text_key], truncation=True)

    tokenized = {}
    for split, subset in ds.items():
        remove_cols = ["sentence"] if "sentence" in subset.column_names else ["text"]
        tokenized[split] = subset.map(_tok, batched=True, remove_columns=remove_cols)
    dd = DatasetDict(tokenized)
    collate = DataCollatorWithPadding(tokenizer=tok)
    return dd, tok, collate


def compute_metrics(eval_pred) -> Dict[str, Any]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # simple positive-class probability from logits
    probs = 1 / (1 + np.exp(-(logits[:, 1] - logits[:, 0])))
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.docs_dir, exist_ok=True)
    os.makedirs(Path(args.save_dir).parent, exist_ok=True)

    # ---------- Load data ----------
    if args.data == "civil":
        raw = load_civil_subset(args.civil_train, args.civil_val, args.civil_test, args.seed)
    else:
        raw = load_prepared(args.prepared_dir)

    tokenized, tokenizer, data_collator = tokenize(raw, args.model)

    # ---------- Model ----------
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # ---------- Training ----------
    training_args = TrainingArguments(
        output_dir="models/tmp_out",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        # transformers 4.56.1 uses eval_strategy
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],  # no wandb
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---------- Evaluate ----------
    test_metrics = trainer.evaluate(tokenized["test"])

    # Also compute a 0.5-threshold classification report on test
    test_logits = trainer.predict(tokenized["test"]).predictions
    probs = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)
    probs_pos = probs[:, 1]
    y_true = np.array(raw["test"]["label"])
    y_pred = (probs_pos >= 0.5).astype(int)

    clf_rep = classification_report(y_true, y_pred, digits=3)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # ---------- Save ----------
    # model + tokenizer
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    # metrics files
    metrics_path = Path(args.docs_dir) / "baseline_metrics.json"
    report_path = Path(args.docs_dir) / "baseline_clf_report.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_metrics": test_metrics,
                "threshold": 0.5,
                "confusion_matrix": cm,
            },
            f,
            indent=2,
        )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(clf_rep)

    print("\n=== Done ===")
    print("Model saved to:", Path(args.save_dir).resolve())
    print("Metrics JSON :", metrics_path.resolve())
    print("Report TXT   :", report_path.resolve())


if __name__ == "__main__":
    main()

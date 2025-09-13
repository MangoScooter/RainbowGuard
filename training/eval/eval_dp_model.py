import os, json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix

# --------- Config (edit if needed) ----------
MODEL_DIR = "outputs/dp_head_v2_strong"   # your saved DP model dir
TEST_CSV  = "data/processed/test.csv"
OUT_TXT   = "docs/day4_dp_head_eval_report.txt"
OUT_JSON  = "docs/day4_dp_head_confusion_matrix.json"
BATCH     = 32
# -------------------------------------------

os.makedirs("docs", exist_ok=True)

# Load test data
df = pd.read_csv(TEST_CSV)
label_col = "label" if "label" in df.columns else ("labels" if "labels" in df.columns else None)
if label_col is None:
    raise ValueError(f"No 'label' or 'labels' column found in {TEST_CSV}.")
df = df.dropna(subset=["text", label_col]).copy()
df[label_col] = df[label_col].astype("int64")
y_true = df[label_col].to_numpy()

# Tokenizer / model
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Build dataset & loader
def to_hf(_df):
    ds = Dataset.from_pandas(_df[["text"]])
    ds = ds.map(lambda b: tok(b["text"], truncation=True), batched=True, remove_columns=["text"])
    return ds.with_format("torch", columns=["input_ids", "attention_mask"])

ds = to_hf(df)
collate = DataCollatorWithPadding(tokenizer=tok)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate, pin_memory=False)

# Inference
preds = []
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
y_pred = np.concatenate(preds)

# Metrics
# Optional names (only used if sizes match); otherwise sklearn will infer numbers 0..N-1
target_names = ["supportive", "harassment", "misgendering", "outing_doxxing"]
kw = {}
if len(np.unique(y_true)) == len(target_names):
    kw["target_names"] = target_names

report_str = classification_report(y_true, y_pred, digits=3, **kw)
cm = confusion_matrix(y_true, y_pred).tolist()

# Save
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(report_str)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"labels": target_names if "target_names" in kw else None, "matrix": cm}, f, indent=2)

print("Saved report to:", OUT_TXT)
print("Saved confusion matrix to:", OUT_JSON)

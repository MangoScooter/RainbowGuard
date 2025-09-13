from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd, numpy as np, json, sys

# ---- Load test CSV and auto-detect label column ----
df = pd.read_csv('data/processed/test.csv')

# Prefer 'labels' if present; else use 'label'; else fail with a clear message.
if 'labels' in df.columns:
    label_col = 'labels'
elif 'label' in df.columns:
    label_col = 'label'
else:
    sys.exit("No label column found. Expected a 'label' or 'labels' column in data/processed/test.csv")

# Drop rows with missing text/labels and force integer labels
df = df.dropna(subset=['text', label_col]).copy()
df[label_col] = df[label_col].astype('int64')

# ---- Build HF dataset with correct label field name ----
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
ds = Dataset.from_pandas(df[['text', label_col]]).rename_column(label_col, 'labels')
ds = ds.map(lambda b: tok(b['text'], truncation=True), batched=True, remove_columns=['text'])

collator = DataCollatorWithPadding(tokenizer=tok)

# ---- Load model (trained if available; else base) ----
try:
    model = AutoModelForSequenceClassification.from_pretrained('outputs/weak_v1', num_labels=4)
except Exception:
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

trainer = Trainer(model=model, data_collator=collator)

# ---- Predict & report ----
pred_logits = trainer.predict(ds).predictions
preds = np.argmax(pred_logits, axis=-1)
y_true = np.array(df[label_col])

id2name = {0:'supportive', 1:'harassment', 2:'misgendering', 3:'outing_doxxing'}
rep = classification_report(y_true, preds, target_names=[id2name[i] for i in range(4)], digits=3, zero_division=0)
cm = confusion_matrix(y_true, preds).tolist()

with open('docs/dayA_weak_class_report.txt','w', encoding='utf-8') as f:
    f.write(rep)
with open('docs/dayA_weak_confusion_matrix.json','w', encoding='utf-8') as f:
    json.dump({'labels': id2name, 'matrix': cm}, f, indent=2)

print(rep)

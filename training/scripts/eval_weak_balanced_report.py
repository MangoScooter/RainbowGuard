from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd, numpy as np, os, json, sys

# ---- Load test CSV and auto-detect label column ----
df = pd.read_csv('data/processed/test.csv')
label_col = 'label' if 'label' in df.columns else ('labels' if 'labels' in df.columns else None)
if label_col is None:
    sys.exit("No label column found. Expected 'label' or 'labels' in data/processed/test.csv")

df = df.dropna(subset=['text', label_col]).copy()
df[label_col] = df[label_col].astype('int64')

# ---- HF dataset (rename to 'labels' for Trainer) ----
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
ds = Dataset.from_pandas(df[['text', label_col]]).rename_column(label_col, 'labels')
ds = ds.map(lambda b: tok(b['text'], truncation=True), batched=True, remove_columns=['text'])
collator = DataCollatorWithPadding(tokenizer=tok)

# ---- Load trained model ----
model = AutoModelForSequenceClassification.from_pretrained('outputs/weak_balanced_v1', num_labels=4)
trainer = Trainer(model=model, data_collator=collator)

# ---- Predict ----
pred_logits = trainer.predict(ds).predictions
y_pred = np.argmax(pred_logits, axis=-1)

# IMPORTANT: use the original column name from the CSV
y_true = np.array(df[label_col])

# ---- Reports ----
target_names = ['supportive','harassment','misgendering','outing_doxxing']
rep = classification_report(y_true, y_pred, target_names=target_names, digits=3, zero_division=0)
cm = confusion_matrix(y_true, y_pred).tolist()

os.makedirs('docs', exist_ok=True)
with open('docs/day3_weak_balanced_class_report.txt','w', encoding='utf-8') as f:
    f.write(rep)
with open('docs/day3_weak_balanced_confusion_matrix.json','w', encoding='utf-8') as f:
    json.dump({'labels': target_names, 'matrix': cm}, f, indent=2)

print(rep)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import DatasetDict, Dataset
import pandas as pd, numpy as np, json, os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

train = pd.read_csv('data/processed/train.csv')
val   = pd.read_csv('data/processed/validation.csv')
test  = pd.read_csv('data/processed/test.csv')

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def tok_fn(b): return tok(b['text'], truncation=True)

ds = DatasetDict({
  'train': Dataset.from_pandas(train[['text','label']]),
  'validation': Dataset.from_pandas(val[['text','label']]),
  'test': Dataset.from_pandas(test[['text','label']])
}).map(tok_fn, batched=True, remove_columns=['text'])

# Start from your Day-2 model if available
model_path = 'outputs/custom_v1'
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
except Exception:
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

def metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    return {'accuracy': acc, 'precision_w': p_w, 'recall_w': r_w, 'f1_w': f1_w}

args = TrainingArguments(
    output_dir='outputs/weak_v1',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy='no',
    logging_steps=100,
    dataloader_pin_memory=False,
    report_to=[]
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds['train'], eval_dataset=ds['validation'],
    tokenizer=tok, compute_metrics=metrics
)

trainer.train()
eval_test = trainer.evaluate(ds['test'])

os.makedirs('docs', exist_ok=True)
with open('docs/dayA_weak_train_metrics.json','w') as f:
    json.dump({k: float(v) for k,v in eval_test.items()}, f, indent=2)

print('Test metrics:', eval_test)

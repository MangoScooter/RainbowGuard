from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import pandas as pd, numpy as np, json, os

# Load processed CSVs
train = pd.read_csv('data/processed/train.csv')
val   = pd.read_csv('data/processed/validation.csv')
test  = pd.read_csv('data/processed/test.csv')

# Minimal privacy-minded preprocessing (truncate + redact basics)
def redact(txt: str) -> str:
    import re
    t = re.sub(r'@\w+', '@user', txt)
    t = re.sub(r'https?://\\S+', '[link]', t)
    t = re.sub(r'\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b', '[phone]', t)
    return t[:480]
for df in (train, val, test):
    df['text'] = df['text'].astype(str).map(redact)

ds = DatasetDict({
  'train': Dataset.from_pandas(train[['text','label']]),
  'validation': Dataset.from_pandas(val[['text','label']]),
  'test': Dataset.from_pandas(test[['text','label']])
})

model_name = 'distilbert-base-uncased'
tok = AutoTokenizer.from_pretrained(model_name)

def tok_fn(batch):
    return tok(batch['text'], truncation=True)
tok_ds = ds.map(tok_fn, batched=True, remove_columns=['text'])

import evaluate
accuracy = evaluate.load('accuracy')
precision = evaluate.load('precision')
recall = evaluate.load('recall')
f1 = evaluate.load('f1')
def metrics(eval_preds):
    import numpy as np
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy.compute(predictions=preds, references=labels)['accuracy'],
        'precision_w': precision.compute(predictions=preds, references=labels, average='weighted')['precision'],
        'recall_w': recall.compute(predictions=preds, references=labels, average='weighted')['recall'],
        'f1_w': f1.compute(predictions=preds, references=labels, average='weighted')['f1'],
    }

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Trainer args that work across versions
args = TrainingArguments(
    output_dir='outputs/custom_privacy_run',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    save_strategy='no',          # avoid version-specific enums
    logging_strategy='steps',
    logging_steps=50,
    dataloader_pin_memory=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds['train'],
    eval_dataset=tok_ds['validation'],
    tokenizer=tok,
    compute_metrics=metrics
)

trainer.train()
eval_test = trainer.evaluate(tok_ds['test'])

os.makedirs('docs', exist_ok=True)
with open('docs/day3_privacy_metrics.json','w') as f:
    json.dump({k: float(v) for k,v in eval_test.items()}, f, indent=2)

print('Test metrics:', eval_test)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import DatasetDict, Dataset
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np, pandas as pd, json, os

train = pd.read_csv('data/processed/train.csv')
val   = pd.read_csv('data/processed/validation.csv')
test  = pd.read_csv('data/processed/test.csv')

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def tok_fn(b): return tok(b['text'], truncation=True)
ds = DatasetDict({
  'train': Dataset.from_pandas(train[['text','label']]),
  'validation': Dataset.from_pandas(val[['text','label']]),
  'test': Dataset.from_pandas(test[['text','label']]),
}).map(tok_fn, batched=True, remove_columns=['text'])

try:
    model = AutoModelForSequenceClassification.from_pretrained('outputs/weak_v1', num_labels=4)
except Exception:
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

# balanced sampler
y = np.array(train['label'])
counts = np.bincount(y, minlength=4)
weights = 1.0/np.maximum(counts,1)
sample_w = weights[y]
sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

def metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {'accuracy':acc, 'precision_w':p_w, 'recall_w':r_w, 'f1_w':f1_w}

args = TrainingArguments(
    output_dir='outputs/weak_balanced_v1_tmp',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    dataloader_pin_memory=False,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],   # will be used by explicit evaluate() call
    tokenizer=tok,
    compute_metrics=metrics
)

# make Trainer use our sampler
trainer._get_train_sampler = lambda *args, **kwargs: sampler

trainer.train()

# explicit eval & save
eval_val  = trainer.evaluate(ds['validation'])
eval_test = trainer.evaluate(ds['test'])

os.makedirs('outputs/weak_balanced_v1', exist_ok=True)
trainer.save_model('outputs/weak_balanced_v1')

os.makedirs('docs', exist_ok=True)
with open('docs/day3_weak_balanced_metrics.json','w') as f:
    json.dump({'val': {k: float(v) for k,v in eval_val.items()},
               'test': {k: float(v) for k,v in eval_test.items()}}, f, indent=2)

print('VAL:', eval_val)
print('TEST:', eval_test)

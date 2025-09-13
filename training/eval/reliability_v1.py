from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import classification_report
import numpy as np, pandas as pd, os, re, json

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
collator = DataCollatorWithPadding(tokenizer=tok)

test = pd.read_csv('data/processed/test.csv')
label_col = 'label' if 'label' in test.columns else 'labels'
ds = Dataset.from_pandas(test[['text', label_col]]).rename_column(label_col, 'labels')
ds = ds.map(lambda b: tok(b['text'], truncation=True), batched=True, remove_columns=['text'])

model = AutoModelForSequenceClassification.from_pretrained('outputs/weak_balanced_v1', num_labels=4)
trainer = Trainer(model=model, data_collator=collator)

logits = trainer.predict(ds).predictions
# softmax
logits = logits - logits.max(axis=1, keepdims=True)
probs  = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
conf   = probs.max(axis=1)
pred   = probs.argmax(axis=1)
y_true = np.array(test[label_col]).astype(int)

# abstain when uncertain
THR = 0.60
pred_abstain = pred.copy()
pred_abstain[conf < THR] = -1

# apply backup rules only to abstained items
texts = test['text'].astype(str).tolist()
rules = {
  3: [r"\b(dox|doxx|doxing|doxxing)\b",
      r"\bout(ing|ed)?\b.*\b(him|her|them|you)\b",
      r"\bposted (address|phone|number|name)\b",
      r"\btold (his|her|their|your) parents\b"],
  2: [r"\bnot a real (boy|girl|man|woman)\b",
      r"\b(real|biological)\s+(man|woman|male|female)\b",
      r"\bdead[- ]?name\b",
      r"\bhe (is|was)\s+a?\s*she\b|\bshe (is|was)\s+a?\s*he\b"],
  1: [r"\b(freak|disgusting|trash|stupid|idiot|gross)\b",
      r"\bgo back\b|\byou don['’]?t belong\b"],
  0: [r"\b(proud of you|you('?| a)re valid|you belong|you matter|we support you)\b"]
}
for i in range(len(texts)):
    if pred_abstain[i] == -1:
        t = texts[i].lower()
        for lab, pats in rules.items():
            if any(re.search(p, t) for p in pats):
                pred_abstain[i] = lab
                break

# fill remaining abstains with original prediction (so we can score end-to-end)
final_pred = np.where(pred_abstain == -1, pred, pred_abstain)

report = classification_report(
    y_true, final_pred,
    target_names=['supportive','harassment','misgendering','outing_doxxing'],
    digits=3, zero_division=0
)

os.makedirs('docs', exist_ok=True)
with open('docs/day3_reliability_report.txt','w', encoding='utf-8') as f:
    f.write(f"abstain_threshold={THR}\n\n{report}")
print(report)

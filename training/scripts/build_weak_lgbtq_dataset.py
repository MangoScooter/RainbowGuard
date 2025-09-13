from datasets import load_dataset
import pandas as pd, re, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

OUT = Path('data/processed'); OUT.mkdir(parents=True, exist_ok=True)

# Load a good-sized slice so it runs on CPU; adjust as you like
ds = load_dataset('civil_comments')

def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r'@\w+', '@user', t)
    t = re.sub(r'https?://\S+', '[link]', t)
    return t.strip()[:480]

# Heuristic keyword sets (v1 — keep simple & precise)
MISGENDER = [
    r"\bnot a real (boy|girl|man|woman)\b",
    r"\bstill a (boy|girl|man|woman)\b",
    r"\bhe (is|was) (a )?she\b", r"\bshe (is|was) (a )?he\b",
    r"\buse (his|her) real name\b", r"\b(deadname|dead-nam)\b"
]
OUTING = [
    r"\b(out|outed|outing)\b.*\b(him|her|them)\b",
    r"\btold (his|her|their) parents\b",
    r"\bposted (his|her|their) (address|number|name)\b",
]
HARASS = [
    r"\b(freak|disgusting|disgusted|ugly|trash|filth|idiot|stupid)\b",
    r"\bgo back\b", r"\byou don['’]t belong\b"
]
SUPPORT = [
    r"\byou're valid\b", r"\bproud of you\b", r"\bhere to help\b",
    r"\bsupport you\b", r"\bwe care\b"
]

def any_match(text, patterns):
    low = text.lower()
    return any(re.search(p, low) for p in patterns)

def weak_label(row):
    t = row['text']
    if any_match(t, OUTING):      return 3   # outing_doxxing
    if any_match(t, MISGENDER):   return 2   # misgendering
    if any_match(t, HARASS):      return 1   # harassment
    if any_match(t, SUPPORT):     return 0   # supportive
    return -1  # unknown

# Build a dataframe with weak labels from train+validation
take_train = ds['train'].to_pandas()[['text']].rename(columns={'text':'text'})
take_val   = ds['validation'].to_pandas()[['text']].rename(columns={'text':'text'})
big = pd.concat([take_train, take_val], ignore_index=True).dropna()
big['text'] = big['text'].map(clean_text)

# Sample to a manageable size (adjust n if you want more)
big = big.sample(n=20000, random_state=42)

big['label'] = big.apply(weak_label, axis=1)
weak = big[big['label'] != -1].copy()

# If you already have hand labels, load them and MERGE (your labels win)
hand = None
try:
    hand = pd.read_csv('data/annotations/to_label.csv')
    hand = hand.rename(columns={c: c.strip().lower() for c in hand.columns})
    name2id = {'supportive':0,'harassment':1,'misgendering':2,'outing_doxxing':3}
    hand['label'] = hand['label_name'].str.strip().str.lower().map(name2id)
    hand = hand[['text','label']].dropna().drop_duplicates()
except Exception:
    pass

combined = weak[['text','label']].drop_duplicates()
if hand is not None and len(hand):
    # Put hand-labeled first so later drop_duplicates keeps your labels
    combined = pd.concat([hand, combined], ignore_index=True).drop_duplicates(subset=['text'], keep='first')

# Split
train_df, tmp = train_test_split(combined, test_size=0.3, stratify=combined['label'], random_state=42)
val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp['label'], random_state=42)

train_df.to_csv(OUT/'train.csv', index=False)
val_df.to_csv(OUT/'validation.csv', index=False)
test_df.to_csv(OUT/'test.csv', index=False)

# Save a tiny report
report = {
    'total_examples': int(len(combined)),
    'counts_by_label': {int(k): int(v) for k,v in combined['label'].value_counts().sort_index().to_dict().items()},
    'splits': {'train': int(len(train_df)), 'validation': int(len(val_df)), 'test': int(len(test_df))}
}
Path('docs').mkdir(exist_ok=True)
with open('docs/dayA_weak_labels_report.json','w') as f:
    json.dump(report, f, indent=2)

print('Done. Weak-labeled sizes:', report)

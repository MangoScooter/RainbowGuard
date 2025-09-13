from datasets import load_dataset
import pandas as pd, re, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json, random

# ---------- Config ----------
POOL_N = 200000          # how many raw comments to scan
TARGET_PER_CLASS = 2000  # cap per class after matching
RANDOM_SEED = 42
# ----------------------------

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

OUT = Path('data/processed'); OUT.mkdir(parents=True, exist_ok=True)
Path('docs').mkdir(exist_ok=True)

# Pattern lists (CONSISTENT NAMES)
SUPPORTIVE = [
    r"\b(proud of you|we support you|i support you)\b",
    r"\byou('?| a)re (so )?(brave|strong|valid|loved|enough)\b",
    r"\byou are not alone\b", r"\byou belong\b", r"\byou matter\b",
    r"\bhere (for|with) you\b", r"\bsending you love\b",
]
HARASSMENT = [
    r"\b(freak|disgusting|filth|trash|stupid|idiot|pervert|weirdo|gross)\b",
    r"\bkill yourself\b", r"\bgo back\b", r"\byou don['’]?t belong\b",
    r"\b(shut up|shut the \w+ up)\b",
]
MISGENDERING = [
    r"\bnot a real (boy|girl|man|woman)\b",
    r"\bstill a (boy|girl|man|woman)\b",
    r"\bborn (a|as) (male|man|boy|female|woman|girl)\b",
    r"\b(real|biological)\s+(man|woman|male|female)\b",
    r"\bhe (is|was)\s+a?\s*she\b", r"\bshe (is|was)\s+a?\s*he\b",
    r"\bdead[- ]?name\b", r"\buse (his|her) real name\b",
    r"\b(trans (men|women) (are|aren't) (men|women))\b",
    r"\bman not a woman\b|\bwoman not a man\b",
    r"\bidentify as (he|she) but\b",
]
OUTING_DOXXING = [
    r"\b(dox|doxx|doxing|doxxing)\b",
    r"\b(out|outed|outing)\b.*\b(him|her|them|you)\b",
    r"\btold (his|her|their|your) parents\b",
    r"\b(posted|shared)\s+(his|her|their|your)\s+(address|phone|number|name)\b",
]

def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r'@\w+', '@user', t)
    t = re.sub(r'https?://\S+', '[link]', t)
    return t.strip()[:480]

def any_match(text: str, patterns) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)

# 0 supportive, 1 harassment, 2 misgendering, 3 outing_doxxing
def weak_label(t: str) -> int:
    if any_match(t, OUTING_DOXXING): return 3
    if any_match(t, MISGENDERING):   return 2
    if any_match(t, HARASSMENT):     return 1
    if any_match(t, SUPPORTIVE):     return 0
    return -1

# ---- Load and sample a large pool ----
ds = load_dataset('civil_comments')
pool = pd.concat([
    ds['train'].to_pandas()[['text']],
    ds['validation'].to_pandas()[['text']],
], ignore_index=True).dropna()
pool['text'] = pool['text'].map(clean_text)

if len(pool) > POOL_N:
    pool = pool.sample(n=POOL_N, random_state=RANDOM_SEED)

pool['label'] = pool['text'].map(weak_label)
weak = pool[pool['label'] != -1].drop_duplicates()

# ---- Merge hand labels if present (hand labels win) ----
try:
    hand = pd.read_csv('data/annotations/to_label.csv')
    hand = hand.rename(columns={c: c.strip().lower() for c in hand.columns})
    name2id = {'supportive':0,'harassment':1,'misgendering':2,'outing_doxxing':3}
    hand['label'] = hand['label_name'].str.strip().str.lower().map(name2id)
    hand = hand[['text','label']].dropna().drop_duplicates()
    combined = pd.concat([hand, weak[['text','label']]], ignore_index=True)\
                 .drop_duplicates(subset=['text'], keep='first')
except Exception:
    combined = weak[['text','label']].copy()

# ---- Balance by capping per class ----
balanced_parts = []
for lab in [0,1,2,3]:
    sub = combined[combined['label']==lab]
    if len(sub) == 0: 
        continue
    if len(sub) > TARGET_PER_CLASS:
        sub = sub.sample(n=TARGET_PER_CLASS, random_state=RANDOM_SEED)
    balanced_parts.append(sub)

balanced = pd.concat(balanced_parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED)

# ---- Stratified split ----
train_df, tmp = train_test_split(balanced, test_size=0.3, stratify=balanced['label'], random_state=RANDOM_SEED)
val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp['label'], random_state=RANDOM_SEED)

OUT.mkdir(parents=True, exist_ok=True)
train_df.to_csv(OUT/'train.csv', index=False)
val_df.to_csv(OUT/'validation.csv', index=False)
test_df.to_csv(OUT/'test.csv', index=False)

report = {
    'counts_by_label': {int(k): int(v) for k,v in balanced['label'].value_counts().sort_index().to_dict().items()},
    'splits': {'train': len(train_df), 'validation': len(val_df), 'test': len(test_df)},
    'pool_n': POOL_N,
    'target_per_class': TARGET_PER_CLASS
}
with open('docs/day3_weak_balanced_report.json','w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print('Balanced counts:', report)

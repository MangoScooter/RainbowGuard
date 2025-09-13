import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json, sys, numpy as np

RAW1 = Path('data/annotations/to_label.csv')
RAW2 = Path('data/annotations/labeled.csv')
LABELS = Path('data/annotations/labels.csv')
OUTDIR = Path('data/processed'); OUTDIR.mkdir(parents=True, exist_ok=True)

lab = pd.read_csv(LABELS)
name2id = {str(r['name']).strip().lower(): int(r['id']) for _, r in lab.iterrows()}

# Load annotations
if RAW2.exists():
    df = pd.read_csv(RAW2)
elif RAW1.exists():
    df = pd.read_csv(RAW1)
else:
    sys.exit("No annotations found: data/annotations/to_label.csv or labeled.csv")

# Normalize columns
df = df.rename(columns={c: c.strip().lower() for c in df.columns})
if 'text' not in df.columns or 'label_name' not in df.columns:
    sys.exit("CSV must have columns: text,label_name")

# Clean + map
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len().between(1, 500)]
df['label_name'] = df['label_name'].astype(str).str.strip().str.lower()

bad = ~df['label_name'].isin(name2id.keys())
if bad.any():
    unknown = sorted(df.loc[bad, 'label_name'].unique())
    sys.exit(f"Unknown labels: {unknown}. Valid: {sorted(name2id.keys())}")

df['label'] = df['label_name'].map(name2id)
df = df[['text','label','label_name']].dropna().drop_duplicates()

N = int(len(df))
if N == 0:
    sys.exit("No usable rows found. Add examples to to_label.csv.")

def can_stratify_frame(frame):
    vc = frame['label'].value_counts()
    return bool(len(vc) > 1 and vc.min() >= 2)

# First split (train vs tmp)
test_size = 0.3 if N >= 10 else 0.5
if can_stratify_frame(df):
    train_df, tmp = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
else:
    train_df, tmp = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)

# Second split (val vs test)
if can_stratify_frame(tmp):
    val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp['label'], random_state=42)
else:
    val_df, test_df = train_test_split(tmp, test_size=0.5, shuffle=True, random_state=42)

# Save CSVs
train_df.to_csv(OUTDIR/'train.csv', index=False)
val_df.to_csv(OUTDIR/'validation.csv', index=False)
test_df.to_csv(OUTDIR/'test.csv', index=False)

# Build a JSON-safe report (cast numpy types -> Python scalars)
counts = df['label_name'].value_counts().to_dict()
counts = {str(k): int(v) for k, v in counts.items()}

report = {
    'counts_total': int(N),
    'counts_by_label': counts,
    'splits': {
        'train': int(len(train_df)),
        'validation': int(len(val_df)),
        'test': int(len(test_df))
    },
    'used_stratify_first': bool(can_stratify_frame(df)),
    'used_stratify_second': bool(can_stratify_frame(tmp))
}

Path('docs').mkdir(exist_ok=True)
with open('docs/day2_split_report.json','w') as f:
    json.dump(report, f, indent=2)
print('Done. Report:', report)

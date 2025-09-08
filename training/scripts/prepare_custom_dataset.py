import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

RAW1 = Path('data/annotations/to_label.csv')      # if you used CSV
RAW2 = Path('data/annotations/labeled.csv')       # if you used Label Studio (CSV export)
LABELS = Path('data/annotations/labels.csv')
OUTDIR = Path('data/processed')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Load labels schema
lab = pd.read_csv(LABELS)
name2id = {r['name']: int(r['id']) for _, r in lab.iterrows()}

# Load whichever file exists
if RAW2.exists():
    df = pd.read_csv(RAW2)
elif RAW1.exists():
    df = pd.read_csv(RAW1)
else:
    raise FileNotFoundError('No annotations found: to_label.csv or labeled.csv')

# Normalize columns
df = df.rename(columns={c: c.strip().lower() for c in df.columns})
assert 'text' in df.columns, 'CSV must have a text column'
assert 'label_name' in df.columns, 'CSV must have a label_name column (schema names)'

# Clean & map labels
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len().between(1, 500)]
df['label_name'] = df['label_name'].str.strip().str.lower()
bad = ~df['label_name'].isin(name2id.keys())
if bad.any():
    raise ValueError(f"Unknown labels found: {sorted(df.loc[bad,'label_name'].unique())}")

df['label'] = df['label_name'].map(name2id)
df = df[['text','label','label_name']].dropna().drop_duplicates()

# Stratified splits
train_df, tmp = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp['label'], random_state=42)

train_df.to_csv(OUTDIR/'train.csv', index=False)
val_df.to_csv(OUTDIR/'validation.csv', index=False)
test_df.to_csv(OUTDIR/'test.csv', index=False)

# Tiny report
report = {
    'counts_total': int(len(df)),
    'counts_by_label': df['label_name'].value_counts().to_dict(),
    'splits': {
        'train': int(len(train_df)),
        'validation': int(len(val_df)),
        'test': int(len(test_df))
    }
}
Path('docs').mkdir(exist_ok=True)
with open('docs/day2_split_report.json','w') as f:
    json.dump(report, f, indent=2)
print('Done. Report:', report)

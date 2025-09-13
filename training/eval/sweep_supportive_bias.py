import os, csv, numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import classification_report

# -------- Config --------
MODEL_DIR = "outputs/dp_head_v2_strong_weighted"  # or "outputs/dp_head_v2_strong"
TEST_CSV  = "data/processed/test.csv"
OUT_CSV   = "docs/day4_supportive_bias_sweep.csv"
SUPPORTIVE_ID = 0
DELTAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
BATCH = 32
# ------------------------

os.makedirs("docs", exist_ok=True)

df = pd.read_csv(TEST_CSV)
lab = "label" if "label" in df.columns else "labels"
y_true = df[lab].astype(int).to_numpy()

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl.to(device)

ds = Dataset.from_pandas(df[["text"]]).map(lambda b: tok(b["text"], truncation=True),
                                           batched=True, remove_columns=["text"]).with_format("torch")
coll = DataCollatorWithPadding(tok)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=coll, pin_memory=False)

def preds_with_delta(delta):
    import numpy as np
    preds=[]
    with torch.no_grad():
        for b in dl:
            b = {k:v.to(device) for k,v in b.items()}
            p = torch.softmax(mdl(**b).logits, dim=-1)
            top_p, top_i = p.max(dim=-1)
            sup_p = p[:, SUPPORTIVE_ID]
            choose_sup = (top_p - sup_p) <= delta
            out = torch.where(choose_sup, torch.full_like(top_i, SUPPORTIVE_ID), top_i)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)

rows=[]
for d in DELTAS:
    y_pred = preds_with_delta(d)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows.append({
        "delta": d,
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
        "supportive_precision": rep.get("0", {}).get("precision", np.nan),
        "supportive_recall":    rep.get("0", {}).get("recall",    np.nan),
        "supportive_f1":        rep.get("0", {}).get("f1-score",  np.nan),
        "harassment_precision": rep.get("1", {}).get("precision", np.nan),
        "harassment_recall":    rep.get("1", {}).get("recall",    np.nan),
        "outing_precision":     rep.get("3", {}).get("precision", np.nan),
        "outing_recall":        rep.get("3", {}).get("recall",    np.nan)
    })

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {OUT_CSV}")
df_out = pd.read_csv(OUT_CSV).sort_values(["supportive_recall","macro_f1"], ascending=[False,False])
print(df_out.head(10).to_string(index=False))

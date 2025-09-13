import os, numpy as np, pandas as pd, torch, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import classification_report

MODEL_DIR = "outputs/dp_head_v2_strong_weighted"   # or dp_head_v2_strong
TEST_CSV  = "data/processed/test.csv"
OUT_TXT   = "docs/day4_dp_head_eval_report_thresh.txt"
SUPPORTIVE_ID = 0      # change if your id2label differs
DELTA = 0.12           # try 0.05–0.20

df = pd.read_csv(TEST_CSV)
lab = "label" if "label" in df.columns else "labels"
y_true = df[lab].astype(int).to_numpy()

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
mdl.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

ds = Dataset.from_pandas(df[["text"]]).map(lambda b: tok(b["text"], truncation=True),
                                           batched=True, remove_columns=["text"]).with_format("torch")
coll = DataCollatorWithPadding(tok)
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=coll, pin_memory=False)

preds = []
with torch.no_grad():
    for b in dl:
        b = {k:v.to(mdl.device) for k,v in b.items()}
        p = torch.softmax(mdl(**b).logits, dim=-1)
        top_p, top_i = p.max(dim=-1)
        sup_p = p[:, SUPPORTIVE_ID]
        choose_sup = (top_p - sup_p) <= DELTA
        preds.append(torch.where(choose_sup, torch.full_like(top_i, SUPPORTIVE_ID), top_i).cpu().numpy())

y_pred = np.concatenate(preds)
rep = classification_report(y_true, y_pred, digits=3)
os.makedirs("docs", exist_ok=True); open(OUT_TXT, "w", encoding="utf-8").write(rep)
print(rep)

import os, json, numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix

CFG = json.load(open("app/inference_config.json", "r", encoding="utf-8"))
MODEL_DIR = CFG["model_dir"]
BIAS = CFG.get("supportive_bias", {"enabled": False})
TEST_CSV  = "data/processed/test.csv"
OUT_TXT   = "docs/day4_final_eval_report.txt"
OUT_JSON  = "docs/day4_final_confusion_matrix.json"

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
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=coll, pin_memory=False)

preds=[]
with torch.no_grad():
    for b in dl:
        b = {k:v.to(device) for k,v in b.items()}
        logits = mdl(**b).logits
        if BIAS.get("enabled", False):
            sid = int(BIAS.get("supportive_id", 0))
            delta = float(BIAS.get("delta", 0.0))
            p = torch.softmax(logits, dim=-1)
            top_p, top_i = p.max(dim=-1)
            sup_p = p[:, sid]
            choose_sup = (top_p - sup_p) <= delta
            out = torch.where(choose_sup, torch.full_like(top_i, sid), top_i)
        else:
            out = logits.argmax(dim=-1)
        preds.append(out.cpu().numpy())
y_pred = np.concatenate(preds)

rep = classification_report(y_true, y_pred, digits=3, zero_division=0)
cm = confusion_matrix(y_true, y_pred).tolist()

os.makedirs("docs", exist_ok=True)
open(OUT_TXT, "w", encoding="utf-8").write(rep + "\n")
json.dump({"matrix": cm}, open(OUT_JSON, "w", encoding="utf-8"), indent=2)

print("Saved:", OUT_TXT)
print("Saved:", OUT_JSON)

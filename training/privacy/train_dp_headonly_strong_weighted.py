import os, json, time, math
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------- Config --------------------
CHECKPOINT = "outputs/weak_balanced_v1"       # starting weights
MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR   = "outputs/dp_head_v2_strong_weighted"
REPORT     = "docs/day4_dp_head_metrics_strong_weighted.json"

# Same privacy as your strong run (ε ~ 0.2)
EPOCHS = 2
BATCH_SIZE = 8
NOISE_MULT = 1.2
MAX_GRAD_NORM = 1.0
SECURE_MODE = False          # set True if you install torchcsprng
LR = 0.2
SEED = 42
# ------------------------------------------------

np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("docs", exist_ok=True); os.makedirs(SAVE_DIR, exist_ok=True)

def load_split(path):
    df = pd.read_csv(path)
    lab = "label" if "label" in df.columns else ("labels" if "labels" in df.columns else None)
    if lab is None: raise ValueError(f"No label column in {path}")
    df = df.dropna(subset=["text", lab]).copy()
    df[lab] = df[lab].astype("int64")
    return df[["text", lab]].rename(columns={lab:"labels"})

train_df = load_split("data/processed/train.csv")
val_df   = load_split("data/processed/validation.csv")
test_df  = load_split("data/processed/test.csv")

tok = AutoTokenizer.from_pretrained(CHECKPOINT if os.path.isdir(CHECKPOINT) else MODEL_NAME)
collate = DataCollatorWithPadding(tokenizer=tok)

def to_hf(df):
    ds = Dataset.from_pandas(df)
    ds = ds.map(lambda b: tok(b["text"], truncation=True), batched=True, remove_columns=["text"])
    return ds.with_format("torch", columns=["input_ids","attention_mask","labels"])

train_ds = to_hf(train_df); val_ds = to_hf(val_df); test_ds = to_hf(test_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.isdir(CHECKPOINT):
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
else:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
model.to(device)

# Freeze everything except classifier head
for name, p in model.named_parameters():
    p.requires_grad = name.startswith("classifier")

# Make DP-compatible if needed
if not ModuleValidator.is_valid(model):
    model = ModuleValidator.fix(model).to(device)
model.train()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=32,        shuffle=False, collate_fn=collate, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=32,        shuffle=False, collate_fn=collate, pin_memory=False)

# ------- class-weighted loss (handle class imbalance) -------
y_train = np.array(train_df["labels"].values)
num_classes = int(max(y_train))+1
counts = np.bincount(y_train, minlength=num_classes)
# Inverse frequency (clip to avoid huge weights if a class is ultra-rare)
weights = 1.0 / np.maximum(counts, 1)
weights = weights / weights.mean()
class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
# ------------------------------------------------------------

head_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(head_params, lr=LR, momentum=0.0)

DELTA = 1.0 / max(len(train_ds), 1)
privacy_engine = PrivacyEngine(secure_mode=SECURE_MODE)
model, optimizer, train_loader = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=train_loader,
    noise_multiplier=NOISE_MULT, max_grad_norm=MAX_GRAD_NORM, grad_sample_mode="hooks",
)

def evaluate(dl):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k,v in batch.items()}
            out = model(**batch)
            ps.append(out.logits.argmax(dim=-1).cpu())
            ys.append(batch["labels"].cpu())
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    a = accuracy_score(y,p)
    pr, rc, f1, _ = precision_recall_fscore_support(y,p,average="weighted",zero_division=0)
    return {"accuracy":a, "precision_w":pr, "recall_w":rc, "f1_w":f1}

# Train
best_f1 = -1.0
for epoch in range(1, EPOCHS+1):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k,v in batch.items()}
        out = model(**batch)
        # Replace default loss with weighted CE
        logits = out.logits
        loss = F.cross_entropy(logits, batch["labels"], weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    val_m = evaluate(val_loader)
    print(f"Epoch {epoch} | val F1_w={val_m['f1_w']:.4f} | eps≈{privacy_engine.get_epsilon(DELTA):.2f}")
    if val_m["f1_w"] > best_f1:
        best_f1 = val_m["f1_w"]
        # unwrap decorated model before saving
        raw_model = getattr(model, "_module", None) or getattr(model, "module", None) or model
        raw_model.save_pretrained(SAVE_DIR)
        tok.save_pretrained(SAVE_DIR)

# Final eval on test
final_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device).eval()
def eval_with(m):
    ys, ps = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            out = m(**batch)
            ps.append(out.logits.argmax(dim=-1).cpu())
            ys.append(batch["labels"].cpu())
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    a = accuracy_score(y,p)
    pr, rc, f1, _ = precision_recall_fscore_support(y,p,average="weighted",zero_division=0)
    return {"accuracy":a,"precision_w":pr,"recall_w":rc,"f1_w":f1}

test_m = eval_with(final_model)
with open(REPORT,"w",encoding="utf-8") as f:
    json.dump({
        "dp":{"epsilon": float(privacy_engine.get_epsilon(DELTA)), "delta": float(DELTA),
              "noise_multiplier": NOISE_MULT, "max_grad_norm": MAX_GRAD_NORM, "secure_mode": SECURE_MODE},
        "class_weights": weights.tolist(),
        "test_metrics": {k: float(v) for k,v in test_m.items()}
    }, f, indent=2)

print("Saved weighted DP model to", SAVE_DIR)
print("Report saved to", REPORT)

import os, json, time, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# ---------- Config ----------
CHECKPOINT = "outputs/weak_balanced_v1"   # start from your strong non-DP model
MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR   = "outputs/dp_head_v1"
REPORT     = "docs/day4_dp_head_metrics.json"
EPOCHS = 3
BATCH_SIZE = 16
LR = 0.2
EPSILON = 8.0
MAX_GRAD_NORM = 1.0
SEED = 42
# ----------------------------

np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("docs", exist_ok=True); os.makedirs(SAVE_DIR, exist_ok=True)

def load_split(path):
    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else ("labels" if "labels" in df.columns else None)
    if label_col is None:
        raise ValueError(f"No label column in {path}")
    df = df.dropna(subset=["text", label_col]).copy()
    df[label_col] = df[label_col].astype("int64")
    return df[["text", label_col]].rename(columns={label_col:"labels"})

train_df = load_split("data/processed/train.csv")
val_df   = load_split("data/processed/validation.csv")
test_df  = load_split("data/processed/test.csv")

tok = AutoTokenizer.from_pretrained(CHECKPOINT if os.path.isdir(CHECKPOINT) else MODEL_NAME)
collate = DataCollatorWithPadding(tokenizer=tok)

def to_hf(df):
    ds = Dataset.from_pandas(df)
    ds = ds.map(lambda b: tok(b["text"], truncation=True), batched=True, remove_columns=["text"])
    ds = ds.with_format("torch", columns=["input_ids","attention_mask","labels"])
    return ds

train_ds = to_hf(train_df); val_ds = to_hf(val_df); test_ds = to_hf(test_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.isdir(CHECKPOINT):
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT).to(device)
else:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4).to(device)

# Freeze everything except classifier head
for name, p in model.named_parameters():
    p.requires_grad = name.startswith("classifier")

# Patch DP-incompatible modules if needed
if not ModuleValidator.is_valid(model):
    model = ModuleValidator.fix(model).to(device)

# ensure training mode before Opacus validates/wraps
model.train()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=collate, pin_memory=False)

head_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(head_params, lr=LR, momentum=0.0)

DELTA = 1.0 / max(len(train_ds), 1)
privacy_engine = PrivacyEngine(secure_mode=False)

def make_private_target_epsilon():
    model.train()
    return privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=train_loader,
        epochs=EPOCHS, target_epsilon=EPSILON, target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM, grad_sample_mode="hooks",
    )

def make_private_fixed_noise(noise_multiplier=0.8):
    model.train()
    return privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=noise_multiplier, max_grad_norm=MAX_GRAD_NORM,
        grad_sample_mode="hooks",
    )

# Try ε-targeting; if it fails, use fixed noise
try:
    model, optimizer, train_loader = make_private_target_epsilon()
except Exception as e:
    print("Epsilon solver failed; falling back to fixed noise. Reason:", repr(e))
    privacy_engine = PrivacyEngine(secure_mode=False)  # reset engine
    model.train()
    model, optimizer, train_loader = make_private_fixed_noise(noise_multiplier=0.8)

def evaluate(dl):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k,v in batch.items()}
            out = model(**batch)
            ps.append(out.logits.argmax(dim=-1).cpu())
            ys.append(batch["labels"].cpu())
    y_true = torch.cat(ys).numpy(); y_pred = torch.cat(ps).numpy()
    acc = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision_w": p_w, "recall_w": r_w, "f1_w": f1_w}

# Cosine LR for the head
total_steps = len(train_loader)*EPOCHS
def lr_at(step):
    return 1e-3 + 0.5*(LR-1e-3)*(1+math.cos(math.pi*step/max(total_steps,1)))

best_f1, step = -1.0, 0
start = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    for batch in train_loader:
        for pg in optimizer.param_groups:
            pg["lr"] = lr_at(step)
        step += 1
        batch = {k: v.to(device) for k,v in batch.items()}
        out = model(**batch)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    val_m = evaluate(val_loader)
    eps_now = privacy_engine.get_epsilon(DELTA)
    print(f"Epoch {epoch} | val F1_w={val_m['f1_w']:.4f} | eps≈{eps_now:.2f}")

    if val_m["f1_w"] > best_f1:
        best_f1 = val_m["f1_w"]
        # unwrap opacus wrapper before saving
        try:
            raw_model = getattr(model, "_module", None) or getattr(model, "module", None) or model
        except Exception:
            raw_model = model
        try:
            privacy_engine.detach()
        except Exception:
            pass
        raw_model.save_pretrained(SAVE_DIR)
        tok.save_pretrained(SAVE_DIR)

elapsed = time.time() - start

# Reload clean model for final test
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
test_m = evaluate(test_loader)
final_eps = privacy_engine.get_epsilon(DELTA)

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump({
        "dp": {"epsilon": float(final_eps), "delta": float(DELTA), "max_grad_norm": MAX_GRAD_NORM, "secure_mode": False},
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "runtime_sec": elapsed,
        "test_metrics": {k: float(v) for k,v in test_m.items()}
    }, f, indent=2)

print("Saved DP (head-only) model to", SAVE_DIR)
print("Report saved to", REPORT)

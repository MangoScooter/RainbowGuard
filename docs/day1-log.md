\## Day 1 Log

\- Set up Python venv and installed: transformers==4.56.1, datasets, evaluate, scikit-learn, accelerate.

\- Verified Transformers signature and corrected TrainingArguments (e.g., `evaluation\_strategy`).

\- Trained baseline classifier with Hugging Face `Trainer`.

\- Ran validation prediction → softmax → threshold sweep (0.05–0.95) to maximize F1.

\- Saved best model and pinned environment (`requirements.txt`).

\- Committed changes and pushed to GitHub (`main`).




import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.multiclass import OneVsRestClassifier

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "../data/train.csv"  # adjust if your notebook is elsewhere
TEXT_COL = "comment_text"

# main binary label (for EDA + stratification)
LABEL_COL = "toxic"

# all six labels for multi-label classification
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

plt.style.use("default")

# -----------------------------
# Load data and basic cleaning
# -----------------------------
df = pd.read_csv(DATA_PATH)

# keep only comment_text + all 6 labels
df = df[[TEXT_COL] + LABEL_COLS].dropna().copy()

# text features
X_text = df[TEXT_COL].astype(str)

# multi-label target matrix (n_samples, 6)
Y = df[LABEL_COLS].astype(int).values

# single-label (toxic) for stratified splitting
y_toxic = df[LABEL_COL].astype(int).values

print("Data shape:", df.shape)
print("Y shape (labels):", Y.shape)
print("y_toxic shape:", y_toxic.shape)

df.head()

# -----------------------------
# EDA: label distribution (toxic)
# -----------------------------
class_counts = df[LABEL_COL].value_counts()
print("Class counts (toxic):\n", class_counts)

plt.figure(figsize=(4, 4))
class_counts.plot(kind="bar")
plt.title("Toxic vs Non-toxic (Counts)")
plt.xlabel("Class (0 = non-toxic, 1 = toxic)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(4, 4))
(df[LABEL_COL].value_counts(normalize=True) * 100).plot(kind="bar")
plt.title("Toxic vs Non-toxic (Percentage)")
plt.xlabel("Class (0 = non-toxic, 1 = toxic)")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.show()

# -----------------------------
# EDA: per-label frequency for all 6 labels
# -----------------------------
label_counts = df[LABEL_COLS].sum()
print("Label positive counts:\n", label_counts)

plt.figure(figsize=(6, 4))
label_counts.plot(kind="bar")
plt.title("Positive count per label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# -----------------------------
# EDA: comment length
# -----------------------------
df["char_len"] = X_text.str.len()

plt.figure(figsize=(6, 4))
plt.hist(df["char_len"], bins=50)
plt.title("Comment length distribution (characters)")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.show()

# toxic vs non-toxic length
non_toxic = df[df[LABEL_COL] == 0]["char_len"]
toxic = df[df[LABEL_COL] == 1]["char_len"]

plt.figure(figsize=(6, 4))
plt.hist(non_toxic, bins=50, alpha=0.7, label="Non-toxic")
plt.hist(toxic, bins=50, alpha=0.7, label="Toxic")
plt.title("Comment length by class (toxic vs non-toxic)")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# -----------------------------
# TF-IDF vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams + bigrams
    max_features=100_000,   # ~80–100k
    min_df=5,
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

print("TF-IDF shape:", X.shape)

# -----------------------------
# Model builder: multi-label One-vs-Rest Logistic Regression
# -----------------------------
def build_model(C=1.0):
    """
    Multi-label model: one-vs-rest Logistic Regression for each label.
    """
    base_clf = LogisticRegression(
        class_weight="balanced",
        C=C,
        max_iter=2000,
        n_jobs=-1
    )
    model = OneVsRestClassifier(base_clf, n_jobs=-1)
    return model

# -----------------------------
# THIS IS MY CV FUNCTION...DONT THINK WE NEED IT BUT ITS HERE
# -----------------------------
# Multi-label cross-validation
# -----------------------------
def run_cv_multilabel(X, Y, y_strat, C=1.0, n_splits=5, random_state=42):
    """
    X: feature matrix (n_samples, n_features)
    Y: label matrix (n_samples, n_labels)
    y_strat: 1D array used for stratification (we use 'toxic')
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    n_labels = Y.shape[1]
    macro_auc_scores = []
    macro_f1_scores = []

    sum_label_auc = np.zeros(n_labels)
    sum_label_f1 = np.zeros(n_labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_strat)):
        print(f"\nFold {fold+1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        model = build_model(C=C)
        model.fit(X_train, Y_train)

        # predicted probabilities for all labels
        Y_proba = model.predict_proba(X_val)  # (n_val, n_labels)
        Y_pred = (Y_proba >= 0.5).astype(int)

        fold_label_aucs = []
        fold_label_f1s = []

        for j in range(n_labels):
            y_true = Y_val[:, j]
            y_p = Y_proba[:, j]
            y_hat = Y_pred[:, j]

            # if label is constant (all 0s) in this fold, roc_auc_score will fail
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_p)
            else:
                auc = np.nan  # skip in macro

            f1 = f1_score(y_true, y_hat, zero_division=0)

            fold_label_aucs.append(auc)
            fold_label_f1s.append(f1)

        fold_label_aucs = np.array(fold_label_aucs, dtype=float)
        macro_auc = np.nanmean(fold_label_aucs)
        macro_f1 = np.mean(fold_label_f1s)

        macro_auc_scores.append(macro_auc)
        macro_f1_scores.append(macro_f1)

        # replace NaNs with 0 for per-label accumulation (rough but OK for summary)
        fold_label_aucs[np.isnan(fold_label_aucs)] = 0.0
        sum_label_auc += fold_label_aucs
        sum_label_f1 += np.array(fold_label_f1s)

        print(f"  Macro AUC: {macro_auc:.4f}, Macro F1: {macro_f1:.4f}")

    # average per-label metrics across folds
    mean_label_auc = sum_label_auc / n_splits
    mean_label_f1 = sum_label_f1 / n_splits

    print("\n===== Per-label Mean Metrics over CV =====")
    for label, auc, f1 in zip(LABEL_COLS, mean_label_auc, mean_label_f1):
        print(f"{label:13s}  AUC: {auc:.4f}   F1: {f1:.4f}")

    print("\n===== Macro Metrics (across labels) =====")
    print(f"Macro AUC: {np.mean(macro_auc_scores):.4f}")
    print(f"Macro F1:  {np.mean(macro_f1_scores):.4f}")

    return macro_auc_scores, macro_f1_scores

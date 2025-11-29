import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

TEXT_COL = "comment_text"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# 1. Load data
df = pd.read_csv("../data/train.csv")
df = df[[TEXT_COL] + LABEL_COLS].dropna().copy()

X_text = df[TEXT_COL].astype(str)
Y = df[LABEL_COLS].astype(int).values

# 2. TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100_000,
    min_df=5,
    stop_words="english"
)
X = vectorizer.fit_transform(X_text)

# 3. Model
def build_model(C=1.0):
    base_clf = LogisticRegression(
        class_weight="balanced",
        C=C,
        max_iter=2000,
        n_jobs=-1
    )
    return OneVsRestClassifier(base_clf, n_jobs=-1)

model = build_model(C=1.0)
model.fit(X, Y)

# Now for the ensemble, you use:
# Y_proba = model.predict_proba(X_val_or_test)

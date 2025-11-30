# imports

# from Kaelyn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# from Preston
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# from Thomas

import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, classification_report


print('imports successful')

path = './project/jigsaw-toxic-comment-classification-challenge/train.csv'



#
# Kaelyn's model
#



TEXT_COL = "comment_text"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# 1. Load data
df = pd.read_csv(path)
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

modelK = build_model(C=1.0)
modelK.fit(X, Y)
print('created model')

# Now for the ensemble, you use:
# Y_proba = model.predict_proba(X_val_or_test)



#
# Preston's Model
#


seed = 1234


def training_split(path_name):
    document = pd.read_csv(path_name)
    return train_test_split(
        document['comment_text'], 
        document[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
        test_size=0.2,
        random_state=seed
    )

def create_models(label_list, model_dict, x_tr, y_tr) -> None:
    for current_label in label_list:
        print(f'creating {current_label} model')
        model = LinearSVC(tol=.0001, fit_intercept=True, C=0.75)
        model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        model.fit(x_tr, y_tr[current_label])
        model_dict[current_label] = model
        print(f'created {current_label} model')


labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
modelsP = dict()

X_tr, X_va, y_tr, y_va = training_split(path)


tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=60000, stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_tr)
X_va_tfidf = tfidf.transform(X_va)

create_models(labels, modelsP, X_train_tfidf, y_tr)
print('models created')



#
# Thomas's model
#


print('running')
train = df#pd.read_csv(path)
#test = pd.read_csv('test.csv')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[^a-z0-9@#%*!?$\-_/ ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train["clean_text"] = train["comment_text"].apply(clean_text)
#test["clean_text"] = test["comment_text"].apply(clean_text)


tfidf2 = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4,6),
    max_features=50000
)

X = tfidf2.fit_transform(train["clean_text"])
#X_test = tfidf.transform(test["clean_text"])
y = train[labels]



#Split training and validation

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

modelT = OneVsRestClassifier(MultinomialNB(alpha=0.1))
print('fitting model')
modelT.fit(X_train, y_train)

y_val_pred_proba = modelT.predict_proba(X_val)
y_val_pred = (y_val_pred_proba > 0.5).astype(int)
print('created model')




# attempting Logistic regression classifier
print('running')


X_tr, X_va, y_tr, y_va = train_test_split(
    df['comment_text'], 
    df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
    test_size=0.2,
    random_state=seed
)
def get_level1_predictions(Kmodel, Pmodel, Tmodel, X):
    # get the predictions for each model
    
    kPred = Kmodel.predict_proba(vectorizer.transform(X))
    #pPred = []

    temp = tfidf.transform(X)
    #for current in labels:
    #    pPred.append(Pmodel[current].predict_proba(temp))
    #pPred = np.transpose(pPred)
    pPred = np.column_stack([Pmodel[label].predict_proba(temp)[:, 1] for label in labels])

    tPred = Tmodel.predict_proba(tfidf2.transform(X))
    
    outputList = np.hstack([kPred, pPred, tPred])
    
    return outputList



X_tr_l1 = []
X_va_l1 = []
X_tr_l1 = get_level1_predictions(modelK, modelsP, modelT, X_tr)
#kPred = modelK.predict_proba(vectorizer.transform(X_tr))
#temp = tfidf.transform(X_tr)
#pPred = np.column_stack([modelsP[label].predict_proba(temp)[:, 1] for label in labels])
#X_tr_l1 = np.hstack([kPred, pPred])
X_va_l1 = get_level1_predictions(modelK, modelsP, modelT, X_va)
print('fitting ensemble')
ensemble = {}
for l in labels:
    model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    model.fit(X_tr_l1, y_tr[l])
    ensemble[l] = model
#print(f'training error: {1 - ensemble.score(X_tr_l1, y_tr)}')
#print(f'validation error: {1 - ensemble.score(X_va_l1, y_va)}')



for current_label in labels:
        print(f'{current_label}:')
        train_error = 1 - ensemble[current_label].score(X_tr_l1, y_tr[current_label])
        print(f'training error: {train_error}')
        val_error = 1 - ensemble[current_label].score(X_va_l1, y_va[current_label])
        print(f'validation error: {val_error}')
        print()

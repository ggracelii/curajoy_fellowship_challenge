# Cyberbullying Detection with Ensemble Machine Learning Methods

# Imports
import re
import nltk
import requests
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

nltk.download('punkt', quiet=True)

# Load and parse dataset
url = "https://raw.githubusercontent.com/eimearfoley/CyberBullyingDetection/refs/heads/master/data/dataset.txt"
lines = requests.get(url).text.splitlines()

data = []
for line in lines:
    m = re.search(r':\s*(True|False)[,\}]?$', line.strip())
    if not m:
        continue
    label = m.group(1) == 'True'
    text = line[:m.start()].strip().strip('"\'')
    data.append((text, label))

df = pd.DataFrame(data, columns=['text', 'label'])

# Preprocess text data
def preprocess(text):
    t = text.lower()
    t = re.sub(r"http\S+|www\S+", '', t)
    t = re.sub(r"[^a-zA-Z\s]", '', t)
    tokens = nltk.word_tokenize(t)
    return ' '.join(tokens)

df['clean'] = df['text'].apply(preprocess)

# TF-IDF vectorizing
vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = vec.fit_transform(df['clean']).toarray()
X = X if isinstance(X, np.ndarray) else X.toarray()
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# SMOTE for handling class imbalance
sm = SMOTE(sampling_strategy=0.8, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Build the ensemble model
clf1 = LogisticRegression(class_weight='balanced', max_iter=300)
clf2 = RandomForestClassifier(class_weight='balanced', n_estimators=200)
clf3 = GaussianNB()
clf4 = GradientBoostingClassifier()
clf5 = RidgeClassifier(class_weight='balanced')

ensemble = StackingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('gnb', clf3),
        ('gb', clf4),
        ('svc', clf5)
    ],
    final_estimator=LogisticRegression(),
    passthrough=True,
    cv=3
)

# Train the ensemble model
param_grid = {
    'final_estimator__C': [1],
    'rf__n_estimators': [100],
    'rf__max_depth': [None]
}

grid = GridSearchCV(
    estimator=ensemble,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1
)

grid.fit(X_res, y_res)
best_ensemble = grid.best_estimator_
print("Best ensemble params:", grid.best_params_)

# Final evaluation
y_prob = best_ensemble.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1s)
best_thresh = thresholds[best_idx]

print(f"\nEnsemble optimal threshold: {best_thresh:.3f} (Precision={precisions[best_idx]:.3f}, Recall={recalls[best_idx]:.3f}, F1={f1s[best_idx]:.3f})")

y_pred = (y_prob > best_thresh).astype(int)

print("\nEnsemble:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Acc {accuracy_score(y_test, y_pred):.3f}, "
      f"Prec {precision_score(y_test, y_pred, zero_division=0):.3f}, "
      f"Rec {recall_score(y_test, y_pred):.3f}, "
      f"F1 {f1_score(y_test, y_pred):.3f}")
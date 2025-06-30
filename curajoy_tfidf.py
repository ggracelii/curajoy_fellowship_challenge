# Cyberbullying Detection with TF-IDF + Forward Feeding Neural Network

# Imports
import re
import nltk
import requests
import numpy as np
import pandas as pd
import tensorflow as tf

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load and parse dataset
url = "https://raw.githubusercontent.com/eimearfoley/CyberBullyingDetection/refs/heads/master/data/dataset.txt"
lines = requests.get(url).text.splitlines()

records = []
for line in lines:
    m = re.search(r':\s*(True|False)[,\}]?$', line.strip())
    if not m:
        continue
    label = m.group(1) == 'True'
    text = line[:m.start()].strip().strip('"\'')
    records.append((text, label))

df = pd.DataFrame(records, columns=['text', 'label'])

# Prpocess text data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"http\S+|www\S+", '', text.lower())
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    tokens = word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)

df['clean'] = df['text'].apply(preprocess)

# TF-IDF vectorizing
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = vectorizer.fit_transform(df['clean']).toarray()
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE for handling class imbalance
sm = SMOTE(sampling_strategy=0.8, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Compute class weights
classes = np.unique(y_res)
cw_values = compute_class_weight('balanced', classes=classes, y=y_res)
class_weight = dict(zip(classes, cw_values))

# Build neural network model
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_t * tf.pow(1 - p_t, gamma) * bce
    return loss

model = Sequential([
    tf.keras.Input(shape=(X_res.shape[1],)),
    Dense(1024, activation='relu'), BatchNormalization(), Dropout(0.5),
    Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
    Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'), BatchNormalization(), Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss(alpha=0.25, gamma=2.0),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# Train the model
model.fit(
    X_res, y_res,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Tune threshold
y_prob = model.predict(X_test).ravel()
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1s)
best_thresh = thresholds[best_idx]

print(f"\nOptimal threshold: {best_thresh:.3f} "
      f"(P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f}, F1={f1s[best_idx]:.3f})")

# Final evaluation
y_pred = (y_prob > best_thresh).astype(int)

print("\nTF-IDF:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Acc: {accuracy_score(y_test, y_pred):.3f}, "
      f"Prec: {precision_score(y_test, y_pred, zero_division=0):.3f}, "
      f"Rec: {recall_score(y_test, y_pred):.3f}, "
      f"F1: {f1_score(y_test, y_pred):.3f}")
# Cyberbullying Detection with BiLSTM

# Imports
import re
import requests
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional,
    GlobalMaxPooling1D, Dense, Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

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

# Process text data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"http\S+|www\S+", '', text.lower())
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    tokens = nltk.word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)

df['clean'] = df['text'].apply(preprocess)

# Tokenization and padding with Keras
MAX_NUM_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean'])
sequences = tokenizer.texts_to_sequences(df['clean'])
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Build BiLSTM model
input_layer = Input(shape=(MAX_LEN,))
x = Embedding(input_dim=MAX_NUM_WORDS, output_dim=128)(input_layer)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), cw))

model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights
)

# Final evaluation
y_pred = (model.predict(X_test).ravel() > 0.5).astype(int)

print("\nBiLSTM Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Acc: {accuracy_score(y_test, y_pred):.3f}, "
      f"Prec: {precision_score(y_test, y_pred, zero_division=0):.3f}, "
      f"Rec: {recall_score(y_test, y_pred):.3f}, "
      f"F1: {f1_score(y_test, y_pred):.3f}")
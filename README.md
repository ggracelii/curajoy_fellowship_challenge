# Cyberbullying Detection Pipeline
This repository contains four machine learning pipelines for cyberbullying detection on social media comments. Each model explores a different approach to text classification, with a focus on maximizing recall while balancing precision and scalability.

## Files
- **`curajoy_bert.py`**
  
Fine-tunes a DistilBERT transformer model using HuggingFace Transformers. This model yields the highest performance and is suited for high-accuracy applications.
  ```bash
  Usage: python curajoy_bert.py
  ```

- **`curajoy_lstm.py`**
  
Trains a BiLSTM model on token-based word embeddings using Keras. This provides a middle ground between traditional and transformer models.

  ```bash
  Usage: python curajoy_lstm.py
  ```

- **`curajoy_tfidf.py`**
  
Builds and evaluates a neural network on TF-IDF features using dense layers and focal loss. Prioritizes higher recall.

  ```bash
  Usage: python curajoy_tfidf.py
  ```

- **`curajoy_ensemble.py`**

Implements a stacked ensemble of classical ML models using TF-IDF features.

  ```bash
  Usage: python curajoy_ensemble.py
  ```

- **`report.pdf`**

Full documentation of the project including dataset details, modeling strategies, evaluation results, and future work.

- **`requirements.txt`**

List of required libraries. Install with:
  ```bash
  pip install -r requirements.txt
  ```

## Author
Grace Li, curaJOY 2025 Impact Fellowship Tech Cohort Finalist

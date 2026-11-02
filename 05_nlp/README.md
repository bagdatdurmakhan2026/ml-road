# 🚀 05_nlp: Natural Language Processing Fundamentals

This module implements core Natural Language Processing (NLP) and text classification algorithms from scratch using NumPy.

## 🛠 Directory Structure

- **`vectorizer.py`**:
  - Custom implementation of the **Bag-of-Words (BoW)** text vectorizer.
  - Extraction of a unique vocabulary from a text corpus.
  - Transformation of text strings into numerical word-frequency vectors.

- **`text_classifier.py`**:
  - Single-layer neural network / linear classifier (`SimpleSpamClassifier`).
  - Implementation of the Forward Pass and loss function (MSE Loss).
  - Model training powered by **Gradient Descent** for binary text classification.

## 🚀 How to Run

1. Test the vectorizer:
   ```bash
   python 05_nlp/vectorizer.py
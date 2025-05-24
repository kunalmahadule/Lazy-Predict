# ⚡ LazyPredict Classifier Demo

This project demonstrates the use of the [`LazyPredict`](https://pypi.org/project/lazypredict/) library to quickly compare the performance of multiple machine learning classifiers on the breast cancer dataset from `sklearn`.

## 📌 What is LazyPredict?

Instead of manually training and evaluating dozens of models, `LazyPredict` runs them all and returns a performance comparison — saving time during the model selection process.

## 🧠 Key Features

- Loads the breast cancer dataset
- Splits the data for training and testing
- Evaluates and compares 25+ ML classifiers using `LazyClassifier`

## 🚀 How to Run

```bash
pip install lazypredict scikit-learn
python lazy_predict.py

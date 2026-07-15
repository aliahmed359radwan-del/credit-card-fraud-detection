# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using supervised classification, built to practice handling severe class imbalance — one of the most common real-world challenges in fraud detection systems.

## Problem

Credit card fraud is extremely rare compared to legitimate transactions, which makes it a genuinely hard classification problem: a model that predicts "not fraud" every single time would still be over 99% accurate while catching zero fraud. This project builds and evaluates models that handle that imbalance properly, rather than relying on misleading accuracy scores.

## Dataset

[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from the Université Libre de Bruxelles (ULB) Machine Learning Group, hosted on Kaggle.

- 284,807 transactions from European cardholders over two days in September 2013
- Only 492 are fraudulent — about 0.17% of all transactions
- Features V1–V28 are PCA-transformed for privacy; only `Time`, `Amount`, and `Class` (the target) are in their original form

The dataset is not included in this repository due to its size — download it from the link above and place it at `data/creditcard.csv` to run the notebook.

## Approach

1. **Exploratory analysis** — checked for missing values, examined the class distribution, and compared transaction amounts between legitimate and fraudulent transactions.
2. **Train/test split before balancing** — the full dataset was split into training and test sets first, preserving the real-world 0.17% fraud rate in the test set. Balancing was applied only to the training data (undersampling legitimate transactions to match the number of fraud cases). This was a deliberate fix over an earlier version of this project, which balanced the entire dataset before splitting — a mistake that inflated test accuracy by testing on an artificially easy 50/50 split instead of realistic, imbalanced data.
3. **Feature scaling** — applied `StandardScaler`, fit only on training data.
4. **Model training** — trained and compared two models: Logistic Regression and Random Forest.
5. **Evaluation** — used precision, recall, F1 score, and a confusion matrix on the untouched, realistically imbalanced test set, since accuracy alone is misleading at this level of class imbalance.

## Results

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.047 | 0.949 | 0.090 |
| Random Forest | 0.037 | 0.949 | 0.071 |

Both models catch roughly 95% of actual fraud cases (recall), but Logistic Regression produces meaningfully fewer false alarms, giving it a better precision and F1 score overall. Random Forest's added complexity did not translate into better performance on this dataset.

**A note on precision:** both models flag a large number of legitimate transactions as fraud (precision is low for both). This is an expected tradeoff of undersampling the training data to a 1:1 class ratio — it makes the model highly sensitive to catching fraud, at the cost of more false positives. In a real deployment, this tradeoff would need to be tuned based on the actual cost of a false alarm versus a missed fraud case (e.g. via classification threshold adjustment or trying `class_weight='balanced'` instead of resampling).

## How to run

1. Download `creditcard.csv` from the [Kaggle dataset link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in a `data/` folder in the project root
2. Install dependencies:
```
pip install numpy pandas scikit-learn joblib
```
3. Open `Creidet Card Fraud Detection.ipynb` and run all cells in order

## Next steps

- Tune the classification threshold to improve precision without sacrificing too much recall
- Try `class_weight='balanced'` as an alternative to undersampling, to compare against the resampled approach
- Add an unsupervised anomaly detection model (e.g. Isolation Forest) trained only on legitimate transactions, and compare its flagged anomalies against the supervised models
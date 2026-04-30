# Credit Risk Scorer

**[🚀 Live Demo → gabetorrijos.github.io/credit-risk-scorer](https://gabetorrijos.github.io/credit-risk-scorer/)**

A machine learning project that predicts loan approval likelihood based on borrower financial data.

## Overview

This project uses logistic regression and random forest models trained on real loan application data to assess credit risk. The model achieves **78.86% accuracy** using logistic regression.

## Key Findings

- **Credit history** is by far the strongest predictor of loan approval
- Logistic regression outperforms random forest on this dataset — simpler models win in finance
- Income alone is a weak predictor; debt-to-income ratio and credit history matter more

## Project Structure

credit-risk-scorer/
├── app/
│   ├── index.html              # Interactive risk scoring UI
│   └── feature_importance.png  # Model visualization
├── data/
│   └── german_credit.csv       # Loan application dataset
├── src/
│   ├── explore.py              # Data exploration
│   ├── preprocess.py           # Data cleaning & feature engineering
│   ├── train.py                # Model training & evaluation
│   └── visualize.py            # Feature importance chart
├── requirements.txt
└── README.md

## Setup

```bash
# Install dependencies
pip3 install -r requirements.txt

# Explore the data
python3 src/explore.py

# Clean and preprocess
python3 src/preprocess.py

# Train and evaluate models
python3 src/train.py

# Generate feature importance chart
python3 src/visualize.py
```

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 78.86% |
| Random Forest | 76.42% |

## Technologies

- Python 3
- pandas — data manipulation
- scikit-learn — machine learning
- matplotlib — visualization
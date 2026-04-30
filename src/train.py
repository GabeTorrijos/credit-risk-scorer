import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.append('.')
from src.preprocess import load_and_clean

# Load clean data
df = load_and_clean('data/german_credit.csv')

# Split into features (X) and target (y)
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data — makes logistic regression converge properly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
print("=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.2%}")
print(classification_report(y_test, lr_preds))

# Model 2: Random Forest (doesn't need scaling)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("\n=== RANDOM FOREST ===")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
print(classification_report(y_test, rf_preds))
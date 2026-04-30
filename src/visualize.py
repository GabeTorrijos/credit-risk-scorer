import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('.')
from src.preprocess import load_and_clean

# Load and prepare data
df = load_and_clean('data/german_credit.csv')
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Get feature importance (coefficients)
importance = pd.Series(model.coef_[0], index=X.columns)
importance = importance.abs().sort_values(ascending=True)

# Plot
plt.figure(figsize=(10, 6))
importance.plot(kind='barh', color='steelblue')
plt.title('Feature Importance - Credit Risk Model', fontsize=14)
plt.xlabel('Importance (absolute coefficient)')
plt.tight_layout()
plt.savefig('app/feature_importance.png', dpi=150)
print("Chart saved to app/feature_importance.png")
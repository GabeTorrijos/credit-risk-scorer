import pandas as pd

def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # Drop Loan_ID — it's just an identifier, not useful for prediction
    df = df.drop(columns=['Loan_ID'])

    # Fill missing values
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

    # Convert text to numbers
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df = pd.get_dummies(df, columns=['Dependents', 'Property_Area'])

    print("=== CLEANED DATA SHAPE ===")
    print(df.shape)
    print("\n=== MISSING VALUES AFTER CLEANING ===")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    df = load_and_clean('data/german_credit.csv')
    print("\n=== PREVIEW ===")
    print(df.head())
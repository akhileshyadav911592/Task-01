import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Remove customerID if present
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric if present
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode target column
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Accuracy
print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

# Classification Report
print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred))

print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred))

# Save best model
if accuracy_score(y_test, rf_pred) > accuracy_score(y_test, lr_pred):
    joblib.dump(rf, "churn_model.pkl")
    print("\nRandom Forest model saved as churn_model.pkl")
else:
    joblib.dump(lr, "churn_model.pkl")
    print("\nLogistic Regression model saved as churn_model.pkl")
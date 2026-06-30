import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Churn Distribution
plt.figure(figsize=(6,4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.savefig("churn_distribution.png")
plt.show()

# Monthly Charges Distribution
plt.figure(figsize=(6,4))
df["MonthlyCharges"].hist(bins=30)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Frequency")
plt.savefig("monthly_charges_distribution.png")
plt.show()

# Tenure Distribution
plt.figure(figsize=(6,4))
df["tenure"].hist(bins=30)
plt.title("Tenure Distribution")
plt.xlabel("Tenure")
plt.ylabel("Frequency")
plt.savefig("tenure_distribution.png")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10,8))
plt.imshow(numeric_df.corr(), aspect="auto")
plt.colorbar()
plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=90)
plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print("EDA completed successfully!")
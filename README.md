# Customer Churn Prediction

## Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a service based on historical customer data. The project uses machine learning algorithms to analyze customer behavior and classify customers as likely to churn or stay.

## Features

* Data preprocessing and cleaning
* Missing value handling
* Exploratory Data Analysis (EDA)
* Data visualization
* Training and evaluation of multiple machine learning models
* Model performance comparison
* Customer churn prediction system
* Interactive web interface using Streamlit

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── train_model.py
├── eda.py
├── customer_churn.csv
├── churn_model.pkl
├── requirements.txt
├── churn_distribution.png
├── monthly_charges_distribution.png
├── tenure_distribution.png
└── correlation_heatmap.png
```

## Dataset

The dataset contains customer information such as:

* Gender
* Senior Citizen Status
* Tenure
* Monthly Charges
* Total Charges
* Contract Details
* Internet Service Information
* Churn Status

## Machine Learning Models Used

1. Logistic Regression
2. Random Forest Classifier

## Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1 Score
* Classification Report

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Customer-Churn-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Model Training

```bash
python train_model.py
```

## Run EDA

```bash
python eda.py
```

## Run the Streamlit Application

```bash
python -m streamlit run app.py
```

## Results

The project compares the performance of multiple machine learning models and selects the best-performing model for customer churn prediction.

## Future Improvements

* Add more machine learning models.
* Improve feature engineering.
* Deploy the application online.
* Add advanced visualizations and dashboards.

## Author

Anand Yadav

## License

This project is intended for educational and internship purposes.

# Loan Approval Prediction Project

Predicts whether a loan application will be approved or not using machine learning.

GitHub URL : https://github.com/harekrishna10/loan-approval-prediction-project

Streamlit Web app Live Demo : 

## Dataset

- `train_u6lujuX_CVtuZ9i.csv` — 614 rows, 13 columns
- Target: `Loan_Status` (1 = approved, 0 = rejected)

## Steps

1. Loaded the dataset and inspected it
2. Data cleaning — removed duplicates, filled missing values (mode for categoricals, median for numericals)
3. Data transformation — encoded categorical variables, dropped Loan_ID
4. EDA — target distribution, approval rates, correlation heatmap
5. Feature selection — created Total_Income, dropped redundant income columns
6. Model building — Logistic Regression, Random Forest, Decision Tree algorithm used for finding the best model.
7. Final pipeline — StandardScaler + Logistic Regression, tuned with GridSearchCV (Best C = YOUR_VALUE, test accuracy YOUR_VALUE), saved as `loan_model.pkl`
8. Streamlit web app (`app.py`) for live predictions

## Model Results


Final tuned model test accuracy: **0.8537**

## How to Run

1. Install packages: `pip install -r requirements.txt`
2. Open `loan_prediction.ipynb` and run all cells
3. Creat new Terminal
4. Start the app: `streamlit run app.py`

## Adding to GitHub

1. git init
2. git add .
3. git commit -m "Loan approval prediction project"
4. git branch -M main
5. git remote add origin https://github.com/harekrishna10/loan-approval-prediction-project.git
6. git push -u origin main

## Adding to Streamlit web app

1. go to https://share.streamlit.io/
2. click on New app
3. Select the github account or sign up with github account
4. Deploy app form will appear
5. In Repo : choose harekrishna10/loan-approval-prediction-project
6. In Branch : choose main
7. In Main file path : choose app.py
8. In App URL : give desired url for you live streamlit web app like: loan-approval-prediction-project-10
9. click on Deploy button
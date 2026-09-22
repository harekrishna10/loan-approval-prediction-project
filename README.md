# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be
approved, based on applicant details such as income, credit history and
property area.

## Dataset

- File: `train_u6lujuX_CVtuZ9i.csv`
- 614 rows × 13 columns
- Target column: `Loan_Status` (Y = approved, N = rejected)

## Project Steps

1. **Data collection** – loaded the labelled loan dataset
2. **Data cleaning** – checked duplicates; filled missing values
   (mode for categorical columns, median for LoanAmount and Loan_Amount_Term)
3. **Encoding** – binary columns and target encoded to 0/1,
   Property_Area one-hot encoded, Loan_ID dropped
4. **EDA** – target distribution, approval rates by category,
   numerical distributions and correlation heatmap
5. **Feature engineering** – created `Total_Income`
   (ApplicantIncome + CoapplicantIncome); dropped the two original
   income columns as redundant
6. **Model development** – 80/20 stratified split (random_state=42);
   compared Logistic Regression, Decision Tree and Random Forest
7. **Evaluation & tuning** – final model: linear SVM
   (StandardScaler + SVC(kernel="linear", probability=True));
   evaluated with confusion matrix, precision, recall, F1, ROC-AUC,
   GridSearchCV and 5-fold cross-validation
8. **Deployment** – Streamlit web app (`app.py`) loads `loan_model.pkl`
   and predicts approval with confidence score

## Model Results (Step 6 comparison, test set)

- Logistic Regression: 0.8537
- Random Forest: 0.8455
- Decision Tree: 0.6992 (overfit: train accuracy 1.0000)

## How to Run

pip install streamlit scikit-learn pandas numpy
streamlit run app.py
Then open http://localhost:8501 in the browser.

## Adding to the github

git init
git add .
git commit -m "Loan approval prediction project"
git remote add origin https://github.com/harekrishna10/loan-approval-prediction-project.git
git branch -M main
git push -u origin main

## Deploying to streamlit web app

1. Go to share.streamlit.io and sign in with your GitHub account
2. Click Create app
3. github-deploy now
4. repository: select github repository
5. Branch: main
6. Main file path: app.py
7. app url: give app url as of your choice
8. save and deploy
9. wait a few minutes — you'll get a public URL like https://loan-approval-prediction-project10.streamlit.app/

## Live Demo

Deployed on Streamlit Community Cloud: https://loan-approval-prediction-project10.streamlit.app/

# Disease Prediction from Medical Data

## Overview
An educational machine-learning classification project using the UCI Heart Disease (Cleveland) dataset. The project predicts a binary disease-positive/disease-negative class from structured medical attributes.

> **Disclaimer:** This project is for educational/research purposes only. It is not a medical diagnosis system and should not be used to make medical decisions.

## Problem Statement
Develop a machine learning classification system that analyzes structured medical data and predicts whether a patient belongs to a disease-positive or disease-negative class.

## Objective
- Inspect and clean a public medical dataset.
- Perform exploratory data analysis.
- Build leakage-aware preprocessing.
- Train Logistic Regression, SVM, Random Forest, and XGBoost.
- Evaluate using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Perform 5-fold stratified cross-validation.
- Tune hyperparameters with GridSearchCV.
- Inspect tree-based feature importance.
- Save an end-to-end preprocessing + model pipeline.

## Dataset
**UCI Heart Disease Dataset (Cleveland data)**

The notebook downloads the dataset using the `ucimlrepo` package. The original target is `num`, with values 0–4. For this binary project, `0` is mapped to negative and values greater than 0 are mapped to positive.

The notebook checks the actual schema before processing rather than inventing column names.

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Google Colab / Jupyter Notebook

## Algorithms
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest Classifier
4. XGBoost Classifier

## Workflow
1. Load dataset
2. Inspect shape, columns, types, missing values, duplicates and statistics
3. Clean duplicate rows and missing markers
4. Convert original target to binary
5. Explore class and feature distributions
6. Separate X and y
7. Perform an 80/20 stratified train-test split
8. Apply imputation, one-hot encoding and scaling in a pipeline
9. Train four classifiers
10. Evaluate test performance
11. Plot confusion matrices and ROC curves
12. Compare models
13. Run 5-fold cross-validation
14. Tune hyperparameters using training data only
15. Inspect feature importance
16. Select a final model using a predefined validation criterion
17. Demonstrate sample prediction
18. Save and reload the model

## How to Run in Google Colab
1. Open Google Colab.
2. Upload `notebooks/Disease_Prediction.ipynb`.
3. Run cells from top to bottom.
4. The notebook installs `ucimlrepo` and XGBoost if required.
5. The UCI dataset is fetched programmatically.

If using your own legitimate CSV instead, upload it through Colab and replace the dataset-loading cell. Do not fabricate medical data.

## Results
The notebook generates the actual results when executed. Results are intentionally not hard-coded because model performance depends on the exact data and preprocessing used.

The comparison includes:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 5-fold cross-validation results

## Sample Prediction
The notebook provides a function that accepts feature values and returns:
- `Model prediction: Positive/Negative`
- predicted probability of the positive class

It does **not** state that a person has a disease.

## Model File
After execution, the notebook saves:
`models/disease_prediction_model.pkl`

The saved file contains the complete preprocessing + final model pipeline.

## Limitations
- Small dataset relative to real-world medical datasets.
- Dataset-specific performance may not generalize.
- Binary target is a simplification of the original target.
- No clinical validation is performed.
- Probabilities may require calibration.
- External validation is necessary for any real-world application.

## Future Improvements
- Larger and more diverse datasets
- Independent external validation
- Class-imbalance strategies
- Broader hyperparameter optimization
- SHAP/explainable AI
- Probability calibration
- Streamlit interface
- Monitoring for data-distribution changes


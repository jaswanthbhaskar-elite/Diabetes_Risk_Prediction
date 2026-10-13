# 🩺 Diabetes Risk Prediction Using Machine Learning

## Overview

Diabetes Risk Prediction is a machine learning project that predicts whether a person is likely to be classified as diabetic based on health-related input features. The project uses two supervised machine learning algorithms — **Logistic Regression** and **Random Forest** — and provides an interactive web application built with Streamlit.

The application allows users to enter health information, generate predictions, view estimated positive-class probabilities, and compare the outputs of both models.

## Objectives

- Analyze and understand the dataset through Exploratory Data Analysis (EDA).
- Perform data cleaning and preprocessing.
- Train and evaluate multiple machine learning classification models.
- Compare model performance using standard evaluation metrics.
- Build a user-friendly web interface for interactive predictions.

## Dataset

The project uses a diabetes prediction dataset containing approximately **96,146 records after cleaning**.

The application uses the following input features:

- Gender
- Age
- Hypertension
- Heart Disease
- Smoking History
- BMI (Body Mass Index)
- HbA1c Level
- Blood Glucose Level

**Target:** Diabetes classification (positive or negative).

## Machine Learning Algorithms

### 1. Logistic Regression
A supervised classification algorithm that estimates the probability of a binary outcome using a logistic function.

### 2. Random Forest
An ensemble learning algorithm that combines predictions from multiple decision trees to perform classification.

## Exploratory Data Analysis

EDA was performed in Google Colab to understand the dataset before model training. The analysis included:

- Dataset structure and data types
- Descriptive statistics
- Data quality and duplicate checks
- Categorical feature distributions using bar charts
- Numerical feature distributions using histograms
- Correlation analysis using a heatmap

## Model Evaluation

The models were evaluated using a held-out test set and cross-validation.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 95.96% | 96.94% |
| Precision | 86.92% | 95.04% |
| Recall | 63.86% | 68.87% |
| F1-score | 73.62% | 79.86% |
| ROC-AUC | 95.99% | 95.62% |
| Mean CV Accuracy | 95.87% | 96.82% |

Based on these evaluation results, Random Forest achieved higher accuracy, precision, recall, and F1-score, while Logistic Regression achieved a slightly higher ROC-AUC.

## Application Workflow

1. The user enters the required health-related information.
2. The application loads the previously trained machine learning pipelines.
3. The selected model generates a binary prediction and estimated probability.
4. The application displays the result.
5. Users can compare the outputs of Logistic Regression and Random Forest.

## Technologies Used

- **Python** — Programming language
- **Google Colab** — Data analysis and model training
- **Pandas and NumPy** — Data manipulation
- **Matplotlib and Seaborn** — Data visualization
- **Scikit-learn** — Preprocessing, model training, and evaluation
- **Joblib** — Saving and loading trained model pipelines
- **Streamlit** — Interactive web application
- **Visual Studio Code** — Local application development

## Project Files

- `app.py` — Streamlit application
- `logistic_regression_pipeline.pkl` — Saved Logistic Regression pipeline
- `random_forest_pipeline.pkl` — Saved Random Forest pipeline
- `feature_metadata.pkl` — Feature metadata used by the application
- `requirements.txt` — Python dependencies, if included
- `*.ipynb` — Jupyter notebook containing data analysis and model development, if included

## Installation and Execution

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 3. Run the application

Ensure that `app.py` and the required `.pkl` files are in the expected locations, then execute:

```bash
streamlit run app.py
```

The application will open in your browser.

## Limitations

- Model predictions depend on the quality and representativeness of the training dataset.
- Performance on the test dataset does not guarantee equivalent performance on new populations.
- Predicted probabilities are model estimates and should not be interpreted as clinically validated individual risk estimates.
- The application is intended for educational and project demonstration purposes, not medical diagnosis.

## Future Enhancements

- Evaluate the models on more diverse datasets.
- Improve probability calibration and assess performance across different groups.
- Add more detailed model interpretation and visualization.
- Deploy the application to a publicly accessible hosting platform.

## Disclaimer

This project is developed for educational and research demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

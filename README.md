# Heart Disease Modeling Project 🫀

**Data Science Fundamentals with Python**  
**Project Duration:** June 2025 – July 2025  
**Dataset:** [UCI Heart Disease Dataset (Cleveland subset)](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## 📌 Overview

This project explores key data science concepts including **classification, regression, dimensionality reduction**, and **unsupervised clustering** using real-world clinical data. The objective is to develop models that predict heart disease and analyze patient health profiles.

---

## 🧠 Key Concepts Applied

- Exploratory Data Analysis (EDA)
- Data Preprocessing (Imputation, Normalization, Encoding)
- Supervised Learning (Logistic Regression, Random Forest)
- Regression Analysis (Multiple Linear Regression)
- Dimensionality Reduction (Principal Component Analysis - PCA)
- Unsupervised Learning (KMeans Clustering)

---

## 📂 Dataset Details

- **Source:** UCI Machine Learning Repository
- **Subset Used:** Cleveland
- **Instances:** 303
- **Features:** 13 clinical features + 1 target
- **Target:** Presence or absence of heart disease (binary classification)

---

## 🧪 Project Workflow

### 1. EDA & Data Preprocessing
- Identified and handled missing values via median imputation.
- Converted multi-class target to binary (0: no disease, 1–4: disease).
- Normalized features using `StandardScaler`.

### 2. Heart Disease Prediction
- Built classification models:
  - Logistic Regression
  - Random Forest Classifier
- Evaluated performance using:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix

### 3. Cholesterol Level Prediction
- Built a multiple linear regression model.
- Predicted serum cholesterol using 12 features.
- Plotted correlation matrix and Seaborn heatmap.
- Analyzed feature influence on cholesterol levels.

### 4. Principal Component Analysis (PCA)
- Reduced dataset dimensionality while retaining 95% variance.
- Visualized explained variance ratios.
- Used PCA output for clustering.

### 5. KMeans Clustering for Patient Profiling
- Applied KMeans clustering on PCA-reduced dataset.
- Determined optimal cluster count using:
  - Elbow Method
  - Silhouette Score
- Visualized clusters using first two PCA components.

---

## 📊 Results & Insights

- Random Forest outperformed Logistic Regression on classification metrics.
- Certain features (e.g., age, chest pain type) were strongly correlated with cholesterol levels.
- PCA efficiently reduced data noise while preserving variance.
- KMeans revealed 3–4 distinct patient clusters based on health patterns.

---

## 🛠️ Tools & Libraries

- Python, Jupyter Notebook  
- Pandas, NumPy, Matplotlib, Seaborn  
- scikit-learn (LogisticRegression, RandomForest, PCA, KMeans, metrics)

---

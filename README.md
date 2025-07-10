# ml_heart_disease_modeling
Data Science Fundamentals with Python
Heart Disease Modeling Project
July 2025
1 Introduction
This project uses the UCI Heart Disease dataset to explore core data science techniques: classification, regression, dimensionality reduction, and unsupervised clustering. You’ll apply everything you’ve learned to analyze patterns
related to heart disease using real clinical data.
2 Dataset Reference
Use the Cleveland dataset available here. It consists of 303 instances, 13 clinical
features and one numerical target (representing the presence or absence of heart
disease).
• Read the dataset description carefully to understand what the 14 different
features represent.
• Use the data loading code provided in the UCI website and import the
dataset directly onto your Jupyter notebook.
3 Tasks and Instructions
3.1 EDA & Data Preprocessing
This is an integral step in any Machine Learning task.
• Start off with Exploratory Data Analysis by looking at the first few
rows, checking out the number of null values, running df.describe(), df.info(),
etc.
• Handle missing values through imputation. Fill the empty cells with the
median of the remaining values in that particular column.
• Convert the num column in the target dataset to binary:
– 0 remains 0 (no disease)
– 1-4 should all be replaced with 1 (presence of disease)
1
• Normalize all the features using the StandardScaler tool from the sklearn.preprocessing
module.
3.2 Heart Disease Prediction
The goal of this task is to create a machine learning model trained on the
given dataset that would predict whether a person has heart disease or not.
Supervised classification algorithms will be used for this purpose:
• Train two models (scikit-learn implementation) after splitting the dataset
into train and test parts:
– Logistic regression
– Random Forest Classifier
• Evaluate and compare the two using:
– Accuracy
– Precision, Recall, F1-Score
– Confusion Matrix
3.3 Cholesterol Level Prediction
In this task, we will build a multiple linear regression model to predict the
serum cholesterol based on the remaining 12 features and also analyze the
major contributing factors to cholesterol:
• Ensure that all the features but the serum cholesterol level are normalized.
• Using scikit-learn, implement a Linear Regression model after splitting
the dataset into train and test datasets.
• Plot a correlation matrix using df.corr() in pandas and visualize a
heatmap using seaborn.heatmap()
• Comment on which features are most correlated with serum cholesterol.
3.4 Principal Component Analysis
The goal of this task is to reduce the dataset’s dimensionality while retaining
most of the variance, for future unsupervised learning tasks.
• Ensure that all the features are normalized.
• Implement PCA from the sklearn.decomposition module.
– Set parameters such that 95% of the variance is retained (feel free to
play around with this value).
• Plot the explained variance ratio for each component.
• Output the shape of the reduced dataset.
2
3.5 Grouping Patients based on Health Profiles
Now we intend to perform K-means clustering, an unsupervised learning
algorithm, on the PCA-reduced dataset, thus grouping patients based on their
health profile.
• Apply KMeans from the sklearn.cluster module on the PCA-reduced dataset.
• Use the Elbow method and the Silhouette score to determine the
optimum number of of clusters.
• Visualize the clusters on a 2D PCA scatterplot by taking only the first
two components.
4 Submission Requirements
• A single Jupyter notebook with clean code and comments.
• Segment the notebook by titling sections in markdown cells
• Further, use markdown cells for your inferences/explanations.
3

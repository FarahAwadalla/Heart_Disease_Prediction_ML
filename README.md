Heart Disease Prediction Project

This project applies machine learning techniques to predict heart disease using the UCI Heart Disease dataset. It covers preprocessing, exploratory data analysis, dimensionality reduction, feature selection, supervised and unsupervised learning, hyperparameter tuning, and model deployment.

📂 Project Structure: 
heart-disease-classification/
│── data/
    ├── heart_disease.csv
│── notebooks/                  
│   ├── 1_preprocessing.ipynb
│   ├── 2_pca.ipynb
│   ├── 3_feature_selection.ipynb
│   ├── 4_supervised_models.ipynb
│   ├── 5_clustering.ipynb
│   ├── 6_hyperparameter_tuning.ipynb
│   ├── 7_model_export.ipynb
│── models/
│   ├── heart_disease_svm_model.pkl
│── results/                    
│   ├── evaluation_metrics.txt
│── requirements.txt
│── README.md
│── .gitignore


⚙️ Workflow:

Data Preprocessing → handle missing values, encode categoricals, scale numerical features.

Exploratory Data Analysis (EDA) → histograms, correlation heatmaps, boxplots.

Dimensionality Reduction (PCA) → reduce features while retaining variance.

Feature Selection → Random Forest importance, RFE, Chi-Square test.

Supervised Learning → Logistic Regression, Decision Tree, Random Forest, SVM.

Unsupervised Learning → K-Means, Hierarchical Clustering.

Hyperparameter Tuning → GridSearchCV and RandomizedSearchCV.

Model Export → final SVM model saved as .pkl pipeline with preprocessing.


🏆 Results:

Best Model: Support Vector Machine (SVM).

Performance:

Accuracy: ~90%

Precision: ~0.89

Recall: ~0.89

F1-score: ~0.89

Logistic Regression also performed strongly (high Recall), but SVM gave the best balance.



🚀 Usage:

Clone the repository and install dependencies:

git clone https://github.com/FarahAwadalla/Heart_Disease_Prediction_ML.git
cd Heart_Disease_Prediction_ML
pip install -r requirements.txt


Run Jupyter notebooks:

jupyter notebook


Load the trained model for predictions:

import joblib
model = joblib.load("models/heart_disease_svm_model.pkl")
prediction = model.predict(sample_data)


📦 Requirements:

Main dependencies (see requirements.txt for full list):

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib

scipy


👩‍💻 Author:

Project developed as part of a AI & machine learning Sprints training program on heart disease prediction.
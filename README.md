# 🔒 AI for Security: IoT23 Network Intrusion Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📝 Project Description

Final project for the **AI for Security** course, focused on analyzing the **IoT-23** dataset for intrusion detection and threat identification in IoT network security. The project implements a comprehensive analysis using advanced Machine Learning techniques, including supervised learning, unsupervised learning, and anomaly detection.

### 🎯 Objectives

- In-depth analysis of IoT network traffic
- Detection of attacks and anomalies using ML algorithms
- Comparison of different classification and clustering techniques
- Performance evaluation of models on real-world data

## 👥 Team: CyberCats

- **Gabriella Bertolino**
- **Guendalina Nardi**
- **Domenico Pittari**
- **Sofia Varotto**

## 📊 Dataset

The project uses the **IoT-23** dataset, a collection of network traffic captured from real IoT devices, containing both benign and malicious traffic. It also includes a comparative analysis with the **CIC-IDS2017** dataset.

## 🛠️ Technologies & Tools

- **Python 3.8+**
- **Jupyter Notebook**
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine Learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **XGBoost** - Advanced gradient boosting

## 🔬 Implemented Methodologies

### 1. **Supervised Learning**

#### Linear Models:
- Naive Bayes (with and without undersampling)
- Support Vector Machine (SVM)

#### Non-Linear Models:
- SVM with RBF kernel
- K-Nearest Neighbors (KNN) with Grid Search
- Decision Tree with optimization
- Random Forest with Grid Search
- Multi-Layer Perceptron (MLP)

### 2. **Boosting**
- AdaBoost
- Gradient Boosting
- XGBoost

### 3. **Unsupervised Learning**

#### Clustering:
- K-Means (with Elbow Method and iterative analysis)
- Hierarchical Clustering (with dendrograms)
- DBSCAN (Density-based clustering)
- t-SNE for dimensionality reduction

### 4. **Anomaly Detection**
- Local Outlier Factor (LOF)
- Isolation Forest
- One-Class SVM
- Elliptic Envelope

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve and AUC
- Homogeneity Score
- Calinski-Harabasz Index
- Silhouette Score

## 📊 Key Results

The notebook includes:
- ✅ Exploratory Data Analysis (EDA)
- ✅ Comparative model visualizations
- ✅ Comparison between supervised and unsupervised algorithms
- ✅ Performance analysis on alternative datasets
- ✅ Conclusions and recommendations


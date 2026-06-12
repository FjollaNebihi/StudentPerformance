# Student Academic Performance Classification and Clustering

This project explores student academic performance using machine learning techniques. The goal is to analyze real-world data, implement multiple classification and clustering algorithms, and compare their performance.

## Project Overview

**Objective:** Gain practical experience implementing machine learning algorithms on real data by building classification models, performing feature engineering, and applying unsupervised learning techniques.

**Dataset:** Students_Academic_Performance_Evaluation_Dataset - contains academic and behavioral factors related to student performance.

---

## Requirements Fulfillment Checklist

### ✅ Classification (Section 1)

#### 1. Four Classifiers Implemented
All classifiers use:
- **Train/Test Split:** 80/20 stratified split (ensures class distribution is preserved)
- **Data Preprocessing:** 
  - Numeric features: Median imputation + StandardScaler normalization
  - Categorical features: Most frequent imputation + OneHotEncoder
- **Feature Selection:** SelectKBest (f_classif scoring) with experimentation on k values
- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation (F1-macro scoring)

**Classifiers Implemented:**

1. **KNN (k-Nearest Neighbors)** - `notebooks/classificators/knn_01.ipynb`
   - Distance-based approach
   - Hyperparameters tuned: n_neighbors, weights, metric, k (feature selection)
   - Evaluation metrics: Accuracy, Precision (macro), Recall (macro), F1-score (macro), Confusion matrix

2. **Logistic Regression** - `notebooks/classificators/logistic_regression_03.ipynb`
   - Linear approach
   - Hyperparameters tuned: C, penalty, solver, k (feature selection)
   - Evaluation metrics: Accuracy, Precision (macro), Recall (macro), F1-score (macro), Confusion matrix

3. **Neural Network (MLP)** - `notebooks/classificators/neural_network_04.ipynb`
   - Deep learning approach
   - Architecture experiments: 2-layer (64, 32) vs 3-layer (128, 64, 32)
   - Activation functions tested: ReLU, Tanh
   - Hyperparameters tuned: hidden_layer_sizes, activation, alpha (L2 regularization), learning_rate_init, k (feature selection)
   - Early stopping enabled to prevent overfitting
   - Evaluation metrics: Accuracy, Precision (macro), Recall (macro), F1-score (macro), Confusion matrix

4. **Random Forest** - `notebooks/classificators/random_forest_02 (2).ipynb`
   - Decision tree ensemble approach
   - Hyperparameters tuned: n_estimators, max_depth, min_samples_split, k (feature selection)
   - Evaluation metrics: Accuracy, Precision (macro), Recall (macro), F1-score (macro), Confusion matrix

#### 2. Feature Selection & Experimentation
- **Method:** SelectKBest with f_classif scoring
- **Tested k values:** 3, 5, 7, 10, and "all" (to evaluate if feature reduction improves performance)
- **Results:** Best k value is selected via GridSearchCV optimization
- **Documentation:** Each classifier notebook documents which features were selected and how many

#### 3. Hyperparameter Tuning
- **Method:** GridSearchCV with 5-fold cross-validation
- **Scoring Metric:** F1-macro (to handle multi-class imbalance)
- **Best Parameters Documented:** Stored in results CSV files for each classifier
- **Tested Configurations:** Comprehensive parameter grids specific to each algorithm

#### 4. Evaluation Metrics
All classifiers evaluated using:
- **Accuracy:** Overall correctness
- **Precision (macro):** True positives across all classes (unweighted average)
- **Recall (macro):** Sensitivity across all classes (unweighted average)
- **F1-Score (macro):** Harmonic mean of precision and recall across all classes
- **Confusion Matrix:** Visual representation of classification errors
- **Classification Report:** Detailed per-class metrics

#### 5. Comparison & Results
- **Comparison Notebook:** `notebooks/classificators/compare_classifiers.ipynb`
- **Merged Results:** All classifier results combined and sorted by F1-macro score
- **Output File:** `results/all_classifiers_comparison.csv`
- **Visualization:** Bar chart comparing metrics across all classifiers

---

### ✅ Clustering (Section 2)

#### 1. K-Means Clustering - `notebooks/clustering/kmeans_01.ipynb`
- **Parameter Experimentation:**
  - Tested k values: 3, 4, 5, 6
  - Initialization: k-means++ (deterministic and faster convergence)
- **Evaluation Metrics:**
  - Silhouette Score (cohesion and separation)
  - Adjusted Rand Index (ARI - comparison with true labels)
  - Normalized Mutual Information (NMI - comparison with true labels)
- **Visualization:** PCA-based 2D scatter plot showing cluster boundaries
- **Findings:** Silhouette score used to select optimal k

#### 2. Hierarchical Clustering - `notebooks/clustering/hierarchical_02.ipynb`
- **Parameter Experimentation:**
  - Number of clusters: 4, 5, 6
  - Linkage methods: ward (minimizes within-cluster variance), average (average distance), complete (maximum distance)
- **Evaluation Metrics:**
  - Silhouette Score
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
- **Visualization:** PCA-based 2D scatter plot showing cluster boundaries
- **Findings:** Best configuration selected based on silhouette score

#### 3. Data Preprocessing for Clustering
- Labels removed before clustering (unsupervised learning)
- Same preprocessing pipeline as classifiers:
  - Median imputation for numeric features
  - StandardScaler normalization
- **Comparison with True Labels:** ARI and NMI metrics compare clustering results with ground truth labels

---

## Project Structure

```
StudentPerformance/
├── README.md                           # This file - comprehensive project documentation
├── data/
│   └── Student_performance_data _.csv  # Dataset (educational records)
├── notebooks/
│   ├── classificators/
│   │   ├── knn_01.ipynb               # KNN classifier with tuning
│   │   ├── logistic_regression_03.ipynb # Logistic Regression with tuning
│   │   ├── neural_network_04.ipynb    # MLP with architecture experiments
│   │   ├── random_forest_02 (2).ipynb # Random Forest with tuning
│   │   └── compare_classifiers.ipynb  # Comparison of all classifiers
│   └── clustering/
│       ├── kmeans_01.ipynb            # K-Means with parameter tuning
│       ├── hierarchical_02.ipynb      # Hierarchical clustering with experiments
│       └── visualizations/            # Cluster visualizations
├── results/
│   ├── knn_results.csv                # KNN performance metrics
│   ├── logistic_regression_results.csv # LR performance metrics
│   ├── neural_network_results.csv     # MLP performance metrics
│   ├── random_forest_results.csv      # RF performance metrics
│   ├── all_classifiers_comparison.csv # Merged comparison table
│   ├── kmeans_results.csv             # K-Means evaluation metrics
│   └── hierarchical_results.csv       # Hierarchical evaluation metrics
├── visualizations/
│   ├── *_confusion_matrix.png         # Confusion matrices for each classifier
│   ├── classifiers_comparison.png     # Bar chart comparing all classifiers
│   ├── kmeans_clusters_pca.png        # K-Means clusters in 2D (PCA)
│   └── hierarchical_clusters_pca.png  # Hierarchical clusters in 2D (PCA)
└── documentation/
    └── student_performance_report.docx # Detailed findings report

```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## Running the Project

### 1. Execute All Classifiers (in order)
```bash
# KNN
jupyter notebook notebooks/classificators/knn_01.ipynb

# Logistic Regression
jupyter notebook notebooks/classificators/logistic_regression_03.ipynb

# Neural Network
jupyter notebook notebooks/classificators/neural_network_04.ipynb

# Random Forest
jupyter notebook notebooks/classificators/random_forest_02\ \(2\).ipynb
```

### 2. Compare Classifiers
```bash
jupyter notebook notebooks/classificators/compare_classifiers.ipynb
```

### 3. Execute Clustering Algorithms (in order)
```bash
# K-Means
jupyter notebook notebooks/clustering/kmeans_01.ipynb

# Hierarchical
jupyter notebook notebooks/clustering/hierarchical_02.ipynb
```

---

## Key Findings

### Classification Results
- **Best Performer:** Determined by F1-macro score from `results/all_classifiers_comparison.csv`
- **Feature Selection Impact:** SelectKBest reduced features while maintaining/improving performance
- **Neural Network Architecture:** 3-layer network (128, 64, 32) with ReLU activation achieved best generalization

### Clustering Results
- **Optimal k (K-Means):** Selected based on silhouette score
- **Best Linkage (Hierarchical):** Ward linkage provided better cluster separation
- **Ground Truth Alignment:** Adjusted Rand Index and NMI show how well unsupervised clusters match actual class labels

---

## Methodology Notes

### Cross-Validation Strategy
- 5-fold stratified cross-validation ensures class balance in each fold
- F1-macro chosen as primary metric due to multi-class nature and potential class imbalance

### Feature Engineering
- SelectKBest with f_classif ensures only statistically significant features are retained
- Reduces model complexity and improves generalization

### Data Preprocessing
- Standardization prevents features with larger scales from dominating distance-based algorithms
- Median imputation chosen over mean to handle potential outliers
- OneHotEncoder with handle_unknown="ignore" handles categorical features safely

### Model Validation
- Train/test split (80/20) provides unbiased performance estimation
- Confusion matrices reveal specific class-wise misclassification patterns
- Classification reports show per-class precision, recall, and F1-scores

---

## References & Algorithms

1. **K-Nearest Neighbors (KNN):** Instance-based learning, distance-based similarity
2. **Logistic Regression:** Linear classification, probability-based decision boundaries
3. **Neural Networks (MLP):** Deep learning, non-linear feature transformations
4. **Random Forest:** Ensemble method, bagging with decision trees
5. **K-Means Clustering:** Unsupervised partitioning, centroid-based grouping
6. **Hierarchical Clustering:** Agglomerative approach, dendrograms, multiple linkage methods

---

## Author
Machine Learning Project - Student Performance Analysis

## Date Completed
2026
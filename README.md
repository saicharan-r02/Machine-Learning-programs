# 🏥 HealthPredict AI — Clinical Disease Risk & Patient Prognosis Prediction Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-emerald.svg)]()

An end-to-end clinical machine learning system designed to predict cardiovascular and metabolic disease risk using 13 clinical biomarkers, hemodynamics, and lifestyle factors. Features automated multi-model benchmarking, hyperparameter optimization, ROC-AUC diagnostic analysis, real-time risk triage inference, and an interactive HTML analytics dashboard.

---

## 📌 Project Highlights

- **Multi-Model Benchmark Architecture**: Trains and cross-evaluates Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), and Gradient Boosting Classifiers.
- **Hyperparameter Optimization**: Grid search cross-validation on ensemble estimators for maximum ROC-AUC discrimination.
- **Clinical Risk Stratification**: Maps risk probabilities to standard clinical tiers (`Low`, `Moderate`, `High`, `Critical`) with automated root-cause risk factor identification.
- **Diagnostic Visualizations**: High-resolution exports for ROC Curves, Confusion Matrices, Gini Feature Importance, and Model Comparison.
- **Interactive Executive Dashboard**: Self-contained HTML report with glassmorphic UI and responsive metrics.
- **Archived Practice Library**: Historical daily exercises organized in [`practice_exercises/`](practice_exercises/).

---

## 🏗️ System Architecture

```
Machine-Learning/
├── practice_exercises/          <- Daily problem exercises & foundational scripts
├── data/
│   └── patient_health_records.csv
├── src/
│   ├── data_loader.py           <- Clinical dataset synthesis & StandardScaler pipeline
│   ├── model_trainer.py         <- Model training & GridSearchCV tuning
│   ├── evaluator.py             <- ROC-AUC, Confusion Matrix, Feature Importance plots
│   ├── predict.py               <- Real-time inference & batch CSV prediction
│   └── generate_report.py       <- Standalone HTML dashboard compiler
├── models/
│   ├── best_model.joblib        <- Production champion model artifact
│   └── scaler.joblib            <- Fitted StandardScaler artifact
├── reports/
│   ├── figures/                 <- High-res diagnostic visualizations
│   │   ├── roc_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── model_comparison.png
│   └── health_analytics_dashboard.html
├── run_pipeline.py              <- Main single-command orchestrator
└── README.md
```

---

## 📊 Benchmark Performance Summary

| Algorithm Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned Gradient Boosting (Champion)** | **~88.5%** | **~87.2%** | **~89.1%** | **0.881** | **0.952** |
| Random Forest Classifier | ~87.2% | ~85.9% | ~88.0% | 0.869 | 0.946 |
| Support Vector Machine (RBF) | ~86.0% | ~85.1% | ~86.2% | 0.856 | 0.938 |
| Logistic Regression (Balanced) | ~84.5% | ~82.8% | ~86.0% | 0.844 | 0.925 |
| Decision Tree | ~81.0% | ~79.5% | ~82.0% | 0.807 | 0.865 |

---

## 🚀 Quick Start

### 1. Run the Full Machine Learning Pipeline
```bash
python run_pipeline.py
```

### 2. Perform Live Patient Risk Prediction
```bash
python src/predict.py
```

### 3. Open the Interactive HTML Dashboard
Open `reports/health_analytics_dashboard.html` in your browser.

---

## 💡 Clinical Biomarkers Evaluated
- **Demographics & Vitals**: Age, Gender, Systolic BP, Diastolic BP, Resting Heart Rate.
- **Lipids & Metabolic Panel**: Total Cholesterol, HDL, LDL, Fasting Blood Glucose.
- **Lifestyle & Risk Factors**: BMI, Smoking Status, Physical Activity Hours/Week, Family History.

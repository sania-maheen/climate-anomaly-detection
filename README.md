

#  X-Climate: Explainable AI for Climate Anomaly Detection

##  Project Overview

**X-Climate** is a machine learning system designed to detect and analyze **climate anomalies** using historical meteorological data. The project combines **ensemble machine learning models** with **Explainable AI (XAI)** techniques to make predictions more transparent and interpretable.

The system identifies abnormal climate events such as:

*  Heatwaves
*  Cold Waves
*  Heavy Rainfall
*  Normal Conditions

To improve interpretability, the project integrates:

* **SHAP (SHapley Additive Explanations)** for global model interpretation
* **LIME (Local Interpretable Model-Agnostic Explanations)** for explaining individual predictions

An interactive **Streamlit dashboard** is provided to visualize anomalies, compare model performance, and explore explanations.

---

#  Objectives

* Detect climate anomalies using machine learning
* Compare ensemble models for anomaly classification
* Improve transparency of predictions using Explainable AI
* Provide an interactive dashboard for climate data analysis

---

#  Technologies Used

* **Python**
* **Scikit-Learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Streamlit**
* **SHAP**
* **LIME**
* **Joblib**

---

#  Dataset

The model is designed to work with **climate datasets containing meteorological variables** such as temperature, humidity, wind speed, and precipitation.

For demonstration purposes, the project uses a **historical climate dataset** that includes measurements such as:

| Feature     | Description               |
| ----------- | ------------------------- |
| T2M_MAX     | Maximum temperature       |
| T2M_MIN     | Minimum temperature       |
| T2M         | Average temperature       |
| RH2M        | Relative humidity         |
| WS2M        | Wind speed                |
| PRECTOTCORR | Precipitation             |
| MONTH       | Month extracted from date |

Although an example dataset was used during development, the framework **can be applied to climate datasets from any region with similar features**.

---

#  Project Workflow

## 1️⃣ Data Preprocessing

* Load raw climate dataset
* Convert **YEAR + DOY** into a proper date
* Handle missing values
* Extract **month information**
* Perform feature engineering using **monthly Z-score deviation**

---

## 2️⃣ Feature Engineering

Deviation features are created to detect unusual climate conditions:

```
Deviation = (Value − Monthly Mean) / Monthly Standard Deviation
```

This helps identify when climate variables deviate significantly from normal seasonal patterns.

---

## 3️⃣ Anomaly Labeling

Climate anomalies are labeled using deviation thresholds.

| Label | Event              |
| ----- | ------------------ |
| 0     | Normal             |
| 1     | Heatwave           |
| 2     | Cold Wave          |
| 3     | Heavy Rainfall     |
| 4     | Drought (optional) |

---

## 4️⃣ Model Training

Two machine learning models are used:

### Random Forest

* Ensemble decision tree model
* Handles complex patterns in climate data
* Robust against overfitting

### Gradient Boosting

* Sequential boosting model
* Improves performance by learning from errors

Both models are evaluated using:

* Precision
* Recall
* F1-Score
* Classification Report

---

# 🔍 Explainable AI (XAI)

Understanding why models make predictions is important in climate analysis.

## SHAP – Global Explanations

SHAP shows how much each feature contributes to model predictions across the entire dataset.

Example insights:

* High **maximum temperature** contributes to heatwave predictions
* High **precipitation levels** contribute to heavy rainfall detection

---

## LIME – Local Explanations

LIME explains **individual predictions**.

It highlights which features influenced the model for a specific data instance, making predictions easier to interpret.

---

#  Streamlit Dashboard

The project includes an interactive **Streamlit dashboard** with multiple sections.

### Overview

* Dataset statistics
* Climate anomaly distribution
* Sample climate data

### Model Performance

* Random Forest vs Gradient Boosting comparison
* Classification reports
* F1-score visualization

### SHAP Explanations

* Global feature importance
* Detailed feature impact plots

### LIME Explanation

* Select a specific date
* View prediction probabilities
* See feature contributions

---

#  Learning Outcomes

Through this project we explored:

* Climate data preprocessing
* Feature engineering for anomaly detection
* Ensemble machine learning models
* Explainable AI techniques
* Interactive data visualization with Streamlit

---

#  Team Project

This project was developed as a **team mini-project** for academic purposes.

AI tools were used during development to assist with learning, experimentation, and implementation.

---

# Future Improvements

* Support real-time climate data
* Extend anomaly types (storms, droughts, extreme humidity)
* Deploy dashboard as a web application
* Integrate deep learning models
* Expand dataset to multiple geographic regions

---



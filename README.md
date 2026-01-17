# 🏏 IPL Player Performance Prediction

## 📌 Project Overview

This project focuses on predicting the expected number of runs scored by an IPL player using historical match data and recent performance trends. The model leverages ball-by-ball IPL data (2008–2025) and applies machine learning techniques to generate accurate predictions.

The project was developed as part of the **Infosys Springboard Internship Program**.

---

## 📂 Dataset

* **Source:** Kaggle – IPL Ball-by-Ball Dataset (2008–2025)
* **Files Used:**

  * `deliveries_updated_ipl_upto_2025.csv`
  * `matches_updated_ipl_upto_2025.csv`

---

## 🧹 Data Preprocessing

* Removed irrelevant and inconsistent records
* Handled missing values
* Converted date fields to datetime format
* Created match-level batting and bowling statistics

---

## ⚙️ Feature Engineering

Key features engineered for model training:

* Average runs in last 5 matches
* Average runs in last 10 matches
* Strike rate in last 5 matches
* Strike rate in last 10 matches
* Rolling averages to capture recent form

---

## 🤖 Machine Learning Model

* Model Type: Regression model
* Target Variable: Runs scored by a player in a match
* Training approach:

  * Train-test split
  * Feature scaling where required
  * Model evaluation using regression metrics

---

## 📊 Model Evaluation

The model performance was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

The results demonstrate the model’s ability to capture recent performance trends effectively.

---

## 🚀 Deployment

The trained model was deployed as an interactive **Streamlit web application**.

### Deployment Platform

* **Streamlit Community Cloud**

### Application Features

* User inputs recent performance metrics
* Model predicts expected runs
* Simple and interactive UI

---

## 🖥️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Repository Structure

```
├── app.py
├── final_runs_prediction_model.pkl
├── requirements.txt
├── README.md
```

---

## 🎯 Internship Milestones Covered

* Data collection and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model development and evaluation
* Model deployment using Streamlit

---

## 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow, from raw data processing to deployment of a predictive web application. It highlights practical data science skills applicable to real-world sports analytics problems.

---

## 👤 Author

**Sai Ram**
Infosys Springboard Intern
